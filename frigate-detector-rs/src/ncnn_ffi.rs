//! Direct ncnn inference via the C API (dlopen'd libncnn.so).
//!
//! Replaces the Python ncnn subprocess: no pipe hop, no numpy, no
//! interpreter. The library is loaded at runtime so the binary still
//! runs (falling back to the Python worker) when libncnn.so is absent.

use std::ffi::{c_char, c_int, c_void, CString};
use std::io;

use libloading::{Library, Symbol};

// Opaque C API handles
type NcnnOption = *mut c_void;
type NcnnNet = *mut c_void;
type NcnnExtractor = *mut c_void;
type NcnnMat = *mut c_void;
type NcnnAllocator = *mut c_void;

macro_rules! sym {
    ($lib:expr, $name:literal, $ty:ty) => {{
        let s: Symbol<$ty> = unsafe { $lib.get($name) }
            .map_err(|e| io::Error::new(io::ErrorKind::NotFound,
                format!("libncnn symbol {} missing: {e}", String::from_utf8_lossy($name))))?;
        unsafe { std::mem::transmute::<Symbol<$ty>, Symbol<'static, $ty>>(s) }
    }};
}

pub struct NcnnFfi {
    // symbols (bound for the lifetime of _lib, which we hold last)
    extractor_create: Symbol<'static, unsafe extern "C" fn(NcnnNet) -> NcnnExtractor>,
    extractor_destroy: Symbol<'static, unsafe extern "C" fn(NcnnExtractor)>,
    extractor_input:
        Symbol<'static, unsafe extern "C" fn(NcnnExtractor, *const c_char, NcnnMat) -> c_int>,
    extractor_extract:
        Symbol<'static, unsafe extern "C" fn(NcnnExtractor, *const c_char, *mut NcnnMat) -> c_int>,
    mat_create_external_3d: Symbol<
        'static,
        unsafe extern "C" fn(c_int, c_int, c_int, *mut c_void, NcnnAllocator) -> NcnnMat,
    >,
    mat_destroy: Symbol<'static, unsafe extern "C" fn(NcnnMat)>,
    mat_get_data: Symbol<'static, unsafe extern "C" fn(NcnnMat) -> *mut c_void>,
    mat_get_w: Symbol<'static, unsafe extern "C" fn(NcnnMat) -> c_int>,
    mat_get_h: Symbol<'static, unsafe extern "C" fn(NcnnMat) -> c_int>,
    mat_get_c: Symbol<'static, unsafe extern "C" fn(NcnnMat) -> c_int>,
    mat_get_elemsize: Symbol<'static, unsafe extern "C" fn(NcnnMat) -> usize>,

    net: NcnnNet,
    in_name: CString,
    out_name: CString,
    model_size: usize,
    f32_buf: Vec<f32>,

    // Drop order: net destroyed via net_destroy before the library unloads.
    net_destroy: Symbol<'static, unsafe extern "C" fn(NcnnNet)>,
    _lib: &'static Library,
}

// The C API net/extractor are only touched from the inference thread.
unsafe impl Send for NcnnFfi {}

fn lib_candidates() -> Vec<String> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("NCNN_LIB") {
        v.push(p);
    }
    v.push("/opt/frigate/libncnn.so".into());
    v.push("libncnn.so".into());
    v
}

impl NcnnFfi {
    pub fn new(param: &str, bin: &str, in_name: &str, out_name: &str, ms: u32) -> io::Result<Self> {
        let lib = lib_candidates()
            .iter()
            .find_map(|p| unsafe { Library::new(p).ok() })
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "libncnn.so not found"))?;
        // Leak the library: it must live for the process lifetime anyway,
        // and 'static symbols keep the borrow checker honest.
        let lib: &'static Library = Box::leak(Box::new(lib));

        let option_create = sym!(lib, b"ncnn_option_create", unsafe extern "C" fn() -> NcnnOption);
        let option_destroy = sym!(lib, b"ncnn_option_destroy", unsafe extern "C" fn(NcnnOption));
        let option_set_num_threads =
            sym!(lib, b"ncnn_option_set_num_threads", unsafe extern "C" fn(NcnnOption, c_int));
        let option_set_vulkan = sym!(
            lib,
            b"ncnn_option_set_use_vulkan_compute",
            unsafe extern "C" fn(NcnnOption, c_int)
        );
        let net_create = sym!(lib, b"ncnn_net_create", unsafe extern "C" fn() -> NcnnNet);
        let net_destroy = sym!(lib, b"ncnn_net_destroy", unsafe extern "C" fn(NcnnNet));
        let net_set_option =
            sym!(lib, b"ncnn_net_set_option", unsafe extern "C" fn(NcnnNet, NcnnOption));
        let net_load_param =
            sym!(lib, b"ncnn_net_load_param", unsafe extern "C" fn(NcnnNet, *const c_char) -> c_int);
        let net_load_model =
            sym!(lib, b"ncnn_net_load_model", unsafe extern "C" fn(NcnnNet, *const c_char) -> c_int);

        let net = unsafe { net_create() };
        let opt = unsafe { option_create() };
        unsafe {
            // 2 threads + passive OpenMP waiting: benchmarked optimal for the
            // CPU-fallback layers on this host (see GPU-DETECTOR.md).
            option_set_num_threads(opt, 2);
            option_set_vulkan(opt, 1);
            net_set_option(net, opt);
        }
        // Optional (Vulkan builds only): pin device 0 explicitly.
        if let Ok(set_dev) = unsafe {
            lib.get::<unsafe extern "C" fn(NcnnNet, c_int)>(b"ncnn_net_set_vulkan_device")
        } {
            unsafe { set_dev(net, 0) };
        }

        let c_param = CString::new(param).unwrap();
        let c_bin = CString::new(bin).unwrap();
        let rp = unsafe { net_load_param(net, c_param.as_ptr()) };
        let rm = unsafe { net_load_model(net, c_bin.as_ptr()) };
        unsafe { option_destroy(opt) };
        if rp != 0 || rm != 0 {
            unsafe { net_destroy(net) };
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("ncnn load failed (param={rp}, model={rm})"),
            ));
        }

        let ms = ms as usize;
        Ok(NcnnFfi {
            extractor_create: sym!(lib, b"ncnn_extractor_create",
                unsafe extern "C" fn(NcnnNet) -> NcnnExtractor),
            extractor_destroy: sym!(lib, b"ncnn_extractor_destroy",
                unsafe extern "C" fn(NcnnExtractor)),
            extractor_input: sym!(lib, b"ncnn_extractor_input",
                unsafe extern "C" fn(NcnnExtractor, *const c_char, NcnnMat) -> c_int),
            extractor_extract: sym!(lib, b"ncnn_extractor_extract",
                unsafe extern "C" fn(NcnnExtractor, *const c_char, *mut NcnnMat) -> c_int),
            mat_create_external_3d: sym!(lib, b"ncnn_mat_create_external_3d",
                unsafe extern "C" fn(c_int, c_int, c_int, *mut c_void, NcnnAllocator) -> NcnnMat),
            mat_destroy: sym!(lib, b"ncnn_mat_destroy", unsafe extern "C" fn(NcnnMat)),
            mat_get_data: sym!(lib, b"ncnn_mat_get_data",
                unsafe extern "C" fn(NcnnMat) -> *mut c_void),
            mat_get_w: sym!(lib, b"ncnn_mat_get_w", unsafe extern "C" fn(NcnnMat) -> c_int),
            mat_get_h: sym!(lib, b"ncnn_mat_get_h", unsafe extern "C" fn(NcnnMat) -> c_int),
            mat_get_c: sym!(lib, b"ncnn_mat_get_c", unsafe extern "C" fn(NcnnMat) -> c_int),
            mat_get_elemsize: sym!(lib, b"ncnn_mat_get_elemsize",
                unsafe extern "C" fn(NcnnMat) -> usize),
            net,
            in_name: CString::new(in_name).unwrap(),
            out_name: CString::new(out_name).unwrap(),
            model_size: ms,
            f32_buf: vec![0.0f32; 3 * ms * ms],
            net_destroy,
            _lib: lib,
        })
    }

    /// Forward pass: fp16 NCHW frame bytes in, raw f32 output blob out.
    /// Same contract as the Python worker's pipe protocol body.
    pub fn forward(&mut self, frame_f16: &[u8]) -> io::Result<Vec<f32>> {
        let n = frame_f16.len() / 2;
        if n != self.f32_buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("expected {} fp16 values, got {n}", self.f32_buf.len()),
            ));
        }
        for (i, chunk) in frame_f16.chunks_exact(2).enumerate() {
            self.f32_buf[i] = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
        }

        let ms = self.model_size as c_int;
        unsafe {
            let in_mat = (self.mat_create_external_3d)(
                ms,
                ms,
                3,
                self.f32_buf.as_mut_ptr() as *mut c_void,
                std::ptr::null_mut(),
            );
            let ex = (self.extractor_create)(self.net);
            let ri = (self.extractor_input)(ex, self.in_name.as_ptr(), in_mat);
            let mut out_mat: NcnnMat = std::ptr::null_mut();
            let re = (self.extractor_extract)(ex, self.out_name.as_ptr(), &mut out_mat);
            (self.mat_destroy)(in_mat);
            if ri != 0 || re != 0 || out_mat.is_null() {
                (self.extractor_destroy)(ex);
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("ncnn forward failed (input={ri}, extract={re})"),
                ));
            }

            let w = (self.mat_get_w)(out_mat) as usize;
            let h = (self.mat_get_h)(out_mat) as usize;
            let c = (self.mat_get_c)(out_mat).max(1) as usize;
            let elemsize = (self.mat_get_elemsize)(out_mat);
            let data = (self.mat_get_data)(out_mat) as *const f32;
            let count = w * h * c;
            let out = if elemsize == 4 && !data.is_null() {
                std::slice::from_raw_parts(data, count).to_vec()
            } else {
                Vec::new()
            };
            (self.mat_destroy)(out_mat);
            (self.extractor_destroy)(ex);
            if out.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("ncnn output not f32 (elemsize={elemsize})"),
                ));
            }
            Ok(out)
        }
    }
}

impl Drop for NcnnFfi {
    fn drop(&mut self) {
        unsafe { (self.net_destroy)(self.net) };
    }
}

/// IEEE 754 half → single conversion (handles subnormals, inf, nan).
#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits & 0x8000) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let f = match exp {
        0 => {
            if frac == 0 {
                sign << 16
            } else {
                // subnormal: normalize
                let mut e = 127 - 15 + 1;
                let mut m = frac;
                while m & 0x400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                (sign << 16) | ((e as u32) << 23) | ((m as u32 & 0x3ff) << 13)
            }
        }
        0x1f => (sign << 16) | 0x7f80_0000 | (frac << 13), // inf / nan
        _ => (sign << 16) | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(f)
}

#[cfg(test)]
mod tests {
    use super::f16_to_f32;

    #[test]
    fn f16_conversion_matches_reference() {
        // (bits, expected) pairs incl. zero, one, subnormal, inf, -2.5
        let cases: [(u16, f32); 6] = [
            (0x0000, 0.0),
            (0x3c00, 1.0),
            (0x3555, 0.333251953125),
            (0xc100, -2.5),
            (0x7c00, f32::INFINITY),
            (0x0001, 5.960464477539063e-8),
        ];
        for (bits, want) in cases {
            let got = f16_to_f32(bits);
            let ok = if want.is_infinite() {
                got == want
            } else {
                (got - want).abs() <= f32::EPSILON * want.abs().max(1.0)
            };
            assert!(ok, "bits {bits:#06x}: got {got}, want {want}");
        }
        assert!(f16_to_f32(0x7e00).is_nan());
    }
}
