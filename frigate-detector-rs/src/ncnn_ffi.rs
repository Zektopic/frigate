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
        let s: Symbol<$ty> = unsafe { $lib.get($name) }.map_err(|e| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!(
                    "libncnn symbol {} missing: {e}",
                    String::from_utf8_lossy($name)
                ),
            )
        })?;
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

/// ncnn/OpenMP CPU thread count: `NCNN_NUM_THREADS` if set to a positive
/// integer, else 2 (benchmarked optimal on the 8-core reference host; small
/// hosts may want 1 so ffmpeg decode keeps a core).
pub fn num_threads() -> i32 {
    std::env::var("NCNN_NUM_THREADS")
        .ok()
        .and_then(|v| v.trim().parse::<i32>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(2)
}

fn lib_candidates() -> Vec<String> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var("NCNN_LIB") {
        v.push(p);
    }
    v.push("/opt/frigate/libncnn.so".into());
    v.push("libncnn.so".into());
    v
}

/// How a request's frame is encoded. The two sizes differ, so the size of
/// the request tells them apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameFormat {
    /// fp16 values, NCHW, already normalised to 0..1: 6 bytes per pixel.
    F16Nchw,
    /// Frigate's raw detector frame: u8, NHWC, 3 bytes per pixel. Normalised
    /// and transposed here, in one pass.
    U8Nhwc,
}

impl FrameFormat {
    pub fn from_len(len: usize, model_size: usize) -> Option<Self> {
        let values = 3 * model_size * model_size;
        if len == 2 * values {
            Some(FrameFormat::F16Nchw)
        } else if len == values {
            Some(FrameFormat::U8Nhwc)
        } else {
            None
        }
    }
}

/// u8 NHWC pixels to f32 NCHW planes scaled to 0..1, with the same
/// arithmetic as `transform_detect_input` in frigate-frame-rs (`v * (1/255)`
/// in f32), so both paths feed the model identical values.
pub fn u8_nhwc_to_f32_nchw(src: &[u8], dst: &mut [f32]) {
    let hw = dst.len() / 3;
    debug_assert_eq!(src.len(), 3 * hw);
    let inv255: f32 = 1.0 / 255.0;
    let (r, rest) = dst.split_at_mut(hw);
    let (g, b) = rest.split_at_mut(hw);
    for (((px, r), g), b) in src.as_chunks::<3>().0.iter().zip(r).zip(g).zip(b) {
        *r = px[0] as f32 * inv255;
        *g = px[1] as f32 * inv255;
        *b = px[2] as f32 * inv255;
    }
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

        let option_create = sym!(
            lib,
            b"ncnn_option_create",
            unsafe extern "C" fn() -> NcnnOption
        );
        let option_destroy = sym!(
            lib,
            b"ncnn_option_destroy",
            unsafe extern "C" fn(NcnnOption)
        );
        let option_set_num_threads = sym!(
            lib,
            b"ncnn_option_set_num_threads",
            unsafe extern "C" fn(NcnnOption, c_int)
        );
        let option_set_vulkan = sym!(
            lib,
            b"ncnn_option_set_use_vulkan_compute",
            unsafe extern "C" fn(NcnnOption, c_int)
        );
        let net_create = sym!(lib, b"ncnn_net_create", unsafe extern "C" fn() -> NcnnNet);
        let net_destroy = sym!(lib, b"ncnn_net_destroy", unsafe extern "C" fn(NcnnNet));
        let net_set_option = sym!(
            lib,
            b"ncnn_net_set_option",
            unsafe extern "C" fn(NcnnNet, NcnnOption)
        );
        let net_load_param = sym!(
            lib,
            b"ncnn_net_load_param",
            unsafe extern "C" fn(NcnnNet, *const c_char) -> c_int
        );
        let net_load_model = sym!(
            lib,
            b"ncnn_net_load_model",
            unsafe extern "C" fn(NcnnNet, *const c_char) -> c_int
        );

        let net = unsafe { net_create() };
        let opt = unsafe { option_create() };
        // NCNN_FFI_NO_VULKAN=1 forces CPU inference (debug/parity checks).
        // The detector plugin sets this when the only Vulkan device is a
        // software one (llvmpipe), which crashes this backend.
        let use_vulkan =
            std::env::var("NCNN_FFI_NO_VULKAN").map_or(1, |v| if v == "1" { 0 } else { 1 });
        unsafe {
            // 2 threads (default) + passive OpenMP waiting: benchmarked
            // optimal for the CPU-fallback layers on this host (see
            // GPU-DETECTOR.md); override with NCNN_NUM_THREADS.
            option_set_num_threads(opt, num_threads());
            option_set_vulkan(opt, use_vulkan);
            net_set_option(net, opt);
        }
        // Optional (Vulkan builds only): pin device 0 explicitly.
        if use_vulkan == 1 {
            if let Ok(set_dev) = unsafe {
                lib.get::<unsafe extern "C" fn(NcnnNet, c_int)>(b"ncnn_net_set_vulkan_device")
            } {
                unsafe { set_dev(net, 0) };
            }
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
            extractor_create: sym!(
                lib,
                b"ncnn_extractor_create",
                unsafe extern "C" fn(NcnnNet) -> NcnnExtractor
            ),
            extractor_destroy: sym!(
                lib,
                b"ncnn_extractor_destroy",
                unsafe extern "C" fn(NcnnExtractor)
            ),
            extractor_input: sym!(
                lib,
                b"ncnn_extractor_input",
                unsafe extern "C" fn(NcnnExtractor, *const c_char, NcnnMat) -> c_int
            ),
            extractor_extract: sym!(
                lib,
                b"ncnn_extractor_extract",
                unsafe extern "C" fn(NcnnExtractor, *const c_char, *mut NcnnMat) -> c_int
            ),
            mat_create_external_3d: sym!(
                lib,
                b"ncnn_mat_create_external_3d",
                unsafe extern "C" fn(c_int, c_int, c_int, *mut c_void, NcnnAllocator) -> NcnnMat
            ),
            mat_destroy: sym!(lib, b"ncnn_mat_destroy", unsafe extern "C" fn(NcnnMat)),
            mat_get_data: sym!(
                lib,
                b"ncnn_mat_get_data",
                unsafe extern "C" fn(NcnnMat) -> *mut c_void
            ),
            mat_get_w: sym!(
                lib,
                b"ncnn_mat_get_w",
                unsafe extern "C" fn(NcnnMat) -> c_int
            ),
            mat_get_h: sym!(
                lib,
                b"ncnn_mat_get_h",
                unsafe extern "C" fn(NcnnMat) -> c_int
            ),
            mat_get_c: sym!(
                lib,
                b"ncnn_mat_get_c",
                unsafe extern "C" fn(NcnnMat) -> c_int
            ),
            mat_get_elemsize: sym!(
                lib,
                b"ncnn_mat_get_elemsize",
                unsafe extern "C" fn(NcnnMat) -> usize
            ),
            net,
            in_name: CString::new(in_name).unwrap(),
            out_name: CString::new(out_name).unwrap(),
            model_size: ms,
            f32_buf: vec![0.0f32; 3 * ms * ms],
            net_destroy,
            _lib: lib,
        })
    }

    /// Forward pass: frame bytes in (see [`FrameFormat`]), raw f32 output
    /// blob out.
    pub fn forward(&mut self, frame: &[u8]) -> io::Result<Vec<f32>> {
        match FrameFormat::from_len(frame.len(), self.model_size) {
            Some(FrameFormat::F16Nchw) => {
                for (dst, bytes) in self.f32_buf.iter_mut().zip(frame.as_chunks::<2>().0) {
                    *dst = f16_to_f32(u16::from_le_bytes(*bytes));
                }
            }
            Some(FrameFormat::U8Nhwc) => u8_nhwc_to_f32_nchw(frame, &mut self.f32_buf),
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!(
                        "expected {} (fp16 NCHW) or {} (u8 NHWC) frame bytes, got {}",
                        2 * self.f32_buf.len(),
                        self.f32_buf.len(),
                        frame.len()
                    ),
                ));
            }
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
                return Err(io::Error::other(format!(
                    "ncnn forward failed (input={ri}, extract={re})"
                )));
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
                return Err(io::Error::other(format!(
                    "ncnn output not f32 (elemsize={elemsize})"
                )));
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
                (sign << 16) | ((e as u32) << 23) | ((m & 0x3ff) << 13)
            }
        }
        0x1f => (sign << 16) | 0x7f80_0000 | (frac << 13), // inf / nan
        _ => (sign << 16) | ((exp + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(f)
}

#[cfg(test)]
mod frame_tests {
    use super::{u8_nhwc_to_f32_nchw, FrameFormat};

    #[test]
    fn frame_format_follows_the_request_size() {
        assert_eq!(
            FrameFormat::from_len(2 * 3 * 640 * 640, 640),
            Some(FrameFormat::F16Nchw)
        );
        assert_eq!(
            FrameFormat::from_len(3 * 640 * 640, 640),
            Some(FrameFormat::U8Nhwc)
        );
        assert_eq!(FrameFormat::from_len(3 * 320 * 320, 640), None);
        assert_eq!(FrameFormat::from_len(0, 640), None);
        // the two encodings of one model size can never be confused
        assert_eq!(
            FrameFormat::from_len(3 * 320 * 320 * 2, 320),
            Some(FrameFormat::F16Nchw)
        );
    }

    #[test]
    fn u8_frame_is_transposed_into_planes() {
        // 2x2 image, pixels (r, g, b) = (10p, 10p + 1, 10p + 2)
        let src: Vec<u8> = (0..4u8)
            .flat_map(|p| [10 * p, 10 * p + 1, 10 * p + 2])
            .collect();
        let mut dst = vec![0.0f32; 12];
        u8_nhwc_to_f32_nchw(&src, &mut dst);
        let inv = 1.0f32 / 255.0;
        let want: Vec<f32> = [0u8, 10, 20, 30, 1, 11, 21, 31, 2, 12, 22, 32]
            .iter()
            .map(|&v| v as f32 * inv)
            .collect();
        assert_eq!(dst, want);
    }

    #[test]
    fn every_u8_value_matches_the_frame_rs_formula_bit_for_bit() {
        // frigate-frame-rs transform_detect_input: `v as f32 * (1.0 / 255.0)`
        let inv255: f32 = 1.0 / 255.0;
        let src: Vec<u8> = (0..=255u8).flat_map(|v| [v, v, v]).collect();
        let mut dst = vec![0.0f32; src.len()];
        u8_nhwc_to_f32_nchw(&src, &mut dst);
        for v in 0..=255u8 {
            let want = (v as f32 * inv255).to_bits();
            for plane in 0..3 {
                assert_eq!(dst[plane * 256 + v as usize].to_bits(), want, "value {v}");
            }
        }
        assert_eq!(dst[0], 0.0);
        assert_eq!(dst[255], 1.0);
    }
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
            (0x3555, 1365.0 / 4096.0), // nearest f16 to 1/3
            (0xc100, -2.5),
            (0x7c00, f32::INFINITY),
            (0x0001, 1.0 / 16_777_216.0), // smallest subnormal, 2^-24
        ];
        // Every f16 is exactly representable as f32, so compare exactly.  (The
        // old `abs() <= EPSILON` tolerance accepted anything below 1.2e-7,
        // so it could not catch a wrong subnormal.)
        for (bits, want) in cases {
            let got = f16_to_f32(bits);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "bits {bits:#06x}: got {got}, want {want}"
            );
        }
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(f16_to_f32(0x03ff), 1023.0 / 16_777_216.0); // largest subnormal
        assert!(f16_to_f32(0x7e00).is_nan());
    }
}

#[cfg(test)]
mod exhaustive_tests {
    use super::f16_to_f32;

    #[test]
    fn f16_conversion_all_bit_patterns() {
        for bits in 0..=u16::MAX {
            let want = half::f16::from_bits(bits).to_f32();
            let got = f16_to_f32(bits);
            if want.is_nan() {
                assert!(got.is_nan(), "bits {bits:#06x}: got {got}, want NaN");
            } else {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "bits {bits:#06x}: got {got}, want {want}"
                );
            }
        }
    }
}
