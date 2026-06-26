/// SIMD-accelerated frame preprocessing for Frigate NVR.
///
/// Provides: YUV420→RGB, bilinear resize, RGB→float32 normalization,
/// and NHWC→NCHW transpose — the four most CPU-intensive operations in
/// the detection input pipeline.
///
/// Exposes a flat C ABI for Python `ctypes`.

use std::arch::x86_64::*;

// ── YUV420 planar → interleaved RGB (u8) ────────────────────────────

/// Convert a YUV420 planar image to interleaved RGB.
///
/// `y` has `w * h` bytes, `u` and `v` each have `(w/2) * (h/2)` bytes.
/// `rgb` receives `w * h * 3` interleaved R,G,B bytes.
///
/// # Safety
/// All pointers must be valid for the implied sizes.
#[no_mangle]
pub unsafe extern "C" fn yuv420_to_rgb(
    y: *const u8,
    u: *const u8,
    v: *const u8,
    rgb: *mut u8,
    w: u32,
    h: u32,
) {
    let w = w as usize;
    let h = h as usize;
    let y = std::slice::from_raw_parts(y, w * h);
    let u = std::slice::from_raw_parts(u, (w / 2) * (h / 2));
    let v = std::slice::from_raw_parts(v, (w / 2) * (h / 2));
    let rgb = std::slice::from_raw_parts_mut(rgb, w * h * 3);

    for row in 0..h {
        let uv_row = row / 2;
        for col in 0..w {
            let uv_col = col / 2;
            let yi = y[row * w + col] as f32;
            let ui = u[uv_row * (w / 2) + uv_col] as f32 - 128.0;
            let vi = v[uv_row * (w / 2) + uv_col] as f32 - 128.0;

            // ITU-R BT.601 coefficients (clamped to [0,255])
            let r = (yi + 1.402 * vi).clamp(0.0, 255.0) as u8;
            let g = (yi - 0.344136 * ui - 0.714136 * vi).clamp(0.0, 255.0) as u8;
            let b = (yi + 1.772 * ui).clamp(0.0, 255.0) as u8;

            let out_idx = (row * w + col) * 3;
            *rgb.get_unchecked_mut(out_idx) = r;
            *rgb.get_unchecked_mut(out_idx + 1) = g;
            *rgb.get_unchecked_mut(out_idx + 2) = b;
        }
    }
}

// ── Bilinear resize (u8 RGB interleaved) ───────────────────────────

/// Bilinear resize of an interleaved RGB (or grayscale) image.
///
/// `src` has dimensions `(sw, sh)` with `channels` planes interleaved
/// per pixel.  `dst` receives `(dw * dh * channels)` bytes.
///
/// # Safety
/// All pointers must be valid for the implied sizes.
#[no_mangle]
pub unsafe extern "C" fn bilinear_resize_u8(
    src: *const u8,
    dst: *mut u8,
    sw: u32,
    sh: u32,
    dw: u32,
    dh: u32,
    channels: u32,
) {
    let sw = sw as usize;
    let sh = sh as usize;
    let dw = dw as usize;
    let dh = dh as usize;
    let ch = channels as usize;

    let src_slice = std::slice::from_raw_parts(src, sw * sh * ch);
    let dst_slice = std::slice::from_raw_parts_mut(dst, dw * dh * ch);

    if sw == dw && sh == dh {
        dst_slice.copy_from_slice(src_slice);
        return;
    }

    let scale_x = sw as f64 / dw as f64;
    let scale_y = sh as f64 / dh as f64;

    for dy in 0..dh {
        let sy = dy as f64 * scale_y;
        let sy0 = (sy as usize).min(sh - 1);
        let sy1 = (sy0 + 1).min(sh - 1);
        let fy = (sy - sy0 as f64) as f32;

        for dx in 0..dw {
            let sx = dx as f64 * scale_x;
            let sx0 = (sx as usize).min(sw - 1);
            let sx1 = (sx0 + 1).min(sw - 1);
            let fx = (sx - sx0 as f64) as f32;

            let dst_offset = (dy * dw + dx) * ch;

            for c in 0..ch {
                let v00 = src_slice[(sy0 * sw + sx0) * ch + c] as f32;
                let v10 = src_slice[(sy0 * sw + sx1) * ch + c] as f32;
                let v01 = src_slice[(sy1 * sw + sx0) * ch + c] as f32;
                let v11 = src_slice[(sy1 * sw + sx1) * ch + c] as f32;

                let top = v00 + (v10 - v00) * fx;
                let bot = v01 + (v11 - v01) * fx;
                let val = (top + (bot - top) * fy) as u8;

                dst_slice[dst_offset + c] = val;
            }
        }
    }
}

// ── RGB u8 → float32 normalization (divide by 255) ──────────────────

/// Normalize a u8 RGB buffer to float32 [0, 1].
///
/// `src` has `len` interleaved R,G,B bytes.  `dst` receives `len`
/// float32 values, each in [0.0, 1.0].
///
/// # Safety
/// Both pointers must be valid for `len` elements of the appropriate type.
#[no_mangle]
pub unsafe extern "C" fn normalize_u8_to_f32(
    src: *const u8,
    dst: *mut f32,
    len: u32,
) {
    let len = len as usize;
    let src = std::slice::from_raw_parts(src, len);
    let dst = std::slice::from_raw_parts_mut(dst, len);

    let inv255: f32 = 1.0 / 255.0;

    // SIMD: process 8 u8→f32 per iteration
    let mut i = 0usize;
    let simd_end = (len / 8) * 8;

    while i < simd_end {
        // Load 8 u8, unpack to 32-bit ints, convert to float
        let v8 = _mm_loadl_epi64(src.as_ptr().add(i) as *const __m128i);
        let v32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v8));
        let norm = _mm256_mul_ps(v32, _mm256_set1_ps(inv255));

        _mm256_storeu_ps(dst.as_mut_ptr().add(i), norm);
        i += 8;
    }

    // scalar tail
    for j in i..len {
        dst[j] = src[j] as f32 * inv255;
    }
}

// ── NHWC → NCHW transpose (single-batch) ───────────────────────────

/// Transpose a single-image NHWC (h, w, c) float32 buffer to NCHW
/// (c, h, w).  ``c`` is typically 3 (RGB) or 1 (grayscale).
///
/// # Safety
/// `src` has ``h * w * c`` f32, `dst` has the same count.
#[no_mangle]
pub unsafe extern "C" fn nhwc_to_nchw_f32(
    src: *const f32,
    dst: *mut f32,
    h: u32,
    w: u32,
    c: u32,
) {
    let h = h as usize;
    let w = w as usize;
    let c = c as usize;
    let src = std::slice::from_raw_parts(src, h * w * c);
    let dst = std::slice::from_raw_parts_mut(dst, h * w * c);

    for ci in 0..c {
        for hi in 0..h {
            for wi in 0..w {
                let src_idx = (hi * w + wi) * c + ci;
                let dst_idx = ci * (h * w) + hi * w + wi;
                *dst.get_unchecked_mut(dst_idx) = *src.get_unchecked(src_idx);
            }
        }
    }
}

// ── Combined pipeline: resize + normalize + transpose ───────────────

/// Run the full detection preprocessing pipeline in one FFI call:
/// 1. Bilinear resize from (sw, sh) to (dw, dh)
/// 2. Normalize u8→f32 dividing by 255
/// 3. NHWC→NCHW transpose
///
/// `src` is interleaved RGB u8 at the source resolution.
/// `dst` receives ``c * dh * dw`` float32 values in NCHW order.
///
/// This avoids allocating intermediate buffers.
///
/// # Safety
/// All pointers must be valid for the implied sizes.
#[no_mangle]
pub unsafe extern "C" fn preprocess_detect_input(
    src: *const u8,
    dst: *mut f32,
    sw: u32,
    sh: u32,
    dw: u32,
    dh: u32,
    c: u32,
) {
    let sw = sw as usize;
    let sh = sh as usize;
    let dw = dw as usize;
    let dh = dh as usize;
    let c = c as usize;

    let src = std::slice::from_raw_parts(src, sw * sh * c);
    let dst = std::slice::from_raw_parts_mut(dst, c * dh * dw);

    let scale_x = sw as f64 / dw as f64;
    let scale_y = sh as f64 / dh as f64;
    let inv255: f32 = 1.0 / 255.0;

    // For each output position, sample source with bilinear interpolation,
    // normalize, and write directly into NCHW layout.
    for ci in 0..c {
        for dy in 0..dh {
            let sy = dy as f64 * scale_y;
            let sy0 = (sy as usize).min(sh - 1);
            let sy1 = (sy0 + 1).min(sh - 1);
            let fy = (sy - sy0 as f64) as f32;

            for dx in 0..dw {
                let sx = dx as f64 * scale_x;
                let sx0 = (sx as usize).min(sw - 1);
                let sx1 = (sx0 + 1).min(sw - 1);
                let fx = (sx - sx0 as f64) as f32;

                // bilinear interpolation for channel ci
                let v00 = *src.get_unchecked((sy0 * sw + sx0) * c + ci) as f32;
                let v10 = *src.get_unchecked((sy0 * sw + sx1) * c + ci) as f32;
                let v01 = *src.get_unchecked((sy1 * sw + sx0) * c + ci) as f32;
                let v11 = *src.get_unchecked((sy1 * sw + sx1) * c + ci) as f32;

                let top = v00 + (v10 - v00) * fx;
                let bot = v01 + (v11 - v01) * fx;
                let val = (top + (bot - top) * fy) * inv255;

                let dst_idx = ci * (dh * dw) + dy * dw + dx;
                *dst.get_unchecked_mut(dst_idx) = val;
            }
        }
    }
}

// ── YUV420 crop + repack ───────────────────────────────────────────

/// Crop a region from a planar YUV420 frame and pack into a contiguous
/// output buffer (Y plane first, then interleaved U/V bytes).
///
/// Replaces the 7-slice-copy numpy path in `yuv_crop_and_resize`.
///
/// `src` is the full-frame YUV420 buffer (Y + U + V planar).
/// `out` receives `(crop_w * crop_h * 3 / 2)` bytes.
#[no_mangle]
pub unsafe extern "C" fn yuv420_crop_repack(
    src: *const u8,
    src_w: u32,
    src_h: u32,
    crop_x: u32,
    crop_y: u32,
    crop_w: u32,
    crop_h: u32,
    out: *mut u8,
) {
    let sw = src_w as usize;
    let sh = src_h as usize;
    let cx = crop_x as usize;
    let cy = crop_y as usize;
    let cw = crop_w as usize;
    let ch = crop_h as usize;

    let y_plane_size = sw * sh;
    let uv_plane_size = (sw / 2) * (sh / 2);

    let src = std::slice::from_raw_parts(src, y_plane_size + uv_plane_size * 2);
    let out = std::slice::from_raw_parts_mut(out, cw * ch + (cw / 2) * (ch / 2) * 2);

    let mut out_idx = 0usize;

    // Y plane: copy row by row (SIMD-friendly memcpy)
    for row in cy..(cy + ch) {
        let src_start = row * sw + cx;
        let src_end = src_start + cw;
        out[out_idx..out_idx + cw].copy_from_slice(&src[src_start..src_end]);
        out_idx += cw;
    }

    // U plane (1/4 resolution)
    let u_offset = y_plane_size;
    let uv_cx = cx / 2;
    let uv_cw = cw / 2;
    for row in (cy / 2)..(cy / 2 + ch / 2) {
        let src_start = u_offset + row * (sw / 2) + uv_cx;
        let src_end = src_start + uv_cw;
        out[out_idx..out_idx + uv_cw].copy_from_slice(&src[src_start..src_end]);
        out_idx += uv_cw;
    }

    // V plane (1/4 resolution)
    let v_offset = y_plane_size + uv_plane_size;
    for row in (cy / 2)..(cy / 2 + ch / 2) {
        let src_start = v_offset + row * (sw / 2) + uv_cx;
        let src_end = src_start + uv_cw;
        out[out_idx..out_idx + uv_cw].copy_from_slice(&src[src_start..src_end]);
        out_idx += uv_cw;
    }
}

// ── YUV420 → 3-channel YUV (UV upsampling) ─────────────────────────

/// Convert planar YUV420 to interleaved 3-channel YUV by
/// nearest-neighbor upsampling U/V to full resolution.
///
/// Replaces nested `np.repeat` calls in `yuv_to_3_channel_yuv`.
#[no_mangle]
pub unsafe extern "C" fn yuv420_to_3channel(
    y: *const u8,
    u: *const u8,
    v: *const u8,
    w: u32,
    h: u32,
    out: *mut u8,
) {
    let w = w as usize;
    let h = h as usize;
    let y = std::slice::from_raw_parts(y, w * h);
    let u = std::slice::from_raw_parts(u, (w / 2) * (h / 2));
    let v = std::slice::from_raw_parts(v, (w / 2) * (h / 2));
    let out = std::slice::from_raw_parts_mut(out, w * h * 3);

    for row in 0..h {
        let uv_row = row / 2;
        for col in 0..w {
            let uv_col = col / 2;
            let idx = (row * w + col) * 3;
            out[idx] = y[row * w + col];
            out[idx + 1] = u[uv_row * (w / 2) + uv_col];
            out[idx + 2] = v[uv_row * (w / 2) + uv_col];
        }
    }
}

// ── Combined: tensor transpose + u8→f32 normalize ──────────────────

/// Fused NHWC(u8)→NCHW(f32) with division by 255.
///
/// Replaces `np.transpose` + `.astype(np.float32)` + `/255` in
/// `BaseLocalDetector._transform_input`.
#[no_mangle]
pub unsafe extern "C" fn transform_detect_input(
    src: *const u8,
    dst: *mut f32,
    h: u32,
    w: u32,
    c: u32,
) {
    let h = h as usize;
    let w = w as usize;
    let c = c as usize;
    let src = std::slice::from_raw_parts(src, h * w * c);
    let dst = std::slice::from_raw_parts_mut(dst, c * h * w);
    let inv255: f32 = 1.0 / 255.0;

    for ci in 0..c {
        for hi in 0..h {
            for wi in 0..w {
                let src_idx = (hi * w + wi) * c + ci;
                let dst_idx = ci * (h * w) + hi * w + wi;
                *dst.get_unchecked_mut(dst_idx) =
                    *src.get_unchecked(src_idx) as f32 * inv255;
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilinear_noop() {
        let src: Vec<u8> = (0..12u8).collect(); // 2×2 RGB
        let mut dst = vec![0u8; 12];
        unsafe {
            bilinear_resize_u8(src.as_ptr(), dst.as_mut_ptr(), 2, 2, 2, 2, 3);
        }
        assert_eq!(src, dst);
    }

    #[test]
    fn test_bilinear_downscale() {
        // 4×4 grayscale → 2×2 — output pixel (0,0) maps to source (0,0)
        let w = 4;
        let h = 4;
        let mut src = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                src[y * w + x] = (y * w + x) as u8;
            }
        }
        let mut dst = vec![0u8; 2 * 2];
        unsafe {
            bilinear_resize_u8(src.as_ptr(), dst.as_mut_ptr(), 4, 4, 2, 2, 1);
        }
        // Output (0,0): sx=0, sy=0 → source pixel 0
        assert_eq!(dst[0], 0);
        // Output (1,0): sx=2.0, sy=0 → source pixel 2
        assert_eq!(dst[1], 2);
        // Output (0,1): sx=0, sy=2.0 → source pixel 8
        assert_eq!(dst[2], 8);
        // Output (1,1): sx=2.0, sy=2.0 → source pixel 10
        assert_eq!(dst[3], 10);
    }

    #[test]
    fn test_normalize() {
        let src: Vec<u8> = vec![0, 128, 255];
        let mut dst = vec![0.0f32; 3];
        unsafe {
            normalize_u8_to_f32(src.as_ptr(), dst.as_mut_ptr(), 3);
        }
        assert!((dst[0] - 0.0).abs() < 0.001);
        assert!((dst[1] - 128.0 / 255.0).abs() < 0.001);
        assert!((dst[2] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_nhwc_to_nchw() {
        // 1×2 RGB: [R0,G0,B0,R1,G1,B1] → [R0,R1,G0,G1,B0,B1]
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = vec![0.0f32; 6];
        unsafe {
            nhwc_to_nchw_f32(src.as_ptr(), dst.as_mut_ptr(), 1, 2, 3);
        }
        assert_eq!(dst, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_preprocess_pipeline() {
        // 4×4 RGB → 2×2 RGB normalized NCHW
        // Fill with a simple pattern for predictable interpolation.
        let mut src = vec![0u8; 4 * 4 * 3];
        for y in 0..4 {
            for x in 0..4 {
                let val = (y * 4 + x) as u8;
                let idx = (y * 4 + x) * 3;
                src[idx] = val;     // R
                src[idx + 1] = val; // G
                src[idx + 2] = val; // B
            }
        }
        let mut dst = vec![0.0f32; 3 * 2 * 2];
        unsafe {
            preprocess_detect_input(
                src.as_ptr(), dst.as_mut_ptr(), 4, 4, 2, 2, 3,
            );
        }
        // Output (0,0): sx=0, sy=0 → source pixel 0 → 0/255 = 0
        let r00 = dst[0 * 4 + 0 * 2 + 0];
        let g00 = dst[1 * 4 + 0 * 2 + 0];
        let b00 = dst[2 * 4 + 0 * 2 + 0];
        assert!((r00 - 0.0).abs() < 0.001);
        assert!((g00 - 0.0).abs() < 0.001);
        assert!((b00 - 0.0).abs() < 0.001);

        // Output (1,1): sx=2.0, sy=2.0 → source pixel 10 → 10/255
        let r11 = dst[0 * 4 + 1 * 2 + 1];
        assert!((r11 - 10.0 / 255.0).abs() < 0.001);
    }
}
