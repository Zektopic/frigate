//! SIMD-accelerated motion pixel pipeline for Frigate NVR.
//!
//! `motion_pixel_pipeline` stands in for the OpenCV + scipy blur, absdiff,
//! threshold, dilate and contour steps of `ImprovedMotionDetector.detect()`.
//! Exposes a flat C ABI so Python can call it via `ctypes`
//! (see `frigate/motion/rust_engine.py`).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ── C ABI types ────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionBox {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

// ── Helpers ────────────────────────────────────────────────────────

#[inline]
#[cfg(target_arch = "x86_64")]
const fn align_up(value: usize, align: usize) -> usize {
    (value + align - 1) & !(align - 1)
}

// ═══════════════════════════════════════════════════════════════════
//  Gaussian blur (3×3 separable approximation, sigma≈1)
// ═══════════════════════════════════════════════════════════════════

/// Separable 3×3 Gaussian blur: horizontal pass → vertical pass.
/// `tmp` must be at least `len` bytes.  `w` is image width.
/// Kernel: [1 2 1]/4 in each direction → effective 3×3 Gaussian ≈ [1,2,1;2,4,2;1,2,1]/16.
#[inline]
unsafe fn gaussian_blur_3x3(src: &[u8], dst: &mut [u8], tmp: &mut [u8], w: usize, h: usize) {
    // horizontal pass: for each row, convolve with [1 2 1]/4
    for y in 0..h {
        let row = y * w;
        for x in 1..(w - 1) {
            let idx = row + x;
            let val = src[idx - 1] as u16 + src[idx] as u16 * 2 + src[idx + 1] as u16;
            tmp[idx] = ((val + 2) / 4) as u8; // rounded division
        }
        // edge pixels: copy
        tmp[row] = src[row];
        tmp[row + w - 1] = src[row + w - 1];
    }

    // vertical pass: for each column, convolve with [1 2 1]/4
    for y in 1..(h - 1) {
        let row = y * w;
        let prev = (y - 1) * w;
        let next = (y + 1) * w;
        for x in 0..w {
            let val = tmp[prev + x] as u16 + tmp[row + x] as u16 * 2 + tmp[next + x] as u16;
            dst[row + x] = ((val + 2) / 4) as u8;
        }
    }
    // top/bottom rows: copy from tmp
    dst[..w].copy_from_slice(&tmp[..w]);
    dst[(h - 1) * w..].copy_from_slice(&tmp[(h - 1) * w..]);
}

// ═══════════════════════════════════════════════════════════════════
//  Scalar kernels — always compiled.
//
//  These are the reference implementations.  On x86_64 the AVX2
//  variants above are selected at *runtime* via the dispatchers below;
//  everywhere else (and on x86_64 CPUs without AVX2, e.g. pre-Haswell
//  Intel and the Celeron/Atom/N-series boxes Frigate is commonly
//  deployed on) these run instead.  Gating SIMD on `cfg(target_arch)`
//  alone is a compile-time check and does not prevent SIGILL.
// ═══════════════════════════════════════════════════════════════════

fn absdiff_scalar(a: &[u8], b: &[u8], dst: &mut [u8], len: usize) {
    for j in 0..len {
        let da = a[j] as i16;
        let db = b[j] as i16;
        dst[j] = (da - db).unsigned_abs() as u8;
    }
}

fn threshold_mask_scalar(buf: &mut [u8], mask: &[u8], thresh: u8, len: usize) {
    for j in 0..len {
        buf[j] = if buf[j] >= thresh && mask[j] == 0 {
            255
        } else {
            0
        };
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Runtime dispatchers.  Call sites keep their original names, so the
//  selection is transparent.  `is_x86_feature_detected!` caches its
//  result in a static after the first call, so the per-frame cost is
//  a single relaxed atomic load.
// ═══════════════════════════════════════════════════════════════════

unsafe fn absdiff_avx2(a: &[u8], b: &[u8], dst: &mut [u8], len: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return absdiff_avx2_impl(a, b, dst, len);
        }
    }
    absdiff_scalar(a, b, dst, len)
}

unsafe fn threshold_mask_avx2(buf: &mut [u8], mask: &[u8], thresh: u8, len: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return threshold_mask_avx2_impl(buf, mask, thresh, len);
        }
    }
    threshold_mask_scalar(buf, mask, thresh, len)
}

// ═══════════════════════════════════════════════════════════════════
//  Core pixel kernels (from original)
// ═══════════════════════════════════════════════════════════════════

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn absdiff_avx2_impl(a: &[u8], b: &[u8], dst: &mut [u8], len: usize) {
    let mut i = 0usize;
    let simd_end = align_up(len.saturating_sub(31), 32) & !31;
    while i < simd_end {
        let va = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        let max_ab = _mm256_max_epu8(va, vb);
        let min_ab = _mm256_min_epu8(va, vb);
        let diff = _mm256_subs_epu8(max_ab, min_ab);
        _mm256_storeu_si256(dst.as_mut_ptr().add(i) as *mut __m256i, diff);
        i += 32;
    }
    for j in i..len {
        let da = a[j] as i16;
        let db = b[j] as i16;
        dst[j] = (da - db).unsigned_abs() as u8;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn threshold_mask_avx2_impl(buf: &mut [u8], mask: &[u8], thresh: u8, len: usize) {
    let thresh_vec = _mm256_set1_epi8(thresh as i8);
    let zero = _mm256_setzero_si256();
    let mut i = 0usize;
    let simd_end = align_up(len.saturating_sub(31), 32) & !31;
    while i < simd_end {
        let v = _mm256_loadu_si256(buf.as_ptr().add(i) as *const __m256i);
        let m = _mm256_loadu_si256(mask.as_ptr().add(i) as *const __m256i);
        // val >= thresh, UNSIGNED.
        //
        // This must not use _mm256_cmpgt_epi8: that is a *signed* byte
        // compare, so every pixel >= 128 reads as negative and was
        // classified as below-threshold — i.e. the AVX2 path silently
        // discarded the strongest motion signals while the scalar tail
        // handled them correctly.  max_epu8(v, t) == v is the unsigned
        // equivalent of v >= t.
        let ge = _mm256_cmpeq_epi8(_mm256_max_epu8(v, thresh_vec), v);
        // mask == 0 ? (mask=0 means UNMASKED = process this pixel)
        let unmasked = _mm256_cmpeq_epi8(m, zero); // m == 0 (unmasked)
                                                   // combine: (val >= thresh) AND (unmasked)
        let res = _mm256_and_si256(ge, unmasked);
        _mm256_storeu_si256(buf.as_mut_ptr().add(i) as *mut __m256i, res);
        i += 32;
    }
    for j in i..len {
        buf[j] = if buf[j] >= thresh && mask[j] == 0 {
            255
        } else {
            0
        };
    }
}

/// One iteration of binary dilation with a 3×3 *cross* (4-neighbour)
/// structuring element: a pixel becomes 255 if it or its left, right, upper
/// or lower neighbour is 255.  Diagonal neighbours are not considered, so
/// this is not `cv2.dilate(img, None)` (3×3 rectangle).  The difference is
/// documented on [`motion_pixel_pipeline`].
///
/// `tmp` must hold at least `w * h` bytes.
fn dilate_cross_3x3(buf: &mut [u8], tmp: &mut [u8], w: usize, h: usize) {
    let len = w * h;
    tmp[..len].copy_from_slice(&buf[..len]);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if buf[idx] == 255 {
                continue;
            }
            let fg = (x > 0 && tmp[idx - 1] == 255)
                || (x + 1 < w && tmp[idx + 1] == 255)
                || (y > 0 && tmp[idx - w] == 255)
                || (y + 1 < h && tmp[idx + w] == 255);
            if fg {
                buf[idx] = 255;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Pixel pipeline — single FFI entry point
// ═══════════════════════════════════════════════════════════════════

/// 4-connected flood-fill contour extraction that also accumulates the
/// total foreground area: the pixel count of every component (matching Python's `total_contour_area`
/// accumulation across ALL contours, not just kept ones).
fn find_contours_with_area(
    pixels: &[u8],
    w: usize,
    h: usize,
    min_area: i32,
) -> (Vec<(i32, i32, i32, i32)>, f32) {
    let mut visited = vec![false; w * h];
    let mut boxes: Vec<(i32, i32, i32, i32)> = Vec::with_capacity(64);
    let mut total_area: f32 = 0.0;
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if pixels[idx] != 255 || visited[idx] {
                continue;
            }
            let mut stack: Vec<(usize, usize)> = Vec::with_capacity(256);
            stack.push((x, y));
            visited[idx] = true;
            let mut min_x = x;
            let mut min_y = y;
            let mut max_x = x;
            let mut max_y = y;
            let mut area: i32 = 0;
            while let Some((cx, cy)) = stack.pop() {
                area += 1;
                if cx < min_x {
                    min_x = cx;
                }
                if cx > max_x {
                    max_x = cx;
                }
                if cy < min_y {
                    min_y = cy;
                }
                if cy > max_y {
                    max_y = cy;
                }
                let neighbors: [(isize, isize); 4] = [
                    (cx as isize - 1, cy as isize),
                    (cx as isize + 1, cy as isize),
                    (cx as isize, cy as isize - 1),
                    (cx as isize, cy as isize + 1),
                ];
                for (nx, ny) in neighbors {
                    if nx >= 0 && (nx as usize) < w && ny >= 0 && (ny as usize) < h {
                        let nidx = (ny as usize) * w + (nx as usize);
                        if !visited[nidx] && pixels[nidx] == 255 {
                            visited[nidx] = true;
                            stack.push((nx as usize, ny as usize));
                        }
                    }
                }
            }
            total_area += area as f32;
            if area >= min_area {
                boxes.push((
                    min_x as i32,
                    min_y as i32,
                    (max_x + 1) as i32,
                    (max_y + 1) as i32,
                ));
            }
        }
    }
    (boxes, total_area)
}

/// Pixel pipeline ONLY — for wiring into ImprovedMotionDetector.detect().
///
/// Replaces the OpenCV steps: gaussian blur → absdiff(avg) → threshold →
/// dilate → contours.  Does NOT touch the running average — Python keeps
/// its `accumulateWeighted` logic (motion_frame_count gating, calibration
/// alpha).  The blur is applied IN-PLACE on `frame` so the caller's buffer
/// matches what the diff saw (Python then averages the blurred frame,
/// exactly like the OpenCV path).
///
/// This approximates the OpenCV path; it is not bit-identical to it:
/// * blur is a 3×3 binomial `[1 2 1]/4` kernel, not
///   `scipy.ndimage.gaussian_filter(sigma=1, radius=blur_radius)`;
/// * a pixel is foreground when `diff >= thresh` (OpenCV `THRESH_BINARY`
///   uses `diff > thresh`);
/// * dilation uses a 3×3 cross, not `cv2.dilate`'s 3×3 rectangle;
/// * components are 4-connected flood fills whose area is the pixel count
///   and are kept when `area >= min_area`, whereas OpenCV traces 8-connected
///   outer contours, measures `cv2.contourArea` and keeps `area > min_area`.
///
/// # Safety
/// * `frame` must be valid for reads and writes of `w * h` bytes and must
///   not alias `avg_frame`, `mask` or `out_boxes`.
/// * `avg_frame` must be valid for reads of `w * h` `f32`s and `mask` for
///   reads of `w * h` bytes.
/// * `out_boxes` must be valid for writes of `max_boxes` [`MotionBox`]es.
/// * `out_total_area` must be null or valid for a write of one `f32`.
///
/// Null pointers (other than `out_total_area`), `w == 0`, `h == 0` and
/// `max_boxes == 0` are rejected and return 0.
///
/// Returns the number of boxes written.
#[no_mangle]
pub unsafe extern "C" fn motion_pixel_pipeline(
    frame: *mut u8,
    avg_frame: *const f32,
    mask: *const u8,
    w: u32,
    h: u32,
    thresh: u8,
    min_area: u32,
    blur_enabled: u8,
    out_boxes: *mut MotionBox,
    max_boxes: u32,
    out_total_area: *mut f32,
) -> u32 {
    let w = w as usize;
    let h = h as usize;
    let len = w * h;

    // Guard the FFI boundary: from_raw_parts on a null pointer is UB
    // even at zero length, and a zero-sized frame underflows the
    // neighbourhood index arithmetic in the blur/dilate kernels.
    if frame.is_null()
        || avg_frame.is_null()
        || mask.is_null()
        || out_boxes.is_null()
        || w == 0
        || h == 0
        || max_boxes == 0
    {
        if !out_total_area.is_null() {
            *out_total_area = 0.0;
        }
        return 0;
    }

    let frame = std::slice::from_raw_parts_mut(frame, len);
    let avg_frame = std::slice::from_raw_parts(avg_frame, len);
    let mask = std::slice::from_raw_parts(mask, len);

    // 1. Gaussian blur in-place (so the caller's frame matches the diff input)
    if blur_enabled != 0 {
        let mut blurred = vec![0u8; len];
        let mut tmp = vec![0u8; len];
        gaussian_blur_3x3(frame, &mut blurred, &mut tmp, w, h);
        frame.copy_from_slice(&blurred);
    }

    // 2. absdiff vs running average (avg converted f32 → u8, like convertScaleAbs)
    let mut avg_u8 = vec![0u8; len];
    for i in 0..len {
        avg_u8[i] = (avg_frame[i] + 0.5).clamp(0.0, 255.0) as u8;
    }
    let mut diff = vec![0u8; len];
    absdiff_avx2(frame, &avg_u8, &mut diff, len);

    // 3. threshold + mask (>= thresh AND unmasked → 255)
    threshold_mask_avx2(&mut diff, mask, thresh, len);

    // 4. dilate 3×3, one iteration
    let mut tmp = vec![0u8; len];
    dilate_cross_3x3(&mut diff, &mut tmp, w, h);

    // 5. contours → boxes + total foreground area
    let (boxes, total_area) = find_contours_with_area(&diff, w, h, min_area as i32);

    if !out_total_area.is_null() {
        *out_total_area = total_area;
    }

    let n = (boxes.len() as u32).min(max_boxes);
    let out_slice = std::slice::from_raw_parts_mut(out_boxes, n as usize);
    for (i, (x1, y1, x2, y2)) in boxes.into_iter().take(n as usize).enumerate() {
        out_slice[i] = MotionBox { x1, y1, x2, y2 };
    }
    n
}

// ═══════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_blur_flat() {
        let src = vec![128u8; 100]; // 10×10 flat image
        let mut dst = vec![0u8; 100];
        let mut tmp = vec![0u8; 100];
        unsafe {
            gaussian_blur_3x3(&src, &mut dst, &mut tmp, 10, 10);
        }
        assert_eq!(&dst, &src); // flat image unchanged by blur
    }

    #[test]
    fn test_absdiff_identical() {
        let data: Vec<u8> = (0..64u8).collect();
        let mut dst = vec![0u8; 64];
        unsafe {
            absdiff_avx2(&data, &data, &mut dst, 64);
        }
        assert!(dst.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_threshold_mask() {
        let mut buf: Vec<u8> = (0..64u8).collect();
        let mask: Vec<u8> = vec![0u8; 64]; // all zeros = unmasked = process all
        unsafe {
            threshold_mask_avx2(&mut buf, &mask, 32, 64);
        }
        for (i, &v) in buf.iter().enumerate() {
            let expected = if i >= 32 { 255 } else { 0 };
            assert_eq!(v, expected, "pixel {i}");
        }
    }

    #[test]
    fn dilate_cross_3x3_skips_diagonals() {
        // A single foreground pixel grows into a plus sign, not a 3×3 block.
        let (w, h) = (5usize, 5usize);
        let mut buf = vec![0u8; w * h];
        buf[2 * w + 2] = 255;
        let mut tmp = vec![0u8; w * h];
        dilate_cross_3x3(&mut buf, &mut tmp, w, h);
        let on: Vec<(usize, usize)> = (0..w * h)
            .filter(|&i| buf[i] == 255)
            .map(|i| (i % w, i / w))
            .collect();
        assert_eq!(on, vec![(2, 1), (1, 2), (2, 2), (3, 2), (2, 3)]);
    }

    #[test]
    fn test_contours_single_rect() {
        let w = 10usize;
        let h = 10usize;
        let mut pixels = vec![0u8; w * h];
        for y in 3..6 {
            for x in 3..6 {
                pixels[y * w + x] = 255;
            }
        }
        let boxes = find_contours_with_area(&pixels, w, h, 0).0;
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0], (3, 3, 6, 6));
    }

    #[test]
    fn test_contours_min_area_filter() {
        let w = 10usize;
        let h = 10usize;
        let mut pixels = vec![0u8; w * h];
        pixels[w + 1] = 255; // 1-pixel dot at (1, 1)
        for y in 5..9 {
            for x in 5..9 {
                pixels[y * w + x] = 255;
            }
        } // 4×4 rect
        let boxes = find_contours_with_area(&pixels, w, h, 10).0;
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0], (5, 5, 9, 9));
    }
}

#[cfg(test)]
mod pixel_pipeline_tests {
    use super::*;

    #[test]
    fn detects_motion_without_touching_average() {
        let w = 64usize;
        let h = 64usize;
        let len = w * h;
        let avg = vec![100.0f32; len];
        let avg_before = avg.clone();
        let mut frame = vec![100u8; len];
        for y in 20..40 {
            for x in 20..40 {
                frame[y * w + x] = 200;
            }
        }
        let mask = vec![0u8; len];
        let mut boxes = vec![
            MotionBox {
                x1: 0,
                y1: 0,
                x2: 0,
                y2: 0
            };
            32
        ];
        let mut total_area: f32 = 0.0;

        let n = unsafe {
            motion_pixel_pipeline(
                frame.as_mut_ptr(),
                avg.as_ptr(),
                mask.as_ptr(),
                w as u32,
                h as u32,
                15,
                10,
                1,
                boxes.as_mut_ptr(),
                32,
                &mut total_area,
            )
        };
        assert!(n >= 1, "should find the bright square");
        assert!(
            total_area >= 350.0,
            "area ~400px expected, got {total_area}"
        );
        assert_eq!(avg, avg_before, "average must not be modified");
        let b = &boxes[0];
        assert!(b.x1 >= 17 && b.x1 <= 22, "x1={}", b.x1);
        assert!(b.y1 >= 17 && b.y1 <= 22, "y1={}", b.y1);
    }

    #[test]
    fn no_motion_on_identical_frame() {
        let w = 32usize;
        let h = 32usize;
        let len = w * h;
        let avg = vec![100.0f32; len];
        let mut frame = vec![100u8; len];
        let mask = vec![0u8; len];
        let mut boxes = vec![
            MotionBox {
                x1: 0,
                y1: 0,
                x2: 0,
                y2: 0
            };
            32
        ];
        let mut total_area: f32 = 0.0;
        let n = unsafe {
            motion_pixel_pipeline(
                frame.as_mut_ptr(),
                avg.as_ptr(),
                mask.as_ptr(),
                w as u32,
                h as u32,
                15,
                10,
                1,
                boxes.as_mut_ptr(),
                32,
                &mut total_area,
            )
        };
        assert_eq!(n, 0);
        assert_eq!(total_area, 0.0);
    }

    #[test]
    fn respects_mask() {
        let w = 32usize;
        let h = 32usize;
        let len = w * h;
        let avg = vec![100.0f32; len];
        let mut frame = vec![100u8; len];
        for y in 5..15 {
            for x in 5..15 {
                frame[y * w + x] = 220;
            }
        }
        let mask = vec![1u8; len]; // fully masked → no motion allowed
        let mut boxes = vec![
            MotionBox {
                x1: 0,
                y1: 0,
                x2: 0,
                y2: 0
            };
            32
        ];
        let mut total_area: f32 = 0.0;
        let n = unsafe {
            motion_pixel_pipeline(
                frame.as_mut_ptr(),
                avg.as_ptr(),
                mask.as_ptr(),
                w as u32,
                h as u32,
                15,
                10,
                0,
                boxes.as_mut_ptr(),
                32,
                &mut total_area,
            )
        };
        assert_eq!(n, 0, "masked pixels must not produce motion");
    }
}

// ═══════════════════════════════════════════════════════════════════
//  AVX2 ↔ scalar parity
//
//  The scalar kernels are the fallback taken on aarch64 and on x86_64
//  CPUs without AVX2.  Before runtime dispatch existed they were dead
//  code on x86 and never executed, so nothing proved they agreed with
//  the SIMD path.  These tests do.
// ═══════════════════════════════════════════════════════════════════
#[cfg(all(test, target_arch = "x86_64"))]
mod simd_parity_tests {
    use super::*;

    /// Deterministic pseudo-random bytes — avoids a dev-dependency.
    fn pseudo_random(n: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as u8
            })
            .collect()
    }

    /// Lengths straddling the 32-byte SIMD stride so both the vector
    /// body and the scalar tail are covered.
    const LENS: [usize; 7] = [0, 1, 31, 32, 33, 64, 1000];

    #[test]
    fn absdiff_scalar_matches_avx2() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        for &len in &LENS {
            let a = pseudo_random(len, 0x1234);
            let b = pseudo_random(len, 0x9876);
            let mut simd = vec![0u8; len];
            let mut scalar = vec![0u8; len];
            unsafe { absdiff_avx2_impl(&a, &b, &mut simd, len) };
            absdiff_scalar(&a, &b, &mut scalar, len);
            assert_eq!(simd, scalar, "absdiff mismatch at len={len}");
        }
    }

    #[test]
    fn threshold_mask_scalar_matches_avx2() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        for &len in &LENS {
            for &thresh in &[0u8, 1, 15, 128, 254, 255] {
                let base = pseudo_random(len, 0xABCD);
                // Mask a scattered third of the pixels.
                let mask: Vec<u8> = (0..len).map(|i| if i % 3 == 0 { 1 } else { 0 }).collect();
                let mut simd = base.clone();
                let mut scalar = base.clone();
                unsafe { threshold_mask_avx2_impl(&mut simd, &mask, thresh, len) };
                threshold_mask_scalar(&mut scalar, &mask, thresh, len);
                assert_eq!(
                    simd, scalar,
                    "threshold mismatch at len={len} thresh={thresh}"
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  FFI boundary guards.
//
//  These crates are cdylibs built with panic = "abort", so anything
//  that panics or reads out of bounds takes the whole Frigate process
//  down rather than failing one frame.  Every C-ABI entry point must
//  therefore reject null pointers and degenerate sizes instead of
//  forming a slice from them.
// ═══════════════════════════════════════════════════════════════════
#[cfg(test)]
mod ffi_guard_tests {
    use super::*;

    #[test]
    fn pixel_pipeline_rejects_null_pointers() {
        let mut boxes = [MotionBox {
            x1: 0,
            y1: 0,
            x2: 0,
            y2: 0,
        }; 4];
        let mut area = -1.0f32;
        let n = unsafe {
            motion_pixel_pipeline(
                std::ptr::null_mut(),
                std::ptr::null(),
                std::ptr::null(),
                64,
                64,
                30,
                10,
                1,
                boxes.as_mut_ptr(),
                4,
                &mut area,
            )
        };
        assert_eq!(n, 0);
        assert_eq!(
            area, 0.0,
            "out_total_area must be initialized on the guard path"
        );
    }

    #[test]
    fn pixel_pipeline_rejects_zero_dimensions() {
        let mut frame = vec![0u8; 16];
        let avg = vec![0f32; 16];
        let mask = vec![0u8; 16];
        let mut boxes = [MotionBox {
            x1: 0,
            y1: 0,
            x2: 0,
            y2: 0,
        }; 4];
        for (w, h) in [(0u32, 8u32), (8, 0), (0, 0)] {
            let n = unsafe {
                motion_pixel_pipeline(
                    frame.as_mut_ptr(),
                    avg.as_ptr(),
                    mask.as_ptr(),
                    w,
                    h,
                    30,
                    10,
                    1,
                    boxes.as_mut_ptr(),
                    4,
                    std::ptr::null_mut(),
                )
            };
            assert_eq!(n, 0, "expected no boxes for {w}x{h}");
        }
    }

    #[test]
    fn pixel_pipeline_rejects_null_output_buffer() {
        let mut frame = vec![0u8; 64];
        let avg = vec![0f32; 64];
        let mask = vec![0u8; 64];
        let n = unsafe {
            motion_pixel_pipeline(
                frame.as_mut_ptr(),
                avg.as_ptr(),
                mask.as_ptr(),
                8,
                8,
                30,
                10,
                1,
                std::ptr::null_mut(),
                4,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(n, 0);
    }

    #[test]
    fn pixel_pipeline_handles_tiny_frames() {
        // Ported from test_rust_ffi_guards.py: (1,1), (1,7), (7,1), (3,5)
        for (h, w) in [(1u32, 1u32), (1, 7), (7, 1), (3, 5)] {
            let len = (h * w) as usize;
            let mut frame: Vec<u8> = (0..len).map(|i| (i * 37 % 256) as u8).collect();
            let avg = vec![0.0f32; len];
            let mask = vec![0u8; len];
            let mut boxes = [MotionBox {
                x1: 0,
                y1: 0,
                x2: 0,
                y2: 0,
            }; 8];
            let mut area = 0.0f32;
            let _ = unsafe {
                motion_pixel_pipeline(
                    frame.as_mut_ptr(),
                    avg.as_ptr(),
                    mask.as_ptr(),
                    w,
                    h,
                    10,
                    0,
                    1,
                    boxes.as_mut_ptr(),
                    8,
                    &mut area,
                )
            };
            assert!(area <= len as f32);
        }
    }
}
