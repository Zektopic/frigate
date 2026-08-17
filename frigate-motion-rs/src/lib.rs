/// SIMD-accelerated motion detection engine for Frigate NVR.
///
/// Replaces the CPU-heavy OpenCV+scipy path with architecture-aware
/// SIMD routines.  Exposes a flat C ABI so Python can call via `ctypes`.

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
//  Histogram-based percentile + contrast stretch
// ═══════════════════════════════════════════════════════════════════

/// Compute the p-th percentile (0..100) of a u8 slice using a 256-bin histogram.
#[inline]
fn percentile_u8(data: &[u8], p: f32) -> u8 {
    let mut hist = [0u32; 256];
    for &v in data {
        hist[v as usize] += 1;
    }
    let target = (data.len() as f32 * p / 100.0).ceil() as u32;
    let mut accum: u32 = 0;
    for (i, &count) in hist.iter().enumerate() {
        accum += count;
        if accum >= target {
            return i as u8;
        }
    }
    255
}

/// Apply contrast stretch: clip to [min_val, max_val], then scale to [0, 255].
/// Processes 32 pixels per SIMD iteration.
#[cfg(target_arch = "x86_64")]
unsafe fn contrast_stretch_simd(buf: &mut [u8], min_val: u8, max_val: u8, len: usize) {
    if min_val >= max_val {
        return;
    }
    let min_v = min_val as f32;
    let range = (max_val - min_val) as f32;
    let scale = 255.0 / range;
    let scale_vec = _mm256_set1_ps(scale);
    let min_vec = _mm256_set1_ps(min_v);
    let zero = _mm256_setzero_ps();
    let ff = _mm256_set1_ps(255.0);

    let mut i = 0usize;
    let simd_end = (len / 8) * 8;

    while i < simd_end {
        // Load 8 u8 → expand to f32
        let v8 = _mm_loadl_epi64(buf.as_ptr().add(i) as *const __m128i);
        let v32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v8));

        // clip, scale, clamp
        let clipped = _mm256_max_ps(v32, min_vec);
        let scaled = _mm256_mul_ps(_mm256_sub_ps(clipped, min_vec), scale_vec);
        let clamped = _mm256_min_ps(_mm256_max_ps(scaled, zero), ff);

        // f32 → i32 → pack to u8
        let vi32 = _mm256_cvtps_epi32(clamped);
        // pack 8×i32 → 8×u16 → 8×u8
        let vi16 = _mm_packus_epi32(
            _mm256_castsi256_si128(vi32),
            _mm256_extracti128_si256::<1>(vi32),
        );
        let vi8 = _mm_packus_epi16(vi16, vi16);
        _mm_storel_epi64(buf.as_mut_ptr().add(i) as *mut __m128i, vi8);

        i += 8;
    }

    // Scalar tail
    for j in i..len {
        let v = buf[j] as f32;
        let v = v.max(min_v);
        let v = ((v - min_v) * scale).clamp(0.0, 255.0);
        buf[j] = v as u8;
    }

    fn not_a_test_debug_step_by_step() {
        let w = 64usize;
        let h = 64usize;
        let len = w * h;
        
        let mut avg = vec![100.0f32; len];
        let mut frame = vec![100u8; len];
        for y in 20..40 {
            for x in 20..40 {
                frame[y * w + x] = 200;
            }
        }
        let mask = vec![0u8; len];
        
        let mut buf: Vec<u8> = frame.clone();
        
        // Step 1: mask
        for i in 0..len {
            if mask[i] != 0 { buf[i] = 0; }
        }
        println!("After mask: buf[22*64+22]={} (should be 200)", buf[22*64+22]);
        
        // Step 4-5: absdiff
        let mut avg_u8 = vec![0u8; len];
        for i in 0..len { avg_u8[i] = avg[i].clamp(0.0, 255.0) as u8; }
        println!("avg_u8[22*64+22]={} (should be 100)", avg_u8[22*64+22]);
        
        let mut diff = vec![0u8; len];
        unsafe { absdiff_avx2(&buf, &avg_u8, &mut diff, len); }
        println!("After absdiff: diff[22*64+22]={} (should be 100)", diff[22*64+22]);
        println!("diff non-zero count: {}", diff.iter().filter(|&&v| v > 0).count());
        println!("diff max: {}", diff.iter().max().unwrap());
        
        // Step 5: threshold + mask
        let thresh: u8 = 15;
        unsafe { threshold_mask_avx2(&mut diff, &mask, thresh, len); }
        let white = diff.iter().filter(|&&v| v == 255).count();
        println!("After threshold({thresh}): {} white pixels (should be 400 for 20x20 rect)", white);
        println!("diff[22*64+22]={} (should be 255)", diff[22*64+22]);
        
        // Step 6: dilate
        let mut dilated = vec![0u8; len];
        unsafe { dilate_3x3(&mut diff, &mut dilated, w, h); }
        let white2 = diff.iter().filter(|&&v| v == 255).count();
        println!("After dilate: {} white pixels", white2);
        
        // Step 7: contours
        let boxes = find_contours_bounding_boxes(&diff, w, h, 10);
        println!("Contours found: {}", boxes.len());
        for (i, b) in boxes.iter().enumerate() {
            println!("  box {i}: ({}, {}, {}, {})", b.0, b.1, b.2, b.3);
        }
        assert!(boxes.len() > 0, "Should find at least 1 contour");
    }
}

#[cfg(not(target_arch = "x86_64"))]
unsafe fn contrast_stretch_scalar(buf: &mut [u8], min_val: u8, max_val: u8, len: usize) {
    if min_val >= max_val {
        return;
    }
    let min_v = min_val as f32;
    let range = (max_val - min_val) as f32;
    let scale = 255.0 / range;
    for v in buf.iter_mut().take(len) {
        let f = (*v as f32).max(min_v);
        *v = ((f - min_v) * scale).clamp(0.0, 255.0) as u8;
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Core pixel kernels (from original)
// ═══════════════════════════════════════════════════════════════════

unsafe fn absdiff_avx2(a: &[u8], b: &[u8], dst: &mut [u8], len: usize) {
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

unsafe fn threshold_mask_avx2(buf: &mut [u8], mask: &[u8], thresh: u8, len: usize) {
    let thresh_vec = _mm256_set1_epi8(thresh as i8);
    let zero = _mm256_setzero_si256();
    let ff = _mm256_set1_epi8(-1i8);
    let mut i = 0usize;
    let simd_end = align_up(len.saturating_sub(31), 32) & !31;
    while i < simd_end {
        let v = _mm256_loadu_si256(buf.as_ptr().add(i) as *const __m256i);
        let m = _mm256_loadu_si256(mask.as_ptr().add(i) as *const __m256i);
        // val >= thresh ?
        let ge = _mm256_cmpgt_epi8(thresh_vec, v);   // thresh > val
        let ge = _mm256_xor_si256(ge, ff);            // val >= thresh
        // mask == 0 ? (mask=0 means UNMASKED = process this pixel)
        let unmasked = _mm256_cmpeq_epi8(m, zero);    // m == 0 (unmasked)
        // combine: (val >= thresh) AND (unmasked)
        let res = _mm256_and_si256(ge, unmasked);
        _mm256_storeu_si256(buf.as_mut_ptr().add(i) as *mut __m256i, res);
        i += 32;
    }
    for j in i..len {
        buf[j] = if buf[j] >= thresh && mask[j] == 0 { 255 } else { 0 };
    }
}

unsafe fn dilate_3x3(buf: &mut [u8], tmp: &mut [u8], w: usize, h: usize) {
    let len = w * h;
    tmp[..len].copy_from_slice(&buf[..len]);
    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            if buf[idx] == 255 {
                continue;
            }
            let mut fg = false;
            if x > 0 && tmp[idx - 1] == 255 {
                fg = true;
            } else if x + 1 < w && tmp[idx + 1] == 255 {
                fg = true;
            } else if y > 0 && tmp[idx - w] == 255 {
                fg = true;
            } else if y + 1 < h && tmp[idx + w] == 255 {
                fg = true;
            }
            if fg {
                *buf.get_unchecked_mut(idx) = 255;
            }
        }
    }
}

fn find_contours_bounding_boxes(
    pixels: &[u8], w: usize, h: usize, min_area: i32,
) -> Vec<(i32, i32, i32, i32)> {
    let mut visited = vec![false; w * h];
    let mut boxes: Vec<(i32, i32, i32, i32)> = Vec::with_capacity(64);
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
                if cx < min_x { min_x = cx; }
                if cx > max_x { max_x = cx; }
                if cy < min_y { min_y = cy; }
                if cy > max_y { max_y = cy; }
                let neighbors: [(isize, isize); 4] = [
                    (cx as isize - 1, cy as isize),
                    (cx as isize + 1, cy as isize),
                    (cx as isize, cy as isize - 1),
                    (cx as isize, cy as isize + 1),
                ];
                for (nx, ny) in neighbors {
                    if nx >= 0 && (nx as usize) < w
                        && ny >= 0 && (ny as usize) < h
                    {
                        let nidx = (ny as usize) * w + (nx as usize);
                        if !visited[nidx] && pixels[nidx] == 255 {
                            visited[nidx] = true;
                            stack.push((nx as usize, ny as usize));
                        }
                    }
                }
            }
            if area >= min_area {
                boxes.push((
                    min_x as i32, min_y as i32,
                    (max_x + 1) as i32, (max_y + 1) as i32,
                ));
            }
        }
    }
    boxes
}

unsafe fn update_average_avx2(
    avg: &mut [f32], frame: &[u8], mask: &[u8], len: usize,
) {
    let mut i = 0usize;
    while i + 7 < len {
        let f8 = _mm_loadl_epi64(frame.as_ptr().add(i) as *const __m128i);
        let f32_lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(f8));
        let a = _mm256_loadu_ps(avg.as_ptr().add(i));
        let m8 = _mm_loadl_epi64(mask.as_ptr().add(i) as *const __m128i);
        let m32 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(m8));
        let m_eq0 = _mm256_cmp_ps::<0>(m32, _mm256_setzero_ps());
        let alpha = _mm256_set1_ps(0.01);
        let one_minus_alpha = _mm256_set1_ps(0.99);
        let blended = _mm256_add_ps(
            _mm256_mul_ps(a, one_minus_alpha),
            _mm256_mul_ps(f32_lo, alpha),
        );
        let res = _mm256_blendv_ps(a, blended, m_eq0);
        _mm256_storeu_ps(avg.as_mut_ptr().add(i), res);
        i += 8;
    }
    for j in i..len {
        if mask[j] == 0 {
            avg[j] = avg[j] * 0.99 + frame[j] as f32 * 0.01;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Full motion pipeline — single FFI entry point
// ═══════════════════════════════════════════════════════════════════

/// Run the complete motion detection pipeline in one FFI call.
///
/// This replicates `ImprovedMotionDetector.detect()`:
/// 1. (optional) contrast stretch via histogram percentiles
/// 2. Apply mask (mask[i] != 0 → set pixel to 0)
/// 3. 3×3 Gaussian blur
/// 4. Absolute difference vs running average
/// 5. Binary threshold
/// 6. 3×3 dilate
/// 7. Connected-component contour extraction
/// 8. Update running average (unmasked pixels, alpha=0.01)
///
/// # Safety
/// All pointers must be valid for `w * h` elements (u8 for frame/mask,
/// f32 for avg_frame).  `out_boxes` has room for `max_boxes` entries.
///
/// Returns the number of boxes written.
#[no_mangle]
pub unsafe extern "C" fn motion_detect_full(
    frame: *const u8,        // grayscale u8, w×h
    avg_frame: *mut f32,     // running avg, w×h f32 (updated in-place)
    mask: *const u8,         // binary mask, w×h u8 (0=process, 1=skip)
    w: u32,
    h: u32,
    thresh: u8,              // binary threshold (0-255)
    min_area: u32,           // min contour area in pixels
    improve_contrast: u8,    // 0=skip, 1=apply contrast stretch
    blur_enabled: u8,        // 0=skip, 1=apply 3×3 gaussian blur
    out_boxes: *mut MotionBox,
    max_boxes: u32,
    out_calibrated: *mut u8, // 0=still calibrating, 1=calibrated
) -> u32 {
    let w = w as usize;
    let h = h as usize;
    let len = w * h;

    let frame = std::slice::from_raw_parts(frame, len);
    let avg_frame = std::slice::from_raw_parts_mut(avg_frame, len);
    let mask = std::slice::from_raw_parts(mask, len);

    // Working buffer — allocated once, reused across stages
    let mut buf: Vec<u8> = Vec::with_capacity(len);
    buf.extend_from_slice(frame);
    let mut tmp: Vec<u8> = vec![0u8; len];

    // Stage 1: Contrast stretch (optional)
    if improve_contrast != 0 {
        let p4 = percentile_u8(&buf, 4.0);
        let p96 = percentile_u8(&buf, 96.0);
        if p4 < p96 {
            #[cfg(target_arch = "x86_64")]
            unsafe { contrast_stretch_simd(&mut buf, p4, p96, len); }
            #[cfg(not(target_arch = "x86_64"))]
            unsafe { contrast_stretch_scalar(&mut buf, p4, p96, len); }
        }
    }

    // Stage 2: Apply mask — mask[i]!=0 means "skip this pixel" (set to 0)
    // This matches Python: resized_frame[self.mask] = [0]
    for i in 0..len {
        if mask[i] != 0 {
            buf[i] = 0; // masked pixels become black (won't trigger motion)
        }
    }

    // Stage 3: Gaussian blur (optional)
    if blur_enabled != 0 {
        let mut blur_tmp = vec![0u8; len];
        unsafe { gaussian_blur_3x3(&buf, &mut tmp, &mut blur_tmp, w, h); }
        std::mem::swap(&mut buf, &mut tmp); // buf now has blurred result
    }

    // Stage 4-5: Absdiff + threshold
    // Convert avg_frame (f32) → u8
    let mut avg_u8: Vec<u8> = vec![0u8; len];
    for i in 0..len {
        avg_u8[i] = avg_frame[i].clamp(0.0, 255.0) as u8;
    }

    unsafe { absdiff_avx2(&buf, &avg_u8, &mut tmp, len); }

    // Stage 5: Threshold + mask (in-place on diff result)
    let mut diff = tmp; // reuse buffer
    unsafe { threshold_mask_avx2(&mut diff, mask, thresh, len); }

    // Stage 6: Dilate
    let mut dilated = vec![0u8; len];
    unsafe { dilate_3x3(&mut diff, &mut dilated, w, h); }

    // Stage 7: Contours → boxes
    let boxes = find_contours_bounding_boxes(&diff, w, h, min_area as i32);
    let total_area: f32 = boxes.iter().map(|(x1,y1,x2,y2)| ((x2-x1)*(y2-y1)) as f32).sum();
    let pct_motion = total_area / (len as f32);

    // Calibration: considered calibrated if <5% motion AND <=4 boxes
    let calibrated = (pct_motion < 0.05 && boxes.len() <= 4) as u8;
    if !out_calibrated.is_null() {
        *out_calibrated = calibrated;
    }

    // Stage 8: Update running average
    unsafe { update_average_avx2(avg_frame, frame, mask, len); }

    // Write output boxes
    let n = (boxes.len() as u32).min(max_boxes);
    let out_slice = std::slice::from_raw_parts_mut(out_boxes, n as usize);
    for (i, (x1, y1, x2, y2)) in boxes.into_iter().take(n as usize).enumerate() {
        out_slice[i] = MotionBox { x1, y1, x2, y2 };
    }
    n
}

/// Legacy entry point — kept for backward compat with existing Python bindings.
#[no_mangle]
pub unsafe extern "C" fn motion_detect(
    frame: *const u8,
    avg_frame: *mut f32,
    mask: *const u8,
    w: u32,
    h: u32,
    thresh: u8,
    min_area: u32,
    alpha: f32,
    out_boxes: *mut MotionBox,
    max_boxes: u32,
) -> u32 {
    let _ = alpha;
    motion_detect_full(
        frame, avg_frame, mask, w, h, thresh, min_area,
        0, // no contrast
        0, // no blur
        out_boxes, max_boxes,
        std::ptr::null_mut(),
    )
}

/// Initialize (or reset) the running-average buffer from a frame.
/// Contour extraction that also accumulates total foreground area.
/// Same flood-fill as `find_contours_bounding_boxes` but sums the pixel
/// count of every component (matching Python's `total_contour_area`
/// accumulation across ALL contours, not just kept ones).
fn find_contours_with_area(
    pixels: &[u8], w: usize, h: usize, min_area: i32,
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
                if cx < min_x { min_x = cx; }
                if cx > max_x { max_x = cx; }
                if cy < min_y { min_y = cy; }
                if cy > max_y { max_y = cy; }
                let neighbors: [(isize, isize); 4] = [
                    (cx as isize - 1, cy as isize),
                    (cx as isize + 1, cy as isize),
                    (cx as isize, cy as isize - 1),
                    (cx as isize, cy as isize + 1),
                ];
                for (nx, ny) in neighbors {
                    if nx >= 0 && (nx as usize) < w
                        && ny >= 0 && (ny as usize) < h
                    {
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
                    min_x as i32, min_y as i32,
                    (max_x + 1) as i32, (max_y + 1) as i32,
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
/// # Safety
/// `frame` (mut), `avg_frame` (read-only) and `mask` must each be valid
/// for `w * h` elements.  `out_boxes` holds `max_boxes` entries;
/// `out_total_area` is a single f32.
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
    dilate_3x3(&mut diff, &mut tmp, w, h);

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

#[no_mangle]
pub unsafe extern "C" fn motion_init_average(
    frame: *const u8,
    avg_frame: *mut f32,
    len: u32,
) {
    let frame = std::slice::from_raw_parts(frame, len as usize);
    let avg = std::slice::from_raw_parts_mut(avg_frame, len as usize);
    for i in 0..(len as usize) {
        avg[i] = frame[i] as f32;
    }
}

#[no_mangle]
pub unsafe extern "C" fn motion_accumulate_weighted(
    src: *const u8,
    avg: *mut f32,
    alpha: f32,
    len: u32,
) {
    if src.is_null() || avg.is_null() || len == 0 {
        return;
    }
    let src_slice = std::slice::from_raw_parts(src, len as usize);
    let avg_slice = std::slice::from_raw_parts_mut(avg, len as usize);
    let one_minus_alpha = 1.0 - alpha;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut i = 0usize;
            let chunks = (len as usize) / 8;
            let alpha_vec = _mm256_set1_ps(alpha);
            let one_minus_alpha_vec = _mm256_set1_ps(one_minus_alpha);

            for _ in 0..chunks {
                let bytes = *(src_slice.as_ptr().add(i) as *const [u8; 8]);
                let u32_vals = [
                    bytes[0] as u32,
                    bytes[1] as u32,
                    bytes[2] as u32,
                    bytes[3] as u32,
                    bytes[4] as u32,
                    bytes[5] as u32,
                    bytes[6] as u32,
                    bytes[7] as u32,
                ];
                let src_i32 = _mm256_loadu_si256(u32_vals.as_ptr() as *const __m256i);
                let src_f32 = _mm256_cvtepi32_ps(src_i32);

                let avg_f32 = _mm256_loadu_ps(avg_slice.as_ptr().add(i));
                let res = _mm256_fmadd_ps(alpha_vec, src_f32, _mm256_mul_ps(one_minus_alpha_vec, avg_f32));
                _mm256_storeu_ps(avg_slice.as_mut_ptr().add(i), res);
                i += 8;
            }

            for j in i..(len as usize) {
                avg_slice[j] = one_minus_alpha * avg_slice[j] + alpha * (src_slice[j] as f32);
            }
            return;
        }
    }

    for i in 0..(len as usize) {
        avg_slice[i] = one_minus_alpha * avg_slice[i] + alpha * (src_slice[i] as f32);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile() {
        let data: Vec<u8> = (0..100u8).collect();
        // histogram-based percentile may differ from numpy by ±1 due to no interpolation
        let p4 = percentile_u8(&data, 4.0);
        let p96 = percentile_u8(&data, 96.0);
        let p50 = percentile_u8(&data, 50.0);
        assert!(p4 == 3 || p4 == 4, "p4={p4}");
        assert!(p96 == 95 || p96 == 96, "p96={p96}");
        assert!(p50 == 49 || p50 == 50, "p50={p50}");
    }

    #[test]
    fn test_gaussian_blur_flat() {
        let src = vec![128u8; 100]; // 10×10 flat image
        let mut dst = vec![0u8; 100];
        let mut tmp = vec![0u8; 100];
        unsafe { gaussian_blur_3x3(&src, &mut dst, &mut tmp, 10, 10); }
        assert_eq!(&dst, &src); // flat image unchanged by blur
    }

    #[test]
    fn test_contrast_stretch() {
        let mut buf: Vec<u8> = (0..64u8).collect();
        let len = buf.len();
        unsafe { contrast_stretch_simd(&mut buf, 16, 48, len); }
        // After stretch: values 0-15 become 0, 48-63 become 255, middle scaled
        assert_eq!(buf[0], 0);
        assert_eq!(buf[63], 255);
    }

    #[test]
    fn test_absdiff_identical() {
        let data: Vec<u8> = (0..64u8).collect();
        let mut dst = vec![0u8; 64];
        unsafe { absdiff_avx2(&data, &data, &mut dst, 64); }
        assert!(dst.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_threshold_mask() {
        let mut buf: Vec<u8> = (0..64u8).collect();
        let mask: Vec<u8> = vec![0u8; 64]; // all zeros = unmasked = process all
        unsafe { threshold_mask_avx2(&mut buf, &mask, 32, 64); }
        for i in 0..64 {
            if i >= 32 { assert_eq!(buf[i], 255, "pixel {i} should be 255"); }
            else { assert_eq!(buf[i], 0, "pixel {i} should be 0"); }
        }
    }

    #[test]
    fn test_contours_single_rect() {
        let w = 10usize; let h = 10usize;
        let mut pixels = vec![0u8; w * h];
        for y in 3..6 { for x in 3..6 { pixels[y * w + x] = 255; } }
        let boxes = find_contours_bounding_boxes(&pixels, w, h, 0);
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0], (3, 3, 6, 6));
    }

    #[test]
    fn test_contours_min_area_filter() {
        let w = 10usize; let h = 10usize;
        let mut pixels = vec![0u8; w * h];
        pixels[1 * w + 1] = 255; // 1-pixel dot
        for y in 5..9 { for x in 5..9 { pixels[y * w + x] = 255; } } // 4×4 rect
        let boxes = find_contours_bounding_boxes(&pixels, w, h, 10);
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0], (5, 5, 9, 9));
    }

    #[test]
    fn test_full_pipeline_no_motion() {
        let w = 16u32; let h = 16u32;
        let len = (w * h) as usize;
        let frame: Vec<u8> = vec![100u8; len];
        let mut avg: Vec<f32> = vec![100.0f32; len];
        let mask: Vec<u8> = vec![0u8; len];
        let mut boxes = vec![MotionBox{x1:0,y1:0,x2:0,y2:0}; 16];
        let mut calibrated: u8 = 0;

        let n = unsafe {
            motion_detect_full(
                frame.as_ptr(), avg.as_mut_ptr(), mask.as_ptr(),
                w, h, 25, 30, 0, 0,
                boxes.as_mut_ptr(), 16, &mut calibrated,
            )
        };
        assert_eq!(n, 0, "no motion expected on identical frame");
        assert_eq!(calibrated, 1, "should be calibrated with no motion");
    }

    #[test]
    fn step_by_step_debug() {
        let w = 64usize;
        let h = 64usize;
        let len = w * h;
        let mut avg = vec![100.0f32; len];
        let mut frame = vec![100u8; len];
        for y in 20..40 { for x in 20..40 { frame[y * w + x] = 200; } }
        let mask = vec![0u8; len];
        let mut buf: Vec<u8> = frame.clone();

        // Stage 1: mask
        for i in 0..len { if mask[i] != 0 { buf[i] = 0; } }
        eprintln!("After mask: buf[22*64+22]={}", buf[22*64+22]);

        // Stage 4: absdiff
        let mut avg_u8 = vec![0u8; len];
        for i in 0..len { avg_u8[i] = avg[i].clamp(0.0, 255.0) as u8; }
        let mut diff = vec![0u8; len];
        unsafe { absdiff_avx2(&buf, &avg_u8, &mut diff, len); }
        eprintln!("After absdiff: non_zero={}, max={}", diff.iter().filter(|&&v| v>0).count(), diff.iter().max().unwrap());
        eprintln!("diff[22*64+22]={}", diff[22*64+22]);

        // Stage 5: threshold
        let thresh: u8 = 15;
        unsafe { threshold_mask_avx2(&mut diff, &mask, thresh, len); }
        let white = diff.iter().filter(|&&v| v==255).count();
        eprintln!("After threshold({thresh}): {white} white pixels");
        eprintln!("diff[22*64+22]={}", diff[22*64+22]);

        // Stage 6: dilate
        let mut dilated = vec![0u8; len];
        unsafe { dilate_3x3(&mut diff, &mut dilated, w, h); }
        eprintln!("After dilate: {} white pixels", diff.iter().filter(|&&v| v==255).count());

        // Stage 7: contours
        let boxes = find_contours_bounding_boxes(&diff, w, h, 10);
        eprintln!("Contours found: {}", boxes.len());
        assert!(boxes.len() > 0, "Should find at least 1 contour");
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;

    #[test]
    fn debug_motion_pipeline() {
        let w = 64u32;
        let h = 64u32;
        let len = (w * h) as usize;
        
        // Background: all 100
        let mut avg = vec![100.0f32; len];
        
        // Frame: background + a bright 20x20 square (simulating motion)
        let mut frame = vec![100u8; len];
        for y in 20..40 {
            for x in 20..40 {
                frame[y as usize * 64 + x as usize] = 200;
            }
        }
        
        // Mask: all zeros (process everything)
        let mask = vec![0u8; len];
        
        // Output
        let mut boxes = vec![MotionBox{x1:0,y1:0,x2:0,y2:0}; 64];
        let mut calibrated: u8 = 0;
        
        let n = unsafe {
            motion_detect_full(
                frame.as_ptr(), avg.as_mut_ptr(), mask.as_ptr(),
                w, h, 15, 10, 
                0, // NO contrast (we want raw diff)
                0, // NO blur (simpler test)
                boxes.as_mut_ptr(), 64, &mut calibrated,
            )
        };
        
        println!("Detections: {n}, calibrated: {calibrated}");
        assert!(n > 0, "Should detect the bright rectangle as motion");
        for i in 0..(n as usize).min(5) {
            let b = &boxes[i];
            println!("  box {i}: x1={}, y1={}, x2={}, y2={}", b.x1, b.y1, b.x2, b.y2);
            // The box should roughly cover the 20x20 rectangle at (20,20)-(40,40)
            assert!(b.x1 >= 18 && b.x1 <= 22, "x1 should be ~20, got {}", b.x1);
            assert!(b.y1 >= 18 && b.y1 <= 22, "y1 should be ~20, got {}", b.y1);
        }
    }
}

#[cfg(test)]
mod pixel_pipeline_tests {
    use super::*;

    #[test]
    fn detects_motion_without_touching_average() {
        let w = 64usize; let h = 64usize; let len = w * h;
        let avg = vec![100.0f32; len];
        let avg_before = avg.clone();
        let mut frame = vec![100u8; len];
        for y in 20..40 { for x in 20..40 { frame[y*w+x] = 200; } }
        let mask = vec![0u8; len];
        let mut boxes = vec![MotionBox{x1:0,y1:0,x2:0,y2:0}; 32];
        let mut total_area: f32 = 0.0;

        let n = unsafe {
            motion_pixel_pipeline(
                frame.as_mut_ptr(), avg.as_ptr(), mask.as_ptr(),
                w as u32, h as u32, 15, 10, 1,
                boxes.as_mut_ptr(), 32, &mut total_area,
            )
        };
        assert!(n >= 1, "should find the bright square");
        assert!(total_area >= 350.0, "area ~400px expected, got {total_area}");
        assert_eq!(avg, avg_before, "average must not be modified");
        let b = &boxes[0];
        assert!(b.x1 >= 17 && b.x1 <= 22, "x1={}", b.x1);
        assert!(b.y1 >= 17 && b.y1 <= 22, "y1={}", b.y1);
    }

    #[test]
    fn no_motion_on_identical_frame() {
        let w = 32usize; let h = 32usize; let len = w * h;
        let avg = vec![100.0f32; len];
        let mut frame = vec![100u8; len];
        let mask = vec![0u8; len];
        let mut boxes = vec![MotionBox{x1:0,y1:0,x2:0,y2:0}; 32];
        let mut total_area: f32 = 0.0;
        let n = unsafe {
            motion_pixel_pipeline(
                frame.as_mut_ptr(), avg.as_ptr(), mask.as_ptr(),
                w as u32, h as u32, 15, 10, 1,
                boxes.as_mut_ptr(), 32, &mut total_area,
            )
        };
        assert_eq!(n, 0);
        assert_eq!(total_area, 0.0);
    }

    #[test]
    fn respects_mask() {
        let w = 32usize; let h = 32usize; let len = w * h;
        let avg = vec![100.0f32; len];
        let mut frame = vec![100u8; len];
        for y in 5..15 { for x in 5..15 { frame[y*w+x] = 220; } }
        let mask = vec![1u8; len]; // fully masked → no motion allowed
        let mut boxes = vec![MotionBox{x1:0,y1:0,x2:0,y2:0}; 32];
        let mut total_area: f32 = 0.0;
        let n = unsafe {
            motion_pixel_pipeline(
                frame.as_mut_ptr(), avg.as_ptr(), mask.as_ptr(),
                w as u32, h as u32, 15, 10, 0,
                boxes.as_mut_ptr(), 32, &mut total_area,
            )
        };
        assert_eq!(n, 0, "masked pixels must not produce motion");
    }

    #[test]
    fn test_accumulate_weighted() {
        let len = 64usize;
        let src: Vec<u8> = (0..len).map(|i| (i * 3 % 256) as u8).collect();
        let mut avg = vec![50.0f32; len];
        let alpha = 0.05f32;

        let expected: Vec<f32> = (0..len)
            .map(|i| (1.0 - alpha) * 50.0 + alpha * (src[i] as f32))
            .collect();

        unsafe {
            motion_accumulate_weighted(src.as_ptr(), avg.as_mut_ptr(), alpha, len as u32);
        }

        for i in 0..len {
            let diff = (avg[i] - expected[i]).abs();
            assert!(diff < 1e-4, "at index {}: avg={} expected={}", i, avg[i], expected[i]);
        }
    }
}
