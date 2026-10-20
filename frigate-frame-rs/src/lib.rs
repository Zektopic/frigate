//! Frame and box kernels for Frigate NVR.
//!
//! Provides YUV420 crop/repack and 3-channel expansion, bilinear resize,
//! u8→f32 normalization (optionally fused with NHWC→NCHW), the FFmpeg
//! frame reader, tracker distances and a few box/region helpers.
//!
//! Exposes a flat C ABI for Python `ctypes` (see `frigate/util/frame_rs.py`).
//! Only kernels that beat their Python/OpenCV fallback once FFI overhead is
//! counted belong here.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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

    // Guard the FFI boundary: from_raw_parts on a null pointer is
    // UB even at zero length, and zero dimensions underflow the
    // `- 1` index clamps below.
    if src.is_null() || dst.is_null() || sw == 0 || sh == 0 || dw == 0 || dh == 0 || ch == 0 {
        return;
    }

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
pub unsafe extern "C" fn normalize_u8_to_f32(src: *const u8, dst: *mut f32, len: u32) {
    let len = len as usize;
    if src.is_null() || dst.is_null() || len == 0 {
        return;
    }

    let src = std::slice::from_raw_parts(src, len);
    let dst = std::slice::from_raw_parts_mut(dst, len);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return normalize_u8_to_f32_avx2(src, dst, len);
        }
    }
    normalize_u8_to_f32_scalar(src, dst, len)
}

/// AVX2 kernel for [`normalize_u8_to_f32`], 8 lanes per iteration.
///
/// # Safety
/// Caller must have verified AVX2 is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn normalize_u8_to_f32_avx2(src: &[u8], dst: &mut [f32], len: usize) {
    let inv255: f32 = 1.0 / 255.0;
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

/// Portable reference kernel for [`normalize_u8_to_f32`].
fn normalize_u8_to_f32_scalar(src: &[u8], dst: &mut [f32], len: usize) {
    let inv255: f32 = 1.0 / 255.0;
    for j in 0..len {
        dst[j] = src[j] as f32 * inv255;
    }
}

// ── NHWC → NCHW transpose (single-batch) ───────────────────────────

// ── Combined pipeline: resize + normalize + transpose ───────────────

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

    // Guard the FFI boundary, and reject a crop window that is not
    // fully contained in the source plane.
    if src.is_null() || out.is_null() || sw == 0 || sh == 0 || cw == 0 || ch == 0 {
        return;
    }
    if cx + cw > sw || cy + ch > sh {
        return;
    }

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
    // The chroma planes are (w/2)*(h/2) (floor), so an odd width or height
    // would index one element past their end; with panic = "abort" that
    // bounds-check panic kills the process.
    if y.is_null()
        || u.is_null()
        || v.is_null()
        || out.is_null()
        || w == 0
        || h == 0
        || !w.is_multiple_of(2)
        || !h.is_multiple_of(2)
    {
        return;
    }

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
    if src.is_null() || dst.is_null() || h == 0 || w == 0 || c == 0 {
        return;
    }

    let hw = h * w;
    let src = std::slice::from_raw_parts(src, hw * c);
    let dst = std::slice::from_raw_parts_mut(dst, c * hw);
    let inv255: f32 = 1.0 / 255.0;

    if c == 3 {
        let (p0, rest) = dst.split_at_mut(hw);
        let (p1, p2) = rest.split_at_mut(hw);
        for i in 0..hw {
            let base = i * 3;
            *p0.get_unchecked_mut(i) = *src.get_unchecked(base) as f32 * inv255;
            *p1.get_unchecked_mut(i) = *src.get_unchecked(base + 1) as f32 * inv255;
            *p2.get_unchecked_mut(i) = *src.get_unchecked(base + 2) as f32 * inv255;
        }
        return;
    }

    for ci in 0..c {
        for hi in 0..h {
            for wi in 0..w {
                let src_idx = (hi * w + wi) * c + ci;
                let dst_idx = ci * hw + hi * w + wi;
                *dst.get_unchecked_mut(dst_idx) = *src.get_unchecked(src_idx) as f32 * inv255;
            }
        }
    }
}

// ── Read frame from FFmpeg stdout directly into buffer ─────────────

/// Read exactly `frame_size` bytes from raw file descriptor `fd` directly
/// into memory location pointed to by `dst` pointer.
///
/// Returns 1 on success, 0 on EOF, and -1 on error.
///
/// # Safety
/// `dst` must be a valid pointer pointing to a buffer of at least `frame_size` bytes.
#[no_mangle]
pub unsafe extern "C" fn read_ffmpeg_frame(fd: i32, dst: *mut u8, frame_size: u32) -> i32 {
    use std::fs::File;
    use std::io::Read;
    use std::os::unix::io::FromRawFd;

    // Validate everything *before* touching the fd.  File::from_raw_fd
    // asserts on a negative descriptor, and this crate is a cdylib
    // built with panic = "abort" — so a closed pipe during an ffmpeg
    // restart would abort the whole Frigate process rather than
    // failing one frame.  The caller already treats <= 0 as an error.
    if fd < 0 || dst.is_null() || frame_size == 0 {
        return -1;
    }

    let mut file = std::mem::ManuallyDrop::new(File::from_raw_fd(fd));
    let slice = std::slice::from_raw_parts_mut(dst, frame_size as usize);
    match file.read_exact(slice) {
        Ok(_) => 1,
        Err(e) => {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                0
            } else {
                -1
            }
        }
    }
}

// ── Tracker association distance ───────────────────────────────────

/// Norfair tracker association distance (port of
/// `frigate.track.norfair_tracker.distance`).
///
/// `det` and `est` each point to 4 f64s `[x1, y1, x2, y2]`.  Returns the
/// euclidean norm of the change vector (relative bottom-center shift plus
/// width/height ratios), or +inf for degenerate / non-finite boxes.
///
/// # Safety
/// `det` and `est` must point to valid arrays of 4 f64s.
#[no_mangle]
pub unsafe extern "C" fn track_distance(det: *const f64, est: *const f64) -> f64 {
    // +inf matches the existing "degenerate input" contract, so the
    // caller treats it as an impossible association rather than a match.
    if det.is_null() || est.is_null() {
        return f64::INFINITY;
    }
    let d = std::slice::from_raw_parts(det, 4);
    let e = std::slice::from_raw_parts(est, 4);

    let ew = e[2] - e[0];
    let eh = e[3] - e[1];
    let dw = d[2] - d[0];
    let dh = d[3] - d[1];
    if !ew.is_finite()
        || !eh.is_finite()
        || !dw.is_finite()
        || !dh.is_finite()
        || ew <= 0.0
        || eh <= 0.0
        || dw <= 0.0
        || dh <= 0.0
    {
        return f64::INFINITY;
    }

    // bottom-center positions, shift relative to estimate size
    let dx = ((d[0] + d[2]) * 0.5 - (e[0] + e[2]) * 0.5) / ew;
    let dy = (d[1].max(d[3]) - e[1].max(e[3])) / eh;

    // width/height ratios normalized to 0
    let wr = ew.max(dw) / ew.min(dw) - 1.0;
    let hr = eh.max(dh) / eh.min(dh) - 1.0;

    (dx * dx + dy * dy + wr * wr + hr * hr).sqrt()
}

/// Vectorized NxM pairwise tracker distance matrix.
#[no_mangle]
pub unsafe extern "C" fn batch_track_distance_matrix(
    dets: *const f64,
    n_dets: usize,
    ests: *const f64,
    n_ests: usize,
    out_matrix: *mut f64,
) {
    if dets.is_null() || ests.is_null() || out_matrix.is_null() || n_dets == 0 || n_ests == 0 {
        return;
    }
    for i in 0..n_dets {
        let det_ptr = dets.add(i * 4);
        for j in 0..n_ests {
            let est_ptr = ests.add(j * 4);
            let dist = track_distance(det_ptr, est_ptr);
            *out_matrix.add(i * n_ests + j) = dist;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Tier A Hot-Path Kernels
// ═══════════════════════════════════════════════════════════════════

#[inline]
fn intersects_i32(a: [i32; 4], b: [i32; 4]) -> bool {
    !(a[2] < b[0] || a[0] > b[2] || a[1] > b[3] || a[3] < b[1])
}

#[inline]
fn inside_i32(a: [i32; 4], b: [i32; 4]) -> bool {
    a[0] >= b[0] && a[1] >= b[1] && a[2] <= b[2] && a[3] <= b[3]
}

#[inline]
fn area_i32(b: [i32; 4]) -> i64 {
    (b[2] as i64 - b[0] as i64 + 1).max(0) * (b[3] as i64 - b[1] as i64 + 1).max(0)
}

#[inline]
fn iou_i32(a: [i32; 4], b: [i32; 4]) -> f32 {
    if !intersects_i32(a, b) {
        return 0.0;
    }
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter_w = (x2 as i64 - x1 as i64 + 1).max(0);
    let inter_h = (y2 as i64 - y1 as i64 + 1).max(0);
    let inter_area = (inter_w * inter_h) as f32;
    if inter_area <= 0.0 {
        return 0.0;
    }
    let union_area = (area_i32(a) + area_i32(b)) as f32 - inter_area;
    if union_area <= 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}

/// Greedy iterative bounding-box union reduction (replaces `frigate.util.object.reduce_boxes`).
/// Writes reduced `[x1, y1, x2, y2]` boxes into `out_boxes` (capacity `n_boxes * 4`)
/// and returns the number of reduced clusters.
#[no_mangle]
pub unsafe extern "C" fn reduce_boxes_i32(
    boxes: *const i32,
    n_boxes: usize,
    iou_threshold: f32,
    out_boxes: *mut i32,
) -> usize {
    if boxes.is_null() || out_boxes.is_null() || n_boxes == 0 {
        return 0;
    }
    let src = std::slice::from_raw_parts(boxes, n_boxes * 4);
    let mut clusters: Vec<[i32; 4]> = Vec::with_capacity(n_boxes);
    for i in 0..n_boxes {
        let b = [src[i * 4], src[i * 4 + 1], src[i * 4 + 2], src[i * 4 + 3]];
        let mut matched = false;
        for cluster in clusters.iter_mut() {
            if iou_i32(b, *cluster) > iou_threshold {
                matched = true;
                cluster[0] = cluster[0].min(b[0]);
                cluster[1] = cluster[1].min(b[1]);
                cluster[2] = cluster[2].max(b[2]);
                cluster[3] = cluster[3].max(b[3]);
            }
        }
        if !matched {
            clusters.push(b);
        }
    }
    let out = std::slice::from_raw_parts_mut(out_boxes, clusters.len() * 4);
    for (i, c) in clusters.iter().enumerate() {
        out[i * 4] = c[0];
        out[i * 4 + 1] = c[1];
        out[i * 4 + 2] = c[2];
        out[i * 4 + 3] = c[3];
    }
    clusters.len()
}

#[inline]
fn cluster_boundary_i32(b: [i32; 4], min_region: i32) -> [i32; 4] {
    let bw = (b[2] - b[0]) as f64;
    let bh = (b[3] - b[1]) as f64;
    let max_region_area = (bw * bh).abs() / 0.1;
    let max_region_size = min_region.max(max_region_area.sqrt() as i32) as f64;

    let cx = bw / 2.0 + b[0] as f64;
    let cy = bh / 2.0 + b[1] as f64;

    let max_x_dist = (max_region_size - (bw / 2.0) * 1.1) as i32 as f64;
    let max_y_dist = (max_region_size - (bh / 2.0) * 1.1) as i32 as f64;

    [
        (cx - max_x_dist) as i32,
        (cy - max_y_dist) as i32,
        (cx + max_x_dist) as i32,
        (cy + max_y_dist) as i32,
    ]
}

#[inline]
fn calculate_region_i32(
    frame_h: i32,
    frame_w: i32,
    xmin: i32,
    ymin: i32,
    xmax: i32,
    ymax: i32,
    model_size: i32,
    multiplier: f64,
) -> [i32; 4] {
    let longest = (xmax - xmin).max(ymax - ymin) as f64 * multiplier;
    let mut size = (longest.floor() as i32 / 4) * 4;
    if size < model_size {
        size = model_size;
    }
    let mut x_offset = ((xmax - xmin) as f64 / 2.0 + xmin as f64 - size as f64 / 2.0) as i32;
    if x_offset < 0 {
        x_offset = 0;
    } else if x_offset > frame_w - size {
        x_offset = (frame_w - size).max(0);
    }
    let mut y_offset = ((ymax - ymin) as f64 / 2.0 + ymin as f64 - size as f64 / 2.0) as i32;
    if y_offset < 0 {
        y_offset = 0;
    } else if y_offset > frame_h - size {
        y_offset = (frame_h - size).max(0);
    }
    [x_offset, y_offset, x_offset + size, y_offset + size]
}

/// Partition `n_boxes` candidate boxes into connected detection clusters
/// (replaces `frigate.util.object.get_cluster_candidates`).
///
/// Writes flattened box indices into `out_flat_indices` (capacity `n_boxes`)
/// and cluster lengths into `out_cluster_lengths` (capacity `n_boxes`).
/// Returns the number of clusters written.
#[no_mangle]
pub unsafe extern "C" fn cluster_candidates_i32(
    boxes: *const i32,
    n_boxes: usize,
    frame_h: i32,
    frame_w: i32,
    min_region: i32,
    out_flat_indices: *mut i32,
    out_cluster_lengths: *mut i32,
) -> usize {
    if boxes.is_null()
        || out_flat_indices.is_null()
        || out_cluster_lengths.is_null()
        || n_boxes == 0
    {
        return 0;
    }
    let raw = std::slice::from_raw_parts(boxes, n_boxes * 4);
    let mut box_arr: Vec<[i32; 4]> = Vec::with_capacity(n_boxes);
    for i in 0..n_boxes {
        box_arr.push([raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]]);
    }

    let mut used = vec![false; n_boxes];
    let mut flat_pos = 0usize;
    let mut num_clusters = 0usize;
    let out_flat = std::slice::from_raw_parts_mut(out_flat_indices, n_boxes);
    let out_lens = std::slice::from_raw_parts_mut(out_cluster_lengths, n_boxes);

    for current_idx in 0..n_boxes {
        if used[current_idx] {
            continue;
        }
        let mut cluster: Vec<usize> = vec![current_idx];
        used[current_idx] = true;
        let cluster_boundary = cluster_boundary_i32(box_arr[current_idx], min_region);

        for compare_idx in 0..n_boxes {
            if used[compare_idx] {
                continue;
            }
            if !inside_i32(box_arr[compare_idx], cluster_boundary) {
                continue;
            }

            // Compute potential cluster region with compare_idx included
            let mut min_x = frame_w.min(box_arr[compare_idx][0]);
            let mut min_y = frame_h.min(box_arr[compare_idx][1]);
            let mut max_x = 0.max(box_arr[compare_idx][2]);
            let mut max_y = 0.max(box_arr[compare_idx][3]);
            for &ci in &cluster {
                let cb = box_arr[ci];
                if cb[0] < min_x {
                    min_x = cb[0];
                }
                if cb[1] < min_y {
                    min_y = cb[1];
                }
                if cb[2] > max_x {
                    max_x = cb[2];
                }
                if cb[3] > max_y {
                    max_y = cb[3];
                }
            }
            let region = calculate_region_i32(
                frame_h, frame_w, min_x, min_y, max_x, max_y, min_region, 1.35,
            );

            let mut should_cluster = true;
            if (region[2] - region[0]) > min_region {
                let reg_area = area_i32(region) as f64;
                let check_box = |idx: usize| -> bool {
                    let b = box_arr[idx];
                    let bw = (b[2] - b[0]) as f64;
                    let bh = (b[3] - b[1]) as f64;
                    (bw * bh) / reg_area >= 0.05
                };
                if reg_area <= 0.0
                    || !check_box(compare_idx)
                    || !cluster.iter().all(|&ci| check_box(ci))
                {
                    should_cluster = false;
                }
            }

            if should_cluster {
                cluster.push(compare_idx);
                used[compare_idx] = true;
            }
        }

        for &idx in &cluster {
            out_flat[flat_pos] = idx as i32;
            flat_pos += 1;
        }
        out_lens[num_clusters] = cluster.len() as i32;
        num_clusters += 1;
    }

    num_clusters
}

/// Consolidate overlapping detections sorted by area ascending within a label group
/// (replaces inner loop of `frigate.util.object.get_consolidated_object_detections`).
///
/// `boxes` has `n * 4` `i32`s (`[x1, y1, x2, y2]`) ordered from smallest area to largest.
/// Writes indices of kept detections into `out_keep` (capacity `n`) and returns kept count.
#[no_mangle]
pub unsafe extern "C" fn consolidate_detections_i32(
    boxes: *const i32,
    n: usize,
    threshold: f32,
    out_keep: *mut i32,
) -> usize {
    if boxes.is_null() || out_keep.is_null() || n == 0 {
        return 0;
    }
    let raw = std::slice::from_raw_parts(boxes, n * 4);
    let out = std::slice::from_raw_parts_mut(out_keep, n);
    let mut kept = 0usize;
    let thresh = threshold as f64;

    for i in 0..n {
        let cur = [raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]];
        let cur_area = area_i32(cur) as f64;
        if cur_area <= 0.0 {
            continue;
        }
        let mut overlap = false;
        for j in (i + 1)..n {
            let oth = [raw[j * 4], raw[j * 4 + 1], raw[j * 4 + 2], raw[j * 4 + 3]];
            let oth_area = area_i32(oth) as f64;
            if oth_area <= 0.0 || (cur_area / oth_area) < 0.05 {
                continue;
            }
            if intersects_i32(cur, oth) {
                let ix1 = cur[0].max(oth[0]);
                let iy1 = cur[1].max(oth[1]);
                let ix2 = cur[2].min(oth[2]);
                let iy2 = cur[3].min(oth[3]);
                let inter_area = area_i32([ix1, iy1, ix2, iy2]) as f64;
                if (inter_area / cur_area) > thresh {
                    overlap = true;
                    break;
                }
            }
        }
        if !overlap {
            out[kept] = i as i32;
            kept += 1;
        }
    }
    kept
}

/// Zero-mean Normalized Cross-Correlation (`cv2.TM_CCOEFF_NORMED`) between two
/// equal-sized `u8` patches `a` and `b` of length `len`.
///
/// Replaces `cv2.matchTemplate(curr_gray, reference_crop, cv2.TM_CCOEFF_NORMED)`
/// in `StationaryMotionClassifier.evaluate`.
#[no_mangle]
pub unsafe extern "C" fn ncc_u8(a: *const u8, b: *const u8, len: usize) -> f64 {
    if a.is_null() || b.is_null() || len == 0 {
        return 0.0;
    }
    let a = std::slice::from_raw_parts(a, len);
    let b = std::slice::from_raw_parts(b, len);

    let mut sum_a: u64 = 0;
    let mut sum_b: u64 = 0;
    for i in 0..len {
        sum_a += a[i] as u64;
        sum_b += b[i] as u64;
    }
    let mean_a = sum_a as f64 / len as f64;
    let mean_b = sum_b as f64 / len as f64;

    let mut num: f64 = 0.0;
    let mut den_a: f64 = 0.0;
    let mut den_b: f64 = 0.0;
    for i in 0..len {
        let da = a[i] as f64 - mean_a;
        let db = b[i] as f64 - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }
    let denom = (den_a * den_b).sqrt();
    if denom <= 1e-12 {
        0.0
    } else {
        (num / denom).clamp(-1.0, 1.0)
    }
}

#[inline]
fn linear_percentile_sorted(sorted: &[i32], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0] as f64;
    }
    let idx = (q / 100.0) * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    sorted[lo] as f64 + frac * (sorted[hi] as f64 - sorted[lo] as f64)
}

/// Compute `[percentile(xmins, 15), percentile(ymins, 15), percentile(xmaxs, 85), percentile(ymaxs, 85)]`
/// with linear interpolation matching `np.percentile` for `NorfairTracker.update_position`.
#[no_mangle]
pub unsafe extern "C" fn percentile_15_85_i32(
    boxes: *const i32,
    n_boxes: usize,
    out_bounds: *mut f64,
) {
    if boxes.is_null() || out_bounds.is_null() || n_boxes == 0 {
        return;
    }
    let raw = std::slice::from_raw_parts(boxes, n_boxes * 4);
    let out = std::slice::from_raw_parts_mut(out_bounds, 4);
    let qs = [15.0, 15.0, 85.0, 85.0];
    let mut col = Vec::with_capacity(n_boxes);
    for c in 0..4 {
        col.clear();
        for i in 0..n_boxes {
            col.push(raw[i * 4 + c]);
        }
        col.sort_unstable();
        out[c] = linear_percentile_sorted(&col, qs[c]);
    }
}

/// Streaming single-pass variance of the 3×3 discrete Laplacian (`[0,1,0; 1,-4,1; 0,1,0]`)
/// with `BORDER_REFLECT_101` border handling, matching `cv2.Laplacian(input, cv2.CV_64F).var()`.
#[no_mangle]
pub unsafe extern "C" fn laplacian_variance_u8(
    src: *const u8,
    w: u32,
    h: u32,
    channels: u32,
) -> f64 {
    let w = w as usize;
    let h = h as usize;
    let ch = channels as usize;
    if src.is_null() || w == 0 || h == 0 || ch == 0 {
        return 0.0;
    }
    let len = w * h * ch;
    let data = std::slice::from_raw_parts(src, len);

    let mut sum: i64 = 0;
    let mut sum_sq: i64 = 0;

    for y in 0..h {
        let ym1 = if y > 0 {
            y - 1
        } else if h > 1 {
            1
        } else {
            0
        };
        let yp1 = if y + 1 < h {
            y + 1
        } else if h > 1 {
            h - 2
        } else {
            0
        };

        for x in 0..w {
            let xm1 = if x > 0 {
                x - 1
            } else if w > 1 {
                1
            } else {
                0
            };
            let xp1 = if x + 1 < w {
                x + 1
            } else if w > 1 {
                w - 2
            } else {
                0
            };

            let c_base = (y * w + x) * ch;
            let t_base = (ym1 * w + x) * ch;
            let b_base = (yp1 * w + x) * ch;
            let l_base = (y * w + xm1) * ch;
            let r_base = (y * w + xp1) * ch;

            for c in 0..ch {
                let center = *data.get_unchecked(c_base + c) as i64;
                let top = *data.get_unchecked(t_base + c) as i64;
                let bot = *data.get_unchecked(b_base + c) as i64;
                let left = *data.get_unchecked(l_base + c) as i64;
                let right = *data.get_unchecked(r_base + c) as i64;

                let lap = top + bot + left + right - 4 * center;
                sum += lap;
                sum_sq += lap * lap;
            }
        }
    }

    let n = len as f64;
    let mean = sum as f64 / n;
    (sum_sq as f64 / n) - mean * mean
}

/// Compute cosine similarity between `query` (`dim` floats) and `n_gallery` vectors
/// in `gallery` (`n_gallery * dim` floats), writing results to `out_scores` (`n_gallery` floats).
#[no_mangle]
pub unsafe extern "C" fn batch_cosine_similarity_f32(
    query: *const f32,
    gallery: *const f32,
    n_gallery: usize,
    dim: usize,
    out_scores: *mut f32,
) {
    if query.is_null() || gallery.is_null() || out_scores.is_null() || n_gallery == 0 || dim == 0 {
        return;
    }
    let q = std::slice::from_raw_parts(query, dim);
    let g = std::slice::from_raw_parts(gallery, n_gallery * dim);
    let out = std::slice::from_raw_parts_mut(out_scores, n_gallery);

    let mut q_norm_sq: f64 = 0.0;
    for &v in q {
        let vf = v as f64;
        q_norm_sq += vf * vf;
    }
    let q_norm = q_norm_sq.sqrt();
    if q_norm <= 1e-12 || !q_norm.is_finite() {
        for o in out.iter_mut() {
            *o = 0.0;
        }
        return;
    }

    for i in 0..n_gallery {
        let row = &g[i * dim..(i + 1) * dim];
        let mut dot: f64 = 0.0;
        let mut g_norm_sq: f64 = 0.0;
        for d in 0..dim {
            let qv = q[d] as f64;
            let gv = row[d] as f64;
            dot += qv * gv;
            g_norm_sq += gv * gv;
        }
        let denom = q_norm * g_norm_sq.sqrt();
        out[i] = if denom <= 1e-12 || !denom.is_finite() {
            0.0
        } else {
            (dot / denom) as f32
        };
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::erasing_op)] // index math documents tensor layout
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
        assert_eq!(dst[0], 0);
        assert_eq!(dst[1], 2);
        assert_eq!(dst[2], 8);
        assert_eq!(dst[3], 10);
    }

    #[test]
    fn test_bilinear_uniform_rgb_64x48_to_32x32() {
        // Ported from test_tracker_rust.py
        let src = vec![200u8; 64 * 48 * 3];
        let mut dst = vec![0u8; 32 * 32 * 3];
        unsafe {
            bilinear_resize_u8(src.as_ptr(), dst.as_mut_ptr(), 48, 64, 32, 32, 3);
        }
        assert!(dst.iter().all(|&v| v == 200));
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
    fn test_transform_detect_input() {
        let src = vec![0u8, 128, 255, 255, 128, 0];
        let mut dst = vec![0.0f32; 6];
        unsafe {
            transform_detect_input(src.as_ptr(), dst.as_mut_ptr(), 1, 2, 3);
        }
        assert!((dst[0] - 0.0).abs() < 1e-6);
        assert!((dst[1] - 1.0).abs() < 1e-6);
        assert!((dst[2] - 128.0 / 255.0).abs() < 1e-6);
        assert!((dst[3] - 128.0 / 255.0).abs() < 1e-6);
        assert!((dst[4] - 1.0).abs() < 1e-6);
        assert!((dst[5] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_yuv420_to_3channel() {
        let y = vec![10u8, 20, 30, 40]; // 2x2
        let u = vec![100u8]; // 1x1
        let v = vec![200u8]; // 1x1
        let mut out = vec![0u8; 2 * 2 * 3];
        unsafe {
            yuv420_to_3channel(y.as_ptr(), u.as_ptr(), v.as_ptr(), 2, 2, out.as_mut_ptr());
        }
        assert_eq!(
            out,
            vec![10, 100, 200, 20, 100, 200, 30, 100, 200, 40, 100, 200]
        );
    }

    #[test]
    fn test_track_distance_and_batch_matrix() {
        // Ported from test_track_distance_rust.py, test_tracker_rust.py, and test_fuzzing.py
        let a = [10.0f64, 20.0, 110.0, 220.0];
        unsafe {
            assert_eq!(track_distance(a.as_ptr(), a.as_ptr()), 0.0);
        }

        let b = [60.0f64, 20.0, 160.0, 220.0];
        unsafe {
            let dist = track_distance(b.as_ptr(), a.as_ptr());
            assert!((dist - 0.5).abs() < 1e-12, "got {dist}");
        }

        let c = [10.0f64, 20.0, 210.0, 220.0];
        unsafe {
            let dist = track_distance(c.as_ptr(), a.as_ptr());
            let want = (0.5f64 * 0.5 + 1.0).sqrt();
            assert!((dist - want).abs() < 1e-12, "got {dist}, want {want}");
        }

        // Batch 2x2 matrix parity with scalar track_distance
        let dets = [10.0f64, 20.0, 110.0, 220.0, 60.0, 20.0, 160.0, 220.0];
        let ests = [10.0f64, 20.0, 110.0, 220.0, 10.0, 20.0, 210.0, 220.0];
        let mut mat = [0.0f64; 4];
        unsafe {
            batch_track_distance_matrix(dets.as_ptr(), 2, ests.as_ptr(), 2, mat.as_mut_ptr());
            for i in 0..2 {
                for j in 0..2 {
                    let want = track_distance(dets.as_ptr().add(i * 4), ests.as_ptr().add(j * 4));
                    assert!((mat[i * 2 + j] - want).abs() < 1e-12);
                }
            }
        }

        // Tiny boxes from test_track_distance_rust.py
        let tiny = [0.0f64, 0.0, 1e-9, 1e-9];
        let unit = [0.0f64, 0.0, 1.0, 1.0];
        unsafe {
            let d = track_distance(tiny.as_ptr(), unit.as_ptr());
            assert!(d.is_finite() && d > 1e8);
        }

        // Degenerate / fuzz cases from test_track_distance_rust.py & test_fuzzing.py
        let extreme_cases = [
            [10.0f64, 20.0, 10.0, 220.0],
            [10.0, 220.0, 110.0, 20.0],
            [f64::NAN, 20.0, 110.0, 220.0],
            [10.0, f64::NAN, 110.0, 220.0],
            [f64::INFINITY, 20.0, 110.0, 220.0],
            [10.0, f64::NEG_INFINITY, 110.0, 220.0],
            [100.0, 100.0, 10.0, 10.0],
            [50.0, 50.0, 50.0, 50.0],
        ];
        for bad in &extreme_cases {
            unsafe {
                assert!(track_distance(bad.as_ptr(), a.as_ptr()).is_infinite());
                assert!(track_distance(a.as_ptr(), bad.as_ptr()).is_infinite());
            }
        }
    }

    #[test]
    fn test_reduce_boxes_i32() {
        let boxes = [
            10, 10, 50, 50, 20, 20, 60, 60, // overlaps with first (IoU > 0)
            100, 100, 140, 140, // disjoint
        ];
        let mut out = [0i32; 12];
        let n = unsafe { reduce_boxes_i32(boxes.as_ptr(), 3, 0.0, out.as_mut_ptr()) };
        assert_eq!(n, 2);
        assert_eq!(&out[0..4], &[10, 10, 60, 60]);
        assert_eq!(&out[4..8], &[100, 100, 140, 140]);
    }

    #[test]
    fn test_cluster_candidates_i32_parity_with_test_video() {
        let run = |boxes: &[[i32; 4]]| -> Vec<Vec<i32>> {
            let flat_boxes: Vec<i32> = boxes.iter().flat_map(|b| b.iter().copied()).collect();
            let mut out_flat = vec![0i32; boxes.len()];
            let mut out_lens = vec![0i32; boxes.len()];
            let n_clusters = unsafe {
                cluster_candidates_i32(
                    flat_boxes.as_ptr(),
                    boxes.len(),
                    1000,
                    2000,
                    160,
                    out_flat.as_mut_ptr(),
                    out_lens.as_mut_ptr(),
                )
            };
            let mut res = Vec::with_capacity(n_clusters);
            let mut pos = 0usize;
            for &len in &out_lens[..n_clusters] {
                let len = len as usize;
                res.push(out_flat[pos..pos + len].to_vec());
                pos += len;
            }
            res
        };

        // 1. test_cluster_candidates
        let c1 = run(&[
            [100, 100, 200, 200],
            [202, 150, 252, 200],
            [900, 900, 950, 950],
        ]);
        assert_eq!(c1.len(), 2);

        // 2. test_cluster_candidates_partition_boxes
        let b2 = [
            [100, 100, 200, 200],
            [202, 150, 252, 200],
            [210, 160, 260, 210],
            [900, 900, 950, 950],
            [905, 905, 955, 955],
        ];
        let c2 = run(&b2);
        let mut assigned: Vec<i32> = c2.into_iter().flatten().collect();
        assigned.sort_unstable();
        assert_eq!(assigned, vec![0, 1, 2, 3, 4]);

        // 3. test_box_too_small_for_cluster
        let c3 = run(&[[100, 100, 600, 600], [655, 100, 700, 145]]);
        assert_eq!(c3.len(), 2);

        // 4. test_redundant_clusters
        let c4 = run(&[[100, 100, 200, 200], [305, 305, 415, 415]]);
        assert_eq!(c4.len(), 2);
        assert!(c4.iter().all(|c| c.len() == 1));

        // 5. test_combine_boxes
        let c5 = run(&[[480, 0, 540, 128], [536, 0, 558, 99]]);
        assert_eq!(c5.len(), 1);

        // 6. test_dont_combine_smaller_boxes
        let c6 = run(&[[460, 0, 561, 144], [565, 0, 586, 71]]);
        assert_eq!(c6.len(), 2);

        // 7. test_dont_combine_boxes
        let c7 = run(&[[460, 0, 532, 129], [586, 0, 606, 46]]);
        assert_eq!(c7.len(), 2);
    }

    #[test]
    fn test_consolidate_detections_i32() {
        // Overlapping boxes where smaller is >70% inside larger -> smaller dropped
        let boxes = [
            1238, 73, 1401, 163, // smaller, inside second
            1209, 73, 1437, 171, // larger
        ];
        let mut keep = [0i32; 2];
        let n = unsafe { consolidate_detections_i32(boxes.as_ptr(), 2, 0.7, keep.as_mut_ptr()) };
        assert_eq!(n, 1);
        assert_eq!(keep[0], 1);

        // Vertically stacked cars from test_video.py -> neither dropped
        let stacked = [954, 312, 1247, 475, 970, 380, 1273, 610];
        let n2 = unsafe { consolidate_detections_i32(stacked.as_ptr(), 2, 0.7, keep.as_mut_ptr()) };
        assert_eq!(n2, 2);

        // Different size objects (<5% area ratio) from test_video.py -> neither dropped
        let diff_size = [
            248, 340, 328, 385, // tiny (3600)
            164, 279, 816, 719, // huge (286880)
        ];
        let n3 =
            unsafe { consolidate_detections_i32(diff_size.as_ptr(), 2, 0.7, keep.as_mut_ptr()) };
        assert_eq!(n3, 2);
    }

    #[test]
    fn test_ncc_u8() {
        let a: Vec<u8> = (0..64u8).collect();
        let inv: Vec<u8> = (0..64u8).map(|v| 255 - v).collect();
        let flat = vec![100u8; 64];
        unsafe {
            let same = ncc_u8(a.as_ptr(), a.as_ptr(), 64);
            let opp = ncc_u8(a.as_ptr(), inv.as_ptr(), 64);
            let zero = ncc_u8(a.as_ptr(), flat.as_ptr(), 64);
            assert!((same - 1.0).abs() < 1e-12);
            assert!((opp - (-1.0)).abs() < 1e-12);
            assert_eq!(zero, 0.0);
        }
    }

    #[test]
    fn test_percentiles_i32() {
        let boxes = [
            10, 20, 100, 200, 20, 30, 110, 210, 30, 40, 120, 220, 100, 200, 300, 400,
        ];
        let mut pct = [0.0f64; 4];
        unsafe {
            percentile_15_85_i32(boxes.as_ptr(), 4, pct.as_mut_ptr());
        }
        // For [10, 20, 30, 100], q=15 -> idx = 0.15 * 3 = 0.45 -> 10 + 0.45 * 10 = 14.5
        assert!((pct[0] - 14.5).abs() < 1e-9);
        // For [100, 110, 120, 300], q=85 -> idx = 0.85 * 3 = 2.55 -> 120 + 0.55 * 180 = 219.0
        assert!((pct[2] - 219.0).abs() < 1e-9);
    }

    #[test]
    fn test_laplacian_variance_u8() {
        let flat = vec![128u8; 32 * 32 * 3];
        let mut checker = vec![0u8; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                if (x + y) % 2 == 0 {
                    checker[y * 32 + x] = 200;
                }
            }
        }
        unsafe {
            assert_eq!(laplacian_variance_u8(flat.as_ptr(), 32, 32, 3), 0.0);
            let v = laplacian_variance_u8(checker.as_ptr(), 32, 32, 1);
            assert!(
                v > 100_000.0,
                "expected high variance on checkerboard, got {v}"
            );
        }
    }

    #[test]
    fn test_batch_cosine_similarity_f32() {
        let query = [1.0f32, 0.0, 0.0];
        let gallery = [
            1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let mut scores = [0.0f32; 4];
        unsafe {
            batch_cosine_similarity_f32(
                query.as_ptr(),
                gallery.as_ptr(),
                4,
                3,
                scores.as_mut_ptr(),
            );
        }
        assert!((scores[0] - 1.0).abs() < 1e-6);
        assert!(scores[1].abs() < 1e-6);
        assert!((scores[2] - (-1.0)).abs() < 1e-6);
        assert_eq!(scores[3], 0.0);
    }
}

// ═══════════════════════════════════════════════════════════════════
//  AVX2 ↔ scalar parity for normalize_u8_to_f32.
// ═══════════════════════════════════════════════════════════════════
#[cfg(all(test, target_arch = "x86_64"))]
mod simd_parity_tests {
    use super::*;

    #[test]
    fn normalize_scalar_matches_avx2() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        for &len in &[0usize, 1, 7, 8, 9, 16, 255, 1000] {
            let src: Vec<u8> = (0..len).map(|i| (i * 37 % 256) as u8).collect();
            let mut simd = vec![0f32; len];
            let mut scalar = vec![0f32; len];
            unsafe { normalize_u8_to_f32_avx2(&src, &mut simd, len) };
            normalize_u8_to_f32_scalar(&src, &mut scalar, len);
            assert_eq!(simd, scalar, "normalize mismatch at len={len}");
        }
    }

    #[test]
    fn normalize_covers_full_u8_range() {
        let src: Vec<u8> = (0..=255u8).collect();
        let len = src.len();
        let mut scalar = vec![0f32; len];
        normalize_u8_to_f32_scalar(&src, &mut scalar, len);
        assert_eq!(scalar[0], 0.0);
        assert_eq!(scalar[255], 1.0);
        assert!(scalar.iter().all(|v| (0.0..=1.0).contains(v)));
    }
}

// ═══════════════════════════════════════════════════════════════════
//  FFI boundary guards & fuzzing tests
// ═══════════════════════════════════════════════════════════════════
#[cfg(test)]
#[allow(clippy::erasing_op)]
mod ffi_guard_tests {
    use super::*;

    #[test]
    fn track_distance_rejects_null() {
        assert!(unsafe { track_distance(std::ptr::null(), std::ptr::null()) }.is_infinite());
    }

    #[test]
    fn read_ffmpeg_frame_rejects_null_and_zero() {
        assert_eq!(
            unsafe { read_ffmpeg_frame(-1, std::ptr::null_mut(), 128) },
            -1
        );
        let mut buf = vec![0u8; 8];
        assert_eq!(unsafe { read_ffmpeg_frame(-1, buf.as_mut_ptr(), 0) }, -1);
    }

    #[test]
    fn bilinear_resize_rejects_zero_dimensions() {
        let src = vec![7u8; 16];
        let mut dst = vec![0u8; 16];
        unsafe { bilinear_resize_u8(src.as_ptr(), dst.as_mut_ptr(), 4, 0, 4, 4, 1) };
        assert!(dst.iter().all(|&v| v == 0));
    }

    #[test]
    fn yuv420_to_3channel_rejects_odd_dimensions() {
        let y = [0u8; 15];
        let uv = [0u8; 2];
        let mut out = [0xABu8; 45];
        unsafe { yuv420_to_3channel(y.as_ptr(), uv.as_ptr(), uv.as_ptr(), 5, 3, out.as_mut_ptr()) };
        assert!(out.iter().all(|&v| v == 0xAB));
    }

    #[test]
    fn crop_repack_rejects_out_of_bounds_window() {
        let src = vec![0u8; 16 * 16 * 3 / 2];
        let mut out = vec![0xABu8; 64];
        unsafe { yuv420_crop_repack(src.as_ptr(), 16, 16, 12, 0, 8, 8, out.as_mut_ptr()) };
        assert!(
            out.iter().all(|&v| v == 0xAB),
            "must not write on an out-of-range crop"
        );
    }

    #[test]
    fn tier_a_kernels_reject_null_and_zero() {
        unsafe {
            assert_eq!(
                reduce_boxes_i32(std::ptr::null(), 4, 0.0, std::ptr::null_mut()),
                0
            );
            assert_eq!(
                cluster_candidates_i32(
                    std::ptr::null(),
                    4,
                    1000,
                    1000,
                    160,
                    std::ptr::null_mut(),
                    std::ptr::null_mut()
                ),
                0
            );
            assert_eq!(
                consolidate_detections_i32(std::ptr::null(), 4, 0.7, std::ptr::null_mut()),
                0
            );
            assert_eq!(ncc_u8(std::ptr::null(), std::ptr::null(), 64), 0.0);
            percentile_15_85_i32(std::ptr::null(), 0, std::ptr::null_mut());
            assert_eq!(laplacian_variance_u8(std::ptr::null(), 0, 0, 0), 0.0);
            batch_cosine_similarity_f32(
                std::ptr::null(),
                std::ptr::null(),
                0,
                0,
                std::ptr::null_mut(),
            );
        }
    }
}
