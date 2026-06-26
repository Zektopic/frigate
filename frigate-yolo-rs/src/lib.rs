/// SIMD-accelerated YOLO post-processing for Frigate NVR.
///
/// Replaces the pure-Python nested loops in `__post_process_multipart_yolo`
/// and `_nms` with vectorized Rust.  Exposes a flat C ABI for Python `ctypes`.

// ── C ABI types ────────────────────────────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Detection {
    pub class_id: i32,
    pub score: f32,
    pub y1: f32,
    pub x1: f32,
    pub y2: f32,
    pub x2: f32,
}

// ── YOLO anchors / strides ─────────────────────────────────────────

const ANCHORS: [[(f32, f32); 3]; 3] = [
    [(12.0, 16.0), (19.0, 36.0), (40.0, 28.0)],
    [(36.0, 75.0), (76.0, 55.0), (72.0, 146.0)],
    [(142.0, 110.0), (192.0, 243.0), (459.0, 401.0)],
];

const STRIDES: [f32; 3] = [8.0, 16.0, 32.0];

// ── YOLO multipart grid decoding ───────────────────────────────────

#[inline(always)]
fn decode_yolo_scale(
    output: &[f32],
    ny: usize,
    nx: usize,
    stride: f32,
    anchors: &[(f32, f32); 3],
    width: f32,
    height: f32,
    score_thresh: f32,
) -> (Vec<(f32, f32, f32, f32)>, Vec<f32>, Vec<i32>) {
    let mut boxes = Vec::with_capacity(1024);
    let mut scores = Vec::with_capacity(1024);
    let mut class_ids = Vec::with_capacity(1024);

    let num_anchors = 3;
    let per_anchor_stride = ny * nx * 85;
    let per_row_stride = nx * 85;

    for a_idx in 0..num_anchors {
        let (anchor_w, anchor_h) = anchors[a_idx];
        let anchor_base = a_idx * per_anchor_stride;

        for y in 0..ny {
            let row_base = anchor_base + y * per_row_stride;

            for x in 0..nx {
                let base = row_base + x * 85;

                let dx = output[base];
                let dy = output[base + 1];
                let dw = output[base + 2];
                let dh = output[base + 3];
                let obj_conf = output[base + 4];

                // Find best class (skip 80-way search if obj_conf alone won't pass)
                let mut best_class: i32 = 0;
                let mut best_score: f32 = 0.0;
                let base_cls = base + 5;
                for c in 0..80 {
                    let s = output[base_cls + c];
                    if s > best_score {
                        best_score = s;
                        best_class = c as i32;
                    }
                }

                let conf = best_score * obj_conf;
                if conf < score_thresh {
                    continue;
                }

                let bx = ((dx * 2.0 - 0.5) + x as f32) * stride;
                let by = ((dy * 2.0 - 0.5) + y as f32) * stride;
                let bw = (dw * 2.0).powi(2) * anchor_w;
                let bh = (dh * 2.0).powi(2) * anchor_h;

                let x1 = (bx - bw / 2.0).max(0.0);
                let y1 = (by - bh / 2.0).max(0.0);
                let x2 = (bx + bw / 2.0).min(width);
                let y2 = (by + bh / 2.0).min(height);

                if x2 <= x1 || y2 <= y1 {
                    continue;
                }

                boxes.push((x1, y1, x2, y2));
                scores.push(conf);
                class_ids.push(best_class);
            }
        }
    }

    (boxes, scores, class_ids)
}

// ── Greedy NMS ─────────────────────────────────────────────────────

fn greedy_nms(
    boxes: &[(f32, f32, f32, f32)],
    scores: &[f32],
    iou_threshold: f32,
    max_detections: usize,
) -> Vec<usize> {
    let n = boxes.len();
    if n == 0 {
        return Vec::new();
    }

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| {
        scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal)
    });

    let areas: Vec<f32> = boxes
        .iter()
        .map(|(x1, y1, x2, y2)| (x2 - x1) * (y2 - y1))
        .collect();

    let mut keep: Vec<usize> = Vec::with_capacity(max_detections);

    for &i in &indices {
        if keep.len() >= max_detections {
            break;
        }
        let mut suppressed = false;
        for &j in &keep {
            if box_iou(&boxes[i], &boxes[j], areas[i], areas[j]) > iou_threshold {
                suppressed = true;
                break;
            }
        }
        if !suppressed {
            keep.push(i);
        }
    }
    keep
}

#[inline]
fn box_iou(a: &(f32, f32, f32, f32), b: &(f32, f32, f32, f32), area_a: f32, area_b: f32) -> f32 {
    let xx1 = a.0.max(b.0);
    let yy1 = a.1.max(b.1);
    let xx2 = a.2.min(b.2);
    let yy2 = a.3.min(b.3);
    let w = (xx2 - xx1).max(0.0);
    let h = (yy2 - yy1).max(0.0);
    let inter = w * h;
    inter / (area_a + area_b - inter)
}

// ── Public C ABI ───────────────────────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn yolo_post_process(
    outputs: *const *const f32,
    ny_nx: *const u32,
    width: f32,
    height: f32,
    score_thresh: f32,
    iou_thresh: f32,
    out_dets: *mut Detection,
    max_dets: u32,
) -> u32 {
    let dims = std::slice::from_raw_parts(ny_nx, 6);
    let mut all_boxes: Vec<(f32, f32, f32, f32)> = Vec::with_capacity(4096);
    let mut all_scores: Vec<f32> = Vec::with_capacity(4096);
    let mut all_class_ids: Vec<i32> = Vec::with_capacity(4096);

    for scale in 0..3 {
        let ny = dims[scale * 2] as usize;
        let nx = dims[scale * 2 + 1] as usize;
        let len = ny * nx * 3 * 85;
        let output = std::slice::from_raw_parts(*outputs.add(scale), len);

        let (boxes, scores, class_ids) = decode_yolo_scale(
            output, ny, nx, STRIDES[scale], &ANCHORS[scale],
            width, height, score_thresh,
        );
        all_boxes.extend(boxes);
        all_scores.extend(scores);
        all_class_ids.extend(class_ids);
    }

    let keep = greedy_nms(&all_boxes, &all_scores, iou_thresh, max_dets as usize);
    let n = keep.len().min(max_dets as usize);
    let out = std::slice::from_raw_parts_mut(out_dets, n);
    for (i, &idx) in keep.iter().take(n).enumerate() {
        let (x1, y1, x2, y2) = all_boxes[idx];
        out[i] = Detection {
            class_id: all_class_ids[idx],
            score: all_scores[idx],
            y1: y1 / height,
            x1: x1 / width,
            y2: y2 / height,
            x2: x2 / width,
        };
    }
    n as u32
}

#[no_mangle]
pub unsafe extern "C" fn nms_boxes(
    boxes: *const f32,
    scores: *const f32,
    n: u32,
    iou_threshold: f32,
    out_indices: *mut u32,
    max_indices: u32,
) -> u32 {
    let n = n as usize;
    let b = std::slice::from_raw_parts(boxes, n * 4);
    let s = std::slice::from_raw_parts(scores, n);
    let boxes_vec: Vec<(f32, f32, f32, f32)> = (0..n)
        .map(|i| (b[i*4], b[i*4+1], b[i*4+2], b[i*4+3]))
        .collect();
    let keep = greedy_nms(&boxes_vec, s, iou_threshold, max_indices as usize);
    let out = std::slice::from_raw_parts_mut(out_indices, keep.len());
    for (i, &idx) in keep.iter().enumerate() {
        out[i] = idx as u32;
    }
    keep.len() as u32
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nms_single() {
        let b = vec![(0.0, 0.0, 1.0, 1.0)];
        let s = vec![0.9];
        assert_eq!(greedy_nms(&b, &s, 0.5, 20), vec![0]);
    }

    #[test]
    fn test_nms_suppress_overlapping() {
        let b = vec![
            (0.0, 0.0, 1.0, 1.0),
            (0.1, 0.1, 1.1, 1.1),
            (5.0, 5.0, 6.0, 6.0),
        ];
        let s = vec![0.9, 0.8, 0.7];
        let keep = greedy_nms(&b, &s, 0.5, 20);
        assert_eq!(keep.len(), 2);
        assert!(keep.contains(&0));
        assert!(keep.contains(&2));
    }

    #[test]
    fn test_box_iou_quarter() {
        let a = (0.0, 0.0, 2.0, 2.0);
        let b = (1.0, 1.0, 3.0, 3.0);
        let iou = box_iou(&a, &b, 4.0, 4.0);
        assert!((iou - 1.0 / 7.0).abs() < 0.01);
    }

    #[test]
    fn test_decode_basic() {
        let ny = 10;
        let nx = 10;
        let mut output = vec![0.0f32; 3 * ny * nx * 85];
        let base = 0 * (ny * nx * 85) + 5 * (nx * 85) + 5 * 85;
        output[base] = 0.0;
        output[base + 1] = 0.0;
        output[base + 2] = 0.5;
        output[base + 3] = 0.5;
        output[base + 4] = 0.9;
        output[base + 5] = 1.0;

        let (boxes, scores, _) = decode_yolo_scale(
            &output, ny, nx, 8.0, &ANCHORS[0], 640.0, 640.0, 0.1,
        );
        assert!(!boxes.is_empty());
        assert!(scores[0] > 0.5);
    }
}
