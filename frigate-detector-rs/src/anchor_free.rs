//! YOLOv8 / YOLO11 post-processing: DFL box decode, class scores and NMS.
//!
//! A port of `_process()` from the Python ncnn worker in
//! `docker/ncnn-vulkan/onnx.py`, which this binary replaces for these
//! models. `testdata/anchor_free_cases.txt` holds that function's output for
//! fixed synthetic inputs, and the tests below check this port against it.
//!
//! The ncnn output has one row of [`CHANNELS`] floats per anchor. Columns
//! 0..64 are 16 DFL bin logits for each box side (left, top, right, bottom)
//! and columns 64..144 are the 80 class logits. Anchors cover the stride 8,
//! 16 and 32 grids in that order, each grid row-major. The DFL distances are
//! in grid cells, so each is scaled by its anchor's stride to get pixels.

use rayon::prelude::*;

/// Floats per anchor row: 4 sides x 16 DFL bins, then 80 class logits.
pub const CHANNELS: usize = 144;
const REG_MAX: usize = 16;
const REG_CHANNELS: usize = 4 * REG_MAX;
const STRIDES: [u32; 3] = [8, 16, 32];
/// Only the best-scoring boxes go into NMS, as in `_process()`.
const NMS_TOP_K: usize = 1000;
/// Rows in the output buffer.
pub const MAX_DETECTIONS: usize = 20;
/// Output buffer length: [`MAX_DETECTIONS`] rows of 6 floats.
pub const OUT_LEN: usize = MAX_DETECTIONS * 6;

/// Sigmoid as `_process()` computes it, with the logit clipped to +-20.
#[inline]
fn sigmoid(logit: f32) -> f32 {
    1.0 / (1.0 + (-logit.clamp(-20.0, 20.0)).exp())
}

/// Expected bin index under the softmax of one side's DFL logits: the
/// distance from the anchor to that side, in grid cells.
///
/// A NaN logit makes the result NaN, so the box fails the size check.
#[inline]
fn dfl_distance(bins: &[f32; REG_MAX]) -> f32 {
    let max = bins.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    let mut exp = [0.0f32; REG_MAX];
    let mut sum = 0.0f32;
    for (slot, &v) in exp.iter_mut().zip(bins) {
        *slot = (v - max).exp();
        sum += *slot;
    }
    exp.iter()
        .enumerate()
        .map(|(bin, &e)| e / sum * bin as f32)
        .sum()
}

/// An anchor whose best class passed the score threshold.
struct Candidate {
    anchor: usize,
    class: usize,
    score: f32,
}

/// A decoded box in model pixels.
#[derive(Clone, Copy)]
struct Detection {
    class: usize,
    score: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    area: f32,
}

impl Detection {
    /// IoU with `other`, computed in the same order as `_process()`.
    #[inline]
    fn iou(&self, other: &Detection) -> f32 {
        let w = (self.x2.min(other.x2) - self.x1.max(other.x1)).max(0.0);
        let h = (self.y2.min(other.y2) - self.y1.max(other.y1)).max(0.0);
        let inter = w * h;
        inter / (self.area + other.area - inter)
    }
}

/// One anchor: its grid-cell centre in model pixels, and its stride.
#[derive(Debug, PartialEq)]
struct Anchor {
    x: f32,
    y: f32,
    stride: f32,
}

/// Decoder for one model input size.
pub struct AnchorFree {
    model_size: f32,
    grid: Vec<Anchor>,
}

impl AnchorFree {
    pub fn new(model_size: u32) -> Self {
        let mut grid = Vec::new();
        for stride in STRIDES {
            let cells = model_size / stride;
            let stride = stride as f32;
            for y in 0..cells {
                for x in 0..cells {
                    grid.push(Anchor {
                        x: (x as f32 + 0.5) * stride,
                        y: (y as f32 + 0.5) * stride,
                        stride,
                    });
                }
            }
        }
        Self {
            model_size: model_size as f32,
            grid,
        }
    }

    /// Number of anchors the model output must have for this input size.
    pub fn anchors(&self) -> usize {
        self.grid.len()
    }

    /// Decodes one inference into `out` as `[class, score, x1, y1, x2, y2]`
    /// rows in model pixels, best score first, with the unused rows zeroed.
    ///
    /// Returns the number of detections written. An empty `raw` (failed
    /// inference) writes none.
    ///
    /// # Errors
    /// When `raw` is not [`anchors`](Self::anchors) rows of [`CHANNELS`]
    /// floats, i.e. the model does not match this layout or input size.
    pub fn process(
        &self,
        raw: &[f32],
        score_thresh: f32,
        nms_thresh: f32,
        out: &mut [f32; OUT_LEN],
    ) -> Result<usize, String> {
        out.fill(0.0);
        if raw.is_empty() {
            return Ok(0);
        }
        if raw.len() != self.anchors() * CHANNELS {
            return Err(format!(
                "yolov8 output has {} floats, expected {} anchors x {CHANNELS} \
                 for input size {}",
                raw.len(),
                self.anchors(),
                self.model_size
            ));
        }

        let candidates: Vec<Candidate> = raw
            .par_chunks_exact(CHANNELS)
            .enumerate()
            .filter_map(|(anchor, row)| best_class(&row[REG_CHANNELS..], score_thresh, anchor))
            .collect();

        let mut boxes: Vec<Detection> = candidates
            .iter()
            .filter_map(|c| self.decode_box(raw, c))
            .collect();

        // Stable sort, so equal scores keep anchor order. `_process()` uses
        // `argsort()[::-1]`, which leaves the order of equal scores
        // unspecified: on exact ties (common with fp16 GPU output) the two
        // can list tied boxes in a different order, or NMS can keep a
        // different one of two near-identical boxes.
        boxes.sort_by(|a, b| b.score.total_cmp(&a.score));
        boxes.truncate(NMS_TOP_K);

        let mut kept: Vec<Detection> = Vec::with_capacity(MAX_DETECTIONS);
        for b in &boxes {
            if kept.len() == MAX_DETECTIONS {
                break;
            }
            // `<=` like `_process()`: a NaN IoU suppresses the box.
            if kept.iter().all(|k| k.iou(b) <= nms_thresh) {
                kept.push(*b);
            }
        }

        for (row, d) in out.as_chunks_mut::<6>().0.iter_mut().zip(&kept) {
            *row = [d.class as f32, d.score, d.x1, d.y1, d.x2, d.y2];
        }
        Ok(kept.len())
    }

    /// Box for `c`, clipped to the model input, or `None` if it is empty.
    fn decode_box(&self, raw: &[f32], c: &Candidate) -> Option<Detection> {
        let start = c.anchor * CHANNELS;
        let mut dist = [0.0f32; 4];
        for (side, bins) in dist
            .iter_mut()
            .zip(raw[start..start + REG_CHANNELS].as_chunks::<REG_MAX>().0)
        {
            *side = dfl_distance(bins);
        }

        let a = &self.grid[c.anchor];
        let ms = self.model_size;
        let x1 = (a.x - dist[0] * a.stride).clamp(0.0, ms);
        let y1 = (a.y - dist[1] * a.stride).clamp(0.0, ms);
        let x2 = (a.x + dist[2] * a.stride).clamp(0.0, ms);
        let y2 = (a.y + dist[3] * a.stride).clamp(0.0, ms);
        // False for NaN as well as for empty boxes.
        if !(x2 > x1 && y2 > y1) {
            return None;
        }
        Some(Detection {
            class: c.class,
            score: c.score,
            x1,
            y1,
            x2,
            y2,
            area: (x2 - x1) * (y2 - y1),
        })
    }
}

/// The best class of one anchor, if its score passes `score_thresh`.
///
/// `_process()` takes the sigmoid of all 80 logits and then the argmax. The
/// sigmoid is monotonic, so the largest logit tells whether the anchor can
/// pass at all, and only candidates pay for 80 sigmoids. Those are still
/// needed: in f32 every logit above about 17 scores exactly 1.0, and
/// `np.argmax` then picks the first of them, not the largest logit. A NaN
/// logit drops the anchor, because `np.argmax` picks the NaN and a NaN
/// score fails the threshold.
#[inline]
fn best_class(logits: &[f32], score_thresh: f32, anchor: usize) -> Option<Candidate> {
    let mut max = f32::NEG_INFINITY;
    let mut has_nan = false;
    for &l in logits {
        has_nan |= l.is_nan();
        max = max.max(l);
    }
    // With NaN ruled out, `max` and its score are ordinary numbers.
    if has_nan || sigmoid(max) <= score_thresh {
        return None;
    }

    let mut class = 0;
    let mut score = f32::NEG_INFINITY;
    for (c, &l) in logits.iter().enumerate() {
        let s = sigmoid(l);
        if s > score {
            score = s;
            class = c;
        }
    }
    (score > score_thresh).then_some(Candidate {
        anchor,
        class,
        score,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCORE_THRESH: f32 = 0.25;
    const NMS_THRESH: f32 = 0.45;
    const NUM_CLASSES: usize = CHANNELS - REG_CHANNELS;

    fn background(anchors: usize) -> Vec<f32> {
        let mut raw = vec![0.0f32; anchors * CHANNELS];
        for row in raw.as_chunks_mut::<CHANNELS>().0 {
            row[REG_CHANNELS..].fill(-10.0);
        }
        raw
    }

    /// Sets an anchor's DFL bins so each side decodes to `cells` grid cells.
    fn set_sides(raw: &mut [f32], anchor: usize, cells: [usize; 4]) {
        let row = &mut raw[anchor * CHANNELS..anchor * CHANNELS + REG_CHANNELS];
        for (side, &bin) in cells.iter().enumerate() {
            row[side * REG_MAX..(side + 1) * REG_MAX].fill(-30.0);
            row[side * REG_MAX + bin] = 30.0;
        }
    }

    #[test]
    fn grid_follows_model_size() {
        assert_eq!(AnchorFree::new(640).anchors(), 8400);
        assert_eq!(AnchorFree::new(320).anchors(), 2100);
        let d = AnchorFree::new(640);
        let a = |x, y, stride| Anchor { x, y, stride };
        assert_eq!(d.grid[0], a(4.0, 4.0, 8.0));
        assert_eq!(d.grid[1], a(12.0, 4.0, 8.0));
        assert_eq!(d.grid[80], a(4.0, 12.0, 8.0));
        assert_eq!(d.grid[6400], a(8.0, 8.0, 16.0)); // first stride-16 cell
        assert_eq!(d.grid[8000], a(16.0, 16.0, 32.0)); // first stride-32 cell
        assert_eq!(*d.grid.last().unwrap(), a(624.0, 624.0, 32.0));
    }

    #[test]
    fn decodes_boxes_in_pixels_at_each_stride() {
        let d = AnchorFree::new(320);
        let mut raw = background(d.anchors());
        // Stride 8, cell (10, 5): centre (84, 44).
        let a8 = 5 * 40 + 10;
        set_sides(&mut raw, a8, [3, 2, 5, 7]);
        raw[a8 * CHANNELS + REG_CHANNELS + 7] = 2.0;
        // Stride 32, cell (4, 3): centre (144, 112). Its grid follows the
        // 40x40 stride-8 and 20x20 stride-16 ones.
        let a32 = 1600 + 400 + 3 * 10 + 4;
        set_sides(&mut raw, a32, [1, 2, 3, 1]);
        raw[a32 * CHANNELS + REG_CHANNELS + 9] = 1.0;

        let mut out = [9.0f32; OUT_LEN];
        assert_eq!(d.process(&raw, SCORE_THRESH, NMS_THRESH, &mut out), Ok(2));
        assert_eq!(&out[..6], &[7.0, sigmoid(2.0), 60.0, 28.0, 124.0, 100.0]);
        assert_eq!(&out[6..12], &[9.0, sigmoid(1.0), 112.0, 48.0, 240.0, 144.0]);
        assert!(out[12..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn nms_keeps_the_best_of_overlapping_boxes() {
        let d = AnchorFree::new(320);
        let mut raw = background(d.anchors());
        // Neighbouring stride-8 cells, 8 px apart, with 32 px wide boxes.
        for (anchor, logit) in [(100, 1.0), (101, 3.0)] {
            set_sides(&mut raw, anchor, [2, 2, 2, 2]);
            raw[anchor * CHANNELS + REG_CHANNELS] = logit;
        }
        // A far-away box survives.
        set_sides(&mut raw, 900, [1, 1, 1, 1]);
        raw[900 * CHANNELS + REG_CHANNELS + 2] = 0.5;

        let mut out = [0.0f32; OUT_LEN];
        assert_eq!(d.process(&raw, SCORE_THRESH, NMS_THRESH, &mut out), Ok(2));
        assert_eq!((out[0], out[1]), (0.0, sigmoid(3.0)));
        assert_eq!((out[6], out[7]), (2.0, sigmoid(0.5)));
    }

    #[test]
    fn saturated_scores_pick_the_first_class_like_argmax() {
        let mut logits = [-10.0f32; NUM_CLASSES];
        logits[3] = 18.0;
        logits[60] = 25.0; // larger logit, but both clip to a score of 1.0
        let c = best_class(&logits, SCORE_THRESH, 0).unwrap();
        assert_eq!((c.class, c.score), (3, 1.0));
    }

    #[test]
    fn nan_logits_drop_the_anchor() {
        let mut logits = [-10.0f32; NUM_CLASSES];
        logits[0] = 5.0;
        logits[50] = f32::NAN;
        assert!(best_class(&logits, SCORE_THRESH, 0).is_none());
    }

    #[test]
    fn empty_output_is_no_detections() {
        let mut out = [1.0f32; OUT_LEN];
        let d = AnchorFree::new(640);
        assert_eq!(d.process(&[], SCORE_THRESH, NMS_THRESH, &mut out), Ok(0));
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn wrong_output_size_is_an_error() {
        let mut out = [0.0f32; OUT_LEN];
        let d = AnchorFree::new(640);
        // A 320 export run at 640, or a model with another class count.
        assert!(d
            .process(
                &vec![0.0; 2100 * CHANNELS],
                SCORE_THRESH,
                NMS_THRESH,
                &mut out
            )
            .is_err());
        assert!(d
            .process(&vec![0.0; 8400 * 85], SCORE_THRESH, NMS_THRESH, &mut out)
            .is_err());
    }

    // ── Parity with the numpy worker ────────────────────────────────
    //
    // `testdata/gen_anchor_free_cases.py` builds the same synthetic outputs
    // with numpy, runs `_process()` from `docker/ncnn-vulkan/onnx.py` on
    // them and records the result. Each case also records a checksum of the
    // input, so a drift between the two generators fails loudly instead of
    // comparing different tensors.

    const FIXTURE: &str = include_str!("../testdata/anchor_free_cases.txt");

    /// SplitMix64 output for counter `i`; the generator script uses the same.
    fn splitmix64(seed: u64, i: u64) -> u64 {
        let mut z = seed.wrapping_add(i.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Top 24 bits of draw `i`: a value in `0..2^24`, exact as an f32.
    fn draw(seed: u64, i: usize) -> i64 {
        (splitmix64(seed, i as u64) >> 40) as i64
    }

    /// Mirror of `synthetic_output()` in the generator script.
    fn synthetic_output(seed: u64, anchors: usize, objects: usize) -> Vec<f32> {
        let n = anchors * CHANNELS;
        let scale = (1 << 21) as f32;
        let mut raw: Vec<f32> = (0..n)
            .map(|i| {
                // DFL logits in [-4, 4); background class logits in [-12, -4).
                let v = (draw(seed, i) - (1 << 23)) as f32 / scale;
                if i % CHANNELS < REG_CHANNELS {
                    v
                } else {
                    v - 8.0
                }
            })
            .collect();
        let at = |anchor: usize, channel: usize| anchor * CHANNELS + channel;

        // Objects: a class logit in [-2, 6) on an anchor and, slightly
        // lower, on its next two anchors, so their boxes overlap for NMS.
        for o in 0..objects {
            let d = |j: usize| draw(seed, n + 4 * o + j);
            let anchor = d(0) as usize % anchors;
            let class = d(1) as usize % NUM_CLASSES;
            let logit = d(2) as f32 / scale - 2.0;
            for k in 0..3 {
                if anchor + k < anchors {
                    raw[at(anchor + k, REG_CHANNELS + class)] = logit - 0.5 * k as f32;
                }
            }
        }

        let d = |j: usize| draw(seed, n + 4 * objects + j);
        // Two classes past the clip, both scoring 1.0.
        let anchor = d(0) as usize % anchors;
        raw[at(anchor, REG_CHANNELS + d(1) as usize % 40)] = 18.0;
        raw[at(anchor, REG_CHANNELS + 40 + d(2) as usize % 40)] = 25.0;
        // A strong class with a NaN class logit beside it.
        let anchor = d(3) as usize % anchors;
        let class = d(4) as usize % NUM_CLASSES;
        raw[at(anchor, REG_CHANNELS + class)] = 5.0;
        raw[at(anchor, REG_CHANNELS + (class + 1) % NUM_CLASSES)] = f32::NAN;
        // A strong class with a NaN DFL bin.
        let anchor = d(5) as usize % anchors;
        raw[at(anchor, REG_CHANNELS)] = 5.0;
        raw[at(anchor, 3)] = f32::NAN;
        raw
    }

    fn checksum(raw: &[f32]) -> u64 {
        raw.iter().enumerate().fold(0u64, |acc, (i, v)| {
            acc.wrapping_add(u64::from(v.to_bits()).wrapping_mul(i as u64 + 1))
        })
    }

    struct Case {
        seed: u64,
        model_size: u32,
        objects: usize,
        checksum: u64,
        rows: Vec<[f32; 6]>,
    }

    fn fixture_cases() -> Vec<Case> {
        let mut cases: Vec<Case> = Vec::new();
        for line in FIXTURE.lines().map(str::trim) {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields[0] == "case" {
                cases.push(Case {
                    seed: fields[1].parse().unwrap(),
                    model_size: fields[2].parse().unwrap(),
                    objects: fields[3].parse().unwrap(),
                    checksum: fields[4].parse().unwrap(),
                    rows: Vec::new(),
                });
            } else {
                let row: Vec<f32> = fields.iter().map(|f| f.parse().unwrap()).collect();
                cases.last_mut().unwrap().rows.push(row.try_into().unwrap());
            }
        }
        cases
    }

    #[test]
    fn matches_numpy_worker() {
        let cases = fixture_cases();
        assert!(cases.len() >= 5, "fixture has {} cases", cases.len());

        for case in &cases {
            let d = AnchorFree::new(case.model_size);
            let raw = synthetic_output(case.seed, d.anchors(), case.objects);
            assert_eq!(
                checksum(&raw),
                case.checksum,
                "seed {}: input differs from the generator script's",
                case.seed
            );

            let mut out = [0.0f32; OUT_LEN];
            let n = d.process(&raw, SCORE_THRESH, NMS_THRESH, &mut out).unwrap();
            assert_eq!(n, case.rows.len(), "seed {}: detection count", case.seed);

            for (k, (got, want)) in out.as_chunks::<6>().0.iter().zip(&case.rows).enumerate() {
                let ctx = format!("seed {} row {k}: got {got:?}, numpy {want:?}", case.seed);
                assert_eq!(got[0], want[0], "class, {ctx}");
                // Scores differ by at most a few ulps (exp implementations);
                // coordinates also by DFL summation order.
                assert!((got[1] - want[1]).abs() <= 2e-6, "score, {ctx}");
                for c in 2..6 {
                    assert!((got[c] - want[c]).abs() <= 1e-3, "coordinate, {ctx}");
                }
            }
        }
    }
}
