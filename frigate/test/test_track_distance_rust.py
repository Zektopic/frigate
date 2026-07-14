"""Parity tests for the Rust tracker association distance."""

import math
import unittest

import numpy as np

from frigate.util.frame_rs import frame_rs_available, track_distance_rust


def python_distance(detection: np.ndarray, estimate: np.ndarray) -> float:
    """Reference implementation (frigate.track.norfair_tracker.distance,
    numpy path, duplicated here so the test does not depend on norfair)."""
    estimate_dim = np.diff(estimate, axis=0).flatten()
    detection_dim = np.diff(detection, axis=0).flatten()

    if (
        not np.all(np.isfinite(estimate_dim))
        or not np.all(np.isfinite(detection_dim))
        or estimate_dim[0] <= 0
        or estimate_dim[1] <= 0
        or detection_dim[0] <= 0
        or detection_dim[1] <= 0
    ):
        return float("inf")

    detection_position = np.array(
        [np.average(detection[:, 0]), np.max(detection[:, 1])]
    )
    estimate_position = np.array([np.average(estimate[:, 0]), np.max(estimate[:, 1])])

    dist = (detection_position - estimate_position).astype(float)
    dist[0] /= estimate_dim[0]
    dist[1] /= estimate_dim[1]

    widths = np.sort([estimate_dim[0], detection_dim[0]])
    heights = np.sort([estimate_dim[1], detection_dim[1]])
    width_ratio = widths[1] / widths[0] - 1.0
    height_ratio = heights[1] / heights[0] - 1.0

    change = np.append(dist, np.array([width_ratio, height_ratio]))
    return float(np.linalg.norm(change))


@unittest.skipUnless(frame_rs_available(), "Rust frame engine not available")
class TestTrackDistanceRust(unittest.TestCase):
    def assert_parity(self, det: np.ndarray, est: np.ndarray) -> None:
        want = python_distance(det, est)
        got = track_distance_rust(det.ravel(), est.ravel())
        if math.isinf(want):
            self.assertTrue(math.isinf(got), f"det={det} est={est}: got {got}")
        else:
            self.assertAlmostEqual(got, want, places=9, msg=f"det={det} est={est}")

    def test_identical_boxes(self):
        box = np.array([[10.0, 20.0], [110.0, 220.0]])
        self.assert_parity(box, box)

    def test_random_boxes_match_python(self):
        rng = np.random.default_rng(42)
        for _ in range(2000):
            x1, y1 = rng.uniform(0, 1000, 2)
            w1, h1 = rng.uniform(1, 500, 2)
            x2, y2 = rng.uniform(0, 1000, 2)
            w2, h2 = rng.uniform(1, 500, 2)
            det = np.array([[x1, y1], [x1 + w1, y1 + h1]])
            est = np.array([[x2, y2], [x2 + w2, y2 + h2]])
            self.assert_parity(det, est)

    def test_degenerate_boxes(self):
        good = np.array([[10.0, 20.0], [110.0, 220.0]])
        zero_w = np.array([[10.0, 20.0], [10.0, 220.0]])
        neg_h = np.array([[10.0, 220.0], [110.0, 20.0]])
        nan_box = np.array([[np.nan, 20.0], [110.0, 220.0]])
        inf_box = np.array([[np.inf, 20.0], [110.0, 220.0]])
        for bad in (zero_w, neg_h, nan_box, inf_box):
            self.assert_parity(bad, good)
            self.assert_parity(good, bad)

    def test_tiny_boxes(self):
        det = np.array([[0.0, 0.0], [1e-9, 1e-9]])
        est = np.array([[0.0, 0.0], [1.0, 1.0]])
        self.assert_parity(det, est)


if __name__ == "__main__":
    unittest.main()
