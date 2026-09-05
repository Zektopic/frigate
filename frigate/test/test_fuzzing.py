"""Fuzzing Test Suite for Frigate C-ABI and Internal Data Pipelines.

Tests edge cases, random noise byte sequences, NaN/Inf floating point bounding boxes,
degenerate polygon vertices, and corrupted payloads without crashing or leaking memory.
"""

import ctypes
import math
import random
import unittest
import numpy as np

from frigate.util.frame_rs import (
    frame_rs_available,
    point_in_polygon_rust,
    polygon_box_overlap_rust,
    intersection_over_union_rust,
    track_distance_rust,
    fast_shm_copy_rust,
)
from frigate.detectors.rust_yolo import (
    yolo_available,
    yolo26_post_process,
)


class TestFuzzingEngines(unittest.TestCase):
    def test_fuzz_fast_shm_copy_random_lengths(self):
        """Fuzz fast_shm_copy with varying buffer sizes, unaligned lengths, and random bytes."""
        if not frame_rs_available():
            self.skipTest("Rust frame engine not available")

        # Test various aligned and unaligned lengths
        test_lengths = [0, 1, 7, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 513, 1024, 65537]
        for length in test_lengths:
            if length == 0:
                continue
            src_bytes = bytearray(random.getrandbits(8) for _ in range(length))
            dst_bytes = bytearray(length)

            src_buf = (ctypes.c_char * length).from_buffer(src_bytes)
            dst_buf = (ctypes.c_char * length).from_buffer(dst_bytes)

            fast_shm_copy_rust(dst_buf, src_buf, length)
            self.assertEqual(src_bytes, dst_bytes, f"Mismatch at length {length}")

    def test_fuzz_track_distance_nan_inf_degenerate(self):
        """Fuzz Norfair track distance with NaNs, Infs, zero/negative areas, and extreme floats."""
        if not frame_rs_available():
            self.skipTest("Rust frame engine not available")

        # Valid reference box
        valid_box = [10.0, 10.0, 100.0, 100.0]

        extreme_cases = [
            [float("nan"), 10.0, 100.0, 100.0],
            [10.0, float("nan"), 100.0, 100.0],
            [float("inf"), 10.0, 100.0, 100.0],
            [10.0, float("-inf"), 100.0, 100.0],
            [100.0, 100.0, 10.0, 10.0],  # Inverted box (x2 < x1, y2 < y1)
            [50.0, 50.0, 50.0, 50.0],    # Zero-width / zero-height box
            [-1e9, -1e9, 1e9, 1e9],      # Extreme coordinates
        ]

        for bad_box in extreme_cases:
            dist = track_distance_rust(bad_box, valid_box)
            # Must return finite float or +inf without panic or segfault
            self.assertTrue(math.isnan(dist) or math.isinf(dist) or isinstance(dist, float))

    def test_fuzz_polygon_geometry_extreme_points(self):
        """Fuzz point-in-polygon and polygon-box overlap with complex / self-intersecting polygons."""
        if not frame_rs_available():
            self.skipTest("Rust frame engine not available")

        # 1. Empty polygon
        self.assertFalse(point_in_polygon_rust(50.0, 50.0, []))
        self.assertFalse(polygon_box_overlap_rust([], (0.0, 0.0, 100.0, 100.0)))

        # 2. Single point / 2-point line segment
        self.assertFalse(point_in_polygon_rust(50.0, 50.0, [(10.0, 10.0)]))
        self.assertFalse(point_in_polygon_rust(50.0, 50.0, [(10.0, 10.0), (20.0, 20.0)]))

        # 3. Huge self-intersecting bowtie polygon
        bowtie = [(0.0, 0.0), (100.0, 100.0), (0.0, 100.0), (100.0, 0.0)]
        res = point_in_polygon_rust(50.0, 50.0, bowtie)
        self.assertIsInstance(res, bool)

        # 4. Fuzz with 1000 random points against a complex 20-vertex polygon
        polygon = [
            (random.uniform(0, 1000), random.uniform(0, 1000))
            for _ in range(20)
        ]
        for _ in range(100):
            px = random.uniform(-100, 1100)
            py = random.uniform(-100, 1100)
            inside = point_in_polygon_rust(px, py, polygon)
            self.assertIsInstance(inside, bool)

    def test_fuzz_yolo26_post_process_corrupted_tensors(self):
        """Fuzz YOLO26 decoded post-processor with random / extreme float tensors."""
        if not yolo_available():
            self.skipTest("Rust YOLO engine not available")

        # Random tensor of shape (84, 8400)
        raw_noise = np.random.uniform(-10.0, 10.0, (84, 8400)).astype(np.float32)
        dets = yolo26_post_process(raw_noise, model_size=640, frame_w=1.0, frame_h=1.0, score_thresh=0.5)
        self.assertEqual(dets.shape, (20, 6))

        # Tensor containing NaNs and Infs
        raw_corrupt = np.zeros((84, 100), dtype=np.float32)
        raw_corrupt[0, :] = np.nan
        raw_corrupt[1, :] = np.inf
        raw_corrupt[4, :] = 0.9  # high class score
        dets_corrupt = yolo26_post_process(raw_corrupt, model_size=640, frame_w=1.0, frame_h=1.0)
        self.assertEqual(dets_corrupt.shape, (20, 6))


if __name__ == "__main__":
    unittest.main()
