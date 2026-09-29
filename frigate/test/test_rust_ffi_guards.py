"""Regression tests for crash-class bugs at the Rust ctypes boundary.

Every native library here is a cdylib built with ``panic = "abort"``, so a
bad length or a Rust panic kills the calling process outright.  These tests
pin the Python-side validation and the Rust fixes that keep them recoverable.
"""

import ctypes
import os
import unittest

import numpy as np

from frigate.detectors.rust_yolo import nms_boxes, yolo_available, yolo_post_process
from frigate.motion.rust_engine import (
    accumulate_weighted,
    motion_available,
    pixel_pipeline,
)
from frigate.util.frame_rs import (
    batch_track_distance_matrix_rust,
    fast_shm_copy_rust,
    frame_rs_available,
    point_in_polygon_rust,
    polygon_box_overlap_rust,
    read_ffmpeg_frame_to_ptr,
)


def _multipart_outputs() -> list[np.ndarray]:
    """Three YOLOv5-style NCHW scales with one planted detection."""
    outs = [np.zeros((1, 255, n, n), np.float32) for n in (80, 40, 20)]
    cell = outs[0].reshape(1, 3, 85, 80, 80)
    cell[0, 0, 0:4, 10, 12] = 0.5  # dx, dy, dw, dh
    cell[0, 0, 4, 10, 12] = 0.9  # objectness
    cell[0, 0, 5 + 3, 10, 12] = 0.95  # class 3
    return outs


class TestRustYoloGuards(unittest.TestCase):
    def setUp(self) -> None:
        if not yolo_available():
            self.skipTest("Rust YOLO engine not available")

    def test_multipart_matches_python_decoder(self) -> None:
        """The Rust path used to return zero detections for every frame."""
        import frigate.util.model as model

        rust = yolo_post_process(_multipart_outputs(), 640, 640)
        python = getattr(model, "__post_process_multipart_yolo")(
            _multipart_outputs(), 640, 640
        )
        self.assertEqual(int((rust[:, 1] > 0).sum()), 1)
        np.testing.assert_allclose(rust[0], python[0], rtol=1e-5, atol=1e-6)

    def test_multipart_rejects_unexpected_shapes(self) -> None:
        with self.assertRaises(ValueError):
            yolo_post_process([np.zeros((1, 80, 80, 255), np.float32)] * 3, 640, 640)
        with self.assertRaises(ValueError):
            yolo_post_process(_multipart_outputs()[:2], 640, 640)

    def test_nms_nan_scores_do_not_abort(self) -> None:
        """A NaN score made sort_unstable_by panic, aborting the process."""
        rng = np.random.default_rng(0)
        for n in (21, 33, 64, 100):
            for _ in range(20):
                xy = rng.uniform(0, 600, (n, 2)).astype(np.float32)
                boxes = np.hstack([xy, xy + 10])
                scores = rng.uniform(0, 1, n).astype(np.float32)
                scores[rng.random(n) < 0.3] = np.nan
                kept = nms_boxes(boxes, scores)
                self.assertFalse(np.isnan(scores[kept]).any())

    def test_nms_rejects_short_box_array(self) -> None:
        with self.assertRaises(ValueError):
            nms_boxes(np.zeros((4, 4), np.float32), np.ones(64, np.float32))


class TestRustMotionGuards(unittest.TestCase):
    def setUp(self) -> None:
        if not motion_available():
            self.skipTest("Rust motion engine not available")

    def test_accumulate_rejects_smaller_average(self) -> None:
        """Rust writes src.size floats; a smaller avg was a heap overflow."""
        with self.assertRaises(ValueError):
            accumulate_weighted(
                np.zeros((64, 64), np.uint8), np.zeros((32, 32), np.float32), 0.5
            )

    def test_accumulate_rejects_average_it_cannot_update_in_place(self) -> None:
        """A float64 / non-contiguous avg used to be copied and the update lost."""
        src = np.full((8, 8), 200, np.uint8)
        with self.assertRaises(TypeError):
            accumulate_weighted(src, np.zeros((8, 8), np.float64), 0.5)
        with self.assertRaises(TypeError):
            accumulate_weighted(src, np.zeros((8, 16), np.float32)[:, ::2], 0.5)

    def test_accumulate_updates_in_place(self) -> None:
        avg = np.zeros((8, 9), np.float32)
        accumulate_weighted(np.full((8, 9), 200, np.uint8), avg, 0.5)
        np.testing.assert_allclose(avg, 100.0)

    def test_pixel_pipeline_rejects_mismatched_mask(self) -> None:
        frame = np.zeros((64, 64), np.uint8)
        with self.assertRaises(ValueError):
            pixel_pipeline(
                frame,
                np.zeros((64, 64), np.float32),
                np.zeros((32, 32), np.uint8),
                10,
                0,
            )

    def test_pixel_pipeline_tiny_and_non_contiguous_frames(self) -> None:
        rng = np.random.default_rng(1)
        for h, w in ((1, 1), (1, 7), (7, 1), (3, 5)):
            wide = rng.integers(0, 256, (h, 2 * w), dtype=np.uint8)
            boxes, area, _ = pixel_pipeline(
                wide[:, ::2],
                np.zeros((h, w), np.float32),
                np.zeros((h, w), np.uint8),
                10,
                0,
            )
            self.assertLessEqual(area, h * w)


class TestRustFrameGuards(unittest.TestCase):
    def setUp(self) -> None:
        if not frame_rs_available():
            self.skipTest("Rust frame engine not available")

    def test_reader_refuses_buffer_smaller_than_frame(self) -> None:
        """A stale, smaller shm segment used to be overrun by read(2)."""
        r, w = os.pipe()
        try:
            backing = bytearray(8192)
            view = memoryview(backing)[:4096]
            os.write(w, b"z" * 8192)
            addr = ctypes.addressof(ctypes.c_char.from_buffer(view))
            self.assertEqual(read_ffmpeg_frame_to_ptr(r, addr, 8192, len(view)), -1)
            self.assertEqual(backing.count(b"z"), 0)
            self.assertEqual(read_ffmpeg_frame_to_ptr(r, addr, 4096, len(view)), 1)
        finally:
            os.close(r)
            os.close(w)

    def test_fast_shm_copy_bounds_length(self) -> None:
        with self.assertRaises(ValueError):
            fast_shm_copy_rust(bytearray(16), bytearray(64), 32)

    def test_batch_distance_validates_box_width(self) -> None:
        with self.assertRaises(ValueError):
            batch_track_distance_matrix_rust([[0, 0, 10]], [[0, 0, 10, 10]])
        # norfair hands over (2, 2) point arrays
        det = np.array([[0.0, 0.0], [10.0, 10.0]])
        out = batch_track_distance_matrix_rust([det], [det])
        self.assertEqual(out.shape, (1, 1))
        self.assertEqual(out[0, 0], 0.0)

    def test_zone_geometry_vectors(self) -> None:
        """Vectors from feat/rust-zone-geometry-engine, on dev's API."""
        tri = [(0.0, 0.0), (100.0, 0.0), (50.0, 100.0)]
        self.assertTrue(point_in_polygon_rust(50.0, 30.0, tri))
        self.assertFalse(point_in_polygon_rust(0.0, 100.0, tri))
        self.assertFalse(point_in_polygon_rust(120.0, 50.0, tri))
        self.assertFalse(point_in_polygon_rust(50.0, -10.0, tri))

        square = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        self.assertTrue(polygon_box_overlap_rust(square, (10.0, 10.0, 90.0, 90.0)))
        self.assertFalse(polygon_box_overlap_rust(square, (200.0, 200.0, 250.0, 250.0)))
        # Edge-only crossing: no vertex or corner containment.
        band = [(-5.0, 1.0), (15.0, 1.0), (15.0, 2.0), (-5.0, 2.0)]
        self.assertTrue(polygon_box_overlap_rust(band, (0.0, 0.0, 10.0, 10.0)))


if __name__ == "__main__":
    unittest.main()
