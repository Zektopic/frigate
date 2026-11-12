"""Bounds checks in the frame_rs ctypes wrappers.

Rust trusts every size it is handed, so the Python wrappers must reject
inputs that would make it read past a buffer or return early and leave an
uninitialized output array behind.
"""

import unittest

import numpy as np

from frigate.util.frame_rs import (
    frame_rs_available,
    yuv420_crop_repack_rust,
    yuv420_region_to_rgb_resize_rust,
    yuv420_to_3channel_rust,
)


@unittest.skipUnless(frame_rs_available(), "Rust frame engine not built")
class TestYuvCropRepackBounds(unittest.TestCase):
    def test_rejects_dimensions_larger_than_frame(self):
        # 64x48 I420 frame, but claim it is 640x480
        frame = np.zeros((72, 64), dtype=np.uint8)
        with self.assertRaises(ValueError):
            yuv420_crop_repack_rust(frame, 640, 480, 0, 0, 32, 32)

    def test_rejects_crop_outside_frame(self):
        frame = np.zeros((72, 64), dtype=np.uint8)
        with self.assertRaises(ValueError):
            yuv420_crop_repack_rust(frame, 64, 48, 48, 0, 32, 32)

    def test_rejects_negative_offsets(self):
        frame = np.zeros((72, 64), dtype=np.uint8)
        with self.assertRaises(ValueError):
            yuv420_crop_repack_rust(frame, 64, 48, -2, 0, 32, 32)

    def test_valid_crop_matches_source(self):
        frame = np.arange(72 * 64, dtype=np.uint32).astype(np.uint8).reshape(72, 64)
        out = yuv420_crop_repack_rust(frame, 64, 48, 8, 8, 16, 16)
        self.assertEqual(out.shape, (24, 16))
        np.testing.assert_array_equal(out[:16], frame[8:24, 8:24])


@unittest.skipUnless(frame_rs_available(), "Rust frame engine not built")
class TestYuvTo3ChannelBounds(unittest.TestCase):
    def test_rejects_odd_width(self):
        with self.assertRaises(ValueError):
            yuv420_to_3channel_rust(np.zeros((6, 5), dtype=np.uint8))

    def test_rejects_non_planar_input(self):
        with self.assertRaises(ValueError):
            yuv420_to_3channel_rust(np.zeros((6, 4, 3), dtype=np.uint8))

    def test_valid_frame_converts(self):
        frame = np.full((6, 4), 7, dtype=np.uint8)
        out = yuv420_to_3channel_rust(frame)
        self.assertEqual(out.shape, (4, 4, 3))
        self.assertTrue((out == 7).all())


@unittest.skipUnless(frame_rs_available(), "Rust frame engine not built")
class TestYuvRegionResizeBounds(unittest.TestCase):
    def test_rejects_odd_source_width(self):
        frame = np.zeros((72, 63), dtype=np.uint8)
        with self.assertRaises(ValueError):
            yuv420_region_to_rgb_resize_rust(frame, (0, 0, 32, 32), 16, 16)

    def test_rejects_degenerate_region(self):
        frame = np.zeros((72, 64), dtype=np.uint8)
        with self.assertRaises(ValueError):
            yuv420_region_to_rgb_resize_rust(frame, (0, 0, 2, 2), 16, 16)

    def test_rejects_non_positive_destination(self):
        frame = np.zeros((72, 64), dtype=np.uint8)
        with self.assertRaises(ValueError):
            yuv420_region_to_rgb_resize_rust(frame, (0, 0, 32, 32), 0, 16)

    def test_out_of_frame_region_is_padded_black(self):
        frame = np.full((72, 64), 200, dtype=np.uint8)
        out = yuv420_region_to_rgb_resize_rust(frame, (-32, -32, 0, 0), 16, 16)
        self.assertEqual(out.shape, (1, 16, 16, 3))
        self.assertTrue((out == 0).all())


if __name__ == "__main__":
    unittest.main()
