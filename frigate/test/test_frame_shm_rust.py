import unittest
import numpy as np

from frigate.util.frame_rs import (
    frame_rs_available,
    intersection_over_union_rust,
    track_distance_rust,
    preprocess_detect_input_rust,
)


class TestFrameShmRust(unittest.TestCase):
    def test_frame_rs_available(self):
        """Ensure Rust frame engine library loads correctly."""
        self.assertTrue(frame_rs_available(), "libfrigate_frame_rs.so should be available")

    def test_iou_rust(self):
        """Test bounding box IoU calculation in Rust."""
        if not frame_rs_available():
            self.skipTest("Rust frame engine not available")

        # Overlapping boxes: box_a [0, 0, 10, 10], box_b [5, 0, 15, 10]
        # Pixel-inclusive area: width=11, height=11 -> area=121. Inter=6*11=66. Union=121+121-66=176 -> 66/176 = 0.375
        box_a = [0.0, 0.0, 10.0, 10.0]
        box_b = [5.0, 0.0, 15.0, 10.0]
        iou = intersection_over_union_rust(box_a, box_b)
        self.assertAlmostEqual(iou, 0.375, places=4)

        # Disjoint boxes
        box_c = [20.0, 20.0, 30.0, 30.0]
        iou_disjoint = intersection_over_union_rust(box_a, box_c)
        self.assertEqual(iou_disjoint, 0.0)

    def test_track_distance_rust(self):
        """Test Norfair association distance in Rust."""
        if not frame_rs_available():
            self.skipTest("Rust frame engine not available")

        # Two identical boxes -> distance = 0
        box = [10.0, 10.0, 50.0, 50.0]
        d = track_distance_rust(box, box)
        self.assertAlmostEqual(d, 0.0, places=4)

    def test_preprocess_detect_input_rust(self):
        """Test zero-copy end-to-end tensor preprocessing in Rust."""
        if not frame_rs_available():
            self.skipTest("Rust frame engine not available")

        w, h = 64, 64
        dst_w, dst_h = 32, 32
        src_bytes = bytes([128] * (w * h * 3))

        out_buf = preprocess_detect_input_rust(src_bytes, w, h, dst_w, dst_h, 3)
        out_np = np.ctypeslib.as_array(out_buf).reshape((3, dst_h, dst_w))

        # Shape must be NCHW: (3, 32, 32)
        self.assertEqual(out_np.shape, (3, 32, 32))
        # With pixel values = 128, normalized value ≈ 128 / 255.0 ≈ 0.50196
        self.assertTrue(np.all(out_np >= 0.0) and np.all(out_np <= 1.0))



if __name__ == "__main__":
    unittest.main()
