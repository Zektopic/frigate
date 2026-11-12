"""Tests for the per-frame box helpers and detector input creation.

These helpers used to dispatch to Rust over ctypes.  For inputs of the size
Frigate sees per frame (one box pair, a handful of motion boxes, a ten-entry
stationary history) the FFI marshalling cost more than the work, so they are
plain Python again.  The tests pin their results, including the edge cases
that a reimplementation is most likely to get wrong.
"""

import unittest
from fractions import Fraction

import cv2
import numpy as np

from frigate.detectors.detector_config import ModelConfig
from frigate.util.image import (
    intersection_over_union,
    yuv_crop_and_resize,
)
from frigate.util.object import (
    average_boxes,
    box_inside,
    box_overlaps,
    create_tensor_input,
    inside_any,
    intersects_any,
    median_of_boxes,
)


def _reference_iou(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Inclusive-pixel IoU computed exactly, then rounded once to float."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return float(Fraction(inter, area_a + area_b - inter))


class TestIntersectionOverUnion(unittest.TestCase):
    def test_matches_exact_double_precision_result(self) -> None:
        # The Rust kernel computed this in f32 (0.638741672...), which is not
        # what the Python implementation returns.
        a, b = (100, 120, 300, 400), (150, 130, 320, 390)
        self.assertEqual(intersection_over_union(a, b), _reference_iou(a, b))
        self.assertEqual(intersection_over_union(a, b), 0.6387416735547236)

    def test_edge_cases(self) -> None:
        cases = [
            ((0, 0, 9, 9), (0, 0, 9, 9)),  # identical -> 1.0
            ((0, 0, 9, 9), (9, 9, 19, 19)),  # share one corner pixel
            ((0, 0, 9, 9), (10, 0, 19, 9)),  # adjacent, no shared pixel
            ((0, 0, 99, 99), (10, 10, 20, 20)),  # containment
            ((5, 5, 5, 5), (5, 5, 5, 5)),  # single pixel boxes
        ]
        for a, b in cases:
            with self.subTest(a=a, b=b):
                self.assertEqual(intersection_over_union(a, b), _reference_iou(a, b))
                self.assertEqual(intersection_over_union(b, a), _reference_iou(a, b))

    def test_random_boxes_match_reference(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(500):
            x1, y1 = rng.integers(0, 500, 2)
            x2, y2 = rng.integers(0, 500, 2)
            a = (int(x1), int(y1), int(x1 + rng.integers(0, 200)), int(y1 + 50))
            b = (int(x2), int(y2), int(x2 + 80), int(y2 + rng.integers(0, 200)))
            self.assertEqual(intersection_over_union(a, b), _reference_iou(a, b))


class TestBoxPredicates(unittest.TestCase):
    def test_overlap_is_inclusive_of_edges(self) -> None:
        self.assertTrue(box_overlaps((0, 0, 10, 10), (10, 10, 20, 20)))
        self.assertFalse(box_overlaps((0, 0, 10, 10), (11, 0, 20, 10)))

    def test_box_inside(self) -> None:
        self.assertTrue(box_inside((0, 0, 10, 10), (0, 0, 10, 10)))
        self.assertTrue(box_inside((0, 0, 10, 10), (2, 2, 8, 8)))
        self.assertFalse(box_inside((0, 0, 10, 10), (2, 2, 11, 8)))
        self.assertFalse(box_inside((2, 2, 8, 8), (0, 0, 10, 10)))

    def test_intersects_any(self) -> None:
        boxes = [(0, 0, 10, 10), (50, 50, 60, 60)]
        self.assertFalse(intersects_any((20, 20, 30, 30), []))
        self.assertFalse(intersects_any((20, 20, 30, 30), boxes))
        self.assertTrue(intersects_any((60, 60, 70, 70), boxes))
        self.assertTrue(intersects_any((5, 5, 6, 6), boxes))

    def test_inside_any(self) -> None:
        boxes = [(0, 0, 10, 10), (50, 50, 60, 60)]
        self.assertFalse(inside_any((1, 1, 2, 2), []))
        self.assertTrue(inside_any((50, 50, 60, 60), boxes))
        self.assertTrue(inside_any((1, 1, 2, 2), boxes))
        # overlapping is not enough, it must be fully contained
        self.assertFalse(inside_any((5, 5, 15, 15), boxes))


class TestBoxStatistics(unittest.TestCase):
    def test_average_boxes(self) -> None:
        boxes = [[10, 20, 100, 200], [20, 30, 110, 210], [30, 40, 120, 221]]
        self.assertEqual(average_boxes(boxes), [20.0, 30.0, 110.0, 631 / 3])

    def test_median_of_boxes_picks_by_area(self) -> None:
        small, mid, large = (0, 0, 9, 9), (0, 0, 19, 19), (0, 0, 29, 29)
        self.assertEqual(median_of_boxes([large, small, mid]), mid)
        # even length picks the upper median, matching int(len / 2)
        self.assertEqual(median_of_boxes([large, small, mid, large]), large)
        self.assertEqual(median_of_boxes([small]), small)


class TestCreateTensorInput(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(11)
        self.height, self.width = 360, 640
        self.frame = rng.integers(
            0, 256, size=(self.height * 3 // 2, self.width), dtype=np.uint8
        )

    def _reference(self, region, code: int, size: int) -> np.ndarray:
        rgb = cv2.cvtColor(yuv_crop_and_resize(self.frame, region), code)
        if rgb.shape[:2] != (size, size):
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        return rgb[np.newaxis]

    def test_matches_opencv_conversion(self) -> None:
        regions = [
            (100, 40, 420, 360),  # in bounds, aligned
            (-40, -20, 280, 300),  # partly outside the frame
            (500, 100, 820, 420),  # runs off the right edge
            (0, 0, 160, 160),  # smaller than the model, upscaled
        ]
        for fmt, code in (
            ("rgb", cv2.COLOR_YUV2RGB_I420),
            ("bgr", cv2.COLOR_YUV2BGR_I420),
        ):
            config = ModelConfig(width=320, height=320, input_pixel_format=fmt)
            for region in regions:
                with self.subTest(fmt=fmt, region=region):
                    out = create_tensor_input(self.frame, config, region)
                    self.assertEqual(out.shape, (1, 320, 320, 3))
                    self.assertEqual(out.dtype, np.uint8)
                    np.testing.assert_array_equal(
                        out, self._reference(region, code, 320)
                    )

    def test_uses_limited_range_bt601(self) -> None:
        # Video-range white (Y=235) and black (Y=16) must map to 255 and 0.
        # The removed Rust kernel used full-range coefficients and produced
        # 235 and 16, i.e. a lower-contrast image for the detector.
        frame = np.full((self.height * 3 // 2, self.width), 128, dtype=np.uint8)
        frame[: self.height, : self.width // 2] = 235
        frame[: self.height, self.width // 2 :] = 16
        config = ModelConfig(width=320, height=320, input_pixel_format="rgb")
        white = create_tensor_input(frame, config, (0, 0, 320, 320))
        black = create_tensor_input(frame, config, (320, 0, 640, 320))
        self.assertTrue(np.all(white == 255))
        self.assertTrue(np.all(black == 0))


if __name__ == "__main__":
    unittest.main()
