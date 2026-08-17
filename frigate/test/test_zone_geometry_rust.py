import unittest

from frigate.util.frame_rs import (
    frame_rs_available,
    point_in_polygon_rust,
    polygon_box_overlap_rust,
)


class TestZoneGeometryRust(unittest.TestCase):
    def test_point_in_polygon_rust(self):
        """Test ray-casting point-in-polygon algorithm in Rust across concave & convex zones."""
        if not frame_rs_available():
            self.skipTest("Rust frame engine not available")

        # Triangle zone: (0, 0), (100, 0), (50, 100)
        tri_zone = [0.0, 0.0, 100.0, 0.0, 50.0, 100.0]

        # Centroid should be inside
        self.assertTrue(point_in_polygon_rust(50.0, 30.0, tri_zone))

        # Outside points
        self.assertFalse(point_in_polygon_rust(0.0, 100.0, tri_zone))
        self.assertFalse(point_in_polygon_rust(120.0, 50.0, tri_zone))
        self.assertFalse(point_in_polygon_rust(50.0, -10.0, tri_zone))

    def test_polygon_box_overlap_rust(self):
        """Test bounding box overlap ratio with arbitrary polygonal zones."""
        if not frame_rs_available():
            self.skipTest("Rust frame engine not available")

        # Square zone: (0, 0) to (100, 100)
        square_zone = [0.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 100.0]

        # Fully contained box -> 1.0
        inside_box = [10.0, 10.0, 90.0, 90.0]
        self.assertEqual(polygon_box_overlap_rust(inside_box, square_zone), 1.0)

        # Fully external box -> 0.0
        outside_box = [200.0, 200.0, 250.0, 250.0]
        self.assertEqual(polygon_box_overlap_rust(outside_box, square_zone), 0.0)


if __name__ == "__main__":
    unittest.main()
