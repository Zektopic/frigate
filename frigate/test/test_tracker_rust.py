import unittest

from frigate.util.frame_rs import (
    frame_rs_available,
    track_distance_rust,
    batch_track_distance_matrix_rust,
)


class TestTrackerRust(unittest.TestCase):
    def test_batch_track_distance_matrix_rust(self):
        """Test vectorized N x M association distance computation in Rust."""
        if not frame_rs_available():
            self.skipTest("Rust frame engine not available")

        box_a = [10.0, 20.0, 110.0, 220.0]
        box_b = [60.0, 20.0, 160.0, 220.0]
        box_c = [10.0, 20.0, 210.0, 220.0]

        dets = [box_a, box_b]
        ests = [box_a, box_c]

        # Calculate matrix in Rust
        matrix = batch_track_distance_matrix_rust(dets, ests)

        # Expected shape: 2 x 2
        self.assertEqual(len(matrix), 2)
        self.assertEqual(len(matrix[0]), 2)

        # Check element-wise equivalence against scalar track_distance_rust
        for i in range(2):
            for j in range(2):
                expected = track_distance_rust(dets[i], ests[j])
                self.assertAlmostEqual(
                    matrix[i][j],
                    expected,
                    places=5,
                    msg=f"Mismatch at ({i}, {j}): {matrix[i][j]} vs {expected}",
                )


if __name__ == "__main__":
    unittest.main()
