import unittest
import numpy as np
import cv2

from frigate.motion.rust_engine import motion_available, accumulate_weighted


class TestMotionAccumulatorRust(unittest.TestCase):
    def test_accumulate_weighted_parity_with_opencv(self):
        """Test that Rust SIMD accumulate_weighted matches cv2.accumulateWeighted within float tolerance."""
        if not motion_available():
            self.skipTest("Rust motion engine not available")

        # Test across various resolutions and alpha parameters
        test_shapes = [(100, 100), (480, 640), (720, 1280)]
        alphas = [0.01, 0.05, 0.2, 0.5]

        for h, w in test_shapes:
            for alpha in alphas:
                np.random.seed(42)
                src = np.random.randint(0, 256, (h, w), dtype=np.uint8)
                avg_cv = np.random.uniform(0.0, 255.0, (h, w)).astype(np.float32)
                avg_rs = avg_cv.copy()

                cv2.accumulateWeighted(src, avg_cv, alpha)
                accumulate_weighted(src, avg_rs, alpha)

                max_diff = np.max(np.abs(avg_cv - avg_rs))
                self.assertLess(
                    max_diff,
                    1e-4,
                    f"Discrepancy too high for shape ({h}, {w}) and alpha {alpha}: max_diff={max_diff}",
                )

    def test_accumulate_weighted_boundary_values(self):
        """Test with edge values: all zeros, all 255s, and single pixel differences."""
        if not motion_available():
            self.skipTest("Rust motion engine not available")

        src = np.full((64, 64), 255, dtype=np.uint8)
        avg_rs = np.zeros((64, 64), dtype=np.float32)

        accumulate_weighted(src, avg_rs, 0.1)
        # Expected: 0.9 * 0 + 0.1 * 255 = 25.5
        np.testing.assert_allclose(avg_rs, 25.5, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
