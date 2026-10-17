import unittest
from unittest.mock import patch

import numpy as np

from frigate.util.frame_rs import (
    batch_track_distance_matrix_rust,
    bilinear_resize_u8_rust,
    frame_rs_available,
    nhwc_to_nchw_f32_rust,
    track_distance_rust,
    yuv420_to_rgb_rust,
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

    def test_norfair_vectorized_frigate_distance_parity(self):
        """Verify VectorizedDistance('frigate') matches ScalarDistance(frigate_distance)."""
        if not frame_rs_available():
            self.skipTest("Rust frame engine not available")

        from norfair.distances import ScalarDistance, get_distance_by_name
        from norfair.tracker import Detection, Tracker

        from frigate.track.norfair_tracker import (
            frigate_distance,
            frigate_vectorized_distance,
        )

        vec_dist = get_distance_by_name("frigate")
        scal_dist = ScalarDistance(frigate_distance)

        t_vec = Tracker(distance_function="frigate", distance_threshold=2.5, initialization_delay=0)
        t_scal = Tracker(distance_function=frigate_distance, distance_threshold=2.5, initialization_delay=0)

        dets1 = [
            Detection(points=np.array([[10.0, 20.0], [110.0, 220.0]]), label="person"),
            Detection(points=np.array([[300.0, 100.0], [450.0, 250.0]]), label="car"),
        ]
        t_vec.update(detections=dets1)
        t_scal.update(detections=dets1)

        dets2 = [
            Detection(points=np.array([[14.0, 22.0], [114.0, 222.0]]), label="person"),
            Detection(points=np.array([[80.0, 30.0], [180.0, 230.0]]), label="person"),
            Detection(points=np.array([[305.0, 102.0], [455.0, 252.0]]), label="car"),
        ]
        m_vec = vec_dist.get_distances(t_vec.tracked_objects, dets2)
        m_scal = scal_dist.get_distances(t_scal.tracked_objects, dets2)
        np.testing.assert_allclose(m_vec, m_scal, rtol=1e-5, atol=1e-5)

        # Verify fallback when _HAS_RUST_DISTANCE is False
        cands = np.array([[14.0, 22.0, 114.0, 222.0], [80.0, 30.0, 180.0, 230.0]])
        objs = np.array([[10.0, 20.0, 110.0, 220.0]])
        with patch("frigate.track.norfair_tracker._HAS_RUST_DISTANCE", False):
            m_py = frigate_vectorized_distance(cands, objs)
        m_rs = frigate_vectorized_distance(cands, objs)
        np.testing.assert_allclose(m_py, m_rs, rtol=1e-6, atol=1e-6)

    def test_new_frame_rs_bindings(self):
        """Test bilinear_resize_u8_rust, yuv420_to_rgb_rust, and nhwc_to_nchw_f32_rust."""
        if not frame_rs_available():
            self.skipTest("Rust frame engine not available")

        # 1. bilinear_resize_u8_rust
        img = np.full((64, 48, 3), 200, dtype=np.uint8)
        resized = bilinear_resize_u8_rust(img, 32, 32)
        self.assertEqual(resized.shape, (32, 32, 3))
        self.assertEqual(resized.dtype, np.uint8)
        np.testing.assert_array_equal(resized, 200)

        # 2. yuv420_to_rgb_rust (neutral gray Y=128, U=128, V=128 -> RGB ~128)
        yuv = np.full((12, 8), 128, dtype=np.uint8)  # 8x8 frame -> I420 height 12
        rgb = yuv420_to_rgb_rust(yuv)
        self.assertEqual(rgb.shape, (8, 8, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        np.testing.assert_allclose(rgb, 128, atol=1)

        # 3. nhwc_to_nchw_f32_rust
        nhwc = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        nchw = nhwc_to_nchw_f32_rust(nhwc)
        self.assertEqual(nchw.shape, (4, 2, 3))
        self.assertTrue(nchw.flags["C_CONTIGUOUS"])
        np.testing.assert_array_equal(nchw, np.transpose(nhwc, (2, 0, 1)))

    def test_arcface_preprocess_bgr_to_rgb_rust(self):
        """Verify ArcfaceEmbedding._preprocess_inputs swaps BGR->RGB and outputs C-contiguous (1, 3, 112, 112)."""
        if not frame_rs_available():
            self.skipTest("Rust frame engine not available")

        from frigate.embeddings.onnx.face_embedding import ArcfaceEmbedding

        embedder = ArcfaceEmbedding.__new__(ArcfaceEmbedding)
        # Pure Blue in BGR (B=255, G=0, R=0) should become pure Blue in channel 2 of RGB (R=-1.0, G=-1.0, B=+1.0)
        bgr_crop = np.zeros((140, 140, 3), dtype=np.uint8)
        bgr_crop[:, :, 0] = 255

        out = embedder._preprocess_inputs([bgr_crop])[0]["data"]
        self.assertEqual(out.shape, (1, 3, 112, 112))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(out.flags["C_CONTIGUOUS"])
        # Channel 0 (R) and Channel 1 (G) should be -1.0, Channel 2 (B) should be +1.0
        self.assertAlmostEqual(float(out[0, 0, 56, 56]), -1.0, places=4)
        self.assertAlmostEqual(float(out[0, 1, 56, 56]), -1.0, places=4)
        self.assertAlmostEqual(float(out[0, 2, 56, 56]), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
