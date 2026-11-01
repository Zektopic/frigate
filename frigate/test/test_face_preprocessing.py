"""Face embedding preprocessing must not depend on the input type or on
whether the Rust frame library is available."""

import io
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from frigate.embeddings.onnx.face_embedding import (
    ARCFACE_INPUT_SIZE,
    FACENET_INPUT_SIZE,
    ArcfaceEmbedding,
    FaceNetEmbedding,
    bilinear_resize_u8_numpy,
)
from frigate.util.frame_rs import frame_rs_available


def _embedder(cls):
    # Skip __init__: no model download/runner needed to test preprocessing.
    return cls.__new__(cls)


def _face(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


class TestBilinearPort(unittest.TestCase):
    @unittest.skipUnless(frame_rs_available(), "Rust frame library not available")
    def test_numpy_port_matches_rust_bit_for_bit(self):
        from frigate.util.frame_rs import bilinear_resize_u8_rust

        rng = np.random.default_rng(1)
        for _ in range(60):
            sh, sw = rng.integers(8, 300, size=2)
            dh, dw = rng.integers(4, 200, size=2)
            ch = int(rng.choice([1, 3]))
            shape = (sh, sw) if ch == 1 else (sh, sw, ch)
            img = rng.integers(0, 256, size=shape, dtype=np.uint8)
            np.testing.assert_array_equal(
                bilinear_resize_u8_numpy(img, int(dw), int(dh)),
                bilinear_resize_u8_rust(img, int(dw), int(dh)),
                err_msg=f"{shape} -> ({dh}, {dw})",
            )

    def test_identity_size_is_a_copy(self):
        img = _face(20, 30)
        out = bilinear_resize_u8_numpy(img, 30, 20)
        np.testing.assert_array_equal(out, img)
        self.assertIsNot(out, img)


class TestFacePreprocessing(unittest.TestCase):
    def _arc(self, raw):
        return _embedder(ArcfaceEmbedding)._preprocess_inputs([raw])[0]["data"]

    def test_arcface_shape_and_range(self):
        x = self._arc(_face(54, 40))
        self.assertEqual(x.shape, (1, 3, ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE))
        self.assertEqual(x.dtype, np.float32)
        self.assertGreaterEqual(x.min(), -1.0)
        self.assertLessEqual(x.max(), 1.0)

    def test_same_with_and_without_rust(self):
        for h, w in [(54, 40), (200, 150), (112, 112), (30, 90)]:
            bgr = _face(h, w, seed=h * w)
            with_rust = self._arc(bgr)
            with patch("frigate.util.frame_rs.frame_rs_available", return_value=False):
                without_rust = self._arc(bgr)
            np.testing.assert_array_equal(with_rust, without_rust)

            fn = _embedder(FaceNetEmbedding)
            a = fn._preprocess_inputs([bgr])
            with patch("frigate.util.frame_rs.frame_rs_available", return_value=False):
                b = fn._preprocess_inputs([bgr])
            self.assertEqual(a.shape, (1, FACENET_INPUT_SIZE, FACENET_INPUT_SIZE, 3))
            np.testing.assert_array_equal(a, b)

    def test_bgr_array_and_encoded_bytes_match(self):
        # Library images are read with cv2.imread (BGR arrays); API uploads
        # arrive as encoded bytes. Both must be embedded identically.
        bgr = _face(64, 48, seed=7)
        ok, png = cv2.imencode(".png", bgr)
        self.assertTrue(ok)
        np.testing.assert_array_equal(self._arc(bgr), self._arc(png.tobytes()))

    def test_channel_order_is_rgb(self):
        bgr = np.zeros((112, 112, 3), np.uint8)
        bgr[:, :, 2] = 255  # red in BGR
        x = self._arc(bgr)
        # channel 0 of the NCHW tensor is R
        self.assertAlmostEqual(float(x[0, 0].mean()), 1.0, places=5)
        self.assertAlmostEqual(float(x[0, 2].mean()), -1.0, places=5)

    def test_grayscale_and_bgra_inputs(self):
        gray = np.full((50, 40), 128, np.uint8)
        self.assertEqual(self._arc(gray).shape, (1, 3, 112, 112))
        bgr = _face(50, 40, seed=3)
        bgra = np.concatenate([bgr, np.full((50, 40, 1), 255, np.uint8)], axis=2)
        np.testing.assert_array_equal(self._arc(bgra), self._arc(bgr))

    def test_pil_input_matches_array(self):
        from PIL import Image

        bgr = _face(60, 45, seed=9)
        pil = Image.open(io.BytesIO(cv2.imencode(".png", bgr)[1].tobytes()))
        np.testing.assert_array_equal(self._arc(bgr), self._arc(pil))


if __name__ == "__main__":
    unittest.main()
