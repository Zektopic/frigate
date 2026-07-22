import unittest
import sys
import math
from unittest.mock import MagicMock, patch


class TestFaceModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mock_numpy = MagicMock()
        mock_numpy.exp = math.exp

        # We mock frigate.config to bypass all the Pydantic and detector imports
        mock_frigate_config = MagicMock()
        mock_frigate_embeddings = MagicMock()
        mock_frigate_log = MagicMock()

        cls.mock_modules = {
            "cv2": MagicMock(),
            "scipy": MagicMock(),
            "scipy.stats": MagicMock(),
            "insightface": MagicMock(),
            "insightface.app": MagicMock(),
            "numpy": mock_numpy,
            "frigate.config": mock_frigate_config,
            "frigate.embeddings.onnx.face_embedding": mock_frigate_embeddings,
            "frigate.log": mock_frigate_log,
        }
        cls.patcher = patch.dict("sys.modules", cls.mock_modules)
        cls.patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def test_similarity_to_confidence_default(self):
        from frigate.data_processing.common.face.model import similarity_to_confidence

        # Default median is 0.3
        # At median, confidence should be 0.5
        conf = similarity_to_confidence(0.3)
        self.assertAlmostEqual(conf, 0.5, places=5)

    def test_similarity_to_confidence_high(self):
        from frigate.data_processing.common.face.model import similarity_to_confidence

        # A very high similarity should map close to 1.0
        conf = similarity_to_confidence(1.0)
        self.assertGreater(conf, 0.99)

    def test_similarity_to_confidence_low(self):
        from frigate.data_processing.common.face.model import similarity_to_confidence

        # A very low similarity (e.g. 0.0 or negative) should map close to 0.0
        conf = similarity_to_confidence(0.0)
        self.assertLess(conf, 0.01)

        conf_neg = similarity_to_confidence(-1.0)
        self.assertLess(conf_neg, 0.001)

    def test_similarity_to_confidence_custom_params(self):
        from frigate.data_processing.common.face.model import similarity_to_confidence

        # Test with custom median, range_width, slope_factor
        conf = similarity_to_confidence(
            cosine_similarity=0.5, median=0.5, range_width=0.4, slope_factor=10
        )
        self.assertAlmostEqual(conf, 0.5, places=5)

        # Above median
        conf_high = similarity_to_confidence(
            cosine_similarity=0.7, median=0.5, range_width=0.4, slope_factor=10
        )
        self.assertGreater(conf_high, 0.9)

        # Below median
        conf_low = similarity_to_confidence(
            cosine_similarity=0.3, median=0.5, range_width=0.4, slope_factor=10
        )
        self.assertLess(conf_low, 0.1)


if __name__ == "__main__":
    unittest.main()
