import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies before any imports
sys.modules['cv2'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['peewee'] = MagicMock()
sys.modules['playhouse'] = MagicMock()
sys.modules['playhouse.sqlite_ext'] = MagicMock()
sys.modules['playhouse.shortcuts'] = MagicMock()
sys.modules['unidecode'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['tzlocal'] = MagicMock()

from frigate.util.rknn_converter import is_rknn_compatible

class TestIsRknnCompatible(unittest.TestCase):
    @patch("frigate.util.rknn_converter.get_soc_type")
    def test_no_soc(self, mock_get_soc_type):
        mock_get_soc_type.return_value = None
        self.assertFalse(is_rknn_compatible("test.onnx"))

    @patch("frigate.util.rknn_converter.get_soc_type")
    def test_unsupported_soc(self, mock_get_soc_type):
        mock_get_soc_type.return_value = "bcm2711"
        self.assertFalse(is_rknn_compatible("test.onnx"))

    @patch("frigate.util.rknn_converter.get_soc_type")
    def test_supported_soc_valid_model_type(self, mock_get_soc_type):
        mock_get_soc_type.return_value = "rk3588"
        self.assertTrue(is_rknn_compatible("test.onnx", model_type="yolox"))

    @patch("frigate.util.rknn_converter.get_soc_type")
    def test_supported_soc_invalid_model_type(self, mock_get_soc_type):
        mock_get_soc_type.return_value = "rk3588"
        self.assertFalse(is_rknn_compatible("test.onnx", model_type="invalid"))

    @patch("frigate.util.rknn_converter.get_soc_type")
    @patch("frigate.util.rknn_converter.get_rknn_model_type")
    def test_supported_soc_valid_inferred_model_type(self, mock_get_rknn_model_type, mock_get_soc_type):
        mock_get_soc_type.return_value = "rk3588"
        mock_get_rknn_model_type.return_value = "yolox"
        self.assertTrue(is_rknn_compatible("yolox.onnx"))

    @patch("frigate.util.rknn_converter.get_soc_type")
    @patch("frigate.util.rknn_converter.get_rknn_model_type")
    def test_supported_soc_invalid_inferred_model_type(self, mock_get_rknn_model_type, mock_get_soc_type):
        mock_get_soc_type.return_value = "rk3588"
        mock_get_rknn_model_type.return_value = None
        self.assertFalse(is_rknn_compatible("unknown.onnx"))

if __name__ == '__main__':
    unittest.main()
