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

import subprocess
from frigate.util.rknn_converter import ensure_rknn_toolkit

class TestEnsureRknnToolkit(unittest.TestCase):
    def setUp(self):
        # We need to save the original builtins.__import__ to call it for non-mocked modules
        self.original_import = __import__
        # Create a dict to track import attempts to test the two-pass logic
        self.import_attempts = {"rknn": 0}

    def _mock_import(self, name, *args, **kwargs):
        if name == "rknn":
            self.import_attempts["rknn"] += 1
            if self.rknn_import_behavior[self.import_attempts["rknn"] - 1] == "fail":
                raise ImportError(f"No module named {name}")
            return MagicMock()
        return self.original_import(name, *args, **kwargs)

    @patch("builtins.__import__")
    @patch("frigate.util.rknn_converter.subprocess.check_call")
    def test_rknn_already_installed(self, mock_check_call, mock_import):
        self.rknn_import_behavior = ["success"]
        mock_import.side_effect = self._mock_import

        result = ensure_rknn_toolkit()

        self.assertTrue(result)
        mock_check_call.assert_not_called()
        self.assertEqual(self.import_attempts["rknn"], 1)

    @patch("builtins.__import__")
    @patch("frigate.util.rknn_converter.subprocess.check_call")
    def test_rknn_dynamic_install_success(self, mock_check_call, mock_import):
        self.rknn_import_behavior = ["fail", "success"]
        mock_import.side_effect = self._mock_import

        result = ensure_rknn_toolkit()

        self.assertTrue(result)
        mock_check_call.assert_called_once()
        self.assertEqual(self.import_attempts["rknn"], 2)

    @patch("builtins.__import__")
    @patch("frigate.util.rknn_converter.subprocess.check_call")
    def test_rknn_dynamic_install_fail_subprocess(self, mock_check_call, mock_import):
        self.rknn_import_behavior = ["fail"]
        mock_import.side_effect = self._mock_import
        mock_check_call.side_effect = subprocess.CalledProcessError(1, "pip")

        result = ensure_rknn_toolkit()

        self.assertFalse(result)
        mock_check_call.assert_called_once()
        self.assertEqual(self.import_attempts["rknn"], 1)

    @patch("builtins.__import__")
    @patch("frigate.util.rknn_converter.subprocess.check_call")
    def test_rknn_dynamic_install_fail_import(self, mock_check_call, mock_import):
        self.rknn_import_behavior = ["fail", "fail"]
        mock_import.side_effect = self._mock_import

        result = ensure_rknn_toolkit()

        self.assertFalse(result)
        mock_check_call.assert_called_once()
        self.assertEqual(self.import_attempts["rknn"], 2)


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
