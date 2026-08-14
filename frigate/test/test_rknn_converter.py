import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies before any imports
from frigate.util.rknn_converter import (
    ensure_rknn_toolkit,
    get_rknn_model_type,
    is_rknn_compatible,
)


class TestGetRknnModelType(unittest.TestCase):
    def test_jina_clip_v1_vision(self):
        self.assertEqual(
            get_rknn_model_type("/models/jina-clip-v1-vision.onnx"),
            "jina-clip-v1-vision",
        )
        self.assertEqual(
            get_rknn_model_type("/models/jina-clip-v1-something-vision.onnx"),
            "jina-clip-v1-vision",
        )
        self.assertNotEqual(
            get_rknn_model_type("/models/jina-clip-v1.onnx"), "jina-clip-v1-vision"
        )
        self.assertNotEqual(
            get_rknn_model_type("/models/vision.onnx"), "jina-clip-v1-vision"
        )

    def test_arcface_r100(self):
        self.assertEqual(get_rknn_model_type("/models/arcface.onnx"), "arcface-r100")
        self.assertEqual(
            get_rknn_model_type("/models/arcface-r100.onnx"), "arcface-r100"
        )

    def test_yolo_variants(self):
        self.assertEqual(get_rknn_model_type("/models/yolov8n.onnx"), "yolov8n.onnx")
        self.assertEqual(get_rknn_model_type("/models/yolox.onnx"), "yolox.onnx")
        self.assertEqual(get_rknn_model_type("/models/yolonas.onnx"), "yolonas.onnx")

    def test_unsupported_models(self):
        self.assertIsNone(get_rknn_model_type("/models/resnet50.onnx"))
        self.assertIsNone(get_rknn_model_type("/models/mobilenet.onnx"))
        self.assertIsNone(get_rknn_model_type("/models/unknown_model.onnx"))


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
    def test_supported_soc_valid_inferred_model_type(
        self, mock_get_rknn_model_type, mock_get_soc_type
    ):
        mock_get_soc_type.return_value = "rk3588"
        mock_get_rknn_model_type.return_value = "yolox"
        self.assertTrue(is_rknn_compatible("yolox.onnx"))

    @patch("frigate.util.rknn_converter.get_soc_type")
    @patch("frigate.util.rknn_converter.get_rknn_model_type")
    def test_supported_soc_invalid_inferred_model_type(
        self, mock_get_rknn_model_type, mock_get_soc_type
    ):
        mock_get_soc_type.return_value = "rk3588"
        mock_get_rknn_model_type.return_value = None
        self.assertFalse(is_rknn_compatible("unknown.onnx"))


class TestEnsureTorch(unittest.TestCase):
    def test_already_installed(self):
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            from frigate.util.rknn_converter import ensure_torch_dependencies

            self.assertTrue(ensure_torch_dependencies())

    @patch("subprocess.check_call")
    def test_install_success(self, mock_check_call):
        original_import = __import__

        def side_effect(name, *args, **kwargs):
            if name == "torch":
                if mock_check_call.called:
                    return MagicMock()
                raise ImportError("No module named 'torch'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=side_effect):
            from frigate.util.rknn_converter import ensure_torch_dependencies

            result = ensure_torch_dependencies()

        self.assertTrue(result)
        mock_check_call.assert_called_once()
        self.assertEqual(mock_check_call.call_args[0][0][0], sys.executable)
        self.assertEqual(mock_check_call.call_args[0][0][1:4], ["-m", "pip", "install"])
        self.assertIn("torch", mock_check_call.call_args[0][0])
        self.assertIn("torchvision", mock_check_call.call_args[0][0])

    @patch("subprocess.check_call")
    def test_install_failure_subprocess(self, mock_check_call):
        original_import = __import__

        def side_effect(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return original_import(name, *args, **kwargs)

        import subprocess

        mock_check_call.side_effect = subprocess.CalledProcessError(1, "pip")

        with patch("builtins.__import__", side_effect=side_effect):
            from frigate.util.rknn_converter import ensure_torch_dependencies

            result = ensure_torch_dependencies()

        self.assertFalse(result)
        mock_check_call.assert_called_once()

    @patch("subprocess.check_call")
    def test_install_failure_import(self, mock_check_call):
        original_import = __import__

        def side_effect(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=side_effect):
            from frigate.util.rknn_converter import ensure_torch_dependencies

            result = ensure_torch_dependencies()

        self.assertFalse(result)
        mock_check_call.assert_called_once()


if __name__ == "__main__":
    unittest.main()
