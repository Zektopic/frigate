"""Tests for the ncnn-vulkan detector plugin helpers (docker/ncnn-vulkan/onnx.py)."""

import importlib.util
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PLUGIN_PATH = Path(__file__).resolve().parents[2] / "docker/ncnn-vulkan/onnx.py"


def _load_plugin():
    spec = importlib.util.spec_from_file_location("ncnn_vulkan_plugin", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _param(cells: int) -> str:
    return (
        "7767517\n"
        "Input                    in0                      0 1 in0\n"
        f"MemoryData               pnnx_262                 0 1 312 0={cells}\n"
        "Concat                   cat_22                   2 1 379 380 out0 0=0\n"
    )


class _FakeGpuInfo:
    def __init__(self, gpu_type: int) -> None:
        self._type = gpu_type

    def type(self) -> int:
        return self._type


@unittest.skipUnless(PLUGIN_PATH.exists(), "ncnn plugin source not available")
class TestNcnnPluginHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plugin = _load_plugin()

    def test_model_size_follows_config(self):
        resolve = self.plugin._resolve_model_input_size
        self.assertEqual(resolve(SimpleNamespace(width=320, height=320)), 320)
        self.assertEqual(resolve(SimpleNamespace(width=640, height=640)), 640)
        self.assertEqual(resolve(SimpleNamespace(width=416, height=416)), 416)

    def test_model_size_falls_back_to_640(self):
        resolve = self.plugin._resolve_model_input_size
        self.assertEqual(resolve(SimpleNamespace(width=320, height=240)), 640)
        self.assertEqual(resolve(SimpleNamespace(width=300, height=300)), 640)
        self.assertEqual(resolve(SimpleNamespace(width=None, height=None)), 640)

    def test_yolo26_detected_at_any_export_size(self):
        detect = self.plugin._detect_architecture
        self.assertEqual(detect(_param(8400)), "yolo26")  # 640
        self.assertEqual(detect(_param(2100)), "yolo26")  # 320
        self.assertEqual(detect(_param(3549)), "yolo26")  # 416
        self.assertEqual(detect(_param(1234)), "yolov5")

    def test_software_vulkan_is_not_a_gpu(self):
        usable = self.plugin._usable_vulkan_gpu

        def fake(count: int, gpu_type: int = 1):
            return SimpleNamespace(
                get_gpu_count=lambda: count,
                get_gpu_info=lambda i: _FakeGpuInfo(gpu_type),
            )

        self.assertFalse(usable(fake(0)))
        self.assertFalse(usable(fake(1, gpu_type=3)))  # llvmpipe
        self.assertTrue(usable(fake(1, gpu_type=0)))  # discrete
        self.assertTrue(usable(fake(1, gpu_type=1)))  # integrated

    def test_num_threads_env(self):
        threads = self.plugin._ncnn_num_threads
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NCNN_NUM_THREADS", None)
            self.assertIsNone(threads())
            os.environ["NCNN_NUM_THREADS"] = "1"
            self.assertEqual(threads(), 1)
            os.environ["NCNN_NUM_THREADS"] = "0"
            self.assertIsNone(threads())
            os.environ["NCNN_NUM_THREADS"] = "many"
            self.assertIsNone(threads())


if __name__ == "__main__":
    unittest.main()
