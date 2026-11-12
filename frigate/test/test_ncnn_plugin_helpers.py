"""Tests for the ncnn-vulkan detector plugin helpers (docker/ncnn-vulkan/onnx.py)."""

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

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


# Reads requests like frigate-detector-rs and answers with one detection
# whose x1 is the size of the request it read.
FAKE_WORKER = r"""
import os, struct, sys
import numpy as np

def read_exact(n):
    buf = b""
    while len(buf) < n:
        chunk = os.read(0, n - len(buf))
        if not chunk:
            sys.exit(0)
        buf += chunk
    return buf

while True:
    size = struct.unpack(">I", read_exact(4))[0]
    read_exact(size)
    out = np.zeros((20, 6), np.float32)
    out[0] = [1, 0.9, size, 0, 0, 0]
    data = out.tobytes()
    os.write(1, struct.pack(">I", len(data)) + data)
"""


@unittest.skipUnless(PLUGIN_PATH.exists(), "ncnn plugin source not available")
class TestNcnnWorkerPayload(unittest.TestCase):
    """What the plugin sends to its inference worker."""

    @classmethod
    def setUpClass(cls):
        cls.plugin = _load_plugin()

    def _detector(self, rust_worker: bool = True, size: int = 4):
        det = object.__new__(self.plugin.ONNXDetector)
        det.model_input_size = size
        det._use_worker = True
        det._worker_accepts_uint8 = rust_worker
        return det

    def test_rust_worker_takes_the_raw_frame(self):
        det = self._detector()
        self.assertTrue(det.accepts_uint8_nhwc)
        frame = np.arange(48, dtype=np.uint8).reshape(1, 4, 4, 3)
        self.assertIs(det._worker_payload(frame), frame)

    def test_python_worker_gets_the_normalised_fp16_tensor(self):
        det = self._detector(rust_worker=False)
        self.assertFalse(det.accepts_uint8_nhwc)
        frame = np.arange(48, dtype=np.uint8).reshape(1, 4, 4, 3)
        payload = det._worker_payload(frame)
        want = (
            np.transpose(frame, (0, 3, 1, 2)).astype(np.float32) * np.float32(1 / 255)
        ).astype(np.float16)
        np.testing.assert_array_equal(payload, want)
        self.assertTrue(payload.flags.c_contiguous)

    def test_in_process_backends_do_not_take_raw_frames(self):
        det = self._detector()
        det._use_worker = False
        self.assertFalse(det.accepts_uint8_nhwc)

    def test_frame_of_another_size_is_not_sent_raw(self):
        frame = np.zeros((1, 8, 8, 3), np.uint8)  # the model is 4x4
        payload = self._detector()._worker_payload(frame)
        self.assertEqual((payload.dtype, payload.shape), (np.float16, (1, 3, 8, 8)))

    def test_float_tensor_bytes_are_what_was_sent_before(self):
        # before: tensor.astype(np.float16).tobytes(), which is C order even
        # for the transposed view Frigate's Python path produces
        nhwc = np.random.default_rng(1).random((1, 4, 4, 3), dtype=np.float32)
        tensor = np.transpose(nhwc, (0, 3, 1, 2))
        payload = self._detector()._worker_payload(tensor)
        self.assertEqual(payload.tobytes(), tensor.astype(np.float16).tobytes())
        memoryview(payload).cast("B")  # writable as one buffer

    def test_write_all_survives_partial_writes(self):
        sink = bytearray()

        def short_write(fd, data):
            chunk = bytes(data[:7])
            sink.extend(chunk)
            return len(chunk)

        with patch.object(self.plugin.os, "write", side_effect=short_write):
            self.plugin.ONNXDetector._write_all(5, np.arange(100, dtype=np.uint8))
        self.assertEqual(bytes(sink), bytes(range(100)))

    def test_round_trip_through_a_worker(self):
        det = self._detector(size=4)
        det._worker = subprocess.Popen(
            [sys.executable, "-c", FAKE_WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

        def stop():
            det._worker.stdin.close()
            det._worker.wait(timeout=5)
            det._worker.stdout.close()

        self.addCleanup(stop)

        # x1 comes back in column 3, divided by the model size
        raw = det._detect_worker(np.zeros((1, 4, 4, 3), np.uint8))
        self.assertAlmostEqual(float(raw[0][3]) * 4, 3 * 4 * 4)
        tensor = det._detect_worker(np.zeros((1, 3, 4, 4), np.float32))
        self.assertAlmostEqual(float(tensor[0][3]) * 4, 2 * 3 * 4 * 4)
