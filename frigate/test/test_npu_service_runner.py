"""Tests for the external NPU service runner."""

import socket
import struct
import threading
import unittest

import numpy as np

from frigate.detectors.npu_service_runner import NpuServiceRunner, parse_npu_device


class FakeNpuService:
    """Answers each request with embedding = per-sample mean repeated `dim` times."""

    def __init__(self, dim: int = 8, fail: bool = False) -> None:
        self.dim, self.fail = dim, fail
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(4)
        self.port = self.srv.getsockname()[1]
        self.requests = 0
        threading.Thread(target=self._serve, daemon=True).start()

    def _recv(self, c, n):
        b = b""
        while len(b) < n:
            chunk = c.recv(n - len(b))
            if not chunk:
                raise ConnectionError
            b += chunk
        return b

    def _serve(self):
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            try:
                while True:
                    hdr = self._recv(c, 20)
                    assert hdr[:4] == b"XFQ1"
                    n, ch, h, w = struct.unpack("<4I", hdr[4:])
                    x = np.frombuffer(self._recv(c, n * ch * h * w * 4), np.float32)
                    self.requests += 1
                    if self.fail:
                        msg = b"boom"
                        c.sendall(b"XFR1" + struct.pack("<iII", -5, 0, len(msg)) + msg)
                        continue
                    means = x.reshape(n, -1).mean(1)
                    out = np.repeat(means[:, None], self.dim, 1).astype(np.float32)
                    c.sendall(
                        b"XFR1" + struct.pack("<iII", 0, n, self.dim) + out.tobytes()
                    )
            except ConnectionError:
                c.close()

    def close(self):
        self.srv.close()


class TestNpuServiceRunner(unittest.TestCase):
    def test_parse_device(self):
        self.assertEqual(parse_npu_device("npu"), ("127.0.0.1", 5800))
        self.assertEqual(parse_npu_device("NPU://10.0.0.2:6000"), ("10.0.0.2", 6000))
        self.assertIsNone(parse_npu_device("GPU"))
        self.assertIsNone(parse_npu_device(None))
        with self.assertRaises(ValueError):
            parse_npu_device("npu://nohostport")

    def test_roundtrip_batch(self):
        svc = FakeNpuService(dim=8)
        r = NpuServiceRunner("m.onnx", "npu", ("127.0.0.1", svc.port))
        x = np.stack([np.full((3, 4, 4), 1.0), np.full((3, 4, 4), 2.0)]).astype(
            np.float32
        )
        out = r.run({"data": x})[0]
        self.assertEqual(out.shape, (2, 8))
        np.testing.assert_allclose(out[:, 0], [1.0, 2.0])
        self.assertEqual(r.get_input_names(), ["data"])
        svc.close()

    def test_service_error_raises(self):
        svc = FakeNpuService(fail=True)
        r = NpuServiceRunner("m.onnx", "npu", ("127.0.0.1", svc.port))
        with self.assertRaises(RuntimeError):
            r.run({"data": np.zeros((1, 3, 4, 4), np.float32)})
        svc.close()

    def test_unavailable_service_returns_none(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()  # nothing listens here
        r = NpuServiceRunner("m.onnx", "npu", ("127.0.0.1", port), timeout=0.5)
        with self.assertLogs("frigate.detectors.npu_service_runner", level="ERROR"):
            self.assertIsNone(r.run({"data": np.zeros((1, 3, 4, 4), np.float32)}))

    def test_reconnects_after_service_restart(self):
        svc = FakeNpuService()
        r = NpuServiceRunner("m.onnx", "npu", ("127.0.0.1", svc.port))
        x = np.ones((1, 3, 4, 4), np.float32)
        r.run({"data": x})
        r._sock.close()  # simulate a dropped connection
        self.assertIsNotNone(r.run({"data": x}))
        svc.close()


if __name__ == "__main__":
    unittest.main()
