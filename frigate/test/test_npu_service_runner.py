"""Tests for the external NPU service runner."""

import socket
import struct
import threading
import unittest

import numpy as np

from frigate.detectors.npu_service_runner import (
    NpuServiceRunner,
    get_npu_service_stats,
    parse_npu_device,
)


class FakeNpuService:
    """Answers each request with embedding = per-sample mean repeated `dim` times."""

    def __init__(
        self, dim: int = 8, fail: bool = False, claim_n: int | None = None
    ) -> None:
        self.dim, self.fail, self.claim_n = dim, fail, claim_n
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
                    if self.claim_n is not None:
                        # header announces far more data than is ever sent
                        c.sendall(
                            b"XFR1" + struct.pack("<iII", 0, self.claim_n, self.dim)
                        )
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

    def test_rejects_response_shape_it_did_not_ask_for(self):
        # A peer claiming ~16 GiB of embeddings must not drive the allocation
        svc = FakeNpuService(dim=1 << 16, claim_n=1 << 16)
        r = NpuServiceRunner("m.onnx", "npu", ("127.0.0.1", svc.port), timeout=0.5)
        with self.assertLogs("frigate.detectors.npu_service_runner", level="ERROR"):
            self.assertIsNone(r.run({"data": np.zeros((1, 3, 4, 4), np.float32)}))
        svc.close()

    def test_rejects_oversized_embedding_dim(self):
        svc = FakeNpuService(dim=(1 << 16) + 1, claim_n=1)
        r = NpuServiceRunner("m.onnx", "npu", ("127.0.0.1", svc.port), timeout=0.5)
        with self.assertLogs("frigate.detectors.npu_service_runner", level="ERROR"):
            self.assertIsNone(r.run({"data": np.zeros((1, 3, 4, 4), np.float32)}))
        svc.close()

    def test_reconnects_after_service_restart(self):
        svc = FakeNpuService()
        r = NpuServiceRunner("m.onnx", "npu", ("127.0.0.1", svc.port))
        x = np.ones((1, 3, 4, 4), np.float32)
        r.run({"data": x})
        r._sock.close()  # simulate a dropped connection
        self.assertIsNotNone(r.run({"data": x}))
        svc.close()


class TestNpuServiceStats(unittest.TestCase):
    def test_stats_from_service(self):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        stats_port = srv.getsockname()[1]
        body = b'{"npu": 42.5, "bo_mem_mib": 286.0, "embeddings_per_s": 7.0, "mean_latency_ms": 130.2, "h_clock_mhz": 1800}'

        def serve():
            c, _ = srv.accept()
            c.recv(1024)
            c.sendall(
                b"HTTP/1.0 200 OK\r\nContent-Type: application/json\r\n\r\n" + body
            )
            c.close()

        threading.Thread(target=serve, daemon=True).start()
        # stats live on the inference port + 1
        s = get_npu_service_stats(("127.0.0.1", stats_port - 1))
        self.assertEqual(s["npu"], 42.5)
        self.assertEqual(s["mem"], "286 MiB")
        self.assertEqual(s["clock_mhz"], 1800)
        srv.close()

    def test_stats_unavailable(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        self.assertIsNone(get_npu_service_stats(("127.0.0.1", port - 1), timeout=0.2))


if __name__ == "__main__":
    unittest.main()
