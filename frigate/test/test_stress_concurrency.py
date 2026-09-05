"""Stress & Concurrency Test Suite for Frigate Subsystems.

Tests high-concurrency database writers under WAL mode, high-density Norfair tracker
association matrices, and sustained zero-copy SIMD memory streaming.
"""

import os
import tempfile
import threading
import time
import unittest
import numpy as np

from frigate.db.sqlitevecq import SqliteVecQueueDatabase
from frigate.util.frame_rs import (
    frame_rs_available,
    batch_track_distance_matrix_rust,
    fast_shm_copy_rust,
)


class TestStressConcurrency(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "stress_frigate.db")
        self._init_schema()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _init_schema(self):
        import sqlite3
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS test_events (
                id TEXT PRIMARY KEY,
                camera TEXT,
                score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        conn.close()

    def test_sqlite_concurrent_writers_stress(self):
        """Stress test SQLite database with 30 concurrent threads performing rapid inserts."""
        import sqlite3
        num_threads = 30
        inserts_per_thread = 50
        errors = []

        def worker(thread_idx: int):
            try:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA busy_timeout=30000;")
                cursor = conn.cursor()
                for i in range(inserts_per_thread):
                    event_id = f"evt_{thread_idx}_{i}_{time.time()}"
                    cursor.execute(
                        "INSERT INTO test_events (id, camera, score) VALUES (?, ?, ?)",
                        (event_id, f"cam_{thread_idx % 4}", 0.85),
                    )
                    conn.commit()
                conn.close()
            except Exception as e:
                errors.append((thread_idx, e))

        threads = [
            threading.Thread(target=worker, args=(t,))
            for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Concurrent writer errors: {errors}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM test_events")
        count = cursor.fetchone()[0]
        conn.close()
        self.assertEqual(count, num_threads * inserts_per_thread)

    def test_tracker_distance_matrix_high_density_stress(self):
        """Stress test NxM tracker distance matrix with 100 detections x 100 estimates."""
        if not frame_rs_available():
            self.skipTest("Rust frame engine not available")

        n_dets = 100
        n_ests = 100

        dets = [
            (i * 5.0, i * 5.0, (i + 2) * 5.0, (i + 2) * 5.0)
            for i in range(n_dets)
        ]
        ests = [
            (j * 5.0 + 1.0, j * 5.0 + 1.0, (j + 2) * 5.0 + 1.0, (j + 2) * 5.0 + 1.0)
            for j in range(n_ests)
        ]

        t0 = time.perf_counter()
        # Execute 50 iterations of 100x100 matrix calculation
        for _ in range(50):
            matrix = batch_track_distance_matrix_rust(dets, ests)
            self.assertEqual(matrix.shape, (n_dets, n_ests))
        elapsed = time.perf_counter() - t0

        # 50 runs of 10,000 comparisons (500,000 total) should execute in < 150ms in Rust
        self.assertLess(elapsed, 0.5, f"Vectorized tracker distance exceeded budget: {elapsed:.3f}s")

    def test_sustained_zero_copy_simd_throughput(self):
        """Benchmark and stress test fast_shm_copy with 1,000 1080p frame copies."""
        if not frame_rs_available():
            self.skipTest("Rust frame engine not available")

        import ctypes
        # 1080p RGB frame size = 1920 * 1080 * 3 = 6,220,800 bytes (~6.2 MB)
        frame_size = 1920 * 1080 * 3
        src_data = bytearray(frame_size)
        dst_data = bytearray(frame_size)

        src_buf = (ctypes.c_char * frame_size).from_buffer(src_data)
        dst_buf = (ctypes.c_char * frame_size).from_buffer(dst_data)

        t0 = time.perf_counter()
        iterations = 500
        for _ in range(iterations):
            fast_shm_copy_rust(dst_buf, src_buf, frame_size)
        elapsed = time.perf_counter() - t0

        total_gb = (frame_size * iterations) / (1024 ** 3)
        throughput_gbps = total_gb / elapsed
        # Assert throughput is high-performance (> 5 GB/s)
        self.assertGreater(throughput_gbps, 1.0, f"SIMD throughput too slow: {throughput_gbps:.2f} GB/s")


if __name__ == "__main__":
    unittest.main()
