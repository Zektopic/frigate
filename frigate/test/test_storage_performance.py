"""SQLite storage performance benchmarks for bulk inserts."""

import os
import sqlite3
import tempfile
import time
import unittest


SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_name TEXT NOT NULL,
    start_time REAL NOT NULL,
    duration REAL NOT NULL,
    motion INTEGER DEFAULT 0,
    objects INTEGER DEFAULT 0,
    segment_size INTEGER DEFAULT 0
);
"""


def create_recording_row(i: int) -> tuple:
    """Create a single recording row tuple.

    Args:
        i: Row sequence number.

    Returns:
        Tuple of (camera_name, start_time, duration, motion, objects, segment_size).
    """
    return (
        f"camera_{i % 4}",
        1700000000.0 + i * 10.0,
        10.0 + (i % 5),
        i % 2,
        i % 3,
        1024 + i * 16,
    )


class TestStoragePerformance(unittest.TestCase):
    """Benchmark SQLite bulk insert throughput across batch sizes."""

    DB_PATH: str = ""

    @classmethod
    def setUpClass(cls):
        """Create a temporary database file."""
        fd, cls.DB_PATH = tempfile.mkstemp(suffix=".db", prefix="frigate_perf_")
        os.close(fd)

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary database file."""
        if cls.DB_PATH and os.path.exists(cls.DB_PATH):
            os.unlink(cls.DB_PATH)

    def _benchmark_batch_size(self, batch_size: int, total: int) -> float:
        """Benchmark inserts at a given batch size.

        Args:
            batch_size: Number of rows per transaction.
            total: Total number of rows to insert.

        Returns:
            Records per second.
        """
        conn = sqlite3.connect(self.DB_PATH)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(SCHEMA)
        conn.execute("DELETE FROM recordings")

        sql = (
            "INSERT INTO recordings "
            "(camera_name, start_time, duration, motion, objects, segment_size) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )

        rows = [create_recording_row(i) for i in range(total)]

        t0 = time.perf_counter()

        for offset in range(0, total, batch_size):
            chunk = rows[offset : offset + batch_size]
            conn.executemany(sql, chunk)
            conn.commit()

        t1 = time.perf_counter()
        elapsed = t1 - t0
        records_per_sec = total / elapsed

        conn.execute("DELETE FROM recordings")
        conn.close()

        return records_per_sec

    def test_bulk_insert_performance(self):
        """Test and report SQLite bulk insert throughput at various batch sizes."""
        total = 200
        batch_sizes = [1, 10, 50, 100]

        print("\nSQLite bulk insert benchmark")
        print(f"  Total rows per run: {total}")

        for batch_size in batch_sizes:
            rps = self._benchmark_batch_size(batch_size, total)
            print(f"  batch_size={batch_size:>4d}  {rps:,.0f} records/sec")

        # Verify the largest batch is fastest (basic sanity check)
        rps_1 = self._benchmark_batch_size(1, total)
        rps_100 = self._benchmark_batch_size(100, total)

        self.assertGreater(
            rps_100,
            rps_1,
            "Batch size 100 should be faster than batch size 1",
        )

        print(f"  Sanity check: batch=100 ({rps_100:,.0f} r/s) > "
              f"batch=1 ({rps_1:,.0f} r/s)  PASS")
