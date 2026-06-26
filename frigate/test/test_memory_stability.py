"""Memory stability tests for frame processing and shared memory lifecycle."""

import gc
import unittest
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import psutil


class TestMemoryStability(unittest.TestCase):
    """Verify no memory leaks during repeated frame processing and SHM operations."""

    def _get_rss_mb(self) -> float:
        """Get current process RSS in megabytes.

        Returns:
            Resident set size in MB.
        """
        return psutil.Process().memory_info().rss / (1024 * 1024)

    def test_simulated_frame_processing_no_leak(self):
        """Run 100 iterations of simulated frame processing, assert RSS growth < 10%."""
        initial_mb = self._get_rss_mb()
        print(f"\n  Initial RSS: {initial_mb:.1f} MB")

        for i in range(100):
            # Simulate frame allocation and processing
            frame = np.random.randint(0, 256, (640, 360, 3), dtype=np.uint8)
            transposed = frame.transpose(2, 0, 1).astype(np.float32) / 255.0
            result = np.expand_dims(transposed, axis=0)

            # Simulate detection output
            detections = np.random.random((10, 6)).astype(np.float32)

            # Release explicitly
            del frame, transposed, result, detections

            # Collect garbage every 20 iterations
            if (i + 1) % 20 == 0:
                gc.collect()

        gc.collect()
        final_mb = self._get_rss_mb()
        growth_pct = ((final_mb - initial_mb) / initial_mb) * 100

        print(f"  Final RSS:   {final_mb:.1f} MB")
        print(f"  Growth:      {growth_pct:+.1f}%")
        print(f"  Limit:       < 10%  {'PASS' if growth_pct < 10 else 'FAIL'}")

        self.assertLess(
            growth_pct,
            10.0,
            f"Memory grew {growth_pct:.1f}% (limit 10%), from "
            f"{initial_mb:.1f}MB to {final_mb:.1f}MB",
        )

    def test_shared_memory_create_close_unlink_lifecycle(self):
        """Test SharedMemory create/close/unlink lifecycle without leaks."""
        initial_mb = self._get_rss_mb()
        print(f"\n  Initial RSS: {initial_mb:.1f} MB")

        num_cycles = 50
        for i in range(num_cycles):
            data = np.zeros((100, 100), dtype=np.uint8)
            shm = SharedMemory(create=True, size=data.nbytes)
            shm.close()
            shm.unlink()
            del data, shm

            if (i + 1) % 10 == 0:
                gc.collect()

        gc.collect()
        final_mb = self._get_rss_mb()
        growth_pct = ((final_mb - initial_mb) / initial_mb) * 100

        print(f"  SHM cycles:  {num_cycles}")
        print(f"  Final RSS:   {final_mb:.1f} MB")
        print(f"  Growth:      {growth_pct:+.1f}%")
        print(f"  Limit:       < 10%  {'PASS' if growth_pct < 10 else 'FAIL'}")

        self.assertLess(
            growth_pct,
            10.0,
            f"Memory grew {growth_pct:.1f}% after {num_cycles} SHM cycles",
        )
