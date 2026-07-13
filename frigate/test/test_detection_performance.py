"""Performance benchmarks for object detection pre-processing pipeline."""

import time
import unittest

import numpy as np


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Simulate detection pre-processing: transpose + normalize + expand_dims.

    Args:
        frame: Input frame as numpy array in HWC format (uint8).

    Returns:
        Preprocessed tensor ready for inference.
    """
    # Transpose from HWC to CHW
    transposed = frame.transpose(2, 0, 1)
    # Normalize to [0, 1]
    normalized = transposed.astype(np.float32) / 255.0
    # Add batch dimension
    return np.expand_dims(normalized, axis=0)


class TestDetectionPerformance(unittest.TestCase):
    """Benchmark detection pre-processing speed at multiple resolutions."""

    ITERATIONS = 50

    def _benchmark_resolution(
        self, width: int, height: int, max_ms: float, label: str
    ) -> float:
        """Run pre-processing benchmark for a given resolution.

        Args:
            width: Frame width in pixels.
            height: Frame height in pixels.
            max_ms: Maximum allowed average time in milliseconds.
            label: Human-readable resolution label for output.

        Returns:
            Average time in milliseconds.
        """
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

        # Warm-up iteration
        preprocess_frame(frame)

        # Timed iterations
        times = []
        for _ in range(self.ITERATIONS):
            t0 = time.perf_counter()
            preprocess_frame(frame)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        avg_ms = sum(times) / len(times)
        min_ms = min(times)
        max_val_ms = max(times)

        print(f"\n  [{label}] {width}x{height} over {self.ITERATIONS} iterations:")
        print(f"    avg={avg_ms:.3f}ms  min={min_ms:.3f}ms  max={max_val_ms:.3f}ms")
        print(f"    limit={max_ms}ms  {'PASS' if avg_ms < max_ms else 'FAIL'}")

        self.assertLess(avg_ms, max_ms, f"{label} avg {avg_ms:.3f}ms >= {max_ms}ms")
        return avg_ms

    def test_preprocessing_300x300(self):
        """Pre-processing at 300x300 should average under 50ms."""
        print("\nDetection pre-processing benchmark")
        self._benchmark_resolution(300, 300, 50.0, "300x300")

    def test_preprocessing_640x360(self):
        """Pre-processing at 640x360 should average under 100ms."""
        self._benchmark_resolution(640, 360, 100.0, "640x360")
