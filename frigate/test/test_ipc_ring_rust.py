import ctypes
import unittest

from frigate.util.frame_rs import (
    frame_rs_available,
    fast_shm_copy_rust,
)


class TestIpcRingRust(unittest.TestCase):
    def test_fast_shm_copy_rust(self):
        """Test zero-copy SIMD memory copy into shared memory buffers."""
        if not frame_rs_available():
            self.skipTest("Rust frame engine not available")

        # Create 1MB test frame
        size = 1024 * 1024
        src_data = bytearray(i % 256 for i in range(size))
        dst_data = bytearray(size)

        src_buf = (ctypes.c_char * size).from_buffer(src_data)
        dst_buf = (ctypes.c_char * size).from_buffer(dst_data)

        fast_shm_copy_rust(dst_buf, src_buf, size)

        self.assertEqual(src_data, dst_data)


if __name__ == "__main__":
    unittest.main()
