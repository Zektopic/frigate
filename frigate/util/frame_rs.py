"""ctypes bindings for the SIMD-accelerated frame utility engine.

Exposes high-performance frame processing and directly reads video frames
from FFmpeg pipes into shared memory buffers.
"""

import ctypes
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_LIB_NAME = "libfrigate_frame_rs.so"
_lib: Optional[ctypes.CDLL] = None
_available: Optional[bool] = None


def _load_lib() -> Optional[ctypes.CDLL]:
    global _lib, _available
    if _available is False:
        return None
    if _lib is not None:
        return _lib
    search = [
        "/opt/frigate",
        os.path.dirname(__file__),
        os.path.join(os.path.dirname(__file__), ".."),
        os.path.join(os.path.dirname(__file__), "..", ".."),
    ]
    for d in search:
        candidate = os.path.join(d, _LIB_NAME)
        if os.path.isfile(candidate):
            try:
                _lib = ctypes.CDLL(candidate)
                _available = True
                logger.debug("Loaded Rust frame engine from %s", candidate)
                return _lib
            except OSError as exc:
                logger.debug("Failed to load %s: %s", candidate, exc)
    _available = False
    return None


def frame_rs_available() -> bool:
    return _load_lib() is not None


def read_ffmpeg_frame_to_ptr(fd: int, ptr_addr: int, frame_size: int) -> int:
    """Reads exactly `frame_size` bytes from FFmpeg raw stdout descriptor `fd`
    directly into memory location `ptr_addr`.

    Returns 1 on success, 0 on EOF, and -1 on error.
    """
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    lib.read_ffmpeg_frame.argtypes = [
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    lib.read_ffmpeg_frame.restype = ctypes.c_int32
    return lib.read_ffmpeg_frame(fd, ptr_addr, frame_size)


def intersection_over_union_rust(box_a, box_b) -> float:
    """Calculate the intersection over union (IoU) of two bounding boxes in Rust.
    Each box should be a sequence of 4 numbers: [x1, y1, x2, y2].
    """
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    lib.intersection_over_union.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.intersection_over_union.restype = ctypes.c_float

    arr_a = (ctypes.c_float * 4)(*box_a)
    arr_b = (ctypes.c_float * 4)(*box_b)

    return float(lib.intersection_over_union(arr_a, arr_b))

