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


def track_distance_rust(detection, estimate) -> float:
    """Norfair association distance between two boxes in Rust.

    Each argument is a sequence of 4 numbers [x1, y1, x2, y2] (flattened
    2x2 norfair points). Returns +inf for degenerate/non-finite boxes,
    matching frigate.track.norfair_tracker.distance.
    """
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    lib.track_distance.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.track_distance.restype = ctypes.c_double

    det = (ctypes.c_double * 4)(*detection)
    est = (ctypes.c_double * 4)(*estimate)

    return float(lib.track_distance(det, est))


def point_in_polygon_rust(px: float, py: float, poly_points: list[float]) -> bool:
    """Ray-casting point in polygon test in Rust."""
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    num_points = len(poly_points) // 2
    arr = (ctypes.c_float * len(poly_points))(*poly_points)

    lib.point_in_polygon.argtypes = [
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
    ]
    lib.point_in_polygon.restype = ctypes.c_int32

    return bool(lib.point_in_polygon(ctypes.c_float(px), ctypes.c_float(py), arr, num_points))


def polygon_box_overlap_rust(
    box: list[float],
    poly_points: list[float],
    grid_samples: int = 4,
) -> float:
    """Computes bounding box polygon overlap fraction (0.0 to 1.0) in Rust."""
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    num_points = len(poly_points) // 2
    box_arr = (ctypes.c_float * 4)(*box)
    poly_arr = (ctypes.c_float * len(poly_points))(*poly_points)

    lib.polygon_box_overlap.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    lib.polygon_box_overlap.restype = ctypes.c_float

    return float(
        lib.polygon_box_overlap(
            box_arr,
            poly_arr,
            ctypes.c_uint32(num_points),
            ctypes.c_uint32(grid_samples),
        )
    )

