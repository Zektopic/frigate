"""ctypes bindings for the SIMD-accelerated frame utility engine.

Exposes high-performance frame processing and directly reads video frames
from FFmpeg pipes into shared memory buffers.
"""

import ctypes
import logging
import os

logger = logging.getLogger(__name__)

_LIB_NAME = "libfrigate_frame_rs.so"
_lib: ctypes.CDLL | None = None
_available: bool | None = None


def _load_lib() -> ctypes.CDLL | None:
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


def point_in_polygon_rust(px: float, py: float, pts: list[tuple[float, float]]) -> bool:
    """Ray-casting point in polygon test in Rust."""
    if len(pts) < 3:
        return False
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    lib.point_in_polygon.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
    ]
    lib.point_in_polygon.restype = ctypes.c_int32

    flat_pts = []
    for x, y in pts:
        flat_pts.extend([float(x), float(y)])
    arr = (ctypes.c_double * len(flat_pts))(*flat_pts)
    return bool(lib.point_in_polygon(px, py, arr, len(pts)))


def polygon_box_overlap_rust(pts: list[tuple[float, float]], box: tuple[float, float, float, float]) -> bool:
    """Check if bounding box [x1, y1, x2, y2] overlaps with polygon in Rust."""
    if len(pts) < 3:
        return False
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    lib.polygon_box_overlap.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.polygon_box_overlap.restype = ctypes.c_int32

    flat_pts = []
    for x, y in pts:
        flat_pts.extend([float(x), float(y)])
    arr_pts = (ctypes.c_double * len(flat_pts))(*flat_pts)
    arr_box = (ctypes.c_double * 4)(*box)
    return bool(lib.polygon_box_overlap(arr_pts, len(pts), arr_box))


def batch_track_distance_matrix_rust(detections: list, estimates: list):
    """Vectorized NxM pairwise tracker distance matrix in Rust."""
    import numpy as np
    n_dets = len(detections)
    n_ests = len(estimates)
    if n_dets == 0 or n_ests == 0:
        return np.zeros((n_dets, n_ests), dtype=np.float64)

    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    lib.batch_track_distance_matrix.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.batch_track_distance_matrix.restype = None

    flat_dets = [float(v) for b in detections for v in b]
    flat_ests = [float(v) for b in estimates for v in b]

    c_dets = (ctypes.c_double * len(flat_dets))(*flat_dets)
    c_ests = (ctypes.c_double * len(flat_ests))(*flat_ests)
    out = np.zeros((n_dets, n_ests), dtype=np.float64)

    lib.batch_track_distance_matrix(
        c_dets,
        ctypes.c_size_t(n_dets),
        c_ests,
        ctypes.c_size_t(n_ests),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    return out


def fast_shm_copy_rust(dst_buf, src_buf, length: int) -> None:
    """Zero-copy SIMD memory copy for shared memory frame transfers."""
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    lib.fast_shm_copy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.fast_shm_copy.restype = None

    dst_ptr = ctypes.addressof(ctypes.c_char.from_buffer(dst_buf))
    src_ptr = ctypes.addressof(ctypes.c_char.from_buffer(src_buf))

    lib.fast_shm_copy(dst_ptr, src_ptr, ctypes.c_size_t(length))

def preprocess_detect_input_rust(
    src_bytes: bytes,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
    channels: int = 3,
) -> ctypes.Array:
    """Zero-copy SIMD preprocess detection input (bilinear resize + normalize + NHWC->NCHW)."""
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    out_size = channels * dst_w * dst_h
    out_buf = (ctypes.c_float * out_size)()

    lib.preprocess_detect_input.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    lib.preprocess_detect_input.restype = None

    src_arr = (ctypes.c_uint8 * len(src_bytes)).from_buffer_copy(src_bytes)

    lib.preprocess_detect_input(
        src_arr,
        out_buf,
        src_w,
        src_h,
        dst_w,
        dst_h,
        channels,
    )
    return out_buf
