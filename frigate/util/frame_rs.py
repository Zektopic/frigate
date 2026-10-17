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
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    search = [
        "/opt/frigate",
        os.path.dirname(__file__),
        os.path.join(os.path.dirname(__file__), ".."),
        repo_root,
        os.path.join(repo_root, "frigate-frame-rs", "target", "release"),
        os.path.join(repo_root, "frigate-frame-rs", "target", "debug"),
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


def read_ffmpeg_frame_to_ptr(
    fd: int, ptr_addr: int, frame_size: int, buffer_size: Optional[int] = None
) -> int:
    """Reads exactly `frame_size` bytes from FFmpeg raw stdout descriptor `fd`
    directly into memory location `ptr_addr`.

    ``buffer_size`` is the capacity of the destination.  Rust trusts
    ``frame_size`` blindly, so a destination smaller than one frame (e.g. a
    stale /dev/shm segment left over from a different detect resolution)
    would be overrun: adjacent mappings are silently corrupted, or read(2)
    fails with EFAULT.  The Python path raised a ValueError in that case, so
    this keeps the failure recoverable.

    Returns 1 on success, 0 on EOF, and -1 on error.
    """
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")
    if ptr_addr == 0 or frame_size <= 0 or frame_size > 0xFFFFFFFF:
        return -1
    if buffer_size is not None and buffer_size < frame_size:
        logger.error(
            "Frame buffer too small for frame: %d < %d bytes", buffer_size, frame_size
        )
        return -1

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

    # A ctypes array initializer rejects >4 values but silently zero-fills
    # fewer, which would turn a malformed box into a wrong answer.
    if len(box_a) != 4 or len(box_b) != 4:
        raise ValueError("boxes must have exactly 4 coordinates")
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


def polygon_box_overlap_rust(
    pts: list[tuple[float, float]], box: tuple[float, float, float, float]
) -> bool:
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

    # Rust reads exactly 4 doubles per box.  reshape() raises on anything
    # else (a 3-value box used to make Rust read past the end of the array)
    # and also accepts norfair's (2, 2) point arrays.
    dets = np.ascontiguousarray(
        np.asarray(detections, dtype=np.float64).reshape(n_dets, 4)
    )
    ests = np.ascontiguousarray(
        np.asarray(estimates, dtype=np.float64).reshape(n_ests, 4)
    )
    out = np.zeros((n_dets, n_ests), dtype=np.float64)

    lib.batch_track_distance_matrix(
        dets.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(n_dets),
        ests.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
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

    # Bound the copy by what both buffers actually hold; Rust trusts
    # `length` and would read/write past either end.
    if (
        length < 0
        or length > memoryview(dst_buf).nbytes
        or length > memoryview(src_buf).nbytes
    ):
        raise ValueError("fast_shm_copy length exceeds a buffer")
    if length == 0:
        return

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
    """Bilinear resize + /255 normalize + NHWC->NCHW in one Rust call.

    Returns a ``c_float`` array of ``channels * dst_h * dst_w`` values.
    """
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")
    if min(src_w, src_h, dst_w, dst_h, channels) <= 0:
        raise ValueError("dimensions must be positive")
    # Rust reads src_w * src_h * channels bytes from src.
    if len(src_bytes) < src_w * src_h * channels:
        raise ValueError(
            f"source holds {len(src_bytes)} bytes, need {src_w * src_h * channels}"
        )

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

    out_buf = (ctypes.c_float * (channels * dst_w * dst_h))()
    src_arr = (ctypes.c_uint8 * len(src_bytes)).from_buffer_copy(src_bytes)
    lib.preprocess_detect_input(src_arr, out_buf, src_w, src_h, dst_w, dst_h, channels)
    return out_buf


def transform_detect_input_rust(tensor_input):
    """Fused NHWC(u8) -> NCHW(f32) normalized to [0.0, 1.0] in a single pass.

    Accepts shape (1, H, W, C) or (H, W, C) uint8 and returns (1, C, H, W)
    or (C, H, W) float32.
    """
    import numpy as np

    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    has_batch = tensor_input.ndim == 4
    if has_batch:
        b, h, w, c = tensor_input.shape
        if b != 1:
            raise ValueError(f"Expected batch size 1, got {b}")
        out_shape = (1, c, h, w)
    elif tensor_input.ndim == 3:
        h, w, c = tensor_input.shape
        out_shape = (c, h, w)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tensor_input.shape}")

    src = np.ascontiguousarray(tensor_input, dtype=np.uint8)
    dst = np.empty(out_shape, dtype=np.float32)

    lib.transform_detect_input.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    lib.transform_detect_input.restype = None

    lib.transform_detect_input(
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(h),
        ctypes.c_uint32(w),
        ctypes.c_uint32(c),
    )
    return dst


def normalize_u8_to_f32_rust(tensor_input):
    """AVX2 SIMD u8 -> f32 normalization (divide by 255.0) preserving shape."""
    import numpy as np

    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    src = np.ascontiguousarray(tensor_input, dtype=np.uint8)
    dst = np.empty(src.shape, dtype=np.float32)

    lib.normalize_u8_to_f32.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
    ]
    lib.normalize_u8_to_f32.restype = None

    lib.normalize_u8_to_f32(
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(src.size),
    )
    return dst


def yuv420_to_3channel_rust(yuv_frame):
    """Convert planar I420 (H*3//2, W) uint8 to 3-channel YUV (H, W, 3) uint8."""
    import numpy as np

    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    height = yuv_frame.shape[0] // 3 * 2
    width = yuv_frame.shape[1]
    src = np.ascontiguousarray(yuv_frame, dtype=np.uint8)
    dst = np.empty((height, width, 3), dtype=np.uint8)

    y_count = height * width
    uv_count = y_count // 4
    base_ptr = src.ctypes.data

    lib.yuv420_to_3channel.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.yuv420_to_3channel.restype = None

    lib.yuv420_to_3channel(
        ctypes.cast(base_ptr, ctypes.POINTER(ctypes.c_uint8)),
        ctypes.cast(base_ptr + y_count, ctypes.POINTER(ctypes.c_uint8)),
        ctypes.cast(base_ptr + y_count + uv_count, ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_uint32(width),
        ctypes.c_uint32(height),
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    return dst


def yuv420_crop_repack_rust(
    frame,
    src_w: int,
    src_h: int,
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
):
    """Crop an in-bounds region from a planar I420 frame and repack into a contiguous I420 buffer."""
    import numpy as np

    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    src = np.ascontiguousarray(frame, dtype=np.uint8)
    dst = np.empty((crop_h + crop_h // 2, crop_w), dtype=np.uint8)

    lib.yuv420_crop_repack.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.yuv420_crop_repack.restype = None

    lib.yuv420_crop_repack(
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_uint32(src_w),
        ctypes.c_uint32(src_h),
        ctypes.c_uint32(crop_x),
        ctypes.c_uint32(crop_y),
        ctypes.c_uint32(crop_w),
        ctypes.c_uint32(crop_h),
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    return dst
