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


def _configure_signatures(lib: ctypes.CDLL) -> None:
    """Configure ctypes argtypes/restype once at load time to avoid per-call setup overhead."""
    if getattr(lib, "_signatures_configured", False):
        return

    u8_p = ctypes.POINTER(ctypes.c_uint8)
    f32_p = ctypes.POINTER(ctypes.c_float)
    f64_p = ctypes.POINTER(ctypes.c_double)
    u32 = ctypes.c_uint32
    i32 = ctypes.c_int32
    f32 = ctypes.c_float
    f64 = ctypes.c_double
    sz = ctypes.c_size_t
    vp = ctypes.c_void_p

    lib.read_ffmpeg_frame.argtypes = [i32, vp, u32]
    lib.read_ffmpeg_frame.restype = i32

    lib.intersection_over_union.argtypes = [f32_p, f32_p]
    lib.intersection_over_union.restype = f32

    lib.track_distance.argtypes = [f64_p, f64_p]
    lib.track_distance.restype = f64

    lib.point_in_polygon.argtypes = [f64, f64, f64_p, sz]
    lib.point_in_polygon.restype = i32

    lib.polygon_box_overlap.argtypes = [f64_p, sz, f64_p]
    lib.polygon_box_overlap.restype = i32

    lib.batch_track_distance_matrix.argtypes = [f64_p, sz, f64_p, sz, f64_p]
    lib.batch_track_distance_matrix.restype = None

    lib.fast_shm_copy.argtypes = [vp, vp, sz]
    lib.fast_shm_copy.restype = None

    lib.preprocess_detect_input.argtypes = [u8_p, f32_p, u32, u32, u32, u32, u32]
    lib.preprocess_detect_input.restype = None

    lib.transform_detect_input.argtypes = [u8_p, f32_p, u32, u32, u32]
    lib.transform_detect_input.restype = None

    lib.normalize_u8_to_f32.argtypes = [u8_p, f32_p, u32]
    lib.normalize_u8_to_f32.restype = None

    lib.yuv420_to_3channel.argtypes = [u8_p, u8_p, u8_p, u32, u32, u8_p]
    lib.yuv420_to_3channel.restype = None

    lib.yuv420_crop_repack.argtypes = [u8_p, u32, u32, u32, u32, u32, u32, u8_p]
    lib.yuv420_crop_repack.restype = None

    lib.bilinear_resize_u8.argtypes = [u8_p, u8_p, u32, u32, u32, u32, u32]
    lib.bilinear_resize_u8.restype = None

    lib.yuv420_to_rgb.argtypes = [u8_p, u8_p, u8_p, u8_p, u32, u32]
    lib.yuv420_to_rgb.restype = None

    i32_p = ctypes.POINTER(ctypes.c_int32)

    lib.nhwc_to_nchw_f32.argtypes = [f32_p, f32_p, u32, u32, u32]
    lib.nhwc_to_nchw_f32.restype = None

    lib.intersects_any_i32.argtypes = [i32_p, i32_p, sz]
    lib.intersects_any_i32.restype = i32

    lib.inside_any_i32.argtypes = [i32_p, i32_p, sz]
    lib.inside_any_i32.restype = i32

    lib.reduce_boxes_i32.argtypes = [i32_p, sz, f32, i32_p]
    lib.reduce_boxes_i32.restype = sz

    lib.cluster_candidates_i32.argtypes = [i32_p, sz, i32, i32, i32, i32_p, i32_p]
    lib.cluster_candidates_i32.restype = sz

    lib.consolidate_detections_i32.argtypes = [i32_p, sz, f32, i32_p]
    lib.consolidate_detections_i32.restype = sz

    lib.yuv420_region_to_rgb_resize.argtypes = [
        u8_p,
        u32,
        u32,
        i32,
        i32,
        i32,
        i32,
        u32,
        u32,
        i32,
        u8_p,
    ]
    lib.yuv420_region_to_rgb_resize.restype = None

    lib.ncc_u8.argtypes = [u8_p, u8_p, sz]
    lib.ncc_u8.restype = f64

    lib.average_and_median_boxes_i32.argtypes = [i32_p, sz, f64_p, i32_p]
    lib.average_and_median_boxes_i32.restype = None

    lib.percentile_15_85_i32.argtypes = [i32_p, sz, f64_p]
    lib.percentile_15_85_i32.restype = None

    lib.laplacian_variance_u8.argtypes = [u8_p, u32, u32, u32]
    lib.laplacian_variance_u8.restype = f64

    lib.batch_cosine_similarity_f32.argtypes = [f32_p, f32_p, sz, sz, f32_p]
    lib.batch_cosine_similarity_f32.restype = None

    lib._signatures_configured = True


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
                _configure_signatures(_lib)
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

    _configure_signatures(lib)
    return lib.read_ffmpeg_frame(fd, ptr_addr, frame_size)


def intersection_over_union_rust(box_a, box_b) -> float:
    """Calculate the intersection over union (IoU) of two bounding boxes in Rust.
    Each box should be a sequence of 4 numbers: [x1, y1, x2, y2].
    """
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    _configure_signatures(lib)

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

    _configure_signatures(lib)

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

    _configure_signatures(lib)

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

    _configure_signatures(lib)

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

    _configure_signatures(lib)

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

    _configure_signatures(lib)

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

    _configure_signatures(lib)

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

    _configure_signatures(lib)

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

    _configure_signatures(lib)

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

    if yuv_frame.ndim != 2:
        raise ValueError(f"Expected planar I420 (H*3//2, W), got {yuv_frame.shape}")

    height = yuv_frame.shape[0] // 3 * 2
    width = yuv_frame.shape[1]
    # Rust silently returns on odd or zero sizes, which would hand back the
    # uninitialized np.empty buffer as if it were a converted frame.
    if height <= 0 or width <= 0 or height % 2 != 0 or width % 2 != 0:
        raise ValueError(
            f"YUV420 dimensions must be positive and even: {width}x{height}"
        )

    src = np.ascontiguousarray(yuv_frame, dtype=np.uint8)
    dst = np.empty((height, width, 3), dtype=np.uint8)

    y_count = height * width
    uv_count = y_count // 4
    base_ptr = src.ctypes.data

    _configure_signatures(lib)

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

    if min(src_w, src_h, crop_w, crop_h) <= 0 or crop_x < 0 or crop_y < 0:
        raise ValueError("crop and source dimensions must be positive")
    if crop_x + crop_w > src_w or crop_y + crop_h > src_h:
        raise ValueError("crop window must lie inside the source frame")

    src = np.ascontiguousarray(frame, dtype=np.uint8)
    # Rust reads a full I420 frame of src_w x src_h from `src`; a frame
    # smaller than the dimensions claimed is an out-of-bounds read.
    needed = src_w * src_h + 2 * ((src_w // 2) * (src_h // 2))
    if src.size < needed:
        raise ValueError(f"frame holds {src.size} bytes, need {needed}")

    dst = np.empty((crop_h + crop_h // 2, crop_w), dtype=np.uint8)

    _configure_signatures(lib)

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


def bilinear_resize_u8_rust(frame, dst_w: int, dst_h: int):
    """Bilinear resize of a 2D (H, W) or 3D (H, W, C) uint8 image in Rust."""
    import numpy as np

    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    if dst_w <= 0 or dst_h <= 0:
        raise ValueError("destination dimensions must be positive")

    if frame.ndim == 2:
        src_h, src_w = frame.shape
        channels = 1
        out_shape = (dst_h, dst_w)
    elif frame.ndim == 3:
        src_h, src_w, channels = frame.shape
        out_shape = (dst_h, dst_w, channels)
    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {frame.shape}")

    if min(src_w, src_h, channels) <= 0:
        raise ValueError("source dimensions must be positive")

    src = np.ascontiguousarray(frame, dtype=np.uint8)
    dst = np.empty(out_shape, dtype=np.uint8)

    _configure_signatures(lib)

    lib.bilinear_resize_u8(
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_uint32(src_w),
        ctypes.c_uint32(src_h),
        ctypes.c_uint32(dst_w),
        ctypes.c_uint32(dst_h),
        ctypes.c_uint32(channels),
    )
    return dst


def yuv420_to_rgb_rust(yuv_frame):
    """Convert planar I420 (H*3//2, W) uint8 to interleaved RGB (H, W, 3) uint8 in Rust."""
    import numpy as np

    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    if yuv_frame.ndim != 2 or yuv_frame.shape[0] % 3 != 0:
        raise ValueError(f"Expected planar I420 (H*3//2, W), got {yuv_frame.shape}")

    height = (yuv_frame.shape[0] * 2) // 3
    width = yuv_frame.shape[1]
    if height <= 0 or width <= 0 or height % 2 != 0 or width % 2 != 0:
        raise ValueError(
            f"YUV420 dimensions must be positive and even: {width}x{height}"
        )

    src = np.ascontiguousarray(yuv_frame, dtype=np.uint8)
    dst = np.empty((height, width, 3), dtype=np.uint8)

    y_count = height * width
    uv_count = y_count // 4
    base_ptr = src.ctypes.data

    _configure_signatures(lib)

    lib.yuv420_to_rgb(
        ctypes.cast(base_ptr, ctypes.POINTER(ctypes.c_uint8)),
        ctypes.cast(base_ptr + y_count, ctypes.POINTER(ctypes.c_uint8)),
        ctypes.cast(base_ptr + y_count + uv_count, ctypes.POINTER(ctypes.c_uint8)),
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_uint32(width),
        ctypes.c_uint32(height),
    )
    return dst


def nhwc_to_nchw_f32_rust(tensor_input):
    """Transpose an NHWC (H, W, C) or (1, H, W, C) float32 tensor to NCHW in Rust."""
    import numpy as np

    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    if tensor_input.ndim == 4:
        b, h, w, c = tensor_input.shape
        if b != 1:
            raise ValueError(f"Expected batch size 1, got {b}")
        out_shape = (1, c, h, w)
    elif tensor_input.ndim == 3:
        h, w, c = tensor_input.shape
        out_shape = (c, h, w)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tensor_input.shape}")

    if min(h, w, c) <= 0:
        raise ValueError("tensor dimensions must be positive")

    src = np.ascontiguousarray(tensor_input, dtype=np.float32)
    dst = np.empty(out_shape, dtype=np.float32)

    _configure_signatures(lib)

    lib.nhwc_to_nchw_f32(
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_uint32(h),
        ctypes.c_uint32(w),
        ctypes.c_uint32(c),
    )
    return dst


def intersects_any_rust(box_a, boxes) -> bool:
    """Return True if `box_a` [x1, y1, x2, y2] intersects any box in `boxes`."""
    from itertools import chain

    n = len(boxes)
    if n == 0:
        return False
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")
    _configure_signatures(lib)

    arr_a = (ctypes.c_int32 * 4)(
        int(box_a[0]), int(box_a[1]), int(box_a[2]), int(box_a[3])
    )
    arr_b = (ctypes.c_int32 * (n * 4))(*(int(v) for v in chain.from_iterable(boxes)))
    return bool(lib.intersects_any_i32(arr_a, arr_b, ctypes.c_size_t(n)))


def inside_any_rust(box_a, boxes) -> bool:
    """Return True if `box_a` [x1, y1, x2, y2] is inside any box in `boxes`."""
    from itertools import chain

    n = len(boxes)
    if n == 0:
        return False
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")
    _configure_signatures(lib)

    arr_a = (ctypes.c_int32 * 4)(
        int(box_a[0]), int(box_a[1]), int(box_a[2]), int(box_a[3])
    )
    arr_b = (ctypes.c_int32 * (n * 4))(*(int(v) for v in chain.from_iterable(boxes)))
    return bool(lib.inside_any_i32(arr_a, arr_b, ctypes.c_size_t(n)))


def reduce_boxes_rust(boxes, iou_threshold: float = 0.0) -> list[list[int]]:
    """Greedy iterative bounding-box union reduction in Rust."""
    from itertools import chain

    n = len(boxes)
    if n == 0:
        return []
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")
    _configure_signatures(lib)

    arr_in = (ctypes.c_int32 * (n * 4))(*(int(v) for v in chain.from_iterable(boxes)))
    arr_out = (ctypes.c_int32 * (n * 4))()
    k = int(
        lib.reduce_boxes_i32(
            arr_in, ctypes.c_size_t(n), ctypes.c_float(iou_threshold), arr_out
        )
    )
    return [
        [
            int(arr_out[i * 4]),
            int(arr_out[i * 4 + 1]),
            int(arr_out[i * 4 + 2]),
            int(arr_out[i * 4 + 3]),
        ]
        for i in range(k)
    ]


def cluster_candidates_rust(
    frame_shape: tuple[int, int], min_region: int, boxes
) -> list[list[int]]:
    """Partition candidate boxes into connected detection clusters in Rust."""
    from itertools import chain

    n = len(boxes)
    if n == 0:
        return []
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")
    _configure_signatures(lib)

    arr_in = (ctypes.c_int32 * (n * 4))(*(int(v) for v in chain.from_iterable(boxes)))
    out_flat = (ctypes.c_int32 * n)()
    out_lens = (ctypes.c_int32 * n)()
    num_clusters = int(
        lib.cluster_candidates_i32(
            arr_in,
            ctypes.c_size_t(n),
            ctypes.c_int32(int(frame_shape[0])),
            ctypes.c_int32(int(frame_shape[1])),
            ctypes.c_int32(int(min_region)),
            out_flat,
            out_lens,
        )
    )
    clusters: list[list[int]] = []
    pos = 0
    for i in range(num_clusters):
        length = int(out_lens[i])
        clusters.append([int(out_flat[pos + j]) for j in range(length)])
        pos += length
    return clusters


def consolidate_detections_rust(boxes, threshold: float = 0.7) -> list[int]:
    """Return kept indices for `boxes` (sorted by area ascending) in Rust."""
    from itertools import chain

    n = len(boxes)
    if n == 0:
        return []
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")
    _configure_signatures(lib)

    arr_in = (ctypes.c_int32 * (n * 4))(*(int(v) for v in chain.from_iterable(boxes)))
    out_keep = (ctypes.c_int32 * n)()
    k = int(
        lib.consolidate_detections_i32(
            arr_in, ctypes.c_size_t(n), ctypes.c_float(threshold), out_keep
        )
    )
    return [int(out_keep[i]) for i in range(k)]


def yuv420_region_to_rgb_resize_rust(
    yuv_frame,
    region: tuple[int, int, int, int],
    dst_w: int,
    dst_h: int,
    is_bgr: bool = False,
):
    """Fused YUV420 region crop + out-of-bounds black pad + RGB/BGR conversion + bilinear resize.

    Returns a (1, dst_h, dst_w, 3) uint8 numpy array ready for `create_tensor_input`.
    """
    import numpy as np

    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    if yuv_frame.ndim != 2:
        raise ValueError(f"Expected planar I420 (H*3//2, W), got {yuv_frame.shape}")
    if dst_w <= 0 or dst_h <= 0:
        raise ValueError("destination dimensions must be positive")

    src_h = (yuv_frame.shape[0] * 2) // 3
    src_w = yuv_frame.shape[1]
    # Rust returns early (leaving `dst` uninitialized) on odd source sizes and
    # degenerate regions, so reject those here and let the caller fall back.
    if src_h <= 0 or src_w <= 0 or src_h % 2 != 0 or src_w % 2 != 0:
        raise ValueError(
            f"YUV420 dimensions must be positive and even: {src_w}x{src_h}"
        )
    if (int(region[3]) - int(region[1])) // 4 * 4 <= 0:
        raise ValueError(f"region {region} is too small")

    src = np.ascontiguousarray(yuv_frame, dtype=np.uint8)
    dst = np.empty((1, dst_h, dst_w, 3), dtype=np.uint8)

    _configure_signatures(lib)

    lib.yuv420_region_to_rgb_resize(
        src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_uint32(src_w),
        ctypes.c_uint32(src_h),
        ctypes.c_int32(int(region[0])),
        ctypes.c_int32(int(region[1])),
        ctypes.c_int32(int(region[2])),
        ctypes.c_int32(int(region[3])),
        ctypes.c_uint32(dst_w),
        ctypes.c_uint32(dst_h),
        ctypes.c_int32(1 if is_bgr else 0),
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    return dst


def ncc_u8_rust(patch_a, patch_b) -> float:
    """Zero-mean Normalized Cross-Correlation (`cv2.TM_CCOEFF_NORMED`) in Rust."""
    import numpy as np

    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    a = np.ascontiguousarray(patch_a, dtype=np.uint8)
    b = np.ascontiguousarray(patch_b, dtype=np.uint8)
    if a.size != b.size or a.size == 0:
        return 0.0

    _configure_signatures(lib)
    return float(
        lib.ncc_u8(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_size_t(a.size),
        )
    )


def average_boxes_rust(boxes) -> list[float]:
    """Compute coordinate-wise float mean of `boxes` in Rust."""
    from itertools import chain

    n = len(boxes)
    if n == 0:
        raise ValueError("boxes must not be empty")
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")
    _configure_signatures(lib)

    arr_in = (ctypes.c_int32 * (n * 4))(*(int(v) for v in chain.from_iterable(boxes)))
    out_avg = (ctypes.c_double * 4)()
    lib.average_and_median_boxes_i32(
        arr_in, ctypes.c_size_t(n), out_avg, ctypes.POINTER(ctypes.c_int32)()
    )
    return [float(out_avg[0]), float(out_avg[1]), float(out_avg[2]), float(out_avg[3])]


def median_of_boxes_rust(boxes) -> list[int]:
    """Return the box with the median area in `boxes` in Rust."""
    from itertools import chain

    n = len(boxes)
    if n == 0:
        raise ValueError("boxes must not be empty")
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")
    _configure_signatures(lib)

    arr_in = (ctypes.c_int32 * (n * 4))(*(int(v) for v in chain.from_iterable(boxes)))
    out_med = (ctypes.c_int32 * 4)()
    lib.average_and_median_boxes_i32(
        arr_in, ctypes.c_size_t(n), ctypes.POINTER(ctypes.c_double)(), out_med
    )
    return [int(out_med[0]), int(out_med[1]), int(out_med[2]), int(out_med[3])]


def percentile_15_85_rust(
    xmins, ymins, xmaxs, ymaxs
) -> tuple[float, float, float, float]:
    """Compute (p15_xmin, p15_ymin, p85_xmax, p85_ymax) with linear interpolation in Rust."""
    from itertools import chain

    n = len(xmins)
    if n == 0:
        raise ValueError("coordinate lists must not be empty")
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")
    _configure_signatures(lib)

    arr_in = (ctypes.c_int32 * (n * 4))(
        *(int(v) for v in chain.from_iterable(zip(xmins, ymins, xmaxs, ymaxs)))
    )
    out_pct = (ctypes.c_double * 4)()
    lib.percentile_15_85_i32(arr_in, ctypes.c_size_t(n), out_pct)
    return (float(out_pct[0]), float(out_pct[1]), float(out_pct[2]), float(out_pct[3]))


def laplacian_variance_u8_rust(image) -> float:
    """Streaming single-pass 3x3 Laplacian variance (`cv2.Laplacian(image, cv2.CV_64F).var()`) in Rust."""
    import numpy as np

    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    if image.ndim == 2:
        h, w = image.shape
        c = 1
    elif image.ndim == 3:
        h, w, c = image.shape
    else:
        raise ValueError(f"Expected 2D or 3D image, got {image.shape}")

    src = np.ascontiguousarray(image, dtype=np.uint8)
    _configure_signatures(lib)
    return float(
        lib.laplacian_variance_u8(
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_uint32(w),
            ctypes.c_uint32(h),
            ctypes.c_uint32(c),
        )
    )


def batch_cosine_similarity_f32_rust(query, gallery):
    """Compute cosine similarity between `query` (dim,) and `gallery` (N, dim) in Rust."""
    import numpy as np

    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust frame engine not available")

    q = np.ascontiguousarray(query, dtype=np.float32).ravel()
    dim = q.size
    g = np.ascontiguousarray(gallery, dtype=np.float32).reshape(-1, dim)
    n_gallery = g.shape[0]
    out = np.zeros(n_gallery, dtype=np.float32)

    _configure_signatures(lib)
    lib.batch_cosine_similarity_f32(
        q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        g.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(n_gallery),
        ctypes.c_size_t(dim),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out
