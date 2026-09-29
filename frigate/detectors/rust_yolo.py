"""ctypes bindings for the SIMD-accelerated YOLO post-processor.

Replaces ``__post_process_multipart_yolo`` and custom NMS with vectorized
Rust implementations.  Library at ``/opt/frigate/libfrigate_yolo_rs.so``.
"""

import ctypes
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

_LIB_NAME = "libfrigate_yolo_rs.so"
_lib: ctypes.CDLL | None = None
_available: bool | None = None


class Detection(ctypes.Structure):
    _fields_ = [
        ("class_id", ctypes.c_int32),
        ("score", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("x1", ctypes.c_float),
        ("y2", ctypes.c_float),
        ("x2", ctypes.c_float),
    ]


def _load_lib() -> ctypes.CDLL | None:
    global _lib, _available
    if _available is False:
        return None
    if _lib is not None:
        return _lib
    for d in ["/opt/frigate", os.path.dirname(__file__)]:
        candidate = os.path.join(d, _LIB_NAME)
        if os.path.isfile(candidate):
            try:
                _lib = ctypes.CDLL(candidate)
                _available = True
                return _lib
            except OSError as exc:
                logger.debug("Failed to load %s: %s", candidate, exc)
    _available = False
    return None


def yolo_available() -> bool:
    return _load_lib() is not None


def yolo_post_process(
    outputs: list[np.ndarray],
    width: int,
    height: int,
    score_thresh: float = 0.4,
    iou_thresh: float = 0.4,
    max_dets: int = 20,
) -> np.ndarray:
    """Run grid decode + NMS on 3 YOLO output scales.

    Returns a ``(20, 6)`` float32 array in Frigate's standard format:
    ``[class_id, score, y1, x1, y2, x2]`` (normalised coordinates).
    """
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust YOLO engine not available")

    if len(outputs) != 3:
        raise ValueError(f"expected 3 YOLO output scales, got {len(outputs)}")

    # The Rust decoder reads each scale as [anchor][y][x][85] (channels
    # last) with exactly 3 * ny * nx * 85 floats.  Detectors hand us the
    # NCHW tensor (1, 255, ny, nx), so:
    #   * take ny/nx from the ORIGINAL 4-D shape (after ravel() every
    #     array is 1-D and the dims all read as 0, which made Rust skip
    #     every scale and return no detections at all);
    #   * reorder to channels-last, matching __post_process_multipart_yolo;
    #   * keep every converted array referenced in `keep_alive` until the
    #     call returns.  Storing `arr.ctypes.data_as(...)` into a ctypes
    #     pointer array copies only the address, so a temporary rebound on
    #     the next loop iteration is freed while Rust still reads it.
    keep_alive: list[np.ndarray] = []
    output_ptrs = (ctypes.POINTER(ctypes.c_float) * 3)()
    ny_nx = (ctypes.c_uint32 * 6)()

    for i, out in enumerate(outputs):
        if out.ndim != 4 or out.shape[0] != 1 or out.shape[1] != 3 * 85:
            raise ValueError(f"unexpected YOLO output shape {out.shape}")
        _, _, ny, nx = out.shape
        arr = np.ascontiguousarray(
            out.reshape(1, 3, 85, ny, nx).transpose(0, 1, 3, 4, 2),
            dtype=np.float32,
        )
        keep_alive.append(arr)
        output_ptrs[i] = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ny_nx[i * 2] = ny
        ny_nx[i * 2 + 1] = nx

    dets = (Detection * max_dets)()

    lib.yolo_post_process.argtypes = [
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(Detection),
        ctypes.c_uint32,
    ]
    lib.yolo_post_process.restype = ctypes.c_uint32

    n = lib.yolo_post_process(
        ctypes.cast(output_ptrs, ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
        ctypes.cast(ny_nx, ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_float(width),
        ctypes.c_float(height),
        ctypes.c_float(score_thresh),
        ctypes.c_float(iou_thresh),
        ctypes.cast(dets, ctypes.POINTER(Detection)),
        ctypes.c_uint32(max_dets),
    )

    # keep_alive must outlive the FFI call above.
    del keep_alive

    result = np.zeros((20, 6), dtype=np.float32)
    for i in range(min(n, 20)):
        result[i] = [
            dets[i].class_id,
            dets[i].score,
            dets[i].y1,
            dets[i].x1,
            dets[i].y2,
            dets[i].x2,
        ]
    return result


def nms_boxes(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.4,
    max_indices: int = 100,
) -> np.ndarray:
    """Greedy NMS on pre-decoded boxes. Returns array of kept indices."""
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust YOLO engine not available")

    boxes = np.ascontiguousarray(boxes, dtype=np.float32).ravel()
    scores = np.ascontiguousarray(scores, dtype=np.float32).ravel()
    n = len(scores)
    # Rust reads n * 4 box coordinates; a shorter box array is a heap
    # over-read.
    if boxes.size != n * 4:
        raise ValueError(f"expected {n * 4} box coordinates, got {boxes.size}")
    if n == 0 or max_indices <= 0:
        return np.zeros(0, dtype=np.int32)
    out_indices = (ctypes.c_uint32 * max_indices)()

    lib.nms_boxes.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32,
    ]
    lib.nms_boxes.restype = ctypes.c_uint32

    kept = lib.nms_boxes(
        boxes.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n,
        iou_threshold,
        ctypes.cast(out_indices, ctypes.POINTER(ctypes.c_uint32)),
        max_indices,
    )

    return np.array(out_indices[:kept], dtype=np.int32)


def yolo26_post_process(
    raw: "np.ndarray",
    model_size: int = 640,
    frame_w: float = 1.0,
    frame_h: float = 1.0,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.45,
) -> "np.ndarray":
    """YOLO26 decoded-output post-process: bbox convert + NMS in Rust.

    Args:
        raw: (84, N) float32 — rows 0-3 are cx,cy,w,h, rows 4-83 scores.
    Returns: (20, 6) float32 [class_id, score, y1, x1, y2, x2] normalised.
    """
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust YOLO engine not available")
    import numpy as np

    if raw.ndim != 2 or raw.shape[0] != 84:
        raise ValueError(f"expected a (84, N) YOLO26 tensor, got {raw.shape}")
    raw = np.ascontiguousarray(raw.T, dtype=np.float32).ravel()
    n = raw.size // 84
    out = np.zeros((20, 6), dtype=np.float32)

    lib.yolo26_post_process.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.yolo26_post_process.restype = None

    lib.yolo26_post_process(
        raw.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n,
        ctypes.c_float(model_size),
        ctypes.c_float(frame_w),
        ctypes.c_float(frame_h),
        ctypes.c_float(score_thresh),
        ctypes.c_float(nms_thresh),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out


def anchor_free_post_process(
    raw: "np.ndarray",
    all_pts: "np.ndarray",
    model_size: int = 640,
    frame_w: int = 640,
    frame_h: int = 640,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.45,
) -> "np.ndarray":
    """YOLOv8/YOLO11 anchor-free post-process: sigmoid + DFL + grid decode + NMS."""
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust YOLO engine not available")
    import numpy as np

    raw = np.ascontiguousarray(raw, dtype=np.float32).ravel()
    pts = np.ascontiguousarray(all_pts, dtype=np.float32).ravel()
    out = np.zeros((20, 6), dtype=np.float32)
    # Rust reads (raw.size // 144) cells and two grid coordinates per
    # cell; anything shorter is an out-of-bounds read.
    n_cells = raw.size // 144
    if raw.size != n_cells * 144 or pts.size < n_cells * 2:
        raise ValueError(
            f"anchor-free tensors mismatch: raw={raw.size} floats, pts={pts.size}"
        )

    lib.yolo_anchor_free_post_process.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.yolo_anchor_free_post_process.restype = ctypes.c_uint32

    lib.yolo_anchor_free_post_process(
        raw.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n_cells,
        pts.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(model_size),
        ctypes.c_float(frame_w),
        ctypes.c_float(frame_h),
        ctypes.c_float(score_thresh),
        ctypes.c_float(nms_thresh),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out
