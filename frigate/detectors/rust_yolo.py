"""ctypes bindings for the SIMD-accelerated YOLO post-processor.

Replaces ``__post_process_multipart_yolo`` and custom NMS with vectorized
Rust implementations.  Library at ``/opt/frigate/libfrigate_yolo_rs.so``.
"""

import ctypes
import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_LIB_NAME = "libfrigate_yolo_rs.so"
_lib: Optional[ctypes.CDLL] = None
_available: Optional[bool] = None


class Detection(ctypes.Structure):
    _fields_ = [
        ("class_id", ctypes.c_int32),
        ("score", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("x1", ctypes.c_float),
        ("y2", ctypes.c_float),
        ("x2", ctypes.c_float),
    ]


def _load_lib() -> Optional[ctypes.CDLL]:
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

    # Build array of output pointers
    output_ptrs = (ctypes.POINTER(ctypes.c_float) * 3)()
    ny_nx = (ctypes.c_uint32 * 6)()

    for i, out in enumerate(outputs):
        out = np.ascontiguousarray(out.ravel().astype(np.float32))
        output_ptrs[i] = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        # Infer ny, nx from shape: (bs, ny, nx, ch) for each scale
        ny_nx[i * 2] = out.shape[-3] if out.ndim >= 3 else 0
        # Shape is actually (1, 255, ny, nx) → we need ny, nx
        if out.ndim >= 4:
            ny_nx[i * 2] = out.shape[2]
            ny_nx[i * 2 + 1] = out.shape[3]
        elif out.ndim >= 3:
            ny_nx[i * 2] = out.shape[1]
            ny_nx[i * 2 + 1] = out.shape[2]

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

    boxes = np.ascontiguousarray(boxes.ravel().astype(np.float32))
    scores = np.ascontiguousarray(scores.ravel().astype(np.float32))
    n = len(scores)
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

    raw = np.ascontiguousarray(raw.T.ravel().astype(np.float32))
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

    raw = np.ascontiguousarray(raw.ravel().astype(np.float32))
    pts = np.ascontiguousarray(all_pts.ravel().astype(np.float32))
    out = np.zeros((20, 6), dtype=np.float32)

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
        raw.size // 144,
        pts.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_float(model_size),
        ctypes.c_float(frame_w),
        ctypes.c_float(frame_h),
        ctypes.c_float(score_thresh),
        ctypes.c_float(nms_thresh),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out
