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


def _configure(lib: ctypes.CDLL) -> None:
    """Declare every prototype once, at load time.

    Assigning ``argtypes`` on every call is wasted work per inference and
    races with concurrent callers of the same function object.
    """
    f32p = ctypes.POINTER(ctypes.c_float)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    f32 = ctypes.c_float
    u32 = ctypes.c_uint32
    lib.yolo_post_process.argtypes = [
        ctypes.POINTER(f32p), u32p, f32, f32, f32, f32, ctypes.POINTER(Detection), u32,
    ]  # fmt: skip
    lib.yolo_post_process.restype = u32
    lib.nms_boxes.argtypes = [f32p, f32p, u32, f32, u32p, u32]
    lib.nms_boxes.restype = u32
    lib.yolo26_post_process.argtypes = [f32p, u32, f32, f32, f32, f32, f32, f32p]
    lib.yolo26_post_process.restype = None
    lib.yolo_anchor_free_post_process.argtypes = [
        f32p, u32, f32p, f32, f32, f32, f32, f32, f32p,
    ]  # fmt: skip
    lib.yolo_anchor_free_post_process.restype = u32


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
        os.path.join(repo_root, "frigate-yolo-rs", "target", "release"),
        os.path.join(repo_root, "frigate-yolo-rs", "target", "debug"),
    ]
    for d in search:
        candidate = os.path.join(d, _LIB_NAME)
        if os.path.isfile(candidate):
            try:
                lib = ctypes.CDLL(candidate)
                _configure(lib)
            except (OSError, AttributeError) as exc:
                logger.debug("Failed to load %s: %s", candidate, exc)
                continue
            _lib = lib
            _available = True
            return _lib
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

    # Build array of output pointers; keep buffers alive across the FFI call.
    # The Rust decoder reads each scale directly in NCHW (1, 3*85, ny, nx) order.
    output_ptrs = (ctypes.POINTER(ctypes.c_float) * 3)()
    ny_nx = (ctypes.c_uint32 * 6)()
    flat_outputs: list[np.ndarray] = []

    for i, out in enumerate(outputs):
        if out.ndim == 4 and out.shape[0] == 1 and out.shape[1] == 3 * 85:
            ny, nx = out.shape[2], out.shape[3]
        elif out.ndim == 3 and out.shape[0] == 3 * 85:
            ny, nx = out.shape[1], out.shape[2]
        else:
            raise ValueError(f"unexpected YOLO output shape {out.shape}")
        ny_nx[i * 2] = ny
        ny_nx[i * 2 + 1] = nx
        flat = np.ascontiguousarray(out, dtype=np.float32).ravel()
        flat_outputs.append(flat)
        output_ptrs[i] = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    dets = (Detection * max_dets)()

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

    # flat_outputs must outlive the FFI call above.
    del flat_outputs

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
        raw: (84, N), (N, 84), (1, 84, N), or (1, N, 84) float32 — 4 bbox + 80 scores.
    Returns: (20, 6) float32 [class_id, score, y1, x1, y2, x2] normalised.
    """
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust YOLO engine not available")

    arr = np.squeeze(raw) if raw.ndim >= 3 else raw
    if arr.ndim != 2 or (arr.shape[0] != 84 and arr.shape[1] != 84):
        raise ValueError(
            f"expected a (84, N) or (N, 84) YOLO26 tensor, got {raw.shape}"
        )
    if arr.shape[0] == 84 and arr.shape[1] != 84:
        arr = arr.T
    raw_flat = np.ascontiguousarray(arr, dtype=np.float32).ravel()
    n = raw_flat.size // 84
    out = np.zeros((20, 6), dtype=np.float32)

    lib.yolo26_post_process(
        raw_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n,
        ctypes.c_float(model_size),
        ctypes.c_float(frame_w),
        ctypes.c_float(frame_h),
        ctypes.c_float(score_thresh),
        ctypes.c_float(nms_thresh),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    return out


_grid_pts_cache: dict[int, "np.ndarray"] = {}


def get_anchor_free_grid_points(model_size: int = 640) -> "np.ndarray":
    """Pre-compute and cache (N_total, 2) anchor-free grid centres for strides (8, 16, 32)."""
    pts = _grid_pts_cache.get(model_size)
    if pts is not None:
        return pts

    grid_pts = []
    for stride in (8, 16, 32):
        g = model_size // stride
        ys, xs = np.mgrid[0:g, 0:g].astype(np.float32)
        grid_pts.append(
            np.column_stack(
                [
                    (xs.reshape(-1) + 0.5) * stride,
                    (ys.reshape(-1) + 0.5) * stride,
                ]
            )
        )
    pts = np.ascontiguousarray(np.concatenate(grid_pts, axis=0), dtype=np.float32)
    _grid_pts_cache[model_size] = pts
    return pts


def anchor_free_post_process(
    raw: "np.ndarray | list[np.ndarray]",
    all_pts: Optional["np.ndarray"] = None,
    model_size: int = 640,
    frame_w: float = 1.0,
    frame_h: float = 1.0,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.45,
) -> "np.ndarray":
    """YOLOv8/YOLO11 anchor-free post-process: sigmoid + DFL + grid decode + NMS."""
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust YOLO engine not available")

    if isinstance(raw, (list, tuple)):
        parts = []
        for t in raw:
            s = np.squeeze(t)
            if s.ndim == 3 and s.shape[0] == 144:
                s = s.reshape(144, -1).T
            elif s.ndim == 3 and s.shape[-1] == 144:
                s = s.reshape(-1, 144)
            parts.append(s)
        arr = np.concatenate(parts, axis=0)
    else:
        arr = np.squeeze(raw) if raw.ndim >= 3 else raw
        if arr.ndim == 2 and arr.shape[0] == 144 and arr.shape[1] != 144:
            arr = arr.T

    if all_pts is None:
        all_pts = get_anchor_free_grid_points(int(model_size))

    raw_flat = np.ascontiguousarray(arr, dtype=np.float32).ravel()
    pts = np.ascontiguousarray(all_pts, dtype=np.float32).ravel()
    out = np.zeros((20, 6), dtype=np.float32)
    # Rust reads (raw.size // 144) cells and two grid coordinates per
    # cell; anything shorter is an out-of-bounds read.
    n_cells = raw_flat.size // 144
    if raw_flat.size != n_cells * 144 or pts.size < n_cells * 2:
        raise ValueError(
            f"anchor-free tensors mismatch: raw={raw_flat.size} floats, pts={pts.size}"
        )

    lib.yolo_anchor_free_post_process(
        raw_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
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
