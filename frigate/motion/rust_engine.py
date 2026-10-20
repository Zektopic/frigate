"""ctypes bindings for the SIMD-accelerated motion detection engine.

:class:`ImprovedMotionDetector` calls :func:`pixel_pipeline` in place of
its OpenCV blur, absdiff, threshold, dilate and contour steps.  Contrast
improvement, masking and the running average stay in Python.

The shared library is at ``/opt/frigate/libfrigate_motion_rs.so``.
"""

import ctypes
import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_LIB_NAME = "libfrigate_motion_rs.so"
_lib: Optional[ctypes.CDLL] = None
_available: Optional[bool] = None


class MotionBox(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.c_int32),
        ("y1", ctypes.c_int32),
        ("x2", ctypes.c_int32),
        ("y2", ctypes.c_int32),
    ]


def _configure(lib: ctypes.CDLL) -> None:
    """Declare every prototype once, at load time.

    Re-assigning ``argtypes`` on each call is wasted work on a per-frame
    path and races with concurrent callers of the same function object.
    """
    u8p = ctypes.POINTER(ctypes.c_uint8)
    f32p = ctypes.POINTER(ctypes.c_float)
    lib.motion_pixel_pipeline.argtypes = [
        u8p, f32p, u8p,
        ctypes.c_uint32, ctypes.c_uint32,  # w, h
        ctypes.c_uint8, ctypes.c_uint32, ctypes.c_uint8,  # thresh, area, blur
        ctypes.POINTER(MotionBox), ctypes.c_uint32,
        f32p,
    ]  # fmt: skip
    lib.motion_pixel_pipeline.restype = ctypes.c_uint32


def _as_u8_plane(name: str, arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Return a C-contiguous uint8 view/copy of ``arr`` with exactly ``shape``.

    Rust derives every buffer length from (w, h); a plane of another shape
    or item size is an out-of-bounds read/write, which in a
    ``panic = "abort"`` cdylib takes the whole process down.
    """
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} != frame shape {shape}")
    if arr.dtype == np.bool_:
        arr = arr.view(np.uint8)
    if arr.dtype != np.uint8:
        raise TypeError(f"{name} must be uint8, got {arr.dtype}")
    return np.ascontiguousarray(arr)


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
        os.path.join(repo_root, "frigate-motion-rs", "target", "release"),
        os.path.join(repo_root, "frigate-motion-rs", "target", "debug"),
    ]
    for d in search:
        candidate = os.path.join(d, _LIB_NAME)
        if os.path.isfile(candidate):
            try:
                lib = ctypes.CDLL(candidate)
                _configure(lib)
                _lib = lib
                _available = True
                logger.debug("Loaded Rust motion engine from %s", candidate)
                return _lib
            except (OSError, AttributeError) as exc:
                logger.debug("Failed to load %s: %s", candidate, exc)
    _available = False
    return None


def motion_available() -> bool:
    return _load_lib() is not None


def pixel_pipeline(
    frame: np.ndarray,
    avg_frame: np.ndarray,
    mask: np.ndarray,
    threshold: int,
    min_area: int,
    blur: bool = True,
    max_boxes: int = 128,
) -> tuple[list[tuple[int, int, int, int]], float, np.ndarray]:
    """Rust pixel pipeline: blur (in-place) → absdiff → threshold → dilate → contours.

    Does NOT modify avg_frame — the caller keeps its accumulateWeighted logic.
    The blur is applied in-place on the (contiguous) frame buffer, matching
    the OpenCV flow where the blurred frame is later averaged into avg_frame.

    Returns ``(boxes, total_contour_area, blurred_frame)``.  Always use the
    returned frame — ``ascontiguousarray`` may have copied the input, in
    which case the caller's original array was NOT blurred.
    """
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust motion engine not available")

    if frame.ndim != 2:
        raise ValueError(f"frame must be 2-D grayscale, got shape {frame.shape}")
    h, w = frame.shape
    frame = _as_u8_plane("frame", frame, (h, w))
    if not frame.flags.writeable:
        frame = frame.copy()  # blurred in place by Rust
    # avg_frame is only read here, so a dtype/layout conversion is safe.
    if avg_frame.shape != (h, w):
        raise ValueError(f"avg_frame shape {avg_frame.shape} != frame shape {(h, w)}")
    avg_frame = np.ascontiguousarray(avg_frame, dtype=np.float32)
    mask = _as_u8_plane("mask", mask, (h, w))
    if max_boxes <= 0:
        return [], 0.0, frame

    boxes = (MotionBox * max_boxes)()
    total_area = ctypes.c_float(0.0)

    n = lib.motion_pixel_pipeline(
        frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        avg_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        w,
        h,
        threshold,
        min_area,
        int(blur),
        ctypes.cast(boxes, ctypes.POINTER(MotionBox)),
        max_boxes,
        ctypes.byref(total_area),
    )
    result = [(boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2) for i in range(n)]
    return result, float(total_area.value), frame
