"""ctypes bindings for the SIMD-accelerated motion detection engine.

Drop-in acceleration for :class:`ImprovedMotionDetector` — call
:func:`detect_motion` instead of the OpenCV + scipy pipeline.

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


def _load_lib() -> Optional[ctypes.CDLL]:
    global _lib, _available
    if _available is False:
        return None
    if _lib is not None:
        return _lib
    search = ["/opt/frigate", os.path.dirname(__file__)]
    for d in search:
        candidate = os.path.join(d, _LIB_NAME)
        if os.path.isfile(candidate):
            try:
                _lib = ctypes.CDLL(candidate)
                _available = True
                logger.debug("Loaded Rust motion engine from %s", candidate)
                return _lib
            except OSError as exc:
                logger.debug("Failed to load %s: %s", candidate, exc)
    _available = False
    return None


def motion_available() -> bool:
    return _load_lib() is not None


def detect_motion(
    frame: np.ndarray,
    avg_frame: np.ndarray,
    mask: np.ndarray,
    threshold: int = 25,
    min_area: int = 30,
    improve_contrast: bool = False,
    blur: bool = True,
    max_boxes: int = 128,
) -> tuple[list[tuple[int, int, int, int]], bool]:
    """Run the full Rust motion-detection pipeline.

    Returns ``(boxes, calibrated)`` where *calibrated* is True when
    motion is <5% and ≤4 boxes.
    """
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust motion engine not available")

    h, w = frame.shape
    frame = np.ascontiguousarray(frame)
    avg_frame = np.ascontiguousarray(avg_frame)
    mask = np.ascontiguousarray(mask)

    boxes = (MotionBox * max_boxes)()
    calibrated = ctypes.c_uint8(0)

    lib.motion_detect_full.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # frame
        ctypes.POINTER(ctypes.c_float),  # avg_frame
        ctypes.POINTER(ctypes.c_uint8),  # mask
        ctypes.c_uint32,
        ctypes.c_uint32,  # w, h
        ctypes.c_uint8,  # threshold
        ctypes.c_uint32,  # min_area
        ctypes.c_uint8,  # improve_contrast
        ctypes.c_uint8,  # blur_enabled
        ctypes.POINTER(MotionBox),  # out_boxes
        ctypes.c_uint32,  # max_boxes
        ctypes.POINTER(ctypes.c_uint8),  # out_calibrated
    ]
    lib.motion_detect_full.restype = ctypes.c_uint32

    n = lib.motion_detect_full(
        frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        avg_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        w,
        h,
        threshold,
        min_area,
        int(improve_contrast),
        int(blur),
        ctypes.cast(boxes, ctypes.POINTER(MotionBox)),
        max_boxes,
        ctypes.byref(calibrated),
    )

    result = [(boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2) for i in range(n)]
    return result, bool(calibrated.value)


def init_average(frame: np.ndarray, avg_frame: np.ndarray) -> None:
    """Initialize / reset the running-average buffer from a frame."""
    lib = _load_lib()
    if lib is None:
        raise RuntimeError("Rust motion engine not available")
    frame = np.ascontiguousarray(frame)
    avg_frame = np.ascontiguousarray(avg_frame)
    lib.motion_init_average.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
    ]
    lib.motion_init_average.restype = None
    lib.motion_init_average(
        frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        avg_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        frame.size,
    )
