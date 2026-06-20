"""Frigate detector plugin using ncnn with Vulkan GPU acceleration.

Supports multiple model architectures:
  - YOLOv5 / YOLOv7:  anchor-based,  3 outputs × 255 channels, Vulkan
  - YOLOv8 / YOLO11:  anchor-free,   1 output  × 144 channels, DFL decode

Model auto-detection:
  The plugin reads the .param file to detect the architecture and picks the
  appropriate post-processing pipeline.  Set ``path`` in the Frigate model
  config to the .param file of your chosen model.

IMPORTANT — lazy ncnn import:
  ncnn's C extension must NOT be imported before Python multiprocessing
  fork (it initialises GPU resources that become invalid in children).
  The import is deliberately deferred to ``ONNXDetector.__init__``, which
  Frigate calls inside the detector child process.
"""

import logging
import os
import re
from typing import Optional

import numpy as np
from pydantic import Field
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig

logger = logging.getLogger(__name__)

DETECTOR_KEY = "onnx"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Strides for both anchor-based (YOLOv5/v7) and anchor-free (YOLOv8/11) heads
_STRIDES = (8, 16, 32)

# YOLOv5s/v7 anchors (width, height) — used when ``255``-channel outputs detected
_YOLOV5_ANCHORS = {
    8: np.array([[10, 13], [16, 30], [33, 23]], dtype=np.float32),
    16: np.array([[30, 61], [62, 45], [59, 119]], dtype=np.float32),
    32: np.array([[116, 90], [156, 198], [373, 326]], dtype=np.float32),
}

# DFL bins for YOLOv8/YOLO11 bounding-box regression
_REG_MAX = 16


# ---------------------------------------------------------------------------
# Detector config
# ---------------------------------------------------------------------------

class ONNXDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: str = Field(default="AUTO", title="Device Type")


# ---------------------------------------------------------------------------
# Helpers — DFL decode (YOLOv8 / YOLO11)
# ---------------------------------------------------------------------------

def _make_grid_points(feat_h: int, feat_w: int, stride: int) -> np.ndarray:
    """(feat_h*feat_w, 2) grid-cell centres for one detection head."""
    ys, xs = np.mgrid[0:feat_h, 0:feat_w].astype(np.float32)
    xs = (xs.reshape(-1) + 0.5) * stride
    ys = (ys.reshape(-1) + 0.5) * stride
    return np.stack([xs, ys], axis=1)


def _dfl_decode(reg_feat: np.ndarray, reg_max: int = _REG_MAX) -> np.ndarray:
    """Distribution Focal Loss decode: (N, 4*reg_max) → (N, 4)."""
    n = reg_feat.shape[0]
    reg_feat = reg_feat.reshape(n, 4, reg_max)
    reg_feat = np.exp(reg_feat - reg_feat.max(axis=-1, keepdims=True))
    reg_feat /= reg_feat.sum(axis=-1, keepdims=True)
    bins = np.arange(reg_max, dtype=np.float32)
    return (reg_feat * bins).sum(axis=-1)


def _post_process_yolov8(
    raw: np.ndarray,
    input_size: int,
    score_thresh: float = 0.25,
    nms_thresh: float = 0.45,
) -> np.ndarray:
    """YOLOv8 / YOLO11 post-processing (anchor-free, DFL).

    Parameters
    ----------
    raw : np.ndarray  shape (N_total, 144)
        Concatenated outputs of the three detection heads.
        144 = 64 (4 sides × 16 DFL bins) + 80 (COCO classes).

    Returns
    -------
    np.ndarray  shape (K, 6)  — [x1, y1, x2, y2, score, class_id]
    """
    reg_raw = raw[:, :64]   # (N, 64)
    cls_raw = raw[:, 64:]   # (N, 80)

    dist = _dfl_decode(reg_raw)  # (N, 4)  left, top, right, bottom

    # Split per stride and convert to boxes
    points_list = []
    box_parts: list[tuple] = []
    offset = 0

    for stride in _STRIDES:
        grid = input_size // stride
        n_cells = grid * grid
        if offset + n_cells > raw.shape[0]:
            break
        pts = _make_grid_points(grid, grid, stride)
        d = dist[offset : offset + n_cells]
        points_list.append(pts)
        box_parts.append((pts[:, 0] - d[:, 0],   # x1
                          pts[:, 1] - d[:, 1],   # y1
                          pts[:, 0] + d[:, 2],   # x2
                          pts[:, 1] + d[:, 3]))  # y2
        offset += n_cells

    if offset == 0:
        return np.zeros((0, 6), dtype=np.float32)

    x1 = np.concatenate([b[0] for b in box_parts])
    y1 = np.concatenate([b[1] for b in box_parts])
    x2 = np.concatenate([b[2] for b in box_parts])
    y2 = np.concatenate([b[3] for b in box_parts])

    # Sigmoid on class logits
    cls_raw = cls_raw - cls_raw.max(axis=1, keepdims=True)
    cls_scores = 1.0 / (1.0 + np.exp(-cls_raw))
    class_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), class_ids]

    # Filter & NMS
    keep = scores > score_thresh
    if not np.any(keep):
        return np.zeros((0, 6), dtype=np.float32)

    x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
    scores, class_ids = scores[keep], class_ids[keep]

    x1, y1 = np.clip(x1, 0, input_size), np.clip(y1, 0, input_size)
    x2, y2 = np.clip(x2, 0, input_size), np.clip(y2, 0, input_size)

    valid = (x2 > x1) & (y2 > y1)
    if not np.any(valid):
        return np.zeros((0, 6), dtype=np.float32)
    x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
    scores, class_ids = scores[valid], class_ids[valid]

    keep_idx = _nms(x1, y1, x2, y2, scores, nms_thresh)
    if keep_idx.size == 0:
        return np.zeros((0, 6), dtype=np.float32)

    return np.stack([
        x1[keep_idx], y1[keep_idx], x2[keep_idx], y2[keep_idx],
        scores[keep_idx], class_ids[keep_idx].astype(np.float32),
    ], axis=1)


# ---------------------------------------------------------------------------
# Helpers — NMS
# ---------------------------------------------------------------------------

def _nms(x1, y1, x2, y2, scores, iou_threshold):
    """Greedy class-agnostic NMS."""
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep)


# ---------------------------------------------------------------------------
# Architecture auto-detection
# ---------------------------------------------------------------------------

def _detect_architecture(param_text: str) -> str:
    """Return 'yolov8' (anchor-free), 'yolov5' (anchor-based), or 'unknown'."""
    if "view_" in param_text and "144" in param_text:
        return "yolov8"
    if "255" in param_text:
        return "yolov5"
    return "yolov5"  # default fallback


def _parse_blobs(param_path: str) -> tuple[str, list[str]]:
    """Return (input_blob_name, [output_blob_names]) from an ncnn .param file.

    NCNN param lines: ``LayerType LayerName InCount OutCount InBlobs... OutBlob Params``
    - An *input blob* is consumed but never produced.
    - An *output blob* is produced but never consumed.
    """
    produced: set[str] = set()
    consumed: set[str] = set()

    with open(param_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                # skip magic number & layer/blob counts (start with digits)
                try:
                    int(line.split()[0])
                    continue
                except (ValueError, IndexError):
                    pass
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            try:
                in_count = int(parts[2])
                out_count = int(parts[3])
            except ValueError:
                continue

            in_start = 4
            in_end = in_start + in_count
            out_start = in_end
            out_end = out_start + out_count

            for i in range(in_start, min(in_end, len(parts))):
                consumed.add(parts[i])
            for i in range(out_start, min(out_end, len(parts))):
                produced.add(parts[i])

    inputs = sorted(consumed - produced)
    outputs = sorted(produced - consumed)
    input_name = inputs[0] if inputs else "in0"

    logger.debug(
        "Auto-detected input=%s, outputs=%s",
        input_name, outputs,
    )
    return input_name, outputs


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class ONNXDetector(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: ONNXDetectorConfig):
        super().__init__(detector_config)

        # ── lazy ncnn import ──────────────────────────────────────────
        # MUST happen here (inside the child process) and NOT at module
        # level.  Frigate imports plugin modules at startup (pre-fork);
        # importing ncnn early initialises GPU resources that become
        # invalid after fork.
        import ncnn
        self.ncnn = ncnn

        # ── resolve model paths ───────────────────────────────────────
        model_dir = os.path.dirname(detector_config.model.path) \
            if detector_config.model.path else "."
        param_path = detector_config.model.path \
            if detector_config.model.path and detector_config.model.path.endswith(".param") \
            else None

        if not param_path or not os.path.exists(param_path):
            raise FileNotFoundError(
                f"NCNN model .param not found: {param_path}. "
                "Set model.path to a .param file in config/model_cache/."
            )

        bin_path = param_path.replace(".param", ".bin")
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"NCNN model .bin not found: {bin_path}")

        # ── detect architecture & outputs ─────────────────────────────
        with open(param_path) as f:
            param_text = f.read()

        self.arch = _detect_architecture(param_text)
        self._in_name, self._out_names = _parse_blobs(param_path)

        self.model_input_size = 640
        self._num_classes = 80
        self._class_names = None

        if self.arch == "yolov8":
            logger.info("NCNN: YOLOv8/YOLO11 anchor-free architecture detected")
            if not self._out_names:
                self._out_names = ["out0"]
            self._anchors = None
        else:
            logger.info("NCNN: YOLOv5/YOLOv7 anchor-based architecture detected")
            self._anchors = dict(_YOLOV5_ANCHORS)

        logger.debug("NCNN: output blobs = %s", self._out_names)

        # ── create network ────────────────────────────────────────────
        gpu_count = ncnn.get_gpu_count()
        if gpu_count > 0:
            logger.info("NCNN: Vulkan GPU detected - %s", ncnn.get_gpu_info(0))
        else:
            logger.warning("NCNN: No Vulkan GPU found, using CPU")

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = True
        self.net.opt.use_fp16_arithmetic = True
        self.net.opt.use_fp16_packed = True
        self.net.opt.use_fp16_storage = True

        self.net.load_param(param_path)
        self.net.load_model(bin_path)

        # ── labels ────────────────────────────────────────────────────
        label_path = detector_config.model.labelmap_path
        if label_path and os.path.exists(label_path):
            with open(label_path) as f:
                self._class_names = [line.strip() for line in f if line.strip()]

        logger.info(
            "NCNN: model loaded (arch=%s, Vulkan=%s)",
            self.arch,
            "on" if gpu_count > 0 else "off",
        )

    # ------------------------------------------------------------------
    def detect_raw(self, tensor_input: np.ndarray) -> np.ndarray:
        """Run inference → (N, 6) detections [x1, y1, x2, y2, score, cls]."""
        _, _, h, w = tensor_input.shape

        mat_in = self.ncnn.Mat(
            (tensor_input.squeeze(0) * 255.0).astype(np.float32)
        )

        with self.net.create_extractor() as ex:
            ex.input(self._in_name, mat_in)

            if self.arch == "yolov8":
                return self._detect_anchor_free(ex, w, h)
            else:
                return self._detect_anchor_based(ex, w, h)

    # ------------------------------------------------------------------
    # Anchor-free (YOLOv8 / YOLO11)
    # ------------------------------------------------------------------
    def _detect_anchor_free(self, ex, w: int, h: int) -> np.ndarray:
        out_name = self._out_names[0] if self._out_names else "out0"
        ret, out0 = ex.extract(out_name)
        if ret != 0:
            logger.warning("NCNN: extract out0 failed (rc=%d)", ret)
            return np.zeros((0, 6), dtype=np.float32)

        raw = np.array(out0).copy()

        if raw.ndim != 2 or raw.shape[1] != 144:
            logger.error(
                "NCNN: unexpected output shape %s (expected (N, 144))", raw.shape
            )
            return np.zeros((0, 6), dtype=np.float32)

        dets = _post_process_yolov8(
            raw, self.model_input_size, score_thresh=0.25, nms_thresh=0.45
        )

        if dets.size == 0:
            return np.zeros((0, 6), dtype=np.float32)

        # Rescale to original frame size
        scale_x = w / self.model_input_size
        scale_y = h / self.model_input_size
        dets[:, 0] *= scale_x
        dets[:, 2] *= scale_x
        dets[:, 1] *= scale_y
        dets[:, 3] *= scale_y

        return dets

    # ------------------------------------------------------------------
    # Anchor-based (YOLOv5 / YOLOv7)
    # ------------------------------------------------------------------
    def _detect_anchor_based(self, ex, w: int, h: int) -> np.ndarray:
        outputs = []
        for name in self._out_names:
            ret, ncnn_mat = ex.extract(name)
            if ret != 0:
                logger.warning("NCNN: extract %s failed (rc=%d)", name, ret)
                return np.zeros((0, 6), dtype=np.float32)
            arr = np.array(ncnn_mat).copy()
            arr = 1.0 / (1.0 + np.exp(-arr))  # sigmoid
            arr = np.expand_dims(arr, 0)
            outputs.append(arr)

        from frigate.util.model import post_process_yolo
        return post_process_yolo(outputs, self.model_input_size, self.model_input_size)
