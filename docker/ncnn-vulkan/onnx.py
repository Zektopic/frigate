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
import select
import struct
import subprocess
import sys
import time

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
    reg_raw = raw[:, :64]  # (N, 64)
    cls_raw = raw[:, 64:]  # (N, 80)

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
        box_parts.append(
            (
                pts[:, 0] - d[:, 0],  # x1
                pts[:, 1] - d[:, 1],  # y1
                pts[:, 0] + d[:, 2],  # x2
                pts[:, 1] + d[:, 3],
            )
        )  # y2
        offset += n_cells

    if offset == 0:
        return _pad_detections(np.zeros((0, 6), dtype=np.float32))

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
        return _pad_detections(np.zeros((0, 6), dtype=np.float32))

    x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
    scores, class_ids = scores[keep], class_ids[keep]

    x1, y1 = np.clip(x1, 0, input_size), np.clip(y1, 0, input_size)
    x2, y2 = np.clip(x2, 0, input_size), np.clip(y2, 0, input_size)

    valid = (x2 > x1) & (y2 > y1)
    if not np.any(valid):
        return _pad_detections(np.zeros((0, 6), dtype=np.float32))
    x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
    scores, class_ids = scores[valid], class_ids[valid]

    keep_idx = _nms(x1, y1, x2, y2, scores, nms_thresh)
    if keep_idx.size == 0:
        return _pad_detections(np.zeros((0, 6), dtype=np.float32))

    # Frigate expects: [label_idx, score, x1, y1, x2, y2]
    return np.stack(
        [
            class_ids[keep_idx].astype(np.float32),
            scores[keep_idx],
            x1[keep_idx],
            y1[keep_idx],
            x2[keep_idx],
            y2[keep_idx],
        ],
        axis=1,
    )


# ---------------------------------------------------------------------------
# Helpers — NMS
# ---------------------------------------------------------------------------


def _pad_detections(dets: np.ndarray, max_det: int = 20) -> np.ndarray:
    """Pad/truncate to exactly max_det rows (Frigate expects fixed-size output)."""
    if dets.shape[0] == 0:
        return np.zeros((max_det, 6), dtype=np.float32)
    if dets.shape[0] >= max_det:
        return dets[:max_det]
    padded = np.zeros((max_det, 6), dtype=np.float32)
    padded[: dets.shape[0]] = dets
    return padded


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
    """Return model architecture from .param file contents."""
    # Ultralytics YOLO26+ export: 84-channel output (4 bbox + 80 classes),
    # post-processing baked in, no DFL/sigmoid needed.
    if "cat_22" in param_text and "8400" in param_text:
        return "yolo26"
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
        input_name,
        outputs,
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

        # ── resolve model path & backend ───────────────────────────────
        model_path = detector_config.model.path
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}. Set model.path in config."
            )

        if model_path.endswith(".onnx"):
            self._init_onnxruntime(model_path, detector_config)
        else:
            self._init_ncnn(model_path, detector_config)

    # ------------------------------------------------------------------
    # ONNX Runtime backend (stable, works with any .onnx model)
    # ------------------------------------------------------------------
    def _init_onnxruntime(self, model_path: str, detector_config) -> None:
        import onnxruntime as ort

        logger.info("ONNX: loading %s", model_path)
        self.backend = "onnxruntime"
        self.arch = "onnx"

        self._ort_sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )

        self._ort_in_name = self._ort_sess.get_inputs()[0].name
        in_shape = self._ort_sess.get_inputs()[0].shape
        self.model_input_size = in_shape[2] if len(in_shape) > 2 else 640
        self._ort_out_name = self._ort_sess.get_outputs()[0].name
        out_shape = self._ort_sess.get_outputs()[0].shape
        logger.info(
            "ONNX: input=%s %s, output=%s %s",
            self._ort_in_name,
            in_shape,
            self._ort_out_name,
            out_shape,
        )

        # Load labels
        label_path = detector_config.model.labelmap_path
        self._class_names = None
        if label_path and os.path.exists(label_path):
            with open(label_path) as f:
                self._class_names = [line.strip() for line in f if line.strip()]

        logger.info("ONNX: model loaded successfully (CPU)")

    # ------------------------------------------------------------------
    # NCNN backend (Vulkan GPU for YOLOv5/v7, CPU for YOLOv8/11)
    # ------------------------------------------------------------------
    def _init_ncnn(self, param_path: str, detector_config) -> None:
        self.backend = "ncnn"

        if not param_path.endswith(".param"):
            raise ValueError(f"NCNN model must be a .param file, got: {param_path}")

        bin_path = param_path.replace(".param", ".bin")
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"NCNN model .bin not found: {bin_path}")

        # ── detect architecture & outputs ───────────────────────────
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
        elif self.arch == "yolo26":
            logger.info("NCNN: YOLO26 decoded-output architecture detected")
            self._out_names = ["out0"]
            self._anchors = None
        else:
            logger.info("NCNN: YOLOv5/YOLOv7 anchor-based architecture detected")
            self._anchors = dict(_YOLOV5_ANCHORS)

        # ── create network ──────────────────────────────────────────
        gpu_count = self.ncnn.get_gpu_count()
        if gpu_count > 0:
            logger.info("NCNN: Vulkan GPU detected - %s", self.ncnn.get_gpu_info(0))
        else:
            logger.warning("NCNN: No Vulkan GPU found, using CPU")

        self.net = self.ncnn.Net()

        if self.arch in ("yolov8", "yolo26"):
            # YOLOv8/YOLO11/YOLO26: Vulkan crashes after fork on AMD RADV.
            # Use worker subprocess — for YOLO26 this is fast (~87ms) because
            # post-processing is baked into the model (no numpy bottleneck).
            self._init_ncnn_worker(param_path, bin_path)
            return
        else:
            # YOLOv5/v7/YOLO26 — Vulkan is stable for these
            self.net.opt.use_vulkan_compute = True
            self.net.opt.use_fp16_arithmetic = True
            self.net.opt.use_fp16_packed = True
            self.net.opt.use_fp16_storage = True

        self.net.load_param(param_path)
        self.net.load_model(bin_path)

        # ── labels ──────────────────────────────────────────────────
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
    # Worker subprocess for YOLOv8/YOLO11 Vulkan (isolates ncnn from
    # Frigate's forked-process environment which triggers segfaults)
    # ------------------------------------------------------------------
    def _init_ncnn_worker(self, param_path: str, bin_path: str) -> None:
        """Spawn a clean subprocess that loads ncnn Vulkan and handles
        inference via pipes.  This bypasses whatever Frigate environment
        quirk (logging, faulthandler, shared memory, ZMQ, ...) causes
        ncnn Vulkan to segfault with YOLOv8/YOLO11 models."""

        # Read param file for architecture detection in the worker
        with open(param_path) as f:
            param_text = f.read()

        arch = _detect_architecture(param_text)
        in_name, out_names = _parse_blobs(param_path)

        worker_code = f"""
import sys, os, struct, numpy as np

param = {param_path!r}
binf = {bin_path!r}
in_name = {in_name!r}
out_names = {out_names!r}
model_size = 640
arch = {arch!r}  # yolo26, yolov8, etc.

# ── load ncnn Vulkan (suppress stdout — used for IPC data) ───
_real_stdout_fd = os.dup(1)
null_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(null_fd, 1)
os.close(null_fd)
import ncnn
ncnn.destroy_gpu_instance()
ncnn.create_gpu_instance()

net = ncnn.Net()
if ncnn.get_gpu_count() > 0:
    dev = ncnn.get_gpu_device(0)
    net.set_vulkan_device(dev)
net.opt.use_vulkan_compute = True
net.opt.use_fp16_arithmetic = True
net.opt.use_fp16_packed = True
net.opt.use_fp16_storage = True
net.load_param(param)
net.load_model(binf)
# Restore stdout for IPC
os.dup2(_real_stdout_fd, 1)
os.close(_real_stdout_fd)
sys.stderr.write(f"NCNN_WORKER_READY\\n")
sys.stderr.flush()

# ── pre-computed constants ────────────────────────────────────
_STRIDES = (8, 16, 32)
_DFL_BINS = np.arange(16, dtype=np.float32)
# Build grid points once (same every frame)
_grid_pts = []
_offset = 0
for _s in _STRIDES:
    _g = model_size // _s
    _n = _g * _g
    _ys, _xs = np.mgrid[0:_g, 0:_g].astype(np.float32)
    _grid_pts.append(np.column_stack([
        (_xs.reshape(-1) + 0.5) * _s,
        (_ys.reshape(-1) + 0.5) * _s,
    ]))
_ALL_PTS = np.concatenate(_grid_pts, axis=0)  # (8400, 2)

def _process(raw, w, h, score_thresh=0.25, nms_thresh=0.45):
    # Key optimisation: pre-filter by class scores BEFORE expensive DFL decode.
    # Most of 8400 candidates have near-zero scores — skip their DFL entirely.
    N = raw.shape[0]
    if N == 0:
        return np.zeros((20, 6), dtype=np.float32)

    # ── Step 1: fast class check (cheap) — find promising candidates ──
    cls = raw[:, 64:]
    # Numerically stable sigmoid: 1/(1+exp(-x))
    # Clamp to avoid overflow in exp
    cls = np.clip(cls, -20.0, 20.0)
    cls = 1.0 / (1.0 + np.exp(-cls))  # sigmoid, in-place
    cid_all = np.argmax(cls, axis=1)
    scores_all = cls[np.arange(N), cid_all]

    # Only keep candidates above threshold — typically 10-200, not 8400
    candidates = np.where(scores_all > score_thresh)[0]
    if not len(candidates):
        return np.zeros((20, 6), dtype=np.float32)

    # ── Step 2: DFL decode ONLY for promising candidates ─────────────
    idx = candidates
    reg = raw[idx][:, :64].reshape(len(idx), 4, 16)
    reg -= reg.max(axis=-1, keepdims=True)
    np.exp(reg, out=reg)
    reg /= reg.sum(axis=-1, keepdims=True)
    dist = (reg * _DFL_BINS).sum(axis=-1)  # (K, 4)

    # ── Step 3: bbox decode (only K candidates) ──────────────────────
    pts = _ALL_PTS[idx]  # (K, 2)
    x1 = pts[:, 0] - dist[:, 0]
    y1 = pts[:, 1] - dist[:, 1]
    x2 = pts[:, 0] + dist[:, 2]
    y2 = pts[:, 1] + dist[:, 3]

    scores = scores_all[idx]
    cid = cid_all[idx]

    # ── clip + valid ─────────────────────────────────────────────────
    np.clip(x1, 0, model_size, out=x1); np.clip(y1, 0, model_size, out=y1)
    np.clip(x2, 0, model_size, out=x2); np.clip(y2, 0, model_size, out=y2)
    valid = (x2 > x1) & (y2 > y1)
    if not valid.any():
        return np.zeros((20, 6), dtype=np.float32)
    x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
    scores, cid = scores[valid], cid[valid]

    # ── NMS ──────────────────────────────────────────────────────────
    K = len(scores)
    if K > 1000:
        topk = scores.argsort()[::-1][:1000]
        x1, y1, x2, y2 = x1[topk], y1[topk], x2[topk], y2[topk]
        scores, cid = scores[topk], cid[topk]
        K = len(scores)

    order = scores.argsort()[::-1]
    keep_idx = []
    areas = (x2 - x1) * (y2 - y1)
    while order.size:
        i = order[0]
        keep_idx.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0., xx2 - xx1) * np.maximum(0., yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= nms_thresh)[0] + 1]

    if not keep_idx:
        return np.zeros((20, 6), dtype=np.float32)

    ki = np.array(keep_idx)
    sx = w / model_size; sy = h / model_size
    dets = np.column_stack([
        cid[ki].astype(np.float32), scores[ki],
        x1[ki] * sx, y1[ki] * sy, x2[ki] * sx, y2[ki] * sy,
    ])
    out = np.zeros((20, 6), dtype=np.float32)
    n = min(len(dets), 20)
    out[:n] = dets[:n]
    return out

# ── YOLO26 handler (decoded output, no DFL/sigmoid needed) ─────
def process_yolo26(raw, fw, score_thresh=0.1, nms_thresh=0.45):
    bboxes = raw[:4, :].T; scores = raw[4:, :].T
    cx,cy,bw,bh = bboxes[:,0],bboxes[:,1],bboxes[:,2],bboxes[:,3]
    x1=cx-bw/2;y1=cy-bh/2;x2=cx+bw/2;y2=cy+bh/2
    cid=np.argmax(scores,axis=1);best=scores[np.arange(scores.shape[0]),cid]
    keep=best>score_thresh
    if not keep.any(): return np.zeros((20,6),dtype=np.float32)
    x1,y1,x2,y2=x1[keep],y1[keep],x2[keep],y2[keep];best,cid=best[keep],cid[keep]
    np.clip(x1,0,fw,out=x1);np.clip(y1,0,fw,out=y1)
    np.clip(x2,0,fw,out=x2);np.clip(y2,0,fw,out=y2)
    v=(x2>x1)&(y2>y1)
    if not v.any(): return np.zeros((20,6),dtype=np.float32)
    x1,y1,x2,y2=x1[v],y1[v],x2[v],y2[v];best,cid=best[v],cid[v]
    order=best.argsort()[::-1];ki=[]
    areas=(x2-x1)*(y2-y1)
    while order.size:
        i=order[0];ki.append(i)
        if order.size==1: break
        xx1=np.maximum(x1[i],x1[order[1:]]);yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]]);yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0.,xx2-xx1)*np.maximum(0.,yy2-yy1)
        order=order[np.where(inter/(areas[i]+areas[order[1:]]-inter)<=nms_thresh)[0]+1]
    if not ki: return np.zeros((20,6),dtype=np.float32)
    ki=np.array(ki);sx=fw/model_size;sy=fw/model_size
    dets=np.column_stack([cid[ki].astype(np.float32),best[ki],x1[ki]*sx,y1[ki]*sy,x2[ki]*sx,y2[ki]*sy])
    out=np.zeros((20,6),dtype=np.float32);n=min(len(dets),20);out[:n]=dets[:n]
    return out

# ── main loop ─────────────────────────────────────────────────
_frame_count = 0
while True:
    # read frame size (4 bytes) then frame data via os.read (fast)
    header = os.read(0, 4)
    if not header: break
    size = struct.unpack('>I', header)[0]
    data = b''
    while len(data) < size:
        chunk = os.read(0, size - len(data))
        if not chunk: break
        data += chunk
    # Frigate sends float16 to halve pipe I/O; convert to float32 for ncnn
    frame = np.frombuffer(data, dtype=np.float16).astype(np.float32).copy()
    _, c, fh, fw = 1, 3, model_size, model_size
    frame = frame.reshape(1, c, fh, fw)
    # Frigate sends 0-1, which is what ultralytics ncnn exports expect.
    # Do NOT scale to 0-255: saturated input makes the model hallucinate
    # high-confidence detections (false positives on empty scenes).

    mat_in = ncnn.Mat(frame)
    with net.create_extractor() as ex:
        ex.input(in_name, mat_in)
        ret, out0 = ex.extract(out_names[0] if out_names else "out0")
        if ret != 0:
            dets = np.zeros((20, 6), dtype=np.float32)
        else:
            raw = np.array(out0).copy()
            if arch == "yolo26":
                dets = process_yolo26(raw, fw)
            else:
                dets = _process(raw, fw, fw)
    result = dets.tobytes()
    os.write(1, struct.pack('>I', len(result)))
    os.write(1, result)
"""
        rust_bin = "/opt/frigate/frigate-detector-rs"
        if self.arch == "yolo26" and os.path.exists(rust_bin):
            logger.info("Spawning Rust NCNN worker: %s", rust_bin)
            self._worker = subprocess.Popen(
                [
                    rust_bin,
                    param_path,
                    bin_path,
                    in_name,
                    out_names[0] if out_names else "out0",
                    str(self.model_input_size),
                    self.arch,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            logger.info("Spawning Python NCNN worker")
            self._worker = subprocess.Popen(
                [sys.executable, "-c", worker_code],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        # Wait for ready signal on stderr (skip ncnn GPU-info lines)
        deadline = time.time() + 30
        ready = False
        while time.time() < deadline:
            r, _, _ = select.select([self._worker.stderr], [], [], 5)
            if not r:
                continue
            line = self._worker.stderr.readline().decode().strip()
            if "NCNN_WORKER_READY" in line or "NCNN_READY" in line:
                ready = True
                break
            if line:
                logger.debug("NCNN worker stderr: %s", line)
        if not ready:
            self._worker.kill()
            raise RuntimeError("NCNN worker did not become ready within 30s")
        logger.info("NCNN: Vulkan inference worker ready (pid=%d)", self._worker.pid)
        self._use_worker = True

    # ------------------------------------------------------------------
    def detect_raw(self, tensor_input: np.ndarray) -> np.ndarray:
        """Run inference → (N, 6) detections [x1, y1, x2, y2, score, cls]."""
        if self.backend == "onnxruntime":
            return self._detect_onnx(tensor_input)

        # Worker subprocess path (YOLOv8/YOLO11 Vulkan)
        if getattr(self, "_use_worker", False):
            return self._detect_worker(tensor_input)

        # NCNN in-process backend (YOLOv5/v7 Vulkan)
        _, _, h, w = tensor_input.shape

        mat_in = self.ncnn.Mat((tensor_input.squeeze(0) * 255.0).astype(np.float32))

        with self.net.create_extractor() as ex:
            ex.input(self._in_name, mat_in)

            if self.arch == "yolo26":
                return self._detect_decoded(ex, w, h)
            if self.arch == "yolov8":
                return self._detect_anchor_free(ex, w, h)
            else:
                return self._detect_anchor_based(ex, w, h)

    # ------------------------------------------------------------------
    # YOLO26 decoded-output inference (post-processing baked into model)
    # ------------------------------------------------------------------
    def _detect_decoded(self, ex, w: int, h: int) -> np.ndarray:
        """Handle Ultralytics YOLO26+ NCNN export: output is (84, N) with
        4 bbox + 80 sigmoid'd class scores — no DFL or anchors needed."""
        out_name = self._out_names[0] if self._out_names else "out0"
        ret, out0 = ex.extract(out_name)
        if ret != 0:
            return _pad_detections(np.zeros((0, 6), dtype=np.float32))
        raw = np.array(out0).copy()  # (84, N)

        if raw.ndim != 2 or raw.shape[0] != 84:
            logger.error("YOLO26: unexpected output shape %s", raw.shape)
            return _pad_detections(np.zeros((0, 6), dtype=np.float32))

        bboxes = raw[:4, :].T  # (N, 4) — cx, cy, w, h
        scores = raw[4:, :].T  # (N, 80) — already sigmoid'd

        # Convert cx,cy,w,h → x1,y1,x2,y2
        cx, cy, bw, bh = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        class_ids = np.argmax(scores, axis=1)
        best_scores = scores[np.arange(scores.shape[0]), class_ids]

        keep = best_scores > 0.25
        if not np.any(keep):
            return _pad_detections(np.zeros((0, 6), dtype=np.float32))
        x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
        best_scores, class_ids = best_scores[keep], class_ids[keep]

        np.clip(x1, 0, self.model_input_size, out=x1)
        np.clip(y1, 0, self.model_input_size, out=y1)
        np.clip(x2, 0, self.model_input_size, out=x2)
        np.clip(y2, 0, self.model_input_size, out=y2)
        valid = (x2 > x1) & (y2 > y1)
        if not np.any(valid):
            return _pad_detections(np.zeros((0, 6), dtype=np.float32))
        x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
        best_scores, class_ids = best_scores[valid], class_ids[valid]

        keep_idx = _nms(x1, y1, x2, y2, best_scores, 0.45)
        if keep_idx.size == 0:
            return _pad_detections(np.zeros((0, 6), dtype=np.float32))

        # Frigate expects: [class_id, score, ymin_norm, xmin_norm, ymax_norm, xmax_norm]
        dets = np.column_stack(
            [
                class_ids[keep_idx].astype(np.float32),
                best_scores[keep_idx],
                y1[keep_idx] / self.model_input_size,
                x1[keep_idx] / self.model_input_size,
                y2[keep_idx] / self.model_input_size,
                x2[keep_idx] / self.model_input_size,
            ]
        )
        return _pad_detections(dets)

    # ------------------------------------------------------------------
    # Worker subprocess inference (YOLOv8/YOLO11 Vulkan)
    # ------------------------------------------------------------------
    @staticmethod
    def _pipe_read(fd: int, nbytes: int) -> bytes:
        """Read exactly *nbytes* from file descriptor, looping on partial reads."""
        buf = b""
        while len(buf) < nbytes:
            chunk = os.read(fd, nbytes - len(buf))
            if not chunk:
                raise OSError("pipe EOF during read")
            buf += chunk
        return buf

    def _detect_worker(self, tensor_input: np.ndarray) -> np.ndarray:
        """Send frame to the persistent ncnn Vulkan worker, get detections.
        Uses float16 to halve pipe I/O (2.5MB vs 5MB per frame)."""
        data = tensor_input.astype(np.float16).tobytes()
        try:
            os.write(self._worker.stdin.fileno(), struct.pack(">I", len(data)))
            os.write(self._worker.stdin.fileno(), data)
            header = self._pipe_read(self._worker.stdout.fileno(), 4)
            size = struct.unpack(">I", header)[0]
            result = self._pipe_read(self._worker.stdout.fileno(), size)
            dets = np.frombuffer(result, dtype=np.float32).reshape(-1, 6).copy()
            # worker output is [class_id, score, x1, y1, x2, y2] in 0..model_input_size range
            # Frigate expects: [class_id, score, ymin_norm, xmin_norm, ymax_norm, xmax_norm]
            if len(dets) > 0:
                x1 = dets[:, 2] / self.model_input_size
                y1 = dets[:, 3] / self.model_input_size
                x2 = dets[:, 4] / self.model_input_size
                y2 = dets[:, 5] / self.model_input_size
                dets[:, 2] = y1
                dets[:, 3] = x1
                dets[:, 4] = y2
                dets[:, 5] = x2
            return dets
        except (BrokenPipeError, OSError) as exc:
            logger.error("NCNN worker: pipe error — %s", exc)
            return _pad_detections(np.zeros((0, 6), dtype=np.float32))

    # ------------------------------------------------------------------
    # ONNX Runtime inference
    # ------------------------------------------------------------------
    def _detect_onnx(self, tensor_input: np.ndarray) -> np.ndarray:
        """YOLOv9 ONNX inference — output already decoded (no DFL needed)."""
        _, _, h, w = tensor_input.shape

        # This model expects 0-255 range, but Frigate normalizes to 0-1.
        # Scale back to 0-255.
        model_input = tensor_input.astype(np.float32) * 255.0

        out = self._ort_sess.run(
            [self._ort_out_name],
            {self._ort_in_name: model_input},
        )[0]  # (1, 84, N_dets)

        # out[0] shape: (84, N_dets) where 84 = 4 bbox + 80 classes
        dets = out[0]  # (84, N)
        bboxes = dets[:4, :].T  # (N, 4) — cx, cy, w, h
        scores = dets[4:, :].T  # (N, 80)

        # Convert cx,cy,w,h → x1,y1,x2,y2
        cx, cy, bw, bh = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Best class per detection
        class_ids = np.argmax(scores, axis=1)
        best_scores = scores[np.arange(scores.shape[0]), class_ids]

        # Filter by score
        keep = best_scores > 0.25
        if not np.any(keep):
            return _pad_detections(np.zeros((0, 6), dtype=np.float32))

        x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
        best_scores = best_scores[keep]
        class_ids = class_ids[keep]

        # Clamp
        x1 = np.clip(x1, 0, 1)
        y1 = np.clip(y1, 0, 1)
        x2 = np.clip(x2, 0, 1)
        y2 = np.clip(y2, 0, 1)

        # Remove degenerate
        valid = (x2 > x1) & (y2 > y1)
        if not np.any(valid):
            return _pad_detections(np.zeros((0, 6), dtype=np.float32))
        x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
        best_scores = best_scores[valid]
        class_ids = class_ids[valid]

        # NMS
        keep_idx = _nms(x1, y1, x2, y2, best_scores, 0.45)
        if keep_idx.size == 0:
            return _pad_detections(np.zeros((0, 6), dtype=np.float32))

        best_scores = best_scores[keep_idx]
        class_ids = class_ids[keep_idx]
        y1_norm = y1[keep_idx]
        x1_norm = x1[keep_idx]
        y2_norm = y2[keep_idx]
        x2_norm = x2[keep_idx]

        # Frigate expects: [label_idx, score, ymin_norm, xmin_norm, ymax_norm, xmax_norm]
        dets = np.stack(
            [
                class_ids.astype(np.float32),
                best_scores,
                y1_norm,
                x1_norm,
                y2_norm,
                x2_norm,
            ],
            axis=1,
        )
        return _pad_detections(dets)

    # ------------------------------------------------------------------
    # Anchor-free (YOLOv8 / YOLO11)
    # ------------------------------------------------------------------
    def _detect_anchor_free(self, ex, w: int, h: int) -> np.ndarray:
        out_name = self._out_names[0] if self._out_names else "out0"
        ret, out0 = ex.extract(out_name)
        if ret != 0:
            logger.warning("NCNN: extract out0 failed (rc=%d)", ret)
            return _pad_detections(np.zeros((0, 6), dtype=np.float32))

        raw = np.array(out0).copy()

        if raw.ndim != 2 or raw.shape[1] != 144:
            logger.error(
                "NCNN: unexpected output shape %s (expected (N, 144))", raw.shape
            )
            return _pad_detections(np.zeros((0, 6), dtype=np.float32))

        dets = _post_process_yolov8(
            raw, self.model_input_size, score_thresh=0.25, nms_thresh=0.45
        )

        if dets.size == 0:
            return _pad_detections(np.zeros((0, 6), dtype=np.float32))

        # Normalise to 0..1 and format as [class_id, score, ymin_norm, xmin_norm, ymax_norm, xmax_norm]
        # dets format from post_process is: [class_id, score, x1, y1, x2, y2]
        x1_norm = dets[:, 2] / self.model_input_size
        y1_norm = dets[:, 3] / self.model_input_size
        x2_norm = dets[:, 4] / self.model_input_size
        y2_norm = dets[:, 5] / self.model_input_size

        norm_dets = np.stack(
            [dets[:, 0], dets[:, 1], y1_norm, x1_norm, y2_norm, x2_norm], axis=1
        )

        return _pad_detections(norm_dets)

    # ------------------------------------------------------------------
    # Anchor-based (YOLOv5 / YOLOv7)
    # ------------------------------------------------------------------
    def _detect_anchor_based(self, ex, w: int, h: int) -> np.ndarray:
        outputs = []
        for name in self._out_names:
            ret, ncnn_mat = ex.extract(name)
            if ret != 0:
                logger.warning("NCNN: extract %s failed (rc=%d)", name, ret)
                return _pad_detections(np.zeros((0, 6), dtype=np.float32))
            arr = np.array(ncnn_mat).copy()
            arr = 1.0 / (1.0 + np.exp(-arr))  # sigmoid
            arr = np.expand_dims(arr, 0)
            outputs.append(arr)

        from frigate.util.model import post_process_yolo

        return post_process_yolo(outputs, self.model_input_size, self.model_input_size)
