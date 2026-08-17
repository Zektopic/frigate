"""Frigate detector plugin using ncnn with Vulkan GPU acceleration.

Uses ncnn's Vulkan backend via RADV (Mesa Vulkan driver),
completely bypassing the unstable ROCm compute stack.

Tested with: AMD Radeon 760M (gfx1103) via RADV GFX1103_R1
"""

import logging
import os

import numpy as np
from pydantic import Field
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig

logger = logging.getLogger(__name__)

DETECTOR_KEY = "ncnn"


class NCNNDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: str = Field(default="AUTO", title="Device Type")


class NCNNDetector(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, detector_config: NCNNDetectorConfig):
        super().__init__(detector_config)

        import ncnn

        self.ncnn = ncnn

        # Find the ncnn model files (.param and .bin) next to the ONNX model
        model_dir = os.path.dirname(detector_config.model.path)
        param_path = os.path.join(model_dir, "yolov5s.ncnn.param")
        bin_path = os.path.join(model_dir, "yolov5s.ncnn.bin")

        if not os.path.exists(param_path) or not os.path.exists(bin_path):
            raise FileNotFoundError(
                f"ncnn model not found: {param_path} / {bin_path}. "
                f"Download from https://github.com/nihui/ncnn-assets"
            )

        logger.info(f"NCNN: loading model from {param_path}")

        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = True
        self.net.opt.use_fp16_arithmetic = True
        self.net.opt.use_fp16_packed = True
        self.net.opt.use_fp16_storage = True

        gpu_count = ncnn.get_gpu_count()
        if gpu_count > 0:
            logger.info(f"NCNN: Vulkan GPU detected - {ncnn.get_gpu_info(0)}")
        else:
            logger.warning("NCNN: No Vulkan GPU found, falling back to CPU")

        self.net.load_param(param_path)
        self.net.load_model(bin_path)

        # YOLOv5s input: 640x640
        self.model_input_size = 640
        self._num_classes = 80
        self._class_names = None

        # YOLOv5s anchors (width, height) per stride
        self._anchors = {
            8: np.array([[10, 13], [16, 30], [33, 23]], dtype=np.float32),
            16: np.array([[30, 61], [62, 45], [59, 119]], dtype=np.float32),
            32: np.array([[116, 90], [156, 198], [373, 326]], dtype=np.float32),
        }

        # Load COCO labels
        label_path = detector_config.model.labelmap_path
        if label_path and os.path.exists(label_path):
            with open(label_path) as f:
                self._class_names = [line.strip() for line in f if line.strip()]

        logger.info(
            f"NCNN: model loaded successfully (Vulkan={'on' if gpu_count > 0 else 'off'})"
        )

    def detect_raw(self, tensor_input: np.ndarray):
        """YOLOv5s inference via ncnn Vulkan, using Frigate's existing YOLO postprocessing."""

        _, _, h, w = tensor_input.shape
        # ncnn YOLOv5s expects 0-255 range, Frigate normalizes to 0-1
        mat_in = self.ncnn.Mat((tensor_input.squeeze(0) * 255.0).astype(np.float32))

        with self.net.create_extractor() as ex:
            ex.input("in0", mat_in)
            ret0, out0_raw = ex.extract("out0")
            if ret0 != 0:
                return np.zeros((0, 6), np.float32)

            arr0 = np.array(out0_raw)
            # Check if this is a single-output YOLO26/11 tensor (e.g. 84x8400 or 8400x84)
            if arr0.ndim == 2 and (arr0.shape[0] == 84 or arr0.shape[1] == 84):
                if arr0.shape[1] == 84 and arr0.shape[0] != 84:
                    arr0 = arr0.T
                from frigate.detectors.rust_yolo import yolo26_post_process, yolo_available
                if yolo_available():
                    return yolo26_post_process(arr0, self.model_input_size, 1.0, 1.0)
                else:
                    from frigate.util.model import post_process_yolo
                    return post_process_yolo([np.expand_dims(arr0, 0)], self.model_input_size, self.model_input_size)

            ret1, out1_raw = ex.extract("out1")
            ret2, out2_raw = ex.extract("out2")
            if ret1 != 0 or ret2 != 0:
                return np.zeros((0, 6), np.float32)

        # Convert multipart YOLOv5s outputs: apply sigmoid
        outputs = []
        for ncnn_mat in [out0_raw, out1_raw, out2_raw]:
            arr = np.array(ncnn_mat)
            arr = 1.0 / (1.0 + np.exp(-arr))
            arr = np.expand_dims(arr, 0)
            outputs.append(arr)

        from frigate.util.model import post_process_yolo
        return post_process_yolo(outputs, self.model_input_size, self.model_input_size)

    @staticmethod
    def _nms(x1, y1, x2, y2, scores, iou_threshold):
        """Simple NMS implementation."""
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
