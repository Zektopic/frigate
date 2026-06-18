"""XDNA NPU detector for AMD Phoenix1 (AIE-ML) using OpenVINO.

This detector leverages the AMD XDNA Neural Processing Unit (NPU) present on
Phoenix1 APUs (Ryzen 5 8600G / Radeon 760M) via OpenVINO's NPU plugin.

Requirements:
- AMD XDNA driver (amdxdna) loaded on the host
- /dev/accel/accel0 or /dev/dri/renderD* device passed through
- OpenVINO with NPU plugin installed in the container
- ONNX model is auto-converted to OpenVINO IR format

The detector uses OpenVINO's NPU device type. If the model is not supported
on NPU, it automatically falls back to the GPU device.
"""

import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from frigate.const import MODEL_CACHE_DIR
from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import (
    BaseDetectorConfig,
    InputDTypeEnum,
    InputTensorEnum,
    ModelTypeEnum,
)
from frigate.util.model import (
    post_process_dfine,
    post_process_rfdetr,
    post_process_yolo,
    post_process_yolox,
)

logger = logging.getLogger(__name__)

DETECTOR_KEY = "xdnpu"


class XdNPUDetectorConfig(BaseDetectorConfig):
    """XDNA NPU detector configuration for AMD AIE-ML NPU."""

    type: Literal[DETECTOR_KEY]
    device: str = Field(
        default="NPU",
        title="Device Type",
        description="Device for inference: 'NPU' (AMD XDNA AIE-ML) or 'GPU' (fallback).",
    )


class XdNPUDetector(DetectionApi):
    """AMD XDNA NPU object detector using OpenVINO.

    Converts ONNX models to OpenVINO IR format and runs inference on the
    AMD XDNA NPU (AIE-ML) present in Phoenix1 APUs. Falls back to GPU
    if the model is not supported on NPU.
    """

    type_key = DETECTOR_KEY

    def __init__(self, detector_config: XdNPUDetectorConfig):
        super().__init__(detector_config)

        try:
            import openvino as ov
        except ImportError:
            logger.error(
                "xdnpu: OpenVINO is not available. "
                "Install openvino package to use the NPU detector."
            )
            raise

        assert detector_config.model.path is not None, (
            "xdnpu: No model.path configured"
        )
        assert detector_config.model.labelmap_path is not None, (
            "xdnpu: No model.labelmap_path configured"
        )

        self.ov_core = ov.Core()
        self.ov = ov

        # Check NPU availability
        available_devices = self.ov_core.available_devices
        logger.info(f"xdnpu: OpenVINO available devices: {available_devices}")

        device = detector_config.device

        if "NPU" in available_devices:
            logger.info("xdnpu: AMD XDNA NPU detected, using NPU device")
            device = "NPU"
        elif "GPU" in available_devices:
            logger.warning(
                "xdnpu: NPU not available, falling back to GPU device"
            )
            device = "GPU"
        else:
            logger.warning(
                "xdnpu: Neither NPU nor GPU available, falling back to CPU"
            )
            device = "CPU"

        self.device = device

        # Convert ONNX model to OpenVINO IR if needed
        model_path = Path(detector_config.model.path)
        self.ov_model_path = self._get_or_convert_model(model_path)

        # Apply NPU-specific optimizations
        if device == "NPU":
            try:
                self.ov_core.set_property(device, {"PERFORMANCE_HINT": "LATENCY"})
                self.ov_core.set_property(device, {"PERF_COUNT": "NO"})
            except Exception as e:
                logger.debug(f"xdnpu: NPU optimization hint not supported: {e}")

            try:
                self.ov_core.set_property(device, {"NPU_TURBO": "YES"})
                logger.info("xdnpu: NPU Turbo mode enabled")
            except Exception as e:
                logger.debug(f"xdnpu: NPU_TURBO not supported by driver: {e}")

        # Compile and warmup model
        try:
            self.compiled_model = self.ov_core.compile_model(
                model=str(self.ov_model_path),
                device_name=device,
            )
            self.infer_request = self.compiled_model.create_infer_request()
            logger.info(
                f"xdnpu: Model compiled for {device} "
                f"(inputs: {[i.get_any_name() for i in self.compiled_model.inputs]})"
            )
        except Exception as e:
            if device == "NPU":
                logger.warning(
                    f"xdnpu: Model compilation failed on NPU ({e}), "
                    "falling back to GPU"
                )
                device = "GPU"
                self.device = device
                self.compiled_model = self.ov_core.compile_model(
                    model=str(self.ov_model_path),
                    device_name=device,
                )
                self.infer_request = self.compiled_model.create_infer_request()
            else:
                raise

        # Store model config for preprocessing
        self.npu_model_type = detector_config.model.model_type
        self.npu_model_px = detector_config.model.input_pixel_format
        self.npu_model_shape = detector_config.model.input_tensor

        if self.npu_model_type == ModelTypeEnum.yolox:
            self.calculate_grids_strides()

        # Warmup inference
        self._warmup(detector_config)
        logger.info(f"xdnpu: Detector ready on {self.device}")

    def _get_or_convert_model(self, model_path: Path) -> Path:
        """Get OpenVINO IR model, converting from ONNX if needed.

        Returns path to the .xml file for the OpenVINO model.
        """
        cache_dir = Path(MODEL_CACHE_DIR) / "xdnpu"
        cache_dir.mkdir(parents=True, exist_ok=True)

        ov_xml_path = cache_dir / f"{model_path.stem}.xml"
        ov_bin_path = cache_dir / f"{model_path.stem}.bin"

        if ov_xml_path.exists() and ov_bin_path.exists():
            logger.info(f"xdnpu: Using cached IR model: {ov_xml_path}")
            return ov_xml_path

        logger.info(f"xdnpu: Converting ONNX model to OpenVINO IR: {model_path}")

        try:
            ov_model = self.ov.convert_model(model_path)
            self.ov.save_model(ov_model, str(ov_xml_path))
            logger.info(f"xdnpu: Model converted and saved to {ov_xml_path}")
            return ov_xml_path
        except Exception as e:
            logger.error(f"xdnpu: Model conversion failed: {e}")
            raise

    def _warmup(self, detector_config: XdNPUDetectorConfig) -> None:
        """Run a warmup inference to front-load compilation costs."""
        if detector_config.model.input_tensor == InputTensorEnum.nchw:
            shape = (1, 3, detector_config.model.height, detector_config.model.width)
        else:
            shape = (1, detector_config.model.height, detector_config.model.width, 3)

        if detector_config.model.input_dtype in (
            InputDTypeEnum.float,
            InputDTypeEnum.float_denorm,
        ):
            dtype = np.float32
        else:
            dtype = np.uint8

        logger.info("xdnpu: Warming up detector (first run may take a moment)...")
        self.detect_raw(np.zeros(shape, dtype=dtype))
        logger.info("xdnpu: Warmup complete")

    def detect_raw(self, tensor_input: np.ndarray):
        """Run inference on the NPU."""
        model_inputs = self.compiled_model.inputs

        if self.npu_model_type == ModelTypeEnum.dfine:
            # Multi-input model: images + orig_target_sizes
            result = self.infer_request.infer({
                model_inputs[0]: tensor_input,
                model_inputs[1]: np.array(
                    [[self.height, self.width]], dtype=np.int64
                ),
            })
            return post_process_dfine(result, self.width, self.height)

        input_name = model_inputs[0].get_any_name()

        # Get output tensors
        result = self.infer_request.infer({input_name: tensor_input})
        tensor_output = list(result.values())

        if self.npu_model_type == ModelTypeEnum.rfdetr:
            return post_process_rfdetr(tensor_output)
        elif self.npu_model_type == ModelTypeEnum.yolonas:
            predictions = tensor_output[0]
            detections = np.zeros((20, 6), np.float32)
            for i, prediction in enumerate(predictions):
                if i == 20:
                    break
                (_, x_min, y_min, x_max, y_max, confidence, class_id) = prediction
                if class_id < 0:
                    break
                detections[i] = [
                    class_id, confidence,
                    y_min / self.height, x_min / self.width,
                    y_max / self.height, x_max / self.width,
                ]
            return detections
        elif self.npu_model_type == ModelTypeEnum.yologeneric:
            return post_process_yolo(tensor_output, self.width, self.height)
        elif self.npu_model_type == ModelTypeEnum.yolox:
            return post_process_yolox(
                tensor_output[0],
                self.width, self.height,
                self.grids, self.expanded_strides,
            )
        else:
            raise Exception(
                f"xdnpu: {self.npu_model_type} is not supported. "
                "See the docs for supported model types."
            )
