"""Facenet Embeddings."""

import logging
import os

import numpy as np

from frigate.const import MODEL_CACHE_DIR
from frigate.detectors.detection_runners import get_optimized_runner
from frigate.embeddings.types import EnrichmentModelTypeEnum
from frigate.log import suppress_stderr_during
from frigate.util.downloader import ModelDownloader

from ...config import FaceRecognitionConfig
from .base_embedding import BaseEmbedding

try:
    from tflite_runtime.interpreter import Interpreter
except ModuleNotFoundError:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ModuleNotFoundError:
        Interpreter = None

logger = logging.getLogger(__name__)

ARCFACE_INPUT_SIZE = 112
FACENET_INPUT_SIZE = 160


def bilinear_resize_u8_numpy(img: np.ndarray, dst_w: int, dst_h: int) -> np.ndarray:
    """Exact NumPy port of the Rust `bilinear_resize_u8` kernel (frigate-frame-rs).

    Same sampling (top-left aligned, ``src = dst * scale``), same f32
    arithmetic and the same truncating cast to uint8, so embeddings don't
    depend on whether the Rust library is available.
    """
    src_h, src_w = img.shape[:2]
    if src_w == dst_w and src_h == dst_h:
        return img.copy()

    sx = np.arange(dst_w, dtype=np.float64) * (src_w / dst_w)
    sy = np.arange(dst_h, dtype=np.float64) * (src_h / dst_h)
    x0 = np.minimum(sx.astype(np.int64), src_w - 1)
    y0 = np.minimum(sy.astype(np.int64), src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    y1 = np.minimum(y0 + 1, src_h - 1)
    fx = (sx - x0).astype(np.float32)
    fy = (sy - y0).astype(np.float32)

    src = img.astype(np.float32)
    if src.ndim == 2:
        src = src[:, :, None]
    fx = fx[None, :, None]
    fy = fy[:, None, None]

    v00 = src[y0][:, x0]
    v10 = src[y0][:, x1]
    v01 = src[y1][:, x0]
    v11 = src[y1][:, x1]
    top = v00 + (v10 - v00) * fx
    bot = v01 + (v11 - v01) * fx
    out = top + (bot - top) * fy
    # Rust `as u8` truncates toward zero and saturates
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out[:, :, 0] if img.ndim == 2 else out


def _resize_u8(img: np.ndarray, dst_w: int, dst_h: int) -> np.ndarray:
    """Resize with the Rust kernel when available, else its exact NumPy port."""
    try:
        from frigate.util.frame_rs import bilinear_resize_u8_rust, frame_rs_available

        if frame_rs_available():
            return bilinear_resize_u8_rust(img, dst_w, dst_h)
    except (ImportError, OSError, RuntimeError, ValueError):
        pass
    return bilinear_resize_u8_numpy(img, dst_w, dst_h)


def letterbox_face_u8(rgb: np.ndarray, target_size: int) -> np.ndarray:
    """Fit an RGB uint8 face crop into a square, centered, normalized to [-1, 1].

    Returns float32 HWC. Every input (live crops, library images, API
    uploads) goes through this, so they are embedded identically.
    """
    height, width = rgb.shape[:2]
    if width != target_size or height != target_size:
        if width > height:
            new_w = target_size
            new_h = int(((height / width) * target_size) // 4 * 4)
        else:
            new_w = int(((width / height) * target_size) // 4 * 4)
            new_h = target_size
        rgb = _resize_u8(rgb, max(new_w, 1), max(new_h, 1))

    og_h, og_w, channels = rgb.shape
    frame = np.zeros((target_size, target_size, channels), dtype=np.float32)
    x_center = (target_size - og_w) // 2
    y_center = (target_size - og_h) // 2
    frame[y_center : y_center + og_h, x_center : x_center + og_w] = rgb
    return (frame / 127.5) - 1.0


class FaceNetEmbedding(BaseEmbedding):
    def __init__(self):
        GITHUB_ENDPOINT = os.environ.get("GITHUB_ENDPOINT", "https://github.com")
        super().__init__(
            model_name="facedet",
            model_file="facenet.tflite",
            download_urls={
                "facenet.tflite": f"{GITHUB_ENDPOINT}/NickM-27/facenet-onnx/releases/download/v1.0/facenet.tflite",
            },
        )
        self.download_path = os.path.join(MODEL_CACHE_DIR, self.model_name)
        self.tokenizer = None
        self.feature_extractor = None
        self.runner = None
        files_names = list(self.download_urls.keys())

        if not all(
            os.path.exists(os.path.join(self.download_path, n)) for n in files_names
        ):
            logger.debug(f"starting model download for {self.model_name}")
            self.downloader = ModelDownloader(
                model_name=self.model_name,
                download_path=self.download_path,
                file_names=files_names,
                download_func=self._download_model,
            )
            self.downloader.ensure_model_files()
        else:
            self.downloader = None
            self._load_model_and_utils()
            logger.debug(f"models are already downloaded for {self.model_name}")

    def _load_model_and_utils(self):
        if self.runner is None:
            if self.downloader:
                self.downloader.wait_for_download()

            # Suppress TFLite delegate creation messages that bypass Python logging
            with suppress_stderr_during("tflite_interpreter_init"):
                self.runner = Interpreter(
                    model_path=os.path.join(MODEL_CACHE_DIR, "facedet/facenet.tflite"),
                    num_threads=2,
                )
                self.runner.allocate_tensors()
            self.tensor_input_details = self.runner.get_input_details()
            self.tensor_output_details = self.runner.get_output_details()

    def _resize_and_letterbox(self, raw_input, target_size: int) -> np.ndarray:
        return letterbox_face_u8(self._to_rgb_u8(raw_input), target_size)

    def _preprocess_inputs(self, raw_inputs):
        frame = self._resize_and_letterbox(raw_inputs[0], FACENET_INPUT_SIZE)
        return np.expand_dims(frame, axis=0)

    def __call__(self, inputs):
        self._load_model_and_utils()
        processed = self._preprocess_inputs(inputs)
        self.runner.set_tensor(self.tensor_input_details[0]["index"], processed)
        self.runner.invoke()
        return self.runner.get_tensor(self.tensor_output_details[0]["index"])


class ArcfaceEmbedding(BaseEmbedding):
    def __init__(self, config: FaceRecognitionConfig):
        GITHUB_ENDPOINT = os.environ.get("GITHUB_ENDPOINT", "https://github.com")
        super().__init__(
            model_name="facedet",
            model_file="arcface.onnx",
            download_urls={
                "arcface.onnx": f"{GITHUB_ENDPOINT}/NickM-27/facenet-onnx/releases/download/v1.0/arcface.onnx",
            },
        )
        self.config = config
        self.download_path = os.path.join(MODEL_CACHE_DIR, self.model_name)
        self.tokenizer = None
        self.feature_extractor = None
        self.runner = None
        files_names = list(self.download_urls.keys())

        if not all(
            os.path.exists(os.path.join(self.download_path, n)) for n in files_names
        ):
            logger.debug(f"starting model download for {self.model_name}")
            self.downloader = ModelDownloader(
                model_name=self.model_name,
                download_path=self.download_path,
                file_names=files_names,
                download_func=self._download_model,
            )
            self.downloader.ensure_model_files()
        else:
            self.downloader = None
            self._load_model_and_utils()
            logger.debug(f"models are already downloaded for {self.model_name}")

    def _load_model_and_utils(self):
        if self.runner is None:
            if self.downloader:
                self.downloader.wait_for_download()

            self.runner = get_optimized_runner(
                os.path.join(self.download_path, self.model_file),
                device=self.config.device or "GPU",
                model_type=EnrichmentModelTypeEnum.arcface.value,
            )

    def _preprocess_inputs(self, raw_inputs):
        frame = letterbox_face_u8(self._to_rgb_u8(raw_inputs[0]), ARCFACE_INPUT_SIZE)
        frame = np.ascontiguousarray(np.transpose(frame, (2, 0, 1)))
        frame = np.expand_dims(frame, axis=0)
        return [{"data": frame}]
