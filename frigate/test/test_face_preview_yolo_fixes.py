"""Tests for face bounding-box clamping, empty-crop guards, preview directory creation, and Rust YOLO shape handling."""

import ctypes
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


class TestFaceAndPreviewFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Ensure heavy runtime modules that may be absent on host are mocked
        cls._mocked: dict[str, object] = {}
        for mod_name in [
            "frigate.config",
            "frigate.comms.embeddings_updater",
            "frigate.comms.event_metadata_updater",
            "frigate.comms.inter_process",
            "frigate.embeddings",
            "frigate.embeddings.onnx",
            "frigate.embeddings.onnx.face_embedding",
            "frigate.ffmpeg_presets",
            "frigate.models",
            "frigate.log",
            "frigate.types",
            "frigate.data_processing.types",
            "scipy",
            "scipy.stats",
        ]:
            if mod_name not in sys.modules:
                cls._mocked[mod_name] = MagicMock()

        # Decorator passthrough for redirect_output_to_logger
        if "frigate.log" in cls._mocked:
            cls._mocked["frigate.log"].redirect_output_to_logger = (  # type: ignore[attr-defined]
                lambda *a, **kw: lambda fn: fn
            )

        cls._patcher = patch.dict("sys.modules", cls._mocked)
        cls._patcher.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._patcher.stop()

    def test_detect_face_negative_coords_do_not_underflow_uint16(self) -> None:
        """Negative x/y from YuNet must clamp to 0 instead of wrapping to ~65535 via uint16."""
        from frigate.data_processing.real_time.face import FaceRealTimeProcessor

        proc = FaceRealTimeProcessor.__new__(FaceRealTimeProcessor)
        proc.face_detector = MagicMock()
        det_row = [-3.5, -2.2, 64.0, 80.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.92]
        proc.face_detector.detect.return_value = (1, [det_row])

        dummy_person = MagicMock()
        dummy_person.shape = (200, 150, 3)
        bbox = proc._FaceRealTimeProcessor__detect_face(dummy_person, threshold=0.5)
        self.assertEqual(bbox, (0, 0, 64, 80))

    def test_detect_face_skips_non_positive_dimensions(self) -> None:
        """Degenerate zero/negative width or height detections are ignored."""
        from frigate.data_processing.real_time.face import FaceRealTimeProcessor

        proc = FaceRealTimeProcessor.__new__(FaceRealTimeProcessor)
        proc.face_detector = MagicMock()
        det_row = [10.0, 10.0, 0.0, 50.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.95]
        proc.face_detector.detect.return_value = (1, [det_row])

        dummy_person = MagicMock()
        dummy_person.shape = (200, 150, 3)
        bbox = proc._FaceRealTimeProcessor__detect_face(dummy_person, threshold=0.5)
        self.assertIsNone(bbox)

    def test_blur_confidence_reduction_handles_empty_crop(self) -> None:
        """Empty face crops must return 0.0 instead of raising OpenCV Laplacian assertion."""
        from frigate.data_processing.common.face.model import FaceRecognizer

        class DummyRecognizer(FaceRecognizer):
            def build(self) -> None:
                pass

            def clear(self) -> None:
                pass

            def classify(self, face_image: np.ndarray) -> tuple[str, float] | None:
                return None

        cfg = MagicMock()
        cfg.face_recognition.blur_confidence_filter = True
        with patch.object(DummyRecognizer, "init_landmark_detector"):
            rec = DummyRecognizer(cfg)

        empty_crop = MagicMock()
        empty_crop.size = 0
        self.assertEqual(rec.get_blur_confidence_reduction(empty_crop), 0.0)

    def test_preview_write_data_respects_record_disabled(self) -> None:
        """PreviewRecorder.write_data must not buffer frames or spawn ffmpeg when record.enabled is False."""
        from frigate.output.preview import PreviewRecorder

        rec = PreviewRecorder.__new__(PreviewRecorder)
        rec.config = MagicMock()
        rec.config.name = "work_room"
        rec.config.record.enabled = False
        rec.camera_name = "work_room"
        rec.start_time = 0
        rec.last_output_time = 0
        rec.offline = False
        rec.output_frames = []
        rec.segment_end = 1000.0
        rec.write_frame_to_cache = MagicMock()

        dummy_frame = MagicMock()
        # First frame when start_time == 0
        rec.write_data([], [], 900.0, dummy_frame)
        self.assertEqual(rec.output_frames, [])
        rec.write_frame_to_cache.assert_not_called()

        # Segment end reached with empty output_frames
        with patch("frigate.output.preview.FFMpegConverter") as mock_conv:
            rec.write_data([], [], 1001.0, dummy_frame)
            mock_conv.assert_not_called()

    def test_ffmpeg_converter_recreates_missing_preview_parent_dir(self) -> None:
        """FFMpegConverter.run must recreate the camera preview directory if cleanup removed it."""
        from frigate.output.preview import FFMpegConverter

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "previews" / "buddha_statue"
            out_path = nested_dir / "100.0-200.0.mp4"
            self.assertFalse(nested_dir.exists())

            conv = FFMpegConverter.__new__(FFMpegConverter)
            conv.config = MagicMock()
            conv.config.name = "buddha_statue"
            conv.frame_times = [100.0, 200.0]
            conv.requestor = MagicMock()
            conv.path = str(out_path)
            conv.ffmpeg_cmd = "ffmpeg -version"

            with patch("frigate.output.preview.sp.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr="")
                conv.run()

            self.assertTrue(nested_dir.is_dir())

    def test_rust_yolo_post_process_extracts_ny_nx_before_ravel(self) -> None:
        """yolo_post_process must read 4D/3D spatial dimensions before flattening outputs to 1D."""
        from frigate.detectors import rust_yolo

        fake_lib = MagicMock()
        captured_ny_nx: list[int] = []

        def fake_post_process(
            output_ptrs, ny_nx_ptr, w, h, score_t, iou_t, dets_ptr, max_dets
        ):
            arr = ctypes.cast(ny_nx_ptr, ctypes.POINTER(ctypes.c_uint32 * 6)).contents
            captured_ny_nx.extend(arr[:])
            return 0

        fake_lib.yolo_post_process.side_effect = fake_post_process

        outputs = []
        mocked_np = not isinstance(getattr(np.zeros((1,)), "ndim", None), int)
        for shape in [(1, 255, 80, 80), (1, 255, 40, 40), (255, 20, 20)]:
            if mocked_np:
                arr = MagicMock()
                arr.shape = shape
                arr.ndim = len(shape)
                arr.ravel.return_value.astype.return_value = arr
            else:
                arr = np.zeros(shape, dtype=np.float32)
            outputs.append(arr)

        if mocked_np:
            dummy_buf = (ctypes.c_float * 1)(0.0)
            dummy_ptr = ctypes.cast(dummy_buf, ctypes.POINTER(ctypes.c_float))
            flat_mock = MagicMock()
            flat_mock.ctypes.data_as.return_value = dummy_ptr
            rust_yolo.np.ascontiguousarray.return_value = flat_mock

        with patch.object(rust_yolo, "_load_lib", return_value=fake_lib):
            rust_yolo.yolo_post_process(outputs, 640, 640)

        self.assertEqual(captured_ny_nx, [80, 80, 40, 40, 20, 20])


class TestRustTier1AndTier2Parity(unittest.TestCase):
    """Parity and regression tests for Tier 1 and Tier 2 Rust kernels in frame_rs and rust_yolo."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._mocked: dict[str, object] = {}
        for mod_name in [
            "onnxruntime",
            "py3nvml",
            "py3nvml.py3nvml",
            "zmq",
            "frigate.version",
            "frigate.ffmpeg_presets",
            "frigate.comms.detections_updater",
            "frigate.comms.inter_process",
            "frigate.comms.object_detector_signaler",
            "frigate.config",
            "frigate.util.process",
        ]:
            if mod_name not in sys.modules:
                cls._mocked[mod_name] = MagicMock()
        cls._patcher = patch.dict("sys.modules", cls._mocked)
        cls._patcher.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._patcher.stop()

    def test_rust_libraries_loaded(self) -> None:
        import importlib.util

        from frigate.detectors.rust_yolo import yolo_available
        from frigate.util.frame_rs import frame_rs_available

        spec = importlib.util.spec_from_file_location(
            "rust_engine_direct",
            str(Path(__file__).resolve().parents[1] / "motion" / "rust_engine.py"),
        )
        assert spec and spec.loader
        rust_engine = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rust_engine)

        self.assertTrue(frame_rs_available(), "libfrigate_frame_rs.so should load")
        self.assertTrue(yolo_available(), "libfrigate_yolo_rs.so should load")
        self.assertTrue(
            rust_engine.motion_available(), "libfrigate_motion_rs.so should load"
        )

    def test_transform_detect_input_and_normalize_parity(self) -> None:
        from frigate.detectors.detector_config import InputDTypeEnum
        from frigate.object_detection.base import BaseLocalDetector
        from frigate.util.frame_rs import (
            normalize_u8_to_f32_rust,
            transform_detect_input_rust,
        )

        rng = np.random.default_rng(123)
        img_4d = rng.integers(0, 256, size=(1, 320, 320, 3), dtype=np.uint8)
        img_3d = img_4d[0]

        # 1. Direct kernel parity: 4D (1, H, W, C) -> (1, C, H, W) float32 0..1
        out_nchw_4d = transform_detect_input_rust(img_4d)
        ref_nchw_4d = img_4d.transpose((0, 3, 1, 2)).astype(np.float32) / 255.0
        np.testing.assert_allclose(out_nchw_4d, ref_nchw_4d, rtol=1e-6, atol=1e-6)

        # 2. Direct kernel parity: 3D (H, W, C) -> (C, H, W) float32 0..1
        out_nchw_3d = transform_detect_input_rust(img_3d)
        ref_nchw_3d = img_3d.transpose((2, 0, 1)).astype(np.float32) / 255.0
        np.testing.assert_allclose(out_nchw_3d, ref_nchw_3d, rtol=1e-6, atol=1e-6)

        # 3. Direct AVX2 SIMD normalize_u8_to_f32_rust parity
        out_nhwc_01 = normalize_u8_to_f32_rust(img_4d)
        ref_nhwc_01 = img_4d.astype(np.float32) / 255.0
        np.testing.assert_allclose(out_nhwc_01, ref_nhwc_01, rtol=1e-6, atol=1e-6)

        # 4. BaseLocalDetector._transform_input parity across transforms and dtypes
        for input_transform in ((0, 3, 1, 2), None):
            for dtype_enum in (
                InputDTypeEnum.float,
                InputDTypeEnum.float_denorm,
                InputDTypeEnum.int,
            ):
                det = BaseLocalDetector.__new__(BaseLocalDetector)
                det.input_transform = input_transform
                det.dtype = dtype_enum
                rust_out = det._transform_input(img_4d.copy())
                with patch(
                    "frigate.util.frame_rs.frame_rs_available", return_value=False
                ):
                    py_out = det._transform_input(img_4d.copy())
                np.testing.assert_allclose(rust_out, py_out, rtol=1e-6, atol=1e-6)

    def test_yuv_to_3_channel_and_crop_parity(self) -> None:
        from frigate.util.image import yuv_crop_and_resize, yuv_to_3_channel_yuv

        rng = np.random.default_rng(456)
        h, w = 240, 320
        yuv = rng.integers(0, 256, size=(h * 3 // 2, w), dtype=np.uint8)

        # 1. yuv_to_3_channel_yuv exact byte parity
        rust_3ch = yuv_to_3_channel_yuv(yuv)
        with patch("frigate.util.frame_rs.frame_rs_available", return_value=False):
            py_3ch = yuv_to_3_channel_yuv(yuv)
        np.testing.assert_array_equal(rust_3ch, py_3ch)

        # 2. yuv_crop_and_resize in-bounds 4-pixel-aligned square crop (Rust fast path)
        rust_crop = yuv_crop_and_resize(yuv, (20, 32, 180, 192), height=80)
        with patch("frigate.util.frame_rs.frame_rs_available", return_value=False):
            py_crop = yuv_crop_and_resize(yuv, (20, 32, 180, 192), height=80)
        np.testing.assert_array_equal(rust_crop, py_crop)

        # 3. yuv_crop_and_resize unaligned y%4==2 crop (falls back to Python get_yuv_crop)
        rust_unaligned = yuv_crop_and_resize(yuv, (20, 30, 180, 190), height=80)
        with patch("frigate.util.frame_rs.frame_rs_available", return_value=False):
            py_unaligned = yuv_crop_and_resize(yuv, (20, 30, 180, 190), height=80)
        np.testing.assert_array_equal(rust_unaligned, py_unaligned)

        # 4. yuv_crop_and_resize out-of-bounds crop (falls back to Python padding)
        rust_oob = yuv_crop_and_resize(yuv, (200, 100, 360, 260), height=80)
        with patch("frigate.util.frame_rs.frame_rs_available", return_value=False):
            py_oob = yuv_crop_and_resize(yuv, (200, 100, 360, 260), height=80)
        np.testing.assert_array_equal(rust_oob, py_oob)

    def test_yolo_multipart_nchw_parity(self) -> None:
        from frigate.util import model as model_mod

        s0 = np.zeros((1, 255, 80, 80), dtype=np.float32)
        s1 = np.zeros((1, 255, 40, 40), dtype=np.float32)
        s2 = np.zeros((1, 255, 20, 20), dtype=np.float32)

        # Scale 0, anchor 0, y=10, x=20, person (class 0)
        s0[0, 0, 10, 20] = 0.5
        s0[0, 1, 10, 20] = 0.5
        s0[0, 2, 10, 20] = 0.6
        s0[0, 3, 10, 20] = 0.7
        s0[0, 4, 10, 20] = 0.92
        s0[0, 5 + 0, 10, 20] = 0.95

        # Scale 1, anchor 2, y=15, x=25, car (class 2)
        base_ch = 2 * 85
        s1[0, base_ch + 0, 15, 25] = 0.4
        s1[0, base_ch + 1, 15, 25] = 0.6
        s1[0, base_ch + 2, 15, 25] = 0.8
        s1[0, base_ch + 3, 15, 25] = 0.5
        s1[0, base_ch + 4, 15, 25] = 0.88
        s1[0, base_ch + 5 + 2, 15, 25] = 0.90

        rust_dets = model_mod.post_process_yolo([s0, s1, s2], 640, 640)
        py_multipart = getattr(model_mod, "__post_process_multipart_yolo")
        py_dets = py_multipart([s0, s1, s2], 640, 640)
        np.testing.assert_allclose(rust_dets, py_dets, rtol=1e-5, atol=1e-6)
        self.assertEqual(int(rust_dets[0, 0]), 0)
        self.assertEqual(int(rust_dets[1, 0]), 2)

    def test_yolo26_and_anchor_free_and_nms_post_process(self) -> None:
        from frigate.util.model import (
            post_process_rfdetr,
            post_process_yolo,
            post_process_yolox,
        )

        # 1. Single-scale 84-ch in both (1, 84, N) and (1, N, 84) layouts
        raw_84_n = np.zeros((1, 84, 100), dtype=np.float32)
        raw_84_n[0, 0:4, 7] = [320.0, 240.0, 100.0, 160.0]
        raw_84_n[0, 4 + 0, 7] = 0.91
        dets_84_n = post_process_yolo([raw_84_n], 640, 640)
        dets_n_84 = post_process_yolo([raw_84_n.transpose(0, 2, 1)], 640, 640)
        np.testing.assert_allclose(dets_84_n, dets_n_84, rtol=1e-6, atol=1e-6)
        self.assertAlmostEqual(float(dets_84_n[0, 1]), 0.91, places=5)
        np.testing.assert_allclose(
            dets_84_n[0, 2:6],
            [160.0 / 640.0, 270.0 / 640.0, 320.0 / 640.0, 370.0 / 640.0],
            rtol=1e-5,
        )

        # 2. Anchor-free 144-ch DFL in both (1, 8400, 144) and 3-scale list layouts
        af_flat = np.full((8400, 144), -10.0, dtype=np.float32)
        # Put high logit at bin 4 for all 4 sides and class 0 at cell 100
        for side in range(4):
            af_flat[100, side * 16 + 4] = 10.0
        af_flat[100, 64 + 0] = 5.0
        dets_af_single = post_process_yolo([af_flat[np.newaxis, ...]], 640, 640)
        self.assertGreater(float(dets_af_single[0, 1]), 0.99)

        # Split into 3 scales (80x80=6400, 40x40=1600, 20x20=400) in NCHW (1, 144, H, W)
        s80 = af_flat[:6400].T.reshape(1, 144, 80, 80)
        s40 = af_flat[6400:8000].T.reshape(1, 144, 40, 40)
        s20 = af_flat[8000:].T.reshape(1, 144, 20, 20)
        dets_af_multi = post_process_yolo([s80, s40, s20], 640, 640)
        np.testing.assert_allclose(dets_af_single, dets_af_multi, rtol=1e-5, atol=1e-5)

        # 3. YOLOX and RF-DETR NMS via Rust nms_boxes
        yolox_preds = np.zeros((1, 10, 85), dtype=np.float32)
        yolox_preds[0, 0, :4] = [10.0, 10.0, np.log(5.0), np.log(5.0)]
        yolox_preds[0, 0, 4] = 0.95
        yolox_preds[0, 0, 5] = 0.95
        grids = np.zeros((1, 10, 2), dtype=np.float32)
        strides = np.full((1, 10, 1), 8.0, dtype=np.float32)
        yolox_dets = post_process_yolox(yolox_preds.copy(), 640, 640, grids, strides)
        self.assertGreater(float(yolox_dets[0, 1]), 0.8)

        rf_boxes = np.zeros((1, 10, 4), dtype=np.float32)
        rf_boxes[0, 0] = [0.5, 0.5, 0.2, 0.2]
        rf_scores = np.full((1, 10, 91), -10.0, dtype=np.float32)
        rf_scores[0, 0, 1] = 10.0
        rf_dets = post_process_rfdetr((rf_boxes, rf_scores))
        self.assertGreater(float(rf_dets[0, 1]), 0.99)


if __name__ == "__main__":
    unittest.main()
