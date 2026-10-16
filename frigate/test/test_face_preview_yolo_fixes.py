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


if __name__ == "__main__":
    unittest.main()
