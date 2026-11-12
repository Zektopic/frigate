"""Tests for the on-demand jsmpeg encoder in frigate.output.camera."""

import threading
import unittest
from unittest.mock import MagicMock, patch

from frigate.output.camera import JsmpegCamera


def _camera_config() -> MagicMock:
    config = MagicMock()
    config.name = "front"
    config.detect.fps = 5
    config.frame_shape = (720, 1280)
    config.live.height = 360
    config.live.quality = 8
    return config


class TestJsmpegLazyEncoder(unittest.TestCase):
    def setUp(self):
        self.stop_event = threading.Event()
        self.converter_patch = patch("frigate.output.camera.FFMpegConverter")
        self.broadcaster_patch = patch("frigate.output.camera.BroadcastThread")
        self.converter_cls = self.converter_patch.start()
        self.broadcaster_cls = self.broadcaster_patch.start()
        self.converter_cls.return_value.process.poll.return_value = None
        self.addCleanup(self.converter_patch.stop)
        self.addCleanup(self.broadcaster_patch.stop)

    def _camera(self) -> JsmpegCamera:
        return JsmpegCamera(_camera_config(), MagicMock(), self.stop_event, None)

    def test_no_encoder_until_first_frame(self):
        camera = self._camera()
        self.converter_cls.assert_not_called()
        self.assertFalse(camera.running)
        camera.stop()  # must be safe without a session

    def test_first_frame_starts_encoder_once(self):
        camera = self._camera()
        camera.write_frame(b"a", now=0.0)
        camera.write_frame(b"b", now=1.0)
        self.assertEqual(self.converter_cls.call_count, 1)
        self.converter_cls.return_value.start.assert_called_once()
        self.broadcaster_cls.return_value.start.assert_called_once()
        # output size follows live.height and the frame aspect ratio
        args = self.converter_cls.call_args.args
        self.assertEqual(args[6:9], (640, 360, 8))
        self.assertTrue(camera.running)

    def test_idle_encoder_stops_and_restarts(self):
        camera = self._camera()
        camera.write_frame(b"a", now=0.0)
        session_stop = self.converter_cls.call_args.args[3]

        camera.check_idle(now=camera.idle_timeout - 1)
        self.assertFalse(session_stop.is_set())

        camera.check_idle(now=camera.idle_timeout + 1)
        self.assertTrue(session_stop.is_set())
        self.assertFalse(camera.running)
        # the process-wide stop event is left alone
        self.assertFalse(self.stop_event.is_set())

        camera.write_frame(b"b", now=camera.idle_timeout + 2)
        self.assertEqual(self.converter_cls.call_count, 2)
        self.assertTrue(camera.running)

    def test_global_stop_reaches_session(self):
        camera = self._camera()
        camera.write_frame(b"a", now=0.0)
        session_stop = self.converter_cls.call_args.args[3]
        self.assertFalse(session_stop.is_set())
        self.stop_event.set()
        self.assertTrue(session_stop.is_set())

    def test_crashed_encoder_is_restarted(self):
        camera = self._camera()
        camera.write_frame(b"a", now=0.0)
        self.converter_cls.return_value.process.poll.return_value = 1
        self.assertFalse(camera.running)
        camera.write_frame(b"b", now=1.0)
        self.assertEqual(self.converter_cls.call_count, 2)

    def test_shutdown_stops_session(self):
        camera = self._camera()
        camera.write_frame(b"a", now=0.0)
        camera.shutdown()
        self.assertFalse(camera.running)


if __name__ == "__main__":
    unittest.main()
