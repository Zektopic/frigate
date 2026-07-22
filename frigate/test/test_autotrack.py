import unittest
from unittest.mock import MagicMock, patch

class TestPTZMovingAtFrameTime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_modules = {
            "cv2": MagicMock(),
            "numpy": MagicMock(),
            "norfair": MagicMock(),
            "norfair.camera_motion": MagicMock(),
            "zmq": MagicMock(),
        }
        cls.patcher = patch.dict("sys.modules", cls.mock_modules)
        cls.patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()

    def test_ptz_not_started(self):
        """PTZ has not started moving."""
        from frigate.ptz.autotrack import ptz_moving_at_frame_time
        self.assertFalse(ptz_moving_at_frame_time(10.0, 0.0, 0.0))

    def test_frame_before_start_time(self):
        """Frame occurs before PTZ start time."""
        from frigate.ptz.autotrack import ptz_moving_at_frame_time
        self.assertFalse(ptz_moving_at_frame_time(5.0, 10.0, 0.0))

    def test_frame_at_start_time(self):
        """Frame occurs exactly at PTZ start time."""
        from frigate.ptz.autotrack import ptz_moving_at_frame_time
        self.assertFalse(ptz_moving_at_frame_time(10.0, 10.0, 0.0))

    def test_frame_after_start_no_stop(self):
        """Frame occurs after PTZ start time, no stop time set."""
        from frigate.ptz.autotrack import ptz_moving_at_frame_time
        self.assertTrue(ptz_moving_at_frame_time(15.0, 10.0, 0.0))

    def test_frame_between_start_and_stop(self):
        """Frame occurs between PTZ start and stop times."""
        from frigate.ptz.autotrack import ptz_moving_at_frame_time
        self.assertTrue(ptz_moving_at_frame_time(15.0, 10.0, 20.0))

    def test_frame_at_stop_time(self):
        """Frame occurs exactly at PTZ stop time."""
        from frigate.ptz.autotrack import ptz_moving_at_frame_time
        self.assertTrue(ptz_moving_at_frame_time(20.0, 10.0, 20.0))

    def test_frame_after_stop_time(self):
        """Frame occurs after PTZ stop time."""
        from frigate.ptz.autotrack import ptz_moving_at_frame_time
        self.assertFalse(ptz_moving_at_frame_time(25.0, 10.0, 20.0))


if __name__ == "__main__":
    unittest.main()
