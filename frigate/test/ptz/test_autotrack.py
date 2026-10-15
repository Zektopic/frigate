import unittest

from frigate.ptz.autotrack import ptz_moving_at_frame_time


class TestPtzAutotrack(unittest.TestCase):
    def test_ptz_moving_at_frame_time_no_motion(self):
        # ptz_start_time initialized to 0.0 (startup)
        self.assertFalse(ptz_moving_at_frame_time(10.0, 0.0, 0.0))

    def test_ptz_moving_at_frame_time_motion_started_no_stop(self):
        # Motion started (start=5.0), but no stop time recorded yet (stop=0.0)
        self.assertTrue(ptz_moving_at_frame_time(10.0, 5.0, 0.0))

    def test_ptz_moving_at_frame_time_before_start(self):
        # Frame time before motion started
        self.assertFalse(ptz_moving_at_frame_time(3.0, 5.0, 10.0))

    def test_ptz_moving_at_frame_time_exact_start(self):
        # Frame time exactly at start time
        self.assertFalse(ptz_moving_at_frame_time(5.0, 5.0, 10.0))

    def test_ptz_moving_at_frame_time_within_bounds(self):
        # Frame time within start and stop bounds
        self.assertTrue(ptz_moving_at_frame_time(7.0, 5.0, 10.0))

    def test_ptz_moving_at_frame_time_exact_stop(self):
        # Frame time exactly at stop time
        self.assertTrue(ptz_moving_at_frame_time(10.0, 5.0, 10.0))

    def test_ptz_moving_at_frame_time_after_stop(self):
        # Frame time after stop time
        self.assertFalse(ptz_moving_at_frame_time(12.0, 5.0, 10.0))
