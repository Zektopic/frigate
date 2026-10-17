"""Tests for the detector handling in FrigateWatchdog."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from frigate.watchdog import (
    DETECTOR_RESTART_WINDOW_S,
    MAX_DETECTOR_RESTARTS,
    FrigateWatchdog,
)


def _dead_detector(exitcode: int = -11) -> SimpleNamespace:
    process = MagicMock()
    process.is_alive.return_value = False
    process.exitcode = exitcode
    return SimpleNamespace(
        detect_process=process,
        detection_start=SimpleNamespace(value=0.0),
        start_or_restart=MagicMock(),
    )


class TestWatchdogDeadDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = _dead_detector()
        self.watchdog = FrigateWatchdog({"gpu": self.detector}, MagicMock())

    @patch("frigate.watchdog.restart_frigate")
    def test_dead_detector_is_restarted_in_place(self, restart_frigate):
        self.watchdog._handle_dead_detector("gpu", self.detector, 1000.0)

        self.detector.start_or_restart.assert_called_once()
        restart_frigate.assert_not_called()

    @patch("frigate.watchdog.restart_frigate")
    def test_repeated_deaths_escalate_to_frigate_restart(self, restart_frigate):
        for i in range(MAX_DETECTOR_RESTARTS):
            self.watchdog._handle_dead_detector("gpu", self.detector, 1000.0 + i)

        restart_frigate.assert_not_called()

        self.watchdog._handle_dead_detector("gpu", self.detector, 1010.0)

        restart_frigate.assert_called_once()
        self.assertEqual(
            self.detector.start_or_restart.call_count, MAX_DETECTOR_RESTARTS
        )

    @patch("frigate.watchdog.restart_frigate")
    def test_old_deaths_expire_from_the_window(self, restart_frigate):
        for i in range(MAX_DETECTOR_RESTARTS):
            self.watchdog._handle_dead_detector("gpu", self.detector, 1000.0 + i)

        later = 1000.0 + MAX_DETECTOR_RESTARTS + DETECTOR_RESTART_WINDOW_S + 1
        self.watchdog._handle_dead_detector("gpu", self.detector, later)

        restart_frigate.assert_not_called()
        self.assertEqual(
            self.detector.start_or_restart.call_count, MAX_DETECTOR_RESTARTS + 1
        )

    @patch("frigate.watchdog.restart_frigate")
    def test_failed_restart_does_not_raise(self, restart_frigate):
        self.detector.start_or_restart.side_effect = OSError("spawn failed")

        self.watchdog._handle_dead_detector("gpu", self.detector, 1000.0)

        restart_frigate.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
