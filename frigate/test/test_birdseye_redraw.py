"""Tests for Birdseye only redrawing cameras whose frame changed."""

import multiprocessing as mp
import unittest
from unittest.mock import patch

import numpy as np

from frigate.config import FrigateConfig
from frigate.output.birdseye import BirdsEyeFrameManager

CAMERAS = ["cam_a", "cam_b", "cam_c"]


def _make_manager() -> BirdsEyeFrameManager:
    config = FrigateConfig(
        **{
            "mqtt": {"host": "mqtt"},
            "birdseye": {"enabled": True, "mode": "continuous"},
            "cameras": {
                name: {
                    "ffmpeg": {
                        "inputs": [
                            {"path": f"rtsp://10.0.0.{i}:554/v", "roles": ["detect"]}
                        ]
                    },
                    "detect": {"height": 360, "width": 640, "fps": 5},
                }
                for i, name in enumerate(CAMERAS)
            },
        }
    )
    return BirdsEyeFrameManager(config, mp.Event())


class TestBirdseyeRedraw(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = _make_manager()
        rng = np.random.default_rng(3)
        self.frames = [
            rng.integers(0, 256, size=(360 * 3 // 2, 640), dtype=np.uint8)
            for _ in range(4)
        ]
        self.frame_time = 1000.0

    def _feed(self, camera: str, frame: np.ndarray) -> None:
        # bypass the 10 fps output cap so every call composes the canvas
        self.manager.last_output_time = 0
        self.manager.update(camera, 1, 1, self.frame_time, frame)
        self.frame_time += 0.05

    def test_unchanged_cameras_are_not_redrawn(self) -> None:
        for i, camera in enumerate(CAMERAS * 2):
            self._feed(camera, self.frames[i % 4])
        self.assertEqual(len(self.manager.active_cameras), len(CAMERAS))

        with patch.object(
            self.manager,
            "copy_to_position",
            wraps=self.manager.copy_to_position,
        ) as copy:
            self._feed("cam_b", self.frames[3])

        self.assertEqual([c.args[1] for c in copy.call_args_list], ["cam_b"])

    def test_incremental_canvas_matches_full_redraw(self) -> None:
        for i in range(20):
            self._feed(CAMERAS[i % len(CAMERAS)], self.frames[(i * 7) % 4])
        incremental = self.manager.frame.copy()

        # repaint every tile from scratch
        self.manager.clear_frame()
        self.manager.update_frame()
        np.testing.assert_array_equal(self.manager.frame, incremental)

    def test_layout_reset_redraws_every_camera(self) -> None:
        for i, camera in enumerate(CAMERAS):
            self._feed(camera, self.frames[i])

        self.manager.clear_frame()
        with patch.object(
            self.manager,
            "copy_to_position",
            wraps=self.manager.copy_to_position,
        ) as copy:
            self.manager.update_frame()

        self.assertEqual(
            sorted(c.args[1] for c in copy.call_args_list), sorted(CAMERAS)
        )


if __name__ == "__main__":
    unittest.main()
