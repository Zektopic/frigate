"""Tests for low-resource mode config defaults and shm frame sizing."""

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from frigate.config import FrigateConfig
from frigate.config.low_resource import apply_low_resource_defaults
from frigate.const import SHM_FRAMES_VAR
from frigate.util.services import (
    calculate_shm_requirements,
    get_requested_shm_frame_count,
)


def _minimal(**extra) -> dict:
    config = {
        "mqtt": {"host": "mqtt"},
        "cameras": {
            "back": {
                "ffmpeg": {
                    "inputs": [
                        {"path": "rtsp://10.0.0.1:554/video", "roles": ["detect"]}
                    ]
                },
                "detect": {"height": 720, "width": 1280, "fps": 5},
            }
        },
    }
    config.update(extra)
    return config


class _DiskUsage:
    def __init__(self, total_mb: float) -> None:
        self.total = int(total_mb * 1048576)
        self.used = 0
        self.free = self.total


class TestLowResourceConfig(unittest.TestCase):
    def test_disabled_by_default_keeps_regular_defaults(self):
        config = FrigateConfig(**_minimal())
        self.assertFalse(config.low_resource.enabled)
        self.assertIsNone(config.low_resource.shm_frames)
        self.assertTrue(config.birdseye.enabled)
        self.assertEqual(config.live.height, 720)
        self.assertTrue(config.cameras["back"].birdseye.enabled)
        self.assertEqual(config.cameras["back"].live.height, 720)

    def test_enabled_applies_cheaper_defaults_to_cameras(self):
        config = FrigateConfig(**_minimal(low_resource={"enabled": True}))
        self.assertFalse(config.birdseye.enabled)
        self.assertEqual(config.live.height, 360)
        self.assertFalse(config.cameras["back"].birdseye.enabled)
        self.assertEqual(config.cameras["back"].live.height, 360)
        # detection settings are never touched
        self.assertEqual(config.cameras["back"].detect.fps, 5)
        self.assertEqual(config.cameras["back"].detect.height, 720)

    def test_explicit_global_values_win(self):
        config = FrigateConfig(
            **_minimal(
                low_resource={"enabled": True},
                birdseye={"enabled": True, "mode": "motion"},
                live={"height": 480},
            )
        )
        self.assertTrue(config.birdseye.enabled)
        self.assertEqual(config.live.height, 480)
        self.assertEqual(config.cameras["back"].live.height, 480)

    def test_explicit_camera_values_win(self):
        raw = _minimal(low_resource={"enabled": True})
        raw["cameras"]["back"]["live"] = {"height": 540}
        config = FrigateConfig(**raw)
        self.assertEqual(config.live.height, 360)
        self.assertEqual(config.cameras["back"].live.height, 540)

    def test_apply_does_not_mutate_input(self):
        raw = _minimal(low_resource={"enabled": True}, live={"quality": 10})
        result = apply_low_resource_defaults(raw)
        self.assertNotIn("birdseye", raw)
        self.assertEqual(raw["live"], {"quality": 10})
        self.assertEqual(result["live"], {"quality": 10, "height": 360})
        self.assertEqual(result["birdseye"], {"enabled": False})

    def test_apply_ignores_non_dict_and_disabled(self):
        self.assertIsNone(apply_low_resource_defaults(None))
        raw = _minimal(low_resource={"enabled": False})
        self.assertIs(apply_low_resource_defaults(raw), raw)

    def test_shm_frames_lower_bound(self):
        with self.assertRaises(ValueError):
            FrigateConfig(**_minimal(low_resource={"shm_frames": 5}))


class TestPotatoExampleConfig(unittest.TestCase):
    EXAMPLE = Path(__file__).resolve().parents[2] / "config/potato.yml.example"

    @unittest.skipUnless(EXAMPLE.exists(), "example config not available")
    def test_example_config_parses(self):
        with open(self.EXAMPLE) as f:
            config = FrigateConfig.parse_yaml(f.read())

        camera = config.cameras["front"]
        self.assertTrue(config.low_resource.enabled)
        self.assertFalse(config.birdseye.enabled)
        self.assertEqual(camera.live.height, 360)
        self.assertEqual(camera.detect.width, 640)
        self.assertEqual(config.model.width, 320)


class TestShmFrameCount(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.pop(SHM_FRAMES_VAR, None)

    def tearDown(self):
        os.environ.pop(SHM_FRAMES_VAR, None)
        if self._env is not None:
            os.environ[SHM_FRAMES_VAR] = self._env

    def test_requested_defaults(self):
        self.assertEqual(get_requested_shm_frame_count(FrigateConfig(**_minimal())), 50)
        self.assertEqual(
            get_requested_shm_frame_count(
                FrigateConfig(**_minimal(low_resource={"enabled": True}))
            ),
            20,
        )

    def test_env_var_and_config_precedence(self):
        os.environ[SHM_FRAMES_VAR] = "30"
        self.assertEqual(
            get_requested_shm_frame_count(
                FrigateConfig(**_minimal(low_resource={"enabled": True}))
            ),
            30,
        )
        self.assertEqual(
            get_requested_shm_frame_count(
                FrigateConfig(**_minimal(low_resource={"shm_frames": 12}))
            ),
            12,
        )

    def test_invalid_env_var_falls_back(self):
        os.environ[SHM_FRAMES_VAR] = "lots"
        self.assertEqual(get_requested_shm_frame_count(FrigateConfig(**_minimal())), 50)

    def test_shm_budget_with_large_shm(self):
        with (
            patch(
                "frigate.util.services.shutil.disk_usage",
                return_value=_DiskUsage(4096),
            ),
            patch("frigate.util.services.get_fs_type", return_value="tmpfs"),
        ):
            normal = calculate_shm_requirements(FrigateConfig(**_minimal()))
            potato = calculate_shm_requirements(
                FrigateConfig(**_minimal(low_resource={"enabled": True}))
            )

        self.assertEqual(normal["shm_frame_count"], 50)
        self.assertEqual(potato["shm_frame_count"], 20)
        self.assertEqual(potato["requested_frame_count"], 20)

    def test_shm_budget_capped_by_small_shm(self):
        # 1 camera at 720p (1.6 MB/frame incl. 2 reserve 720p cams => 4.8 MB)
        with (
            patch(
                "frigate.util.services.shutil.disk_usage",
                return_value=_DiskUsage(128),
            ),
            patch("frigate.util.services.get_fs_type", return_value="tmpfs"),
        ):
            stats = calculate_shm_requirements(FrigateConfig(**_minimal()))

        self.assertLess(stats["shm_frame_count"], 50)
        self.assertEqual(stats["requested_frame_count"], 50)


class TestShmWarning(unittest.TestCase):
    def _count(self, stats: dict) -> int:
        from frigate.camera.maintainer import CameraMaintainer

        maintainer = CameraMaintainer.__new__(CameraMaintainer)
        maintainer.config = MagicMock()
        with (
            patch(
                "frigate.camera.maintainer.calculate_shm_requirements",
                return_value=stats,
            ),
            self.assertLogs("frigate.camera.maintainer", level="DEBUG") as logs,
        ):
            count = maintainer._CameraMaintainer__calculate_shm_frame_count()
        self.warnings = [r for r in logs.records if r.levelname == "WARNING"]
        return count

    def _stats(self, count: int, requested: int) -> dict:
        return {
            "total": 256,
            "available": 200,
            "camera_frame_size": 4.8,
            "shm_frame_count": count,
            "requested_frame_count": requested,
            "min_shm": 150,
        }

    def test_no_warning_for_deliberately_small_ring(self):
        self.assertEqual(self._count(self._stats(12, 12)), 12)
        self.assertEqual(self.warnings, [])

    def test_warning_when_shm_caps_request(self):
        self.assertEqual(self._count(self._stats(15, 50)), 15)
        self.assertEqual(len(self.warnings), 1)


if __name__ == "__main__":
    unittest.main()
