"""Tests for the resource monitor: the Rust collector, its conversion to
Frigate's stats layout, and the psutil fallback."""

import asyncio
import ctypes
import os
import subprocess
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

from frigate.stats import rust_stats
from frigate.stats.util import set_cpu_stats, set_gpu_stats, set_npu_usages
from frigate.util import services
from frigate.util.services import get_cpu_stats

BURN = "import time\nwhile True:\n    pass\n"


def native_doc(**overrides):
    """A collection as frigate-stats-rs returns it."""
    doc = {
        "version": 1,
        "system": {"cpu": 12.5, "mem": 40.0, "mem_total_kb": 1000, "cpus": 4},
        "processes": {
            "10": {
                "ppid": 1,
                "name": "python3",
                "cmdline": "python3 -u -m frigate",
                "cpu": 5.3,
                "cpu_average": 4.1,
                "mem": 0.6,
                "tree_cpu": 40.0,
                "tree_mem": 3.0,
            },
            "11": {
                "ppid": 10,
                "name": "python3",
                "cmdline": "frigate.detector:ncnn_vulkan",
                "cpu": 1.5,
                "cpu_average": 1.0,
                "mem": 0.1,
                "tree_cpu": 33.3,
                "tree_mem": 0.3,
            },
            "12": {
                "ppid": 11,
                "name": "frigate-detecto",
                "cmdline": "/opt/frigate/frigate-detector-rs m.param m.bin",
                "cpu": 31.8,
                "cpu_average": 8.5,
                "mem": 0.2,
                "tree_cpu": 31.8,
                "tree_mem": 0.2,
            },
            "20": {
                "ppid": 10,
                "name": "ffmpeg",
                "cmdline": "ffmpeg -i rtsp://admin:hunter2@192.168.1.5/stream",
                "cpu": 3.0,
                "cpu_average": 2.0,
                "mem": 0.3,
                "tree_cpu": 3.0,
                "tree_mem": 0.3,
            },
        },
        "gpus": [],
        "npus": [],
    }
    doc.update(overrides)
    return doc


def amd_gpu(**overrides):
    gpu = {
        "card": "card1",
        "pdev": "0000:0f:00.0",
        "driver": "amdgpu",
        "node_present": True,
        "busy": 66.8,
        "busy_samples": 14,
        "vram_used": 536870912,
        "vram_total": 2147483648,
        "temp": 57.6,
        "power": 44.14,
        "has_decode_engine": False,
        "engines": {"gfx": 4.0, "enc": 12.5, "compute": 30.0},
        "clients": {
            "12": {"vram_kb": 100, "gtt_kb": 0, "engines": {"compute": 30.0}},
            "20": {"vram_kb": 50, "gtt_kb": 0, "engines": {"gfx": 4.0, "enc": 12.5}},
            "30": {"vram_kb": 10, "gtt_kb": 0, "engines": {}},
        },
    }
    gpu.update(overrides)
    return gpu


class TestCpuUsages(unittest.TestCase):
    def test_layout_matches_the_psutil_collector(self):
        usages = rust_stats.cpu_usages(native_doc())
        self.assertEqual(usages["frigate.full_system"], {"cpu": "12.5", "mem": "40.0"})
        self.assertEqual(
            usages["10"],
            {
                "cpu": "5.3",
                "cpu_average": "4.1",
                "mem": "0.6",
                "cmdline": "python3 -u -m frigate",
            },
        )
        # the child that does the inference is listed
        self.assertIn("12", usages)

    def test_camera_credentials_are_masked(self):
        cmdline = rust_stats.cpu_usages(native_doc())["20"]["cmdline"]
        self.assertNotIn("hunter2", cmdline)
        self.assertIn("192.168.1.5", cmdline)

    def test_get_cpu_stats_uses_the_native_collection(self):
        with patch.object(services.psutil, "process_iter") as process_iter:
            usages = get_cpu_stats(native_doc())
        process_iter.assert_not_called()
        self.assertEqual(usages["12"]["cpu"], "31.8")

    def test_tree_usage_covers_the_child_processes(self):
        doc = native_doc()
        self.assertEqual(rust_stats.tree_usage(doc, 11), {"cpu": "33.3", "mem": "0.3"})
        self.assertIsNone(rust_stats.tree_usage(doc, 999))
        self.assertIsNone(rust_stats.tree_usage(doc, None))
        self.assertIsNone(rust_stats.tree_usage(None, 11))


class TestAmdGpuUsages(unittest.TestCase):
    def test_unified_video_queue_is_reported_as_decode(self):
        usages = rust_stats.amd_gpu_usages(native_doc(gpus=[amd_gpu()]))
        self.assertEqual(
            usages,
            {
                "amd-vaapi": {
                    "vendor": "amd",
                    "gpu": "66.8%",
                    "mem": "25.0%",
                    "dec": "12.5%",
                    "compute": "30.0%",
                    "temp": 58,
                    "power": 44.14,
                    "clients": {"12": "30.0%", "20": "16.5%"},
                }
            },
        )

    def test_separate_decode_engine_keeps_encode_apart(self):
        gpu = amd_gpu(
            has_decode_engine=True,
            engines={"dec": 20.0, "enc": 5.0, "enc_1": 2.5, "jpeg": 1.0},
        )
        entry = rust_stats.amd_gpu_usages(native_doc(gpus=[gpu]))["amd-vaapi"]
        self.assertEqual(entry["dec"], "21.0%")
        self.assertEqual(entry["enc"], "7.5%")
        self.assertNotIn("compute", entry)

    def test_gpu_without_a_device_node_is_not_reported(self):
        gpus = [amd_gpu(node_present=False)]
        self.assertEqual(rust_stats.amd_gpu_usages(native_doc(gpus=gpus)), {})

    def test_other_drivers_are_left_alone(self):
        gpus = [amd_gpu(driver="i915")]
        self.assertEqual(rust_stats.amd_gpu_usages(native_doc(gpus=gpus)), {})

    def test_suspended_gpu_has_blank_readings(self):
        gpu = amd_gpu(engines={}, clients={})
        for key in ("busy", "vram_used", "vram_total", "temp", "power"):
            del gpu[key]
        entry = rust_stats.amd_gpu_usages(native_doc(gpus=[gpu]))["amd-vaapi"]
        self.assertEqual(entry, {"vendor": "amd", "gpu": "", "mem": ""})

    def test_second_gpu_gets_its_own_key(self):
        gpus = [amd_gpu(), amd_gpu(card="card2", pdev="0000:03:00.0", busy=1.0)]
        usages = rust_stats.amd_gpu_usages(native_doc(gpus=gpus))
        self.assertEqual(list(usages), ["amd-vaapi", "amd-vaapi-card2"])
        self.assertEqual(usages["amd-vaapi-card2"]["gpu"], "1.0%")


def xdna(**overrides):
    npu = {
        "driver": "amdxdna",
        "accessible": True,
        "runtime_status": "active",
        "util": 80.0,
        "mem_bytes": 192 * 1024 * 1024,
        "contexts": 3,
        "command_rate": 412.5,
        "h_clock_mhz": 1800,
    }
    npu.update(overrides)
    return npu


class TestXdnaNpuUsage(unittest.TestCase):
    def test_accessible_npu(self):
        usage = rust_stats.xdna_npu_usage(native_doc(npus=[xdna()]))
        self.assertEqual(
            usage,
            {
                "npu": 80.0,
                "mem": "192 MiB",
                "clock_mhz": 1800,
                "contexts": 3,
                "commands_per_s": 412.5,
                "state": "active",
            },
        )

    def test_npu_without_a_device_node_is_not_reported(self):
        doc = native_doc(npus=[xdna(accessible=False)])
        self.assertIsNone(rust_stats.xdna_npu_usage(doc))
        self.assertIsNone(rust_stats.xdna_npu_usage(native_doc()))


def fake_config(hwaccel_args="preset-vaapi", amd_gpu_stats=True, face_device=None):
    config = MagicMock()
    camera = MagicMock()
    camera.ffmpeg.hwaccel_args = hwaccel_args
    camera.ffmpeg.inputs = []
    config.cameras = {"front": camera}
    config.telemetry.stats.amd_gpu_stats = amd_gpu_stats
    config.telemetry.stats.network_bandwidth = False
    config.detectors = {}
    config.face_recognition.enabled = face_device is not None
    config.face_recognition.device = face_device
    return config


class TestStatsIntegration(unittest.TestCase):
    def test_amd_gpu_comes_from_the_native_collection(self):
        all_stats: dict = {}
        with (
            patch("frigate.stats.util.get_amd_gpu_stats") as radeontop,
            patch("frigate.stats.util._is_amd_vaapi", return_value=True),
        ):
            asyncio.run(
                set_gpu_stats(
                    fake_config(), all_stats, {}, native_doc(gpus=[amd_gpu()])
                )
            )
        radeontop.assert_not_called()
        self.assertEqual(all_stats["gpu_usages"]["amd-vaapi"]["gpu"], "66.8%")

    def test_gpu_that_only_runs_the_detector_is_reported(self):
        # no hwaccel at all: the hwaccel scan alone would report nothing
        all_stats: dict = {}
        asyncio.run(
            set_gpu_stats(
                fake_config(hwaccel_args=[]),
                all_stats,
                {},
                native_doc(gpus=[amd_gpu()]),
            )
        )
        self.assertIn("amd-vaapi", all_stats["gpu_usages"])

    def test_amd_gpu_stats_can_be_disabled(self):
        all_stats: dict = {}
        with patch("frigate.stats.util._is_amd_vaapi", return_value=True):
            asyncio.run(
                set_gpu_stats(
                    fake_config(amd_gpu_stats=False),
                    all_stats,
                    {},
                    native_doc(gpus=[amd_gpu()]),
                )
            )
        self.assertNotIn("gpu_usages", all_stats)

    def test_radeontop_is_the_fallback_without_the_library(self):
        all_stats: dict = {}
        with (
            patch(
                "frigate.stats.util.get_amd_gpu_stats",
                return_value={"gpu": "4.17%", "mem": "60.37%"},
            ),
            patch("frigate.stats.util._is_amd_vaapi", return_value=True),
        ):
            asyncio.run(set_gpu_stats(fake_config(), all_stats, {}, None))
        self.assertEqual(
            all_stats["gpu_usages"]["amd-vaapi"],
            {"vendor": "amd", "gpu": "4.17%", "mem": "60.37%"},
        )

    def test_npu_from_the_driver_without_a_service(self):
        all_stats: dict = {}
        asyncio.run(set_npu_usages(fake_config(), all_stats, native_doc(npus=[xdna()])))
        self.assertEqual(all_stats["npu_usages"]["amd-xdna"]["npu"], 80.0)

    def test_service_figures_win_over_the_driver(self):
        all_stats: dict = {}
        service = {
            "npu": 12.5,
            "mem": "183 MiB",
            "embeddings_per_s": 0.4,
            "mean_latency_ms": 131.0,
            "clock_mhz": None,
            "state": "active",
        }
        with patch(
            "frigate.detectors.npu_service_runner.get_npu_service_stats",
            return_value=service,
        ):
            asyncio.run(
                set_npu_usages(
                    fake_config(face_device="npu"),
                    all_stats,
                    native_doc(npus=[xdna()]),
                )
            )
        entry = all_stats["npu_usages"]["amd-xdna"]
        self.assertEqual(entry["npu"], 12.5)  # time in NPU calls, not occupancy
        self.assertEqual(entry["mean_latency_ms"], 131.0)
        self.assertEqual(entry["contexts"], 3)  # only the driver knows
        self.assertEqual(entry["clock_mhz"], 1800)  # the service had none

    def test_no_npu_entry_without_driver_or_service(self):
        all_stats: dict = {}
        asyncio.run(set_npu_usages(fake_config(), all_stats, native_doc()))
        self.assertNotIn("npu_usages", all_stats)

    def test_cpu_stats_from_the_native_collection(self):
        all_stats: dict = {}
        asyncio.run(set_cpu_stats(all_stats, native_doc()))
        self.assertEqual(all_stats["cpu_usages"]["11"]["cpu"], "1.5")


class BurnerTestCase(unittest.TestCase):
    """Runs a child process that keeps one core busy."""

    def setUp(self):
        self.burner = subprocess.Popen(["python3", "-c", BURN])
        self.addCleanup(self._stop)

    def _stop(self):
        self.burner.kill()
        self.burner.wait()


@unittest.skipUnless(os.path.isdir("/proc/self"), "needs procfs")
class TestPsutilFallback(BurnerTestCase):
    def test_child_cpu_is_measured_between_two_collections(self):
        # Every collection used to build new psutil.Process objects, whose
        # first cpu_percent() is always 0.0, so every process read as idle.
        services._cpu_stat_processes.clear()
        get_cpu_stats()
        time.sleep(1.0)
        usages = get_cpu_stats()
        pid = str(self.burner.pid)
        self.assertIn(pid, usages)
        self.assertGreater(float(usages[pid]["cpu"]), 50.0)

    def test_exited_processes_are_forgotten(self):
        services._cpu_stat_processes.clear()
        get_cpu_stats()
        self.assertIn(self.burner.pid, services._cpu_stat_processes)
        self._stop()
        usages = get_cpu_stats()
        self.assertNotIn(str(self.burner.pid), usages)
        self.assertNotIn(self.burner.pid, services._cpu_stat_processes)


@unittest.skipUnless(rust_stats.stats_rs_available(), "frigate-stats-rs not built")
class TestNativeCollector(BurnerTestCase):
    def test_child_cpu_and_tree_totals(self):
        rust_stats.collect()
        time.sleep(1.0)
        doc = rust_stats.collect()
        self.assertEqual(doc["version"], 1)

        child = doc["processes"][str(self.burner.pid)]
        self.assertGreater(child["cpu"], 50.0)
        self.assertEqual(child["ppid"], os.getpid())
        self.assertIn("python3", child["cmdline"])

        me = doc["processes"][str(os.getpid())]
        self.assertGreaterEqual(me["tree_cpu"], child["cpu"])
        self.assertGreaterEqual(me["tree_rss_kb"], me["rss_kb"] + child["rss_kb"])

        system = doc["system"]
        self.assertGreater(system["mem_total_kb"], 0)
        self.assertGreater(system["cpus"], 0)
        self.assertTrue(0.0 <= system["cpu"] <= 100.0)
        self.assertTrue(0.0 < system["mem"] <= 100.0)

    def test_first_reading_of_a_process_is_not_zero(self):
        # give the child a moment of history, then collect once
        time.sleep(0.5)
        doc = rust_stats.collect()
        self.assertGreater(doc["processes"][str(self.burner.pid)]["cpu"], 20.0)

    def test_short_buffer_is_grown_without_collecting_again(self):
        rust_stats.collect()
        time.sleep(1.0)
        with patch.object(rust_stats, "_buffer", ctypes.create_string_buffer(8)):
            doc = rust_stats.collect()
            self.assertGreater(len(rust_stats._buffer), 8)
        # usage still spans the full second, not the retry
        self.assertGreater(doc["processes"][str(self.burner.pid)]["cpu"], 50.0)

    def test_frigate_layout_from_a_real_collection(self):
        doc = rust_stats.collect()
        usages = get_cpu_stats(doc)
        self.assertIn("frigate.full_system", usages)
        entry = usages[str(os.getpid())]
        self.assertEqual(set(entry), {"cpu", "cpu_average", "mem", "cmdline"})
        float(entry["cpu"]), float(entry["cpu_average"]), float(entry["mem"])
        # these never raise, whatever hardware the test machine has
        rust_stats.amd_gpu_usages(doc)
        rust_stats.xdna_npu_usage(doc)


if __name__ == "__main__":
    sys.exit(unittest.main())
