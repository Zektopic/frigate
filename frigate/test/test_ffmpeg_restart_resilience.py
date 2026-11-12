"""Tests for detect ffmpeg restart resilience (shared decoder resets).

No real ffmpeg or GPU is used. Process lifecycle tests use fake ``Popen``
objects, plus a few tiny real child processes (``sleep``) to verify that
nothing is left as a zombie and no file descriptors or threads leak.
"""

import logging
import os
import shutil
import subprocess as sp
import sys
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from frigate.log import LogPipe
from frigate.util.ffmpeg import reap_ffmpeg, stop_ffmpeg, terminate_ffmpeg
from frigate.video import ffmpeg as video_ffmpeg
from frigate.video.ffmpeg import CameraWatchdog, capture_frames
from frigate.video.restart_coordinator import (
    MAX_RESTART_BACKOFF,
    MAX_SHARED_RESTART_STAGGER,
    REGISTRY_RETENTION,
    SHARED_RESTART_JITTER,
    SHARED_RESTART_STAGGER,
    STABLE_RUN_SECONDS,
    FfmpegRestartCoordinator,
)

CAMERAS = [f"cam{i}" for i in range(8)]
RETRY_INTERVAL = 10.0


class FakeClock:
    """Stand-in for ``datetime`` in frigate.video.ffmpeg with a settable time."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.t = start

    def now(self, tz=None):
        return SimpleNamespace(timestamp=lambda: self.t)


class FakePopen:
    """Minimal ``Popen`` double recording lifecycle calls."""

    def __init__(
        self,
        exits_on_term: bool = True,
        exits_on_kill: bool = True,
        running: bool = True,
    ) -> None:
        self.pid = 4242
        self.calls: list[str] = []
        self.exits_on_term = exits_on_term
        self.exits_on_kill = exits_on_kill
        self.returncode = None if running else 1
        self._signal: str | None = None
        self.stdin = None
        self.stdout = MagicMock()
        self.stderr = None

    def poll(self):
        self.calls.append("poll")
        return self.returncode

    def terminate(self):
        self.calls.append("terminate")
        self._signal = "term"

    def kill(self):
        self.calls.append("kill")
        self._signal = "kill"

    def _wait(self, timeout):
        if (self._signal == "term" and self.exits_on_term) or (
            self._signal == "kill" and self.exits_on_kill
        ):
            self.returncode = -15 if self._signal == "term" else -9
            return self.returncode
        raise sp.TimeoutExpired("ffmpeg", timeout)

    def communicate(self, timeout=None):
        self.calls.append(f"communicate({timeout})")
        self._wait(timeout)
        return (b"", None)

    def wait(self, timeout=None):
        self.calls.append(f"wait({timeout})")
        return self._wait(timeout)


def make_watchdog(
    name: str,
    registry: dict | None,
    clock: FakeClock,
    rng=lambda: 0.5,
) -> CameraWatchdog:
    """Build a watchdog without its ZMQ / config plumbing."""
    watchdog = CameraWatchdog.__new__(CameraWatchdog)
    watchdog.logger = logging.getLogger(f"watchdog.{name}")
    watchdog.config = SimpleNamespace(name=name)
    watchdog.sleeptime = RETRY_INTERVAL
    watchdog.restart_coordinator = FfmpegRestartCoordinator(
        name, RETRY_INTERVAL, registry, rng=rng
    )
    watchdog._restart_at = None
    watchdog._unreaped_processes = []
    watchdog.reconnect_timestamps = __import__("collections").deque()
    watchdog.reconnects = None
    watchdog.capture_thread = None
    watchdog.ffmpeg_detect_process = FakePopen()
    watchdog.logpipe = MagicMock()
    watchdog.started_at: list[float] = []

    def fake_start():
        watchdog.started_at.append(clock.t)
        watchdog.ffmpeg_detect_process = FakePopen()

    watchdog.start_ffmpeg_detect = fake_start
    return watchdog


class TestRestartCoordinator(unittest.TestCase):
    def test_single_camera_first_failure_restarts_immediately(self) -> None:
        coordinator = FfmpegRestartCoordinator("front", RETRY_INTERVAL, {})
        coordinator.report_failure(100.0)
        plan = coordinator.plan_restart(100.0)

        self.assertEqual(plan.delay, 0.0)
        self.assertEqual(plan.attempt, 1)
        self.assertFalse(plan.shared)
        self.assertFalse(plan.log_summary)

    def test_without_registry_never_shared(self) -> None:
        coordinator = FfmpegRestartCoordinator("front", RETRY_INTERVAL, None)
        coordinator.report_failure(100.0)
        plan = coordinator.plan_restart(100.0)

        self.assertEqual(plan.delay, 0.0)
        self.assertFalse(plan.shared)

    def test_simultaneous_failures_are_staggered_by_arrival(self) -> None:
        registry: dict = {}
        coordinators = [
            FfmpegRestartCoordinator(c, RETRY_INTERVAL, registry, rng=lambda: 0.5)
            for c in CAMERAS
        ]

        for i, coordinator in enumerate(coordinators):
            coordinator.report_failure(100.0 + i * 0.1)

        plans = [c.plan_restart(105.0) for c in coordinators]
        jitter = 0.5 * SHARED_RESTART_JITTER

        for rank, plan in enumerate(plans):
            self.assertTrue(plan.shared)
            self.assertEqual(plan.peers, tuple(CAMERAS))
            self.assertAlmostEqual(plan.delay, rank * SHARED_RESTART_STAGGER + jitter)

        self.assertEqual(sum(p.log_summary for p in plans), 1)

    def test_stagger_is_capped(self) -> None:
        registry: dict = {}
        cams = [f"cam{i:02d}" for i in range(40)]
        coordinators = [
            FfmpegRestartCoordinator(c, RETRY_INTERVAL, registry, rng=lambda: 1.0)
            for c in cams
        ]
        for i, coordinator in enumerate(coordinators):
            coordinator.report_failure(100.0 + i * 0.01)

        delays = [c.plan_restart(101.0).delay for c in coordinators]
        self.assertLessEqual(
            max(delays), MAX_SHARED_RESTART_STAGGER + SHARED_RESTART_JITTER
        )

    def test_failures_outside_window_are_independent(self) -> None:
        registry: dict = {}
        a = FfmpegRestartCoordinator("a", RETRY_INTERVAL, registry)
        b = FfmpegRestartCoordinator("b", RETRY_INTERVAL, registry)

        a.report_failure(100.0)
        a.plan_restart(100.0)
        b.report_failure(200.0)
        plan = b.plan_restart(200.0)

        self.assertFalse(plan.shared)
        self.assertEqual(plan.delay, 0.0)

    def test_repeated_failures_back_off_and_cap(self) -> None:
        coordinator = FfmpegRestartCoordinator("front", RETRY_INTERVAL, None)
        now = 1000.0
        delays = []

        for _ in range(8):
            coordinator.report_failure(now)
            plan = coordinator.plan_restart(now)
            delays.append(plan.delay)
            now += plan.delay + 25.0

        self.assertEqual(delays, [0.0, 10.0, 20.0, 40.0, 60.0, 60.0, 60.0, 60.0])
        self.assertEqual(max(delays), MAX_RESTART_BACKOFF)

    def test_backoff_resets_after_stable_run(self) -> None:
        coordinator = FfmpegRestartCoordinator("front", RETRY_INTERVAL, None)
        coordinator.plan_restart(0.0)
        self.assertEqual(coordinator.plan_restart(30.0).delay, 10.0)

        plan = coordinator.plan_restart(40.0 + STABLE_RUN_SECONDS)
        self.assertEqual(plan.attempt, 1)
        self.assertEqual(plan.delay, 0.0)

    def test_registry_errors_degrade_to_local_backoff(self) -> None:
        registry = MagicMock()
        registry.__setitem__.side_effect = BrokenPipeError()
        registry.items.side_effect = EOFError()
        coordinator = FfmpegRestartCoordinator("front", RETRY_INTERVAL, registry)

        coordinator.report_failure(100.0)
        plan = coordinator.plan_restart(100.0)

        self.assertEqual(plan.delay, 0.0)
        self.assertFalse(plan.shared)

    def test_old_registry_entries_are_pruned(self) -> None:
        registry: dict = {"old_cam": 0.0, ("__shared_stall_summary__", 0.0): "x"}
        coordinator = FfmpegRestartCoordinator("front", RETRY_INTERVAL, registry)

        now = REGISTRY_RETENTION + 10.0
        coordinator.report_failure(now)
        coordinator.plan_restart(now)

        self.assertEqual(registry, {"front": now})

    def test_shared_registry_across_processes(self) -> None:
        """A real manager dict (as used in production) works end to end."""
        import multiprocessing as mp

        with mp.Manager() as manager:
            registry = manager.dict()
            a = FfmpegRestartCoordinator("a", RETRY_INTERVAL, registry)
            b = FfmpegRestartCoordinator("b", RETRY_INTERVAL, registry)
            a.report_failure(100.0)
            b.report_failure(101.0)

            plan_a = a.plan_restart(102.0)
            plan_b = b.plan_restart(102.0)

        self.assertTrue(plan_a.shared and plan_b.shared)
        self.assertEqual(plan_a.log_summary + plan_b.log_summary, 1)
        self.assertGreater(plan_b.delay, plan_a.delay)


class TestTerminateFfmpeg(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = logging.getLogger("test.terminate")

    def test_graceful_exit(self) -> None:
        proc = FakePopen()
        self.assertTrue(terminate_ffmpeg(proc, self.logger))
        self.assertNotIn("kill", proc.calls)
        proc.stdout.close.assert_called_once()

    def test_escalates_to_kill_and_reaps(self) -> None:
        proc = FakePopen(exits_on_term=False)
        self.assertTrue(stop_ffmpeg(proc, self.logger))
        self.assertEqual(
            [c for c in proc.calls if c != "poll"],
            ["terminate", "communicate(30)", "kill", "wait(10)"],
        )
        self.assertEqual(proc.returncode, -9)
        proc.stdout.close.assert_called_once()

    def test_wait_without_draining(self) -> None:
        proc = FakePopen(exits_on_term=False)
        terminate_ffmpeg(proc, self.logger, drain_output=False)
        self.assertIn("wait(30)", proc.calls)
        self.assertNotIn("communicate(30)", proc.calls)

    def test_survivor_is_reported_and_reaped_later(self) -> None:
        proc = FakePopen(exits_on_term=False, exits_on_kill=False)

        self.assertFalse(terminate_ffmpeg(proc, self.logger))
        proc.stdout.close.assert_not_called()
        self.assertFalse(reap_ffmpeg(proc))

        # the driver call finally returns and SIGKILL takes effect
        proc.returncode = -9
        self.assertTrue(reap_ffmpeg(proc))
        proc.stdout.close.assert_called_once()

    def test_already_dead_process_is_not_signalled(self) -> None:
        proc = FakePopen(running=False)
        self.assertTrue(terminate_ffmpeg(proc, self.logger))
        self.assertNotIn("terminate", proc.calls)
        self.assertNotIn("kill", proc.calls)

    def test_close_pipes_can_be_deferred(self) -> None:
        proc = FakePopen()
        terminate_ffmpeg(proc, self.logger, close_pipes=False)
        proc.stdout.close.assert_not_called()

    @unittest.skipUnless(sys.platform.startswith("linux"), "needs /proc")
    def test_real_process_ignoring_sigterm_is_killed_and_reaped(self) -> None:
        proc = sp.Popen(
            [
                sys.executable,
                "-c",
                "import signal, sys, time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "print('ready', flush=True)\n"
                "time.sleep(60)",
            ],
            stdout=sp.PIPE,
            stdin=sp.DEVNULL,
            start_new_session=True,
        )
        proc.stdout.readline()

        self.assertTrue(terminate_ffmpeg(proc, self.logger, timeout=0.5))
        self.assertEqual(proc.returncode, -9)
        self.assertTrue(proc.stdout.closed)
        self.assertFalse(os.path.exists(f"/proc/{proc.pid}"))


@patch.object(video_ffmpeg, "datetime")
class TestWatchdogRestarts(unittest.TestCase):
    def _install_clock(self, mock_datetime) -> FakeClock:
        clock = FakeClock()
        mock_datetime.now.side_effect = clock.now
        return clock

    def test_single_camera_stall_unchanged(self, mock_datetime) -> None:
        clock = self._install_clock(mock_datetime)
        watchdog = make_watchdog("front", {}, clock)
        old = watchdog.ffmpeg_detect_process

        with self.assertLogs("watchdog.front", level="INFO") as logs:
            watchdog.reset_capture_thread()

        self.assertEqual(watchdog.started_at, [clock.t])
        self.assertIsNone(watchdog._restart_at)
        self.assertEqual(old.calls[:3], ["poll", "terminate", "communicate(30)"])
        watchdog.logpipe.dump.assert_called_once_with()
        self.assertTrue(
            any(
                "last 100 lines prior to exit" in m for m in logs.output if "ERROR" in m
            )
        )
        self.assertFalse(any("Multiple cameras" in m for m in logs.output))

    def test_simultaneous_stall_restarts_staggered(self, mock_datetime) -> None:
        clock = self._install_clock(mock_datetime)
        registry: dict = {}
        watchdogs = [make_watchdog(c, registry, clock) for c in CAMERAS]

        # every watchdog notices the stall within ~2 seconds
        with self.assertLogs(level="DEBUG") as logs:
            for watchdog in watchdogs:
                watchdog.reset_capture_thread()
                clock.t += 0.25

            # drive the 1 Hz watchdog loops until everything restarted
            for _ in range(60):
                clock.t += 1.0
                for watchdog in watchdogs:
                    watchdog._start_deferred_restart(clock.t)

        starts = [w.started_at for w in watchdogs]
        self.assertTrue(all(len(s) == 1 for s in starts))
        start_times = [s[0] for s in starts]

        # restarts happen in arrival order, spaced by at least the stagger
        self.assertEqual(start_times, sorted(start_times))
        gaps = [b - a for a, b in zip(start_times[1:], start_times[2:])]
        self.assertTrue(all(g >= SHARED_RESTART_STAGGER - 1.0 for g in gaps), gaps)
        self.assertGreater(start_times[-1] - start_times[0], 10.0)

        # one summary instead of N x 100 ffmpeg lines at error level
        summaries = [m for m in logs.output if "Multiple cameras" in m]
        self.assertEqual(len(summaries), 1)
        self.assertTrue(summaries[0].startswith("WARNING"))
        for watchdog in watchdogs[1:]:
            watchdog.logpipe.dump.assert_called_once_with(logging.DEBUG)

    def test_repeated_failures_back_off(self, mock_datetime) -> None:
        clock = self._install_clock(mock_datetime)
        watchdog = make_watchdog("front", None, clock)
        delays = []

        for _ in range(7):
            watchdog.reset_capture_thread(terminate=False)
            restart_at = watchdog._restart_at
            delays.append(0.0 if restart_at is None else restart_at - clock.t)

            if restart_at is not None:
                clock.t = restart_at
                self.assertTrue(watchdog._start_deferred_restart(clock.t))

            # the new ffmpeg fails again 20 seconds later
            clock.t += 20.0

        self.assertEqual(delays, [0.0, 10.0, 20.0, 40.0, 60.0, 60.0, 60.0])
        self.assertEqual(len(watchdog.started_at), 7)

    def test_fps_overflow_is_not_a_shared_event(self, mock_datetime) -> None:
        clock = self._install_clock(mock_datetime)
        registry: dict = {}
        make_watchdog("other", registry, clock).reset_capture_thread()
        watchdog = make_watchdog("front", registry, clock)

        watchdog.reset_capture_thread(drain_output=False, device_failure=False)

        self.assertNotIn("front", registry)
        self.assertEqual(watchdog.started_at, [clock.t])

    def test_unkillable_process_is_reaped_later(self, mock_datetime) -> None:
        clock = self._install_clock(mock_datetime)
        watchdog = make_watchdog("front", None, clock)
        stuck = FakePopen(exits_on_term=False, exits_on_kill=False)
        watchdog.ffmpeg_detect_process = stuck

        watchdog.reset_capture_thread()
        self.assertEqual(watchdog._unreaped_processes, [stuck])

        watchdog._reap_unreaped_processes()
        self.assertEqual(watchdog._unreaped_processes, [stuck])

        stuck.returncode = -9
        watchdog._reap_unreaped_processes()
        self.assertEqual(watchdog._unreaped_processes, [])
        stuck.stdout.close.assert_called_once()

    def test_stop_all_cancels_pending_restart(self, mock_datetime) -> None:
        clock = self._install_clock(mock_datetime)
        watchdog = make_watchdog("front", None, clock)
        watchdog.ffmpeg_other_processes = []
        watchdog._restart_at = clock.t + 30

        watchdog.stop_all_ffmpeg()

        self.assertIsNone(watchdog._restart_at)
        self.assertIsNone(watchdog.ffmpeg_detect_process)


@unittest.skipUnless(sys.platform.startswith("linux"), "needs /proc")
@unittest.skipUnless(shutil.which("sleep"), "needs sleep")
class TestRestartLeaks(unittest.TestCase):
    """Real child processes and LogPipe across many restarts."""

    def _open_fds(self) -> int:
        return len(os.listdir("/proc/self/fd"))

    def test_no_fd_thread_or_zombie_leak_across_restarts(self) -> None:
        clock = FakeClock()
        watchdog = make_watchdog("front", None, clock)
        logpipe = LogPipe("ffmpeg.front.detect")
        watchdog.logpipe = logpipe
        spawned: list[sp.Popen] = []

        def real_start():
            proc = sp.Popen(
                [shutil.which("sleep"), "30"],
                stdout=sp.PIPE,
                stderr=logpipe,
                stdin=sp.DEVNULL,
                start_new_session=True,
            )
            spawned.append(proc)
            watchdog.ffmpeg_detect_process = proc

            # stands in for CameraCaptureRunner: reads stdout until EOF
            def reader():
                while proc.stdout.read(4096):
                    pass

            watchdog.capture_thread = threading.Thread(target=reader, daemon=True)
            watchdog.capture_thread.start()

        watchdog.start_ffmpeg_detect = real_start

        with patch.object(video_ffmpeg, "datetime") as mock_datetime:
            mock_datetime.now.side_effect = clock.now
            real_start()
            time.sleep(0.2)
            baseline_fds = self._open_fds()
            baseline_threads = threading.active_count()

            for _ in range(50):
                # each restart is spaced past the stable window: no backoff
                clock.t += STABLE_RUN_SECONDS + 1
                watchdog.reset_capture_thread()

            time.sleep(0.2)
            self.assertLessEqual(self._open_fds(), baseline_fds)
            self.assertLessEqual(threading.active_count(), baseline_threads)

            stop_ffmpeg(watchdog.ffmpeg_detect_process, watchdog.logger)
            watchdog.capture_thread.join(timeout=5)

        logpipe.close()
        logpipe.join(timeout=5)

        self.assertEqual(len(spawned), 51)
        self.assertTrue(all(p.returncode is not None for p in spawned))
        self.assertTrue(all(p.stdout.closed for p in spawned))
        # reaped children no longer have a /proc entry (no zombies)
        self.assertFalse(any(os.path.exists(f"/proc/{p.pid}") for p in spawned))
        self.assertFalse(logpipe.is_alive())


class TestCaptureReadFailures(unittest.TestCase):
    def test_failed_reads_do_not_spin_or_spam(self) -> None:
        polls = {"n": 0}

        def poll():
            polls["n"] += 1
            return None if polls["n"] < 20 else 1

        process = MagicMock()
        process.stdout.read.side_effect = OSError("pipe closed")
        process.poll.side_effect = poll
        config = SimpleNamespace(name="front", enabled=True)
        frame_manager = MagicMock()
        frame_manager.write.return_value = bytearray(4)
        stop_event = threading.Event()
        value = lambda: SimpleNamespace(value=0.0)  # noqa: E731

        with (
            patch.object(video_ffmpeg, "CameraConfigUpdateSubscriber"),
            patch("frigate.util.frame_rs.frame_rs_available", return_value=False),
            patch.object(video_ffmpeg, "READ_FAILURE_BACKOFF", 0.001),
            self.assertLogs("frigate.video.ffmpeg", level="ERROR") as logs,
        ):
            capture_frames(
                process,
                config,
                2,
                0,
                (2, 2),
                frame_manager,
                MagicMock(),
                value(),
                value(),
                value(),
                stop_event,
            )

        unable = [m for m in logs.output if "Unable to read frames" in m]
        self.assertEqual(len(unable), 1)
        self.assertTrue(any("is not running" in m for m in logs.output))


if __name__ == "__main__":
    unittest.main()
