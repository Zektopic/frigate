import datetime
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from multiprocessing.synchronize import Event as MpEvent

from frigate.object_detection.base import ObjectDetectProcess
from frigate.util.process import FrigateProcess
from frigate.util.services import restart_frigate

logger = logging.getLogger(__name__)

MAX_RESTARTS = 5
RESTART_WINDOW_S = 60

# A dead detection process is restarted in place. Only when it keeps dying
# is the whole container restarted: that takes the web UI, go2rtc and every
# camera down with it, so it is the last resort rather than the first one.
MAX_DETECTOR_RESTARTS = 3
DETECTOR_RESTART_WINDOW_S = 300


@dataclass
class MonitoredProcess:
    """A process monitored by the watchdog for automatic restart."""

    name: str
    process: FrigateProcess
    factory: Callable[[], FrigateProcess]
    on_restart: Callable[[FrigateProcess], None] | None = None
    restart_timestamps: deque[float] = field(
        default_factory=lambda: deque(maxlen=MAX_RESTARTS)
    )
    clean_exit_logged: bool = False

    def is_restarting_too_fast(self, now: float) -> bool:
        while (
            self.restart_timestamps
            and now - self.restart_timestamps[0] > RESTART_WINDOW_S
        ):
            self.restart_timestamps.popleft()
        return len(self.restart_timestamps) >= MAX_RESTARTS


class FrigateWatchdog(threading.Thread):
    def __init__(
        self,
        detectors: dict[str, ObjectDetectProcess],
        stop_event: MpEvent,
    ):
        super().__init__(name="frigate_watchdog")
        self.detectors = detectors
        self.stop_event = stop_event
        self._monitored: list[MonitoredProcess] = []
        self._restart_times: dict[str, float] = {}
        self._detector_deaths: dict[str, deque[float]] = {}

    def register(
        self,
        name: str,
        process: FrigateProcess,
        factory: Callable[[], FrigateProcess],
        on_restart: Callable[[FrigateProcess], None] | None = None,
    ) -> None:
        """Register a FrigateProcess for monitoring and automatic restart."""
        self._monitored.append(
            MonitoredProcess(
                name=name,
                process=process,
                factory=factory,
                on_restart=on_restart,
            )
        )

    def _check_process(self, entry: MonitoredProcess) -> None:
        if entry.process.is_alive():
            return

        exitcode = entry.process.exitcode
        if exitcode == 0:
            if not entry.clean_exit_logged:
                logger.info("Process %s exited cleanly, not restarting", entry.name)
                entry.clean_exit_logged = True
            return

        logger.warning(
            "Process %s (PID %s) exited with code %s",
            entry.name,
            entry.process.pid,
            exitcode,
        )

        now = datetime.datetime.now(datetime.UTC).timestamp()

        if entry.is_restarting_too_fast(now):
            logger.error(
                "Process %s restarting too frequently (%d times in %ds), backing off",
                entry.name,
                MAX_RESTARTS,
                RESTART_WINDOW_S,
            )
            return

        try:
            entry.process.close()
            new_process = entry.factory()
            new_process.start()

            entry.process = new_process
            entry.restart_timestamps.append(now)

            if entry.on_restart:
                entry.on_restart(new_process)

            logger.info("Restarted %s (PID %s)", entry.name, new_process.pid)
        except Exception:
            logger.exception("Failed to restart %s", entry.name)

    def _handle_dead_detector(
        self, name: str, detector: ObjectDetectProcess, now: float
    ) -> None:
        """Restart a detection process that exited, escalating if it keeps dying.

        This used to call restart_frigate() on the first death, which sends
        SIGTERM to s6 and restarts the whole container -- the API, nginx and
        websocket go away with it and the UI has to reconnect or be reloaded.
        A negative exit code is the signal that killed it (-11 SIGSEGV,
        -6 SIGABRT, -9 SIGKILL e.g. from the OOM killer).
        """
        process = detector.detect_process
        exitcode = process.exitcode if process is not None else None

        deaths = self._detector_deaths.setdefault(name, deque())
        while deaths and now - deaths[0] > DETECTOR_RESTART_WINDOW_S:
            deaths.popleft()

        if len(deaths) >= MAX_DETECTOR_RESTARTS:
            logger.error(
                "Detection process %s exited with code %s and has died %d times "
                "in %ds. Exiting Frigate...",
                name,
                exitcode,
                len(deaths) + 1,
                DETECTOR_RESTART_WINDOW_S,
            )
            restart_frigate()
            return

        deaths.append(now)
        logger.warning(
            "Detection process %s exited with code %s. Restarting detection process...",
            name,
            exitcode,
        )

        try:
            detector.start_or_restart()
        except Exception:
            logger.exception("Failed to restart detection process %s", name)
            return

        self._restart_times[name] = now

    def run(self) -> None:
        time.sleep(10)
        while not self.stop_event.wait(10):
            now = datetime.datetime.now(datetime.UTC).timestamp()

            # check the detection processes
            for name, detector in self.detectors.items():
                detection_start = detector.detection_start.value  # type: ignore[attr-defined]
                # issue https://github.com/python/typeshed/issues/8799
                # from mypy 0.981 onwards
                if detection_start > 0.0 and now - detection_start > 10:
                    last_restart = self._restart_times.get(name, 0.0)
                    if now - last_restart < 35:
                        logger.debug(
                            "Skipping stuck check for %s (restarted %.1fs ago)",
                            name,
                            now - last_restart,
                        )
                    else:
                        logger.info(
                            "Detection appears to be stuck. Restarting detection process..."
                        )
                        detector.start_or_restart()
                        self._restart_times[name] = now
                elif (
                    detector.detect_process is not None
                    and not detector.detect_process.is_alive()
                ):
                    self._handle_dead_detector(name, detector, now)

            for entry in self._monitored:
                self._check_process(entry)

        logger.info("Exiting watchdog...")
