"""Coordinate detect ffmpeg restarts across cameras.

Hardware decoders (VAAPI, QSV, NVDEC, ...) are shared by every camera on a
host. When the device resets (for example an amdgpu ``vcn_unified_0 ring
timeout``) decoding stops for all cameras at once. Each camera's watchdog
runs in its own capture process and used to restart its ffmpeg as soon as it
noticed the failure, so every camera hit the device at the same instant while
it was still recovering, which can prolong the reset storm.

This module provides :class:`FfmpegRestartCoordinator`. Each camera reports
its stalls/crashes to a small registry shared between the capture processes
(a manager ``dict``). When several cameras fail within
:data:`SHARED_STALL_WINDOW` seconds the event is treated as a shared device
event and restarts are staggered by arrival order with a little random jitter.
Independently, repeated restarts of the same camera are delayed with a capped
exponential backoff. A single camera failing once is restarted immediately,
exactly as before.
"""

import logging
import random
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

SHARED_STALL_WINDOW = 30.0
"""Seconds within which failures of different cameras count as one event.

Covers the 20s no-frames threshold drift between watchdogs plus cameras whose
ffmpeg crashes outright a few seconds earlier than others stall."""

SHARED_STALL_MIN_CAMERAS = 2
"""Number of cameras failing within the window that marks a shared event."""

SHARED_RESTART_STAGGER = 2.0
"""Seconds between consecutive camera restarts during a shared event."""

MAX_SHARED_RESTART_STAGGER = 30.0
"""Upper bound for the arrival-order stagger, so large installs stay bounded."""

SHARED_RESTART_JITTER = 2.0
"""Maximum random jitter (seconds) added to restarts during a shared event."""

MAX_RESTART_BACKOFF = 60.0
"""Upper bound (seconds) for the per-camera exponential backoff."""

STABLE_RUN_SECONDS = 300.0
"""A camera that ran this long since its last restart starts backoff afresh."""

REGISTRY_RETENTION = 600.0
"""Registry entries older than this are pruned."""

_SUMMARY_KEY = "__shared_stall_summary__"


@dataclass(frozen=True)
class RestartPlan:
    """When and why a camera's detect ffmpeg should be restarted."""

    delay: float
    attempt: int
    shared: bool = False
    peers: tuple[str, ...] = field(default_factory=tuple)
    log_summary: bool = False


class FfmpegRestartCoordinator:
    """Per-camera restart planner backed by an optional cross-process registry.

    ``registry`` maps camera name to the wall-clock timestamp of that camera's
    most recent failure, plus summary bookkeeping keys. It is normally a
    ``SyncManager().dict()`` shared by every capture process; ``None`` (or any
    registry error) degrades to per-camera backoff only.
    """

    def __init__(
        self,
        camera_name: str,
        retry_interval: float,
        registry: MutableMapping[Any, Any] | None = None,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self.camera_name = camera_name
        self.retry_interval = retry_interval
        self.registry = registry
        self._rng = rng
        self._attempt = 0
        self._last_restart_at: float | None = None
        self._reported_at: float | None = None

    def report_failure(self, now: float) -> None:
        """Record that this camera stopped producing frames at ``now``."""
        self._reported_at = now

        if self.registry is None:
            return

        try:
            self.registry[self.camera_name] = now
        except (OSError, EOFError) as e:
            logger.debug("Unable to update ffmpeg stall registry: %s", e)

    def plan_restart(self, now: float) -> RestartPlan:
        """Return the restart plan for this camera's current failure."""
        if (
            self._last_restart_at is None
            or now - self._last_restart_at >= STABLE_RUN_SECONDS
        ):
            self._attempt = 1
        else:
            self._attempt += 1

        backoff = 0.0
        if self._attempt > 1:
            backoff = min(
                self.retry_interval * 2 ** (self._attempt - 2), MAX_RESTART_BACKOFF
            )

        peers, rank, event_start = self._shared_event_members()
        shared = len(peers) >= SHARED_STALL_MIN_CAMERAS
        stagger = 0.0
        log_summary = False

        if shared:
            stagger = min(rank * SHARED_RESTART_STAGGER, MAX_SHARED_RESTART_STAGGER)
            stagger += self._rng() * SHARED_RESTART_JITTER
            log_summary = self._claim_summary(event_start)

        self._reported_at = None
        delay = backoff + stagger
        self._last_restart_at = now + delay
        return RestartPlan(
            delay=delay,
            attempt=self._attempt,
            shared=shared,
            peers=peers,
            log_summary=log_summary,
        )

    def _shared_event_members(self) -> tuple[tuple[str, ...], int, float]:
        """Return (cameras in this event ordered by arrival, our rank, start)."""
        reported_at = self._reported_at

        if self.registry is None or reported_at is None:
            return (), 0, 0.0

        try:
            snapshot = dict(self.registry.items())
        except (OSError, EOFError) as e:
            logger.debug("Unable to read ffmpeg stall registry: %s", e)
            return (), 0, 0.0

        self._prune(snapshot, reported_at)
        members = sorted(
            (ts, name)
            for name, ts in snapshot.items()
            if isinstance(name, str)
            and isinstance(ts, (int, float))
            and abs(ts - reported_at) <= SHARED_STALL_WINDOW
        )
        names = tuple(name for _, name in members)
        rank = names.index(self.camera_name) if self.camera_name in names else 0
        event_start = members[0][0] if members else reported_at
        return names, rank, event_start

    def _claim_summary(self, event_start: float) -> bool:
        """Return True for exactly one camera per shared event."""
        if self.registry is None:
            return False

        key = (_SUMMARY_KEY, round(event_start, 3))
        try:
            # setdefault runs atomically inside the manager process
            return self.registry.setdefault(key, self.camera_name) == self.camera_name
        except (OSError, EOFError) as e:
            logger.debug("Unable to update ffmpeg stall registry: %s", e)
            return False

    def _prune(self, snapshot: dict[Any, Any], now: float) -> None:
        cutoff = now - REGISTRY_RETENTION

        for key, value in snapshot.items():
            ts = key[1] if isinstance(key, tuple) else value

            if isinstance(ts, (int, float)) and ts < cutoff:
                try:
                    self.registry.pop(key, None)
                except (OSError, EOFError):
                    return
