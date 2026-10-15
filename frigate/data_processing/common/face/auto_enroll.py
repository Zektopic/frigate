"""Guarded automatic enrolment of recognised faces into the face library.

Frigate "trains" face recognition by averaging the embeddings of the images in
``FACE_DIR/<name>/``. Auto-enrolment adds new images of people who are already
in the library, so recognition keeps up with haircuts, glasses, lighting and
camera angles without manual training.

Wrongly enrolling a face slowly teaches the model the wrong person, so every
candidate has to pass all of these guardrails:

* only people who already exist in the library (never ``unknown``, never a new
  name),
* per-frame: score >= ``min_score`` and at least ``min_margin`` above the
  runner-up person, face area >= ``min_area``,
* per tracked object: at least ``min_frames`` such frames, every recognition of
  that object agreeing on the same name, and enrolment only once the object has
  ended (so a track that later flips to someone else enrols nothing),
* per person: at most one enrolment per ``cooldown_seconds`` and at most
  ``max_images`` auto-enrolled images (oldest auto images are rotated out;
  images added by the user are never touched).

Auto-enrolled files are named ``<name>_auto_<timestamp>.webp`` so they can be
identified and removed from the Train UI like any other image.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)

AUTO_TAG = "_auto_"


@dataclass(frozen=True)
class AutoEnrollSettings:
    enabled: bool = False
    min_score: float = 0.95
    min_margin: float = 0.15
    min_frames: int = 3
    min_area: int = 2500
    max_images: int = 40
    cooldown_seconds: int = 900


@dataclass
class _Candidate:
    score: float
    timestamp: float
    image: object  # np.ndarray, kept opaque so this module needs no numpy


@dataclass
class _Track:
    names: set[str] = field(default_factory=set)
    qualifying: int = 0
    # Only the best crop is kept, so memory stays bounded for long tracks.
    best: _Candidate | None = None


class FaceLibrary(Protocol):
    """Minimal file-system view of the face library (injectable for tests)."""

    def people(self) -> set[str]: ...

    def list_files(self, name: str) -> list[str]: ...

    def write(self, name: str, filename: str, image: object) -> None: ...

    def delete(self, name: str, filename: str) -> None: ...


class DirFaceLibrary:
    """FaceLibrary backed by ``FACE_DIR``."""

    def __init__(self, root: str, writer: Callable[[str, object], None]) -> None:
        self.root = root
        self._writer = writer

    def people(self) -> set[str]:
        if not os.path.isdir(self.root):
            return set()
        return {
            d
            for d in os.listdir(self.root)
            if d != "train" and os.path.isdir(os.path.join(self.root, d))
        }

    def list_files(self, name: str) -> list[str]:
        folder = os.path.join(self.root, name)
        return os.listdir(folder) if os.path.isdir(folder) else []

    def write(self, name: str, filename: str, image: object) -> None:
        self._writer(os.path.join(self.root, name, filename), image)

    def delete(self, name: str, filename: str) -> None:
        try:
            os.unlink(os.path.join(self.root, name, filename))
        except FileNotFoundError:
            pass


class AutoEnroller:
    def __init__(
        self,
        settings: AutoEnrollSettings,
        library: FaceLibrary,
        on_enrolled: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings
        self.library = library
        self.on_enrolled = on_enrolled
        self._tracks: dict[str, _Track] = {}
        self._last_enrolled: dict[str, float] = {}

    def observe(
        self,
        object_id: str,
        name: str,
        score: float,
        runner_up: float,
        area: int,
        image: object,
        timestamp: float,
    ) -> None:
        """Record one recognition of a tracked object."""
        if not self.settings.enabled:
            return

        track = self._tracks.setdefault(object_id, _Track())
        # Every recognition counts towards consistency, even weak ones: a track
        # that was ever recognised as someone else must not be enrolled.
        if name and name != "unknown":
            track.names.add(name)

        if (
            name
            and name != "unknown"
            and score >= self.settings.min_score
            and score - runner_up >= self.settings.min_margin
            and area >= self.settings.min_area
        ):
            track.qualifying += 1
            if track.best is None or score > track.best.score:
                # Frames live in reused shared memory: keep a private copy.
                owned = image.copy() if hasattr(image, "copy") else image
                track.best = _Candidate(score, timestamp, owned)

    def finish(self, object_id: str, now: float) -> str | None:
        """Decide on enrolment when a tracked object ends.

        Returns the enrolled name, or None.
        """
        track = self._tracks.pop(object_id, None)
        if not self.settings.enabled or track is None:
            return None

        if len(track.names) != 1:
            if len(track.names) > 1:
                logger.debug(
                    "auto-enroll: %s had conflicting names %s", object_id, track.names
                )
            return None

        name = next(iter(track.names))
        if track.qualifying < self.settings.min_frames or track.best is None:
            return None
        if name not in self.library.people():
            return None
        last = self._last_enrolled.get(name)
        if last is not None and now - last < self.settings.cooldown_seconds:
            return None

        best = track.best
        filename = f"{name}{AUTO_TAG}{best.timestamp:.3f}.webp"
        self.library.write(name, filename, best.image)
        self._last_enrolled[name] = now
        self._rotate(name)
        logger.info(
            "auto-enroll: added %s (score %.2f, %d qualifying frames)",
            filename,
            best.score,
            track.qualifying,
        )

        if self.on_enrolled:
            self.on_enrolled(name)
        return name

    def _rotate(self, name: str) -> None:
        autos = sorted(f for f in self.library.list_files(name) if AUTO_TAG in f)
        # Filenames embed the timestamp, so lexical order is chronological.
        for old in autos[: max(0, len(autos) - self.settings.max_images)]:
            self.library.delete(name, old)
