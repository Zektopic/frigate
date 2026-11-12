"""Low-resource ("potato PC") tuning for small hosts.

When ``low_resource.enabled`` is set, :func:`apply_low_resource_defaults`
rewrites the raw config dict *before* validation and fills in cheaper
defaults for keys the user has not set explicitly. Explicit values always
win, and camera-level overrides keep working because the injected values
live at the global level and propagate down like any user-set global.
"""

import copy
from typing import Any, Optional

from pydantic import Field

from frigate.const import MIN_SHM_FRAMES

from .base import FrigateBaseModel

__all__ = [
    "LowResourceConfig",
    "LOW_RESOURCE_DEFAULTS",
    "apply_low_resource_defaults",
]

# Global defaults applied when low_resource.enabled is true and the key is
# absent from the user's config. Keep this list to settings that only trade
# away optional quality/features, never detection correctness.
LOW_RESOURCE_DEFAULTS: dict[str, Any] = {
    # Birdseye composes every camera into an extra stream and keeps an
    # extra ffmpeg encoder alive.
    "birdseye": {"enabled": False},
    # jsmpeg fallback live view: encode a quarter of the pixels of 720p.
    "live": {"height": 360},
}


class LowResourceConfig(FrigateBaseModel):
    enabled: bool = Field(
        default=False,
        title="Enable low-resource mode",
        description="Apply cheaper defaults for small hosts (fewer shared memory frames, Birdseye off, lower jsmpeg live resolution). Values set explicitly in the config always take precedence.",
    )
    shm_frames: Optional[int] = Field(
        default=None,
        ge=MIN_SHM_FRAMES,
        le=500,
        title="Shared memory frames per camera",
        description="Number of decoded frames buffered per camera in /dev/shm. Defaults to 50, or 20 when low-resource mode is enabled. Overrides the SHM_MAX_FRAMES environment variable.",
    )


def _fill_missing(target: dict[str, Any], defaults: dict[str, Any]) -> None:
    for key, value in defaults.items():
        if isinstance(value, dict):
            existing = target.get(key)
            if existing is None:
                target[key] = copy.deepcopy(value)
            elif isinstance(existing, dict):
                _fill_missing(existing, value)
        elif key not in target:
            target[key] = value


def apply_low_resource_defaults(data: Any) -> Any:
    """Return the raw config with low-resource defaults filled in.

    Only acts on plain dicts whose ``low_resource.enabled`` is truthy; the
    input is not mutated.
    """
    if not isinstance(data, dict):
        return data

    low_resource = data.get("low_resource")
    if not isinstance(low_resource, dict) or not low_resource.get("enabled"):
        return data

    result = dict(data)
    for key, defaults in LOW_RESOURCE_DEFAULTS.items():
        section = result.get(key)
        if section is None:
            result[key] = copy.deepcopy(defaults)
        elif isinstance(section, dict):
            section = copy.deepcopy(section)
            _fill_missing(section, defaults)
            result[key] = section

    return result
