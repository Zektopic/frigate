"""ctypes bindings for the Rust resource monitor (frigate-stats-rs).

One call per stats interval returns per-process CPU and memory for Frigate's
process tree, AMD GPU usage (sysfs and DRM fdinfo) and AMD XDNA NPU usage. The
library keeps the counters of the previous call, so every figure is measured
over the whole interval between two collections.

When the library is not present every function reports that, and the callers
fall back to psutil and radeontop.
"""

import ctypes
import json
import logging
import os
import threading
from typing import Any

from frigate.util.builtin import clean_camera_user_pass

logger = logging.getLogger(__name__)

_LIB_NAME = "libfrigate_stats_rs.so"
# The JSON layout this module understands (frigate_stats_abi_version).
_ABI_VERSION = 1
# Helpers that are supervised outside of the main process, so they are not
# among its descendants. Matched against the process name.
_EXTRA_PROCESS_NAMES = b"go2rtc,certsync"

_lib: ctypes.CDLL | None = None
_available: bool | None = None
_lock = threading.Lock()
_buffer = ctypes.create_string_buffer(256 * 1024)


def _load_lib() -> ctypes.CDLL | None:
    global _lib, _available

    if _available is False:
        return None

    if _lib is not None:
        return _lib

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    search = [
        "/opt/frigate",
        repo_root,
        os.path.join(repo_root, "frigate-stats-rs", "target", "release"),
        os.path.join(repo_root, "frigate-stats-rs", "target", "debug"),
    ]

    for directory in search:
        candidate = os.path.join(directory, _LIB_NAME)

        if not os.path.isfile(candidate):
            continue

        try:
            lib = ctypes.CDLL(candidate)
            lib.frigate_stats_abi_version.argtypes = []
            lib.frigate_stats_abi_version.restype = ctypes.c_uint32
            lib.frigate_stats_collect.argtypes = [
                ctypes.c_int32,
                ctypes.c_char_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            lib.frigate_stats_collect.restype = ctypes.c_int64
            lib.frigate_stats_last.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            lib.frigate_stats_last.restype = ctypes.c_int64
        except (OSError, AttributeError) as exc:
            logger.debug("Failed to load %s: %s", candidate, exc)
            continue

        version = lib.frigate_stats_abi_version()

        if version != _ABI_VERSION:
            logger.warning(
                "%s has stats layout %s, expected %s; using the Python collectors",
                candidate,
                version,
                _ABI_VERSION,
            )
            continue

        _lib = lib
        _available = True
        logger.debug("Loaded Rust stats collector from %s", candidate)
        return _lib

    _available = False
    return None


def stats_rs_available() -> bool:
    return _load_lib() is not None


def collect(root_pid: int = 0) -> dict[str, Any] | None:
    """Collect usage for the process tree under root_pid (0: this process).

    Returns None when the library is unavailable or the collection failed.
    """
    global _buffer

    lib = _load_lib()

    if lib is None:
        return None

    with _lock:
        length = lib.frigate_stats_collect(
            root_pid, _EXTRA_PROCESS_NAMES, _buffer, len(_buffer)
        )

        if length > len(_buffer):
            # Fetch the same document again rather than collecting a new one,
            # which would measure usage over the last few microseconds.
            _buffer = ctypes.create_string_buffer(int(length) * 2)
            length = lib.frigate_stats_last(_buffer, len(_buffer))

        if length <= 0 or length > len(_buffer):
            logger.debug("Rust stats collection failed (%s)", length)
            return None

        raw = _buffer.raw[:length]

    try:
        return json.loads(raw)
    except ValueError:
        logger.debug("Rust stats collector returned invalid JSON")
        return None


def cpu_usages(native: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Convert a collection to the `cpu_usages` stats entry."""
    system = native.get("system", {})
    usages: dict[str, dict[str, str]] = {
        "frigate.full_system": {
            "cpu": str(system.get("cpu", 0.0)),
            "mem": str(system.get("mem", 0.0)),
        }
    }

    for pid, process in native.get("processes", {}).items():
        usages[pid] = {
            "cpu": str(process["cpu"]),
            "cpu_average": str(process["cpu_average"]),
            "mem": str(process["mem"]),
            "cmdline": clean_camera_user_pass(process["cmdline"]),
        }

    return usages


def tree_usage(native: dict[str, Any] | None, pid: Any) -> dict[str, str] | None:
    """CPU and memory of a process together with all of its descendants.

    A detector that runs inference in a child process (frigate-detector-rs)
    uses almost no CPU itself, so its own figures would always read as idle.
    """
    if native is None or pid is None:
        return None

    process = native.get("processes", {}).get(str(pid))

    if process is None:
        return None

    return {"cpu": str(process["tree_cpu"]), "mem": str(process["tree_mem"])}


def _percent(value: float) -> str:
    return f"{round(value, 2)}%"


def amd_gpu_usages(native: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert a collection to `gpu_usages` entries for AMD GPUs.

    Only GPUs whose device node is visible are reported: sysfs lists the
    host's hardware even when it was not passed into the container.
    """
    usages: dict[str, dict[str, Any]] = {}
    cards = [
        gpu
        for gpu in native.get("gpus", [])
        if gpu.get("driver") == "amdgpu" and gpu.get("node_present")
    ]

    for index, gpu in enumerate(cards):
        # the first GPU keeps the key radeontop stats were published under
        key = "amd-vaapi" if index == 0 else f"amd-vaapi-{gpu['card']}"
        entry: dict[str, Any] = {"vendor": "amd", "gpu": "", "mem": ""}

        if gpu.get("busy") is not None:
            entry["gpu"] = _percent(gpu["busy"])

        if gpu.get("vram_total"):
            entry["mem"] = _percent(100 * gpu.get("vram_used", 0) / gpu["vram_total"])

        engines: dict[str, float] = gpu.get("engines", {})
        encode = sum(v for k, v in engines.items() if k.startswith("enc"))
        decode = sum(v for k, v in engines.items() if k.startswith(("dec", "jpeg")))

        if gpu.get("has_decode_engine"):
            entry["dec"] = _percent(min(decode, 100.0))
            entry["enc"] = _percent(min(encode, 100.0))
        elif encode or decode or engines:
            # One queue serves both on newer video cores, and the kernel
            # accounts all of it as "enc". Decoding is what Frigate does with
            # it, so that is how it is labelled.
            entry["dec"] = _percent(min(encode + decode, 100.0))

        if "compute" in engines:
            entry["compute"] = _percent(engines["compute"])

        if gpu.get("temp") is not None:
            entry["temp"] = round(gpu["temp"])

        if gpu.get("power") is not None:
            entry["power"] = gpu["power"]

        clients = {
            pid: _percent(min(sum(client.get("engines", {}).values()), 100.0))
            for pid, client in gpu.get("clients", {}).items()
            if client.get("engines")
        }

        if clients:
            entry["clients"] = clients

        usages[key] = entry

    return usages


def xdna_npu_usage(native: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a collection to an `npu_usages` entry for an AMD XDNA NPU.

    Needs /dev/accel/accel0 inside the container; without it the NPU is only
    visible through whatever service is driving it.
    """
    for npu in native.get("npus", []):
        if not npu.get("accessible"):
            continue

        return {
            "npu": npu["util"],
            "mem": f"{npu['mem_bytes'] / (1024 * 1024):.0f} MiB",
            "clock_mhz": npu["h_clock_mhz"],
            "contexts": npu["contexts"],
            "commands_per_s": npu["command_rate"],
            "state": npu["runtime_status"],
        }

    return None
