"""Benchmark Birdseye composition with 8 continuously active cameras.

Each 1280x720 camera delivers 5 fps, so the manager sees 40 updates/s.
Reports the cost of an update that composes the canvas (the 10 fps output
cap is bypassed) and the CPU used per second with the cap in place.

Run from the repository root inside the Frigate image:
    python3 testing-scripts/benchmark_birdseye.py
"""

import multiprocessing as mp
import time

import numpy as np

from frigate.config import FrigateConfig
from frigate.output.birdseye import BirdsEyeFrameManager

N_CAM = 8
FRAME_INTERVAL = 1 / (N_CAM * 5)


def main() -> None:
    cameras = {
        f"cam{i}": {
            "ffmpeg": {
                "inputs": [{"path": f"rtsp://10.0.0.{i}:554/v", "roles": ["detect"]}]
            },
            "detect": {"height": 720, "width": 1280, "fps": 5},
        }
        for i in range(N_CAM)
    }
    config = FrigateConfig(
        **{
            "mqtt": {"host": "mqtt"},
            "birdseye": {"enabled": True, "mode": "continuous"},
            "cameras": cameras,
        }
    )
    manager = BirdsEyeFrameManager(config, mp.Event())
    rng = np.random.default_rng(0)
    frames = [
        rng.integers(0, 256, (720 * 3 // 2, 1280), dtype=np.uint8) for _ in range(4)
    ]
    frame_time = 1000.0

    def feed(i: int, uncapped: bool) -> None:
        nonlocal frame_time
        if uncapped:
            manager.last_output_time = 0
        manager.update(f"cam{i % N_CAM}", 1, 1, frame_time, frames[i % 4])
        frame_time += FRAME_INTERVAL

    # make every camera active and compute the layout
    for i in range(N_CAM * 3):
        feed(i, uncapped=True)

    def composing(n: int = 400) -> float:
        start = time.perf_counter()
        for i in range(n):
            feed(i, uncapped=True)
        return (time.perf_counter() - start) / n * 1e6

    print(f"composing update: {min(composing() for _ in range(5)):.0f} us/call")

    busy = 0.0
    calls = 0
    seconds = 3.0
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        start = time.perf_counter()
        feed(calls, uncapped=False)
        busy += time.perf_counter() - start
        calls += 1
        time.sleep(FRAME_INTERVAL)
    print(f"{N_CAM * 5} frames/s in, 10 fps out: {busy / seconds * 1e3:.1f} ms CPU/s")
    print("canvas checksum", int(manager.frame.astype(np.int64).sum()))


if __name__ == "__main__":
    main()
