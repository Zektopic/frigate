"""Record the numpy worker's YOLOv8 output for the Rust parity test.

Runs ``_process()`` from the Python ncnn worker embedded in
``docker/ncnn-vulkan/onnx.py`` on synthetic model outputs and prints the
detections as ``anchor_free_cases.txt``. ``src/anchor_free.rs`` builds the
same outputs from the same seeds and checks its port against these rows.

Run it with the numpy that production uses (1.26), from the repo root:

    docker run --rm -v "$PWD:/src:ro" -w /src python:3.11-slim sh -c \\
        'pip install -q numpy==1.26.4 >&2 && python -I \\
        frigate-detector-rs/testdata/gen_anchor_free_cases.py' \\
        > frigate-detector-rs/testdata/anchor_free_cases.txt
"""

import sys
from pathlib import Path

import numpy as np

CHANNELS = 144
REG_CHANNELS = 64
NUM_CLASSES = 80

# (seed, model size, objects). 600 objects put more than 1000 boxes past
# the score threshold, which exercises the top-1000 cut before NMS.
CASES = [
    (1, 640, 40),
    (2, 640, 0),
    (3, 320, 30),
    (4, 640, 300),
    (5, 640, 600),
]

WORKER_START = "# ── pre-computed constants"
WORKER_END = "# ── YOLO26 handler"


def load_process(plugin_path: Path, model_size: int):
    """Return the worker's ``_process`` with its grid built for ``model_size``."""
    source = plugin_path.read_text()
    start = source.index(WORKER_START)
    end = source.index(WORKER_END, start)
    code = source[start:end]
    # The worker is an f-string; this part must not depend on interpolation.
    if "{" in code or "}" in code:
        sys.exit("worker _process() uses f-string fields; update this script")

    namespace = {"np": np, "model_size": model_size}
    exec(code, namespace)  # noqa: S102 - our own source, not user input
    return namespace["_process"]


def splitmix64(seed: int, idx: np.ndarray) -> np.ndarray:
    """SplitMix64 output for each counter in ``idx`` (wrapping uint64)."""
    with np.errstate(over="ignore"):
        z = np.uint64(seed) + (idx + np.uint64(1)) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def draws(seed: int, start: int, count: int) -> list[int]:
    """Top 24 bits of draws ``start..start+count``."""
    idx = np.arange(start, start + count, dtype=np.uint64)
    return [int(v) for v in splitmix64(seed, idx) >> np.uint64(40)]


def synthetic_output(seed: int, anchors: int, objects: int) -> np.ndarray:
    """Mirror of ``synthetic_output()`` in ``src/anchor_free.rs``."""
    n = anchors * CHANNELS
    k = (splitmix64(seed, np.arange(n, dtype=np.uint64)) >> np.uint64(40)).astype(
        np.int64
    )
    # DFL logits in [-4, 4); background class logits in [-12, -4).
    values = (k - 2**23).astype(np.float64) / 2**21
    values = values.reshape(anchors, CHANNELS)
    values[:, REG_CHANNELS:] -= 8.0
    raw = values.astype(np.float32)

    for o in range(objects):
        d = draws(seed, n + 4 * o, 3)
        anchor = d[0] % anchors
        cls = d[1] % NUM_CLASSES
        logit = d[2] / 2**21 - 2.0
        for step in range(3):
            if anchor + step < anchors:
                raw[anchor + step, REG_CHANNELS + cls] = logit - 0.5 * step

    d = draws(seed, n + 4 * objects, 6)
    # Two classes past the clip, both scoring 1.0.
    anchor = d[0] % anchors
    raw[anchor, REG_CHANNELS + d[1] % 40] = 18.0
    raw[anchor, REG_CHANNELS + 40 + d[2] % 40] = 25.0
    # A strong class with a NaN class logit beside it.
    anchor = d[3] % anchors
    cls = d[4] % NUM_CLASSES
    raw[anchor, REG_CHANNELS + cls] = 5.0
    raw[anchor, REG_CHANNELS + (cls + 1) % NUM_CLASSES] = np.nan
    # A strong class with a NaN DFL bin.
    anchor = d[5] % anchors
    raw[anchor, REG_CHANNELS] = 5.0
    raw[anchor, 3] = np.nan
    return raw


def checksum(raw: np.ndarray) -> int:
    """Sum of bit pattern x (index + 1), wrapping, as in the Rust test."""
    bits = raw.reshape(-1).view(np.uint32).astype(np.uint64)
    idx = np.arange(1, bits.size + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return int(np.sum(bits * idx, dtype=np.uint64))


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    plugin = repo / "docker" / "ncnn-vulkan" / "onnx.py"

    print("# Output of _process() in docker/ncnn-vulkan/onnx.py for the synthetic")
    print("# model outputs built by gen_anchor_free_cases.py (see its docstring).")
    print(f"# numpy {np.__version__}")
    print("# case <seed> <model size> <objects> <input checksum> <detections>")
    print("# <class> <score> <x1> <y1> <x2> <y2>")

    for seed, model_size, objects in CASES:
        process = load_process(plugin, model_size)
        anchors = sum((model_size // s) ** 2 for s in (8, 16, 32))
        raw = synthetic_output(seed, anchors, objects)
        sig = checksum(raw)  # before _process, which may not modify it anyway

        with np.errstate(invalid="ignore"):
            dets = process(raw.copy(), model_size, model_size)
        rows = [row for row in dets if row[1] > 0]

        print(f"case {seed} {model_size} {objects} {sig} {len(rows)}")
        for row in rows:
            print(" ".join(repr(float(v)) for v in row))


if __name__ == "__main__":
    main()
