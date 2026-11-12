# Python to Rust Migration & Roadmap

This document outlines the current division of labor between Python and Rust in the Zektopic Frigate custom repository, and identifies candidate components for future migration to Rust.

---

## 🦀 Active Rust Components

These performance-critical components have been successfully migrated to Rust to optimize hot paths:

1. **`frigate-detector-rs` & `frigate-yolo-rs`**:
   * Rayon-parallel and hardware-optimized post-processing (DFL decoding, Non-Maximum Suppression (NMS), coordinate scaling, and class filters) for YOLO models.
   * **Direct ncnn C-API inference** (2026-07-13): dlopens `/opt/frigate/libncnn.so` (built in a dedicated Docker stage, version-matched to pip ncnn) and runs the Vulkan forward pass in-process — no Python subprocess, no second pipe hop, fp16→f32 converted in Rust (exhaustively unit-tested). Python subprocess retained as automatic fallback when the lib is absent.
   * Parity-verified against the PyTorch model. This surfaced a long-standing accuracy bug: pyncnn maps 4D numpy input to a `dims=4 c=1` Mat, systematically crushing class scores (chair 0.56 → refrigerator 0.11). The FFI's 3D mat is correct; the Python fallback now reshapes to 3D as well.
   * **YOLOv8 / YOLO11 raw-DFL exports** (2026-11-08): `frigate-detector-rs <...> yolov8` decodes the `(N, 144)` head (DFL boxes, class sigmoid, NMS) in `src/anchor_free.rs`, so the ncnn-vulkan plugin no longer needs the Python worker for these models (it stays as the fallback when the binary is missing). Parity is tested against the numpy worker's `_process()` via `testdata/gen_anchor_free_cases.py`. The port found that `_process()` never multiplied DFL distances by the stride (every box was at most 15 px a side); both are fixed. The same bug remains in `frigate-yolo-rs`'s `yolo_anchor_free_post_process` (used by `frigate/util/model.py`).
   * Debug switch: `NCNN_FFI_NO_VULKAN=1` forces CPU inference for parity checks.
   * ncnn worker pinned to `num_threads=2` with `OMP_WAIT_POLICY=PASSIVE` — default OpenMP threading oversubscribed the CPU on Vulkan-fallback layers and spin-waited between parallel regions (~2× slower inference at ~174% CPU). See GPU-DETECTOR.md § ncnn thread tuning.
2. **`frigate-frame-rs` tracker distance** (2026-07-13): the norfair association distance (`track_distance`) runs in Rust — 8.3x faster per call (21.9us → 2.6us), executed per detection×tracked-object pair per frame. Parity-fuzzed against the numpy implementation (2000 random pairs + NaN/inf/degenerate cases) in `frigate/test/test_track_distance_rust.py`.
3. **`frigate-motion-rs`**:
   * Evaluates high-framerate raw video frames to isolate motion regions (contour finding, thresholding).
   * Reduces CPU usage compared to Python's OpenCV motion detection loops.
   * `motion_pixel_pipeline` FFI (blur → absdiff → threshold → dilate → contours) wired into `ImprovedMotionDetector.detect()`, replacing the OpenCV/scipy pixel math per frame; all Python post-processing (PTZ, calibration, `accumulateWeighted`) retained.
3. **`frigate-frame-rs`**:
   * Handles frame processing, scaling, crop operations, and pixel format conversions directly inside shared memory (`/dev/shm`).
4. **`frigate-stats-rs`** (2026-11-11): the resource monitor behind the System page and `/api/stats`.
   * One call per stats interval (`frigate/stats/rust_stats.py`) returns per-process CPU and memory for Frigate's process tree, AMD GPU usage and AMD XDNA NPU usage as JSON. The library keeps the previous counters, so every figure covers the whole interval between two collections.
   * **Fixes per-process CPU reading 0.0 for everything.** The Python collector built new `psutil.Process` objects on every collection, and the first `cpu_percent()` of a new object is always 0.0. (The psutil fallback is fixed too: it now keeps the objects.)
   * **Detectors report their whole process tree.** Inference runs in the `frigate-detector-rs` child, which was not listed at all, so the detector always looked idle.
   * **AMD GPU without `radeontop`**: device load (mean of 1 Hz readings of `gpu_busy_percent`, not one instantaneous sample), VRAM, temperature and power from sysfs; decode and compute time per process from DRM fdinfo. On video cores with a unified queue (VCN 4, e.g. Radeon 760M) the kernel accounts decode as `enc`; it is shown as decode. A GPU that only runs the detector (no VAAPI hwaccel) is now reported as well.
   * **AMD XDNA NPU** straight from the `amdxdna` driver when `/dev/accel/accel0` is passed into the container (contexts, commands per second, buffer memory, column occupancy). A runtime-suspended NPU is never queried, since any query wakes it. The npu-face service's own figures still take precedence when it is in use.
   * Intel, NVIDIA and Rockchip stats are unchanged and still collected in Python.

---

## 🐍 Remaining Python Components

The core coordination layer and non-performance-critical logic remain in Python:

1. **FastAPI Web Server**:
   * API routing, authentication, WebSocket connections, and real-time event dispatching (`frigate/api/`).
2. **Configuration Validation**:
   * YAML parsing and schema validation using Pydantic (`frigate/config/`).
3. **Database & ORM**:
   * SQLite persistence for events and review segments via Peewee ORM (`frigate/models.py`, `migrations/`).
4. **Recording Retention & Clip Assembly**:
   * Segment collection, folder monitoring, clip generation, and disk storage maintenance (`frigate/recordings/`, `frigate/events/`).
5. **Process Supervision**:
   * Monitoring/respawning helper processes (`go2rtc`, FFmpeg capture, Rust detector binary).
6. **Telemetry & Stats**:
   * The stats snapshot, history and Prometheus export. The collection itself is in `frigate-stats-rs`; Intel, NVIDIA and Rockchip GPU stats are still read by Python helpers.
7. **Semantic Search & Face Recognition**:
   * Embedding computations and database indexing (`frigate/data_processing/`).

---

## 🚀 Future Rust Migration Candidates

To achieve a minimal CPU/RAM footprint and completely remove the Python GIL overhead in critical pipelines, the following modules are planned for future Rust migration:

### 1. Frame Capture Pipe Reader (`frigate/video/ffmpeg.py`)
* **Current State**: **Migrated to Rust FFI** (`read_ffmpeg_frame`). FFmpeg stdout pipe reading is offloaded directly to Rust via raw file descriptors and buffer pointers, eliminating Python-level raw byte handling and GC overhead.

### 2. Tracker & Association Association (`frigate/object_processing.py`)
* **Current State**: **IoU and association distance migrated to Rust FFI** (`intersection_over_union`, `track_distance`), both with Python fallbacks. The Kalman filter stays in norfair — replacing it means reimplementing norfair's object lifecycle for little gain.
* **Zone evaluation / telemetry**: evaluated 2026-07-13 and deliberately NOT migrated — no zones are configured (the loop never runs) and its geometry test is already OpenCV C++; telemetry polls every ~10s with descendant-only scanning. No measurable win available.
* **Rust Target**: Complete parallel tracking engine in Rust for swift tracking computations across dozens of active objects.

### 3. Event Loop Triggers (`frigate/events/`)
* **Current State**: Evaluates object confidence, zone overlaps, and thresholds to register new events and review segments.
* **Rust Target**: Shift event state classification (start, active, end) to Rust to prevent GIL lag during busy periods.

### 4. Telemetry Collector
* **Current State**: **Migrated to Rust** (`frigate-stats-rs`, see above) for process, AMD GPU and AMD XDNA NPU stats. psutil and `radeontop` remain only as the fallback when the library is missing.
* **Remaining**: Intel (i915/xe fdinfo), NVIDIA (NVML) and Rockchip collectors are still Python; they could move behind the same call, but need that hardware to verify.
