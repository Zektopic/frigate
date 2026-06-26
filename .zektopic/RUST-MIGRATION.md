# Python to Rust Migration & Roadmap

This document outlines the current division of labor between Python and Rust in the Zektopic Frigate custom repository, and identifies candidate components for future migration to Rust.

---

## 🦀 Active Rust Components

These performance-critical components have been successfully migrated to Rust to optimize hot paths:

1. **`frigate-detector-rs` & `frigate-yolo-rs`**:
   * Rayon-parallel and hardware-optimized post-processing (DFL decoding, Non-Maximum Suppression (NMS), coordinate scaling, and class filters) for YOLO models.
   * Runs model inference via a decoupled python subprocess worker to prevent GPU crashes on AMD RADV and bypass Python GIL.
2. **`frigate-motion-rs`**:
   * Evaluates high-framerate raw video frames to isolate motion regions (contour finding, thresholding).
   * Reduces CPU usage compared to Python's OpenCV motion detection loops.
3. **`frigate-frame-rs`**:
   * Handles frame processing, scaling, crop operations, and pixel format conversions directly inside shared memory (`/dev/shm`).

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
   * Resource statistics aggregator (queries CPU, GPU, memory stats via `psutil`/helpers).
7. **Semantic Search & Face Recognition**:
   * Embedding computations and database indexing (`frigate/data_processing/`).

---

## 🚀 Future Rust Migration Candidates

To achieve a minimal CPU/RAM footprint and completely remove the Python GIL overhead in critical pipelines, the following modules are planned for future Rust migration:

### 1. Frame Capture Pipe Reader (`frigate/video/capture.py`)
* **Current State**: Python reads raw video bytes from FFmpeg pipes, parses frame boundaries, and pushes them into shared memory.
* **Rust Target**: Native Rust pipe reader doing zero-copy writes directly into `/dev/shm`, lowering pipe IO overhead under high FPS/resolution.

### 2. Tracker & Association Association (`frigate/object_processing.py`)
* **Current State**: Frame-to-frame box tracking (overlapping bounding boxes, Kalman filters, and tracks association) runs in Python.
* **Rust Target**: Parallel tracking engine in Rust for swift tracking computations across dozens of active objects.

### 3. Event Loop Triggers (`frigate/events/`)
* **Current State**: Evaluates object confidence, zone overlaps, and thresholds to register new events and review segments.
* **Rust Target**: Shift event state classification (start, active, end) to Rust to prevent GIL lag during busy periods.

### 4. Telemetry Collector
* **Current State**: Aggregates CPU, GPU, and camera stats via Python processes.
* **Rust Target**: A lightweight background stats-gathering thread, freeing up Python overhead.
