# Frigate Architectural & Performance Improvements

Based on the audit log and codebase analysis, the following architectural and performance improvements are recommended for implementation in the future to improve Frigate's performance, stability, and compatibility on lower-end systems (e.g., AMD Ryzen APUs, limited RAM/SHM environments).

## 1. Video Ingestion / Capture Loop Optimizations (`frigate/video/ffmpeg.py`)
- **Zero-Copy Memory Allocation**: Replace `stdout.read` with `stdout.readinto(frame_buffer)` to read decoded frame bytes directly into shared memory. This eliminates `bytes` allocation and `frame_buffer[:]` copying on every frame, reducing CPU overhead and garbage collection pressure.
- **Throttling IPC Checks**: The configuration updates loop via ZMQ currently checks for updates on every frame. Throttle `config_subscriber.check_for_updates()` to ~2.0 second intervals.
- **Throttling Value Updates**: Synchronizing multiprocessing `Value` variables (like FPS) demands kernel locks. Update these atomic variables at 1Hz instead of per-frame.

## 2. Shared Memory Management (`frigate/app.py` & Sizing)
- **Robust Cleanup**: The current `UntrackedSharedMemory` cleanup aborts entirely if a single `shm.unlink()` call fails (e.g., due to an already unlinked segment by a subprocess). Wrap `unlink()` in a `try/except` block to ensure all orphaned segments are properly garbage-collected.
- **SHM Depletion Safety**: Frigate calculates SHM block sizes using `1.5 bytes per pixel * frames * cameras`. Enforce stricter constraints or automatic `emptyDir` RAM limits to prevent Docker containers from silently failing with `No space left on device` when the system's `/dev/shm` max size is exceeded.

## 3. Object Detection Array Layout (`frigate/object_detection/base.py`)
- **Contiguous Memory**: Passing non-contiguous arrays (created via `np.transpose()`) to C-bindings for inference results in implicit memory copies inside the bindings. Ensure array layouts are flattened using `np.ascontiguousarray()` before handing them to inference frameworks. Quantized model architectures (`int8`) should also be preferred over `float32` models for non-GPU inference on APUs.

## 4. SQLite Database IO (`frigate/db/sqlitevecq.py` & `frigate/app.py`)
- **Queue Database Pragmas**: The background queue database lacks I/O optimization pragmas. Add `PRAGMA journal_mode=WAL;`, `PRAGMA synchronous=NORMAL;`, and `PRAGMA temp_store=MEMORY;` to the background Peewee connection block to prevent write-locking on slow disks.
- **Avoid Auto-Vacuum FULL**: The database initializes with `"auto_vacuum": "FULL"`. Given Frigate's frequent deletion of events/clips, this causes massive write-amplification. Switch to `"INCREMENTAL"` or `"NONE"`, relying on the periodic 2-week `VACUUM` routine instead.
- **Dual Connection Pool Collision**: Both `SqliteVecQueueDatabase` and `RecordProcess` spin up separate connection pools onto the same SQLite file. This risks diverging pragma states. A single connection-pool architecture with explicit connection lending is advised.

## 5. Security & Parallel Processing Concerns
- **SyncManager Pickle Safety**: `multiprocessing.Manager()` uses `pickle` for its IPC serialization. Unauthenticated data injected from MQTT queues traverses these pipelines, exposing a remote code execution risk. Avoid passing raw payloads via SyncManager queues.
- **Storage Cleanup Race Conditions**: `reduce_storage_consumption()` deletes media from disk, then performs a bulk database deletion. A concurrent `RecordingCleanup` process can delete clips in the middle of this loop, leading to missing database elements. Transactions must be locked between disk and database ops.

## 6. Stability During Restarts
- **Watchdog Detection Gap**: Restarting detectors takes up to 30 seconds to join processes. During this time, the watchdog ignores missing inferences. Detection pipelines shouldn't be entirely suspended during graceful degradation.
