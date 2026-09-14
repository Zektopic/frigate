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

## 7. Testing Environment & Mock Automation Optimization
- **Frontend Deprecation (`punycode`)**: Vite and Vitest runners are emitting multiple node `DEP0040` warnings. Audit dependencies (such as `whatwg-url` and `tr46`) inside `web/package.json`. Major versions of these libraries have replaced `punycode` with userland native URL methods and should be bumped to clean CI/CD output logs.
- **Docker Test Infrastructure (overlayfs)**: On some Linux hosts, `docker buildx build` inside `make run_tests` fails with a mount source `overlay` error (`invalid argument`). To enable seamless native execution, introduce a fallback target in the Makefile allowing `DOCKER_BUILDKIT=0` execution or explicitly utilizing the `vfs` storage driver.
- **Python Backend Testing (Fallback Refactoring)**:
  - **Environment Variables & Permissions**: Local test executions throw `PermissionError: [Errno 13] Permission denied: '/config'`. Set an overridden `CONFIG_DIR` pointing to a local `/tmp/frigate-test-config` inside `test_runner.py` or aggressively mock `frigate.const.MODEL_CACHE_DIR` directly during test initialization to bypass host restricted read/writes.
  - **Peewee Database Mocks**: Tests covering `test_storage.py` and `test_video.py` fail heavily (e.g. `AttributeError: type object 'Recordings' has no attribute 'insert'`). `test_runner.py` needs an expanded mock ORM to intercept SQLite execution and correctly chain methods like `.where()`, `.order_by()`, and `.insert().execute()`.
  - **Pydantic V2 Mock Refactoring**: Update `MockBaseModel` to support strict parsing of metadata values and error types.

## Future Actionable Improvements

### 1. Robust Python Backend Test Environment
- **Issue**: Running `test_runner.py` directly relies on fragile, global `sys.modules` patching. Mocked objects like `ModuleMock` break type expectations (e.g., in `pathvalidate` or `cv2.dnn.NMSBoxes`), leading to cascading false negatives.
- **Action**: Implement a standard Python `tox` or `pytest` setup that installs minimal mock representations (via dedicated local stub libraries or using `pytest-mock`) instead of dynamic overriding. Alternatively, separate tests that require native bindings (e.g., `cv2`, `numpy`) into a dedicated suite that only runs under Docker, leaving pure Python logic tests to run blazingly fast locally without excessive mocking.

### 2. Docker BuildKit Bypass for Testing
- **Issue**: Running `make run_tests` locally currently fails during the build step due to `overlayfs` mount issues from BuildKit on certain host kernel environments.
- **Action**: Update `Makefile` inside the `run_tests` target (or create a new `run_tests_local` target) to export `DOCKER_BUILDKIT=0` or explicitly use the standard `docker build` instead of `docker buildx build`. This provides an immediate fallback for engineers encountering local container storage driver faults during test invocation.

### 3. Pydantic v2 Schema Testing Compatibility
- **Issue**: Testing HTTP and configuration logic natively often fails because the mocked `BaseModel` lacks validation parity (e.g., missing `RootModel` logic or missing internal Pydantic error structures).
- **Action**: Stop mocking `pydantic` globally. Since it relies heavily on native Rust core components in v2, add it as an explicit development dependency (`pip install pydantic pydantic-core`) for local testing environments. This allows exact structural validation during unit testing and ensures config objects (e.g., `FrigateConfig`) correctly trigger actual `ValidationError`s.

### 4. Separation of Frontend Testing Workflows
- **Issue**: Mixed testing tooling (Vitest and Playwright) in the `web/` directory leads to configuration overlap and test resolution failures if executed at the root level indiscriminately.
- **Action**: Add explicit npm scripts to the root `package.json` separating integration runs from unit tests (e.g., `"test:unit": "vitest run src/"` and `"test:e2e": "playwright test"`). Ensure automated CI strictly invokes these separate targets to prevent framework collisions.
