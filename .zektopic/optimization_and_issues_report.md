# Frigate NVR: Comprehensive Performance Optimization & Code Quality Blueprint

This document contains a hardware-specific optimization plan for your **AMD Ryzen 5 3500U APU (Radeon Vega Graphics)**, system-level configurations, Go bridge optimizations, and concrete codebase improvements designed to maximize efficiency in resource-constrained environments (restricted CPU cores, slow disk storage, low memory, and integrated GPUs).

---

## Part 1: Hardware-Specific Configurations for AMD Ryzen 5 3500U

Your PC is equipped with an **AMD Ryzen 5 3500U** mobile processor (4 cores, 8 threads, AVX2 support) and an integrated **Radeon Vega GPU**. To prevent CPU exhaustion, implement the following settings:

### 1. Hardware Video Decoding (VA-API)
Offload video stream decoding from the CPU to the integrated GPU using the Video Acceleration API.
*   **Frigate Config (`frigate.yaml`)**:
    ```yaml
    ffmpeg:
      hwaccel_args: preset-vaapi
    ```
*   **Docker Container Configuration**:
    Map `/dev/dri/renderD128` (and card0) into the container to enable hardware access:
    ```yaml
    services:
      frigate:
        devices:
          - /dev/dri/renderD128:/dev/dri/renderD128
          - /dev/dri/card0:/dev/dri/card0
    ```

### 2. OpenVINO CPU Acceleration for AI Object Detection
While the Vega GPU lacks native TensorRT/ROCm support in standard containers, OpenVINO includes a CPU execution engine that leverages your processor's **AVX2** instruction set. This is significantly faster and uses less CPU than standard CPU-based TensorFlow Lite.
*   **Frigate Config (`frigate.yaml`)**:
    ```yaml
    detectors:
      ov:
        type: openvino
        device: CPU
    ```

### 3. Stream Ingestion Tuning
Avoid passing high-resolution streams to detection nodes:
*   Configure the `detect` role on a lower-resolution sub-stream (e.g., 640x480 or 1280x720) capped at **5 FPS**.
*   Configure the `record` role on your high-resolution main stream.
*   Ensure `go2rtc` stream configurations use `video=copy` to pass streams through without re-encoding.
*   **Frigate Config (`frigate.yaml`)**:
    ```yaml
    cameras:
      front_door:
        ffmpeg:
          inputs:
            - path: rtsp://...
              roles:
                - detect
        detect:
          width: 1280
          height: 720
          fps: 5
    ```

---

## Part 2: Codebase Optimization (Python & SQLite Patches)

The following sections contain code-level audits and concrete optimizations targeting CPU hot paths, redundant memory copies, and database locking.

### 1. Video Ingestion Capture Loop (`frigate/video/ffmpeg.py`)

#### 🔴 CPU Bottleneck: `stdout.read` Bytes Allocations & Copies
*   **Location**: [ffmpeg.py:L95](file:///home/manu/Documents/github-orgs/Zektopic/frigate/frigate/video/ffmpeg.py#L95)
*   **Issue**: Calling `ffmpeg_process.stdout.read(frame_size)` allocates a new `bytes` object (e.g., 1–3MB) every frame, which the slice assignment (`frame_buffer[:] = ...`) then copies to shared memory. This triggers continuous garbage collection pressure and CPU-intensive copying at frame-rate.
*   **Optimization**: Use `readinto()` to read directly from the stdout stream into the pre-allocated shared memory buffer.
*   **Code Patch**:
    ```diff
    -            try:
    -                frame_buffer[:] = ffmpeg_process.stdout.read(frame_size)
    -            except Exception:
    +            try:
    +                # Read directly into shared memory buffer (zero allocation & zero copy)
    +                bytes_read = ffmpeg_process.stdout.readinto(frame_buffer)
    +                if bytes_read != frame_size:
    +                    raise OSError(f"Incomplete read: expected {frame_size} bytes, got {bytes_read}")
    +            except Exception:
    ```

#### 🟡 CPU Overhead: High-Frequency ZMQ Config Polling
*   **Location**: [ffmpeg.py:L80-L86](file:///home/manu/Documents/github-orgs/Zektopic/frigate/frigate/video/ffmpeg.py#L80-L86)
*   **Issue**: `get_enabled_state()` checks for config updates via ZMQ on **every frame**. Polling IPC sockets at 10–30Hz per camera wastes CPU cycles.
*   **Optimization**: Rate-limit the updates check to once every 2 seconds.
*   **Code Patch**:
    ```diff
    -    def get_enabled_state():
    -        """Fetch the latest enabled state from ZMQ."""
    -        config_subscriber.check_for_updates()
    -        return config.enabled
    +    last_config_check = 0.0
    +    def get_enabled_state():
    +        nonlocal last_config_check
    +        now = time.monotonic()
    +        if now - last_config_check > 2.0:
    +            config_subscriber.check_for_updates()
    +            last_config_check = now
    +        return config.enabled
    ```

#### 🟡 CPU Overhead: Shared Value Updates Throttling
*   **Location**: [ffmpeg.py:L89-L91](file:///home/manu/Documents/github-orgs/Zektopic/frigate/frigate/video/ffmpeg.py#L89-L91)
*   **Issue**: Updating multiprocessing atomic `Value` variables on every frame requires kernel-level locks, whereas watchdogs only monitor them at 1Hz.
*   **Optimization**: Throttle updates to once per second.
*   **Code Patch**:
    ```diff
    -            fps.value = frame_rate.eps()
    -            skipped_fps.value = skipped_eps.eps()
    -            current_frame.value = datetime.now().timestamp()
    +            now_ts = datetime.now().timestamp()
    +            current_frame.value = now_ts
    +            if frame_index % int(max(config.detect.fps, 1)) == 0:
    +                fps.value = frame_rate.eps()
    +                skipped_fps.value = skipped_eps.eps()
    ```

---

### 2. Object Detection Flow (`frigate/object_detection/base.py`)

#### 🔴 CPU Bottleneck: Array Transposition Copies
*   **Location**: [base.py:L68-L78](file:///home/manu/Documents/github-orgs/Zektopic/frigate/frigate/object_detection/base.py#L68-L78)
*   **Issue**: `np.transpose` returns non-contiguous memory views. Passing this to C-bindings (ONNX/TFLite) forces a contiguous memory copy.
*   **Optimization**:
    1.  Prefer quantized integer models (`dtype: int`) to avoid float division and casting overhead.
    2.  For float models, ensure layouts are contiguous using `np.ascontiguousarray()` before passing them to detection libraries.

---

### 3. SQLite Vector Queue Database (`frigate/db/sqlitevecq.py`)

#### 🔴 Database I/O Blockers: Missing Pragmas on Background Queue Thread
*   **Location**: [sqlitevecq.py:L17-L25](file:///home/manu/Documents/github-orgs/Zektopic/frigate/frigate/db/sqlitevecq.py#L17-L25)
*   **Issue**: Background Peewee write connections do not configure optimization pragmas. Under heavy clip-writing loops on slow drives, transaction logs will block on disk syncs (`fsync`).
*   **Optimization**: Apply WAL journal modes, loose synchronization, and memory-backed temporary storage.
*   **Code Patch**:
    ```diff
         def _connect(self, *args: Any, **kwargs: Any) -> sqlite3.Connection:
             conn: sqlite3.Connection = super()._connect(*args, **kwargs)
    +        conn.execute("PRAGMA journal_mode=WAL;")
    +        conn.execute("PRAGMA synchronous=NORMAL;")
    +        conn.execute("PRAGMA mmap_size=104857600;") # 100MB memory map
    +        conn.execute("PRAGMA temp_store=MEMORY;")
             if self.load_vec_extension:
                 self._load_vec_extension(conn)
             self._register_regexp(conn)
             return conn
    ```

---

### 4. Process Management & Database Vacuuming (`frigate/app.py`)

#### 🔴 CPU Scheduling Overhead: Too Many Python Subprocesses
*   **Issue**: Spawning separate OS processes for minor background roles (`EventCleanup`, `RecordingCleanup`, `StorageMaintainer`) incurs process overhead and context-switching cost on low-core APUs.
*   **Optimization**: Set non-critical background processes (such as statistics and cleanups) to lower scheduling priorities (`os.nice(10)` or higher) to prevent them from starving the detection threads.

#### 🔴 Write Amplification: FULL Auto-Vacuum
*   **Location**: [app.py:L264](file:///home/manu/Documents/github-orgs/Zektopic/frigate/frigate/app.py#L264)
*   **Issue**: `"auto_vacuum": "FULL"` forces SQLite to reorganize pages on every deletion. With Frigate frequently deleting recording chunks, this causes massive write amplification on flash memory.
*   **Optimization**: Change `"auto_vacuum"` to `"INCREMENTAL"` or `"NONE"`. Frigate already executes a full `VACUUM` every 2 weeks, making transaction-level auto-vacuuming redundant.

---

## Part 3: CPU-Bound Features Tuning

Toggle or lower the footprint of heavier secondary features in your `frigate.yaml` configuration:

1.  **Audio Detection**:
    If unused, disable globally to avoid constant audio thread execution:
    ```yaml
    audio:
      enabled: false
    ```
2.  **Semantic Search & Embeddings**:
    Avoid loading heavy ONNX embeddings models locally if memory is below 8GB:
    ```yaml
    semantic_search:
      enabled: false
    ```
    If enabled, set `model_size: small` and disable reindexing on startup (`reindex: false`).
3.  **Generative AI (GenAI)**:
    Ensure `genai` tasks are offloaded to external APIs (Gemini/OpenAI) rather than executing local LLMs (via Llama.cpp or Ollama) on your Ryzen APU CPU.

---

## Part 4: Telegram Bridge Optimization (`frigate-telegram`)

If you are using the Go-based `frigate-telegram` bridge, adjust its configurations (`internal/config/config.go` or environment variables) to decrease resource requirements on the host:

1.  **Skip Local Media Caching & Downloading**:
    Avoid downloading heavy clips or previews by restricting notifications to text or small thumbnails:
    *   `SEND_TEXT_EVENT: True`
    *   `INCLUDE_THUMBNAIL_EVENT: True`
    *   `INCLUDE_CLIP_EVENT: False`
    *   `INCLUDE_PREVIEW_EVENT: False`
2.  **Reduce Checking Frequency**:
    Increase checking intervals to lower thread wakeups:
    *   `SLEEP_TIME`: Increase from 5s to **10s** or **15s**.
    *   `WATCH_DOG_SLEEP_TIME`: Increase from 3s to **6s**.
3.  **Scope Filters**:
    Limit the processed cameras using `FRIGATE_EXCLUDE_CAMERA` or `FRIGATE_EXCLUDE_LABEL` to prevent the Go bridge from parsing unwanted events.

---

## Part 5: Code Quality & Security Audit Findings

Below is the summary of security and logical issues discovered in your codebase:

### 1. `verify_password` AssertionError Crash / Bypass
*   **File**: `frigate/api/auth.py`
*   **Issue**: Uses `assert algorithm == PASSWORD_HASH_ALGORITHM`. If user data is corrupted or formatted with a legacy algorithm, the application will raise an unhandled `AssertionError` resulting in an HTTP 500 error. If python is run with `-O` compiler flags, the check is skipped entirely.
*   **Fix**:
    ```python
    if algorithm != PASSWORD_HASH_ALGORITHM:
        return False
    ```

### 2. CSRF Mitigation Bypass on Missing Origin Header
*   **File**: `frigate/api/fastapi_app.py`
*   **Issue**: CSRF protection returns `True` (bypasses validation) if the `Origin` header is missing from the incoming request.
*   **Fix**: Validate both `Origin` and `Referer` headers, and verify that `x-csrf-token` matches a secure session token.

### 3. VLM Watch Context Memory Leak
*   **File**: `frigate/jobs/vlm_watch.py`
*   **Issue**: If VLM responses fail to parse due to malformed JSON, the runner returns early but fails to pop the appended frame from `self.conversation`. Repeated failures lead to context window blowup and token bloat.
*   **Fix**: Add history cleanup in the `except` block to pop the last turns.
