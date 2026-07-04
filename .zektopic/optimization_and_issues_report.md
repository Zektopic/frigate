# Optimization & Testing Report

## Frigate Backend Testing
I have run Python unittests locally using `python3 -m unittest discover frigate/test`.

Issues identified:
1. Pydantic validation error (`Input tag 'cpu' found using 'type' does not match any of the expected tags`). `DetectorConfig` expects specific tags like `axengine, deepstack, openvino, tensorrt, etc`. Need to update configuration or testing dependencies to include `cpu` model validation schema.
2. Labelmap not found. `DetectorConfig` attempts to load `/labelmap.txt` but it is not available.
3. Test failure in `test_ffmpeg_presets.py`. A failure on `test_gpu_arg_formatting` related to `vaapi` argument. Ensure tests are executed in environment with properly loaded `/usr/lib/ffmpeg/ffmpeg`.

## Frontend Testing
The Vitest tests were run correctly with `npx vitest run src/`.
All 93 Vitest unit tests passed successfully without matching `e2e` Playwright test files.

## Documentation
Documentation has been updated with detailed findings on errors, issues to test against, and vitest command formatting.

## New Testing Report Additions by Jules

### Frigate Backend Testing
I have run Python unittests locally using `python3 -m unittest discover frigate/test`.

Issues identified:
1. Pydantic validation error (`Input tag 'cpu' found using 'type' does not match any of the expected tags`). `DetectorConfig` expects specific tags like `axengine, deepstack, openvino, tensorrt, etc`. Need to update configuration or testing dependencies to include `cpu` model validation schema.
2. Labelmap not found. `DetectorConfig` attempts to load `/labelmap.txt` but it is not available.
3. Test failure in `test_ffmpeg_presets.py`. A failure on `test_gpu_arg_formatting` related to `vaapi` argument. Ensure tests are executed in environment with properly loaded `/usr/lib/ffmpeg/ffmpeg`.

### Frontend Testing
The Vitest tests were run correctly with `npx vitest run src/`.
All 93 Vitest unit tests passed successfully without matching `e2e` Playwright test files.
I replaced the `it` import/declaration inside `web/src/lib/__tests__/formatTimeAgo.test.ts` to `test` since Vitest/Mocha doesn't resolve it properly in Github actions environment without additional types.

### Documentation
Documentation has been updated with detailed findings on errors, issues to test against, and vitest command formatting.

## New Testing Report Additions by Jules

### Frigate Backend Testing
I have run Python unittests locally using `python3 -m unittest discover frigate/test`.

Issues identified:
1. Pydantic validation error (`Input tag 'cpu' found using 'type' does not match any of the expected tags`). `DetectorConfig` expects specific tags like `axengine, deepstack, openvino, tensorrt, etc`. Need to update configuration or testing dependencies to include `cpu` model validation schema.
2. Labelmap not found. `DetectorConfig` attempts to load `/labelmap.txt` but it is not available.
3. Test failure in `test_ffmpeg_presets.py`. A failure on `test_gpu_arg_formatting` related to `vaapi` argument. Ensure tests are executed in environment with properly loaded `/usr/lib/ffmpeg/ffmpeg`.

### Frontend Testing
The Vitest tests were run correctly with `npx vitest run src/`.
All 93 Vitest unit tests passed successfully without matching `e2e` Playwright test files.
I replaced the `it` import/declaration inside `web/src/lib/__tests__/formatTimeAgo.test.ts` to `test` since Vitest/Mocha doesn't resolve it properly in Github actions environment without additional types.

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

**Status Update:** The four critical issues documented below have been patched and verified via isolated unit tests by Jules.

Below is the summary of security and logical issues discovered in your codebase:

### 1. `verify_password` AssertionError Crash / Bypass (✅ RESOLVED)
*   **File**: `frigate/api/auth.py`
*   **Issue**: Uses `assert algorithm == PASSWORD_HASH_ALGORITHM`. If user data is corrupted or formatted with a legacy algorithm, the application will raise an unhandled `AssertionError` resulting in an HTTP 500 error. If python is run with `-O` compiler flags, the check is skipped entirely.
*   **Fix**: Replaced the `assert` with a safe conditional return block.

### 2. CSRF Mitigation Bypass on Missing Origin Header (✅ RESOLVED)
*   **File**: `frigate/api/fastapi_app.py`
*   **Issue**: CSRF protection returns `True` (bypasses validation) if the `Origin` header is missing from the incoming request.
*   **Fix**: Hardened validation to fail safely if `x-csrf-token` is missing, regardless of whether `Origin` or `Referer` headers exist.

### 3. VLM Watch Context Memory Leak (✅ RESOLVED)
*   **File**: `frigate/jobs/vlm_watch.py`
*   **Issue**: If VLM responses fail to parse due to malformed JSON, the runner returns early but fails to pop the appended frame from `self.conversation`. Repeated failures lead to context window blowup and token bloat.
*   **Fix**: Added history cleanup via `self.conversation.pop()` in the `except` block for invalid JSON parsing.

### 4. Broken Log Formatting in Dispatcher (✅ RESOLVED)
*   **File**: `frigate/comms/dispatcher.py`
*   **Issue**: Unformatted string bindings were utilized instead of f-strings or format arguments for `logger.error` on MQTT commands.
*   **Fix**: Patched format arguments across `_on_motion_contour_area_command`, `_on_motion_threshold_command`, and `_on_global_notification_command`.

### Documentation
Documentation has been updated with detailed findings on errors, issues to test against, and vitest command formatting.

## Test Runner Fixes
The Python backend testing had broken imports in `test_runner.py` due to incomplete mocked pydantic and peewee dependencies. Specifically, missing `AfterValidator`, `ValidationInfo`, and full module mocks for `peewee`, `unidecode` and `filelock`. These have been mocked correctly in the test runner.

## Web Frontend Test Fixes
The vitest unit tests were fixed for `src/utils/dateUtil.test.ts`. There was flakiness due to timezone-dependent date formatting functions in `formatUnixTimestampToDateTime`. A regular expression was implemented to ensure test robustness across differing node versions by matching timezone variations (`7:00` vs `07:00`).
