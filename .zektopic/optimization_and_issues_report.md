## Testing Status and Roadmap Update

### 1. Frontend Tests
- **Execution**: Successfully ran `cd web && npm ci && npm run test -- --run src/`.
- **Result**: All 138 tests across 13 files passed locally.
- **Issues Found**: Node deprecation warnings (e.g., `DEP0040`) for the `punycode` module are present in the output. This is caused by older dependencies (like `tr46` or `whatwg-url`) that need to be bumped.

### 2. Python Backend Tests (Local Runner)
- **Execution**: Executed `python3 test_runner.py` outside of the Docker container.
- **Result**: The script executes but results in numerous errors and failures (e.g., ~28 failures, ~198 errors out of 710 tests).
- **Issues Found**: The root cause is the extreme complexity of correctly mocking C-extensions (`numpy`, `cv2`, `librosa`) and accurately replicating Pydantic v2 schemas and validation logic using simple `sys.modules` patching. For instance, `MockPydanticValidationError` fails to accurately replicate Pydantic v2's schema validation, leading to false positives in configuration tests (e.g., `test_profiles.py`).

### 3. Python Backend Tests (Docker / Native)
- **Execution**: Attempted to run the backend test suite via `make run_tests` (which relies on `docker buildx build`).
- **Result**: The build process fails on the host environment.
- **Issues Found**: A Docker BuildKit error related to `overlayfs` mounts (`mount source: "overlay"... err: invalid argument`) prevents the container from building. Even with `DOCKER_BUILDKIT=0`, the Makefile explicitly invokes `buildx`, which bypasses the flag and still encounters the storage driver error.

### 4. Mypy Type Checking (Local)
- **Execution**: Installed `mypy` and ran `python3 -m mypy --config-file frigate/mypy.ini frigate`.
- **Result**: Found 61 type errors.
- **Issues Found**: Errors include subclassing `Any` (e.g., `BaseModel`, `SqliteQueueDatabase`), returning `Any` from explicitly typed functions, indexable type issues, and numerous unused `type: ignore` comments.

### 5. Rust Tests
- **Execution**: Executed `cargo test` across `frigate-detector-rs`, `frigate-frame-rs`, `frigate-motion-rs`, and `frigate-yolo-rs`.
- **Result**: All 26 tests passed successfully.
- **Issues Found**: A few minor warnings for unused code or unnecessary `mut` variables exist in the Rust libraries.

---

## Actionable Roadmap of Future Implementations and Improvements

#### A. Backend & Mock Architecture Refactoring
- **Address Docker Mount Issues**: To properly test the backend natively without brittle mocks, the host environment's Docker storage driver (e.g., switching from `overlayfs` to `vfs` or properly configuring `containerd`) must be addressed, or the project needs a specific test target that strictly disables `buildx`/BuildKit.
- **Refine the Fallback Mock Engine**: If local `test_runner.py` execution remains a requirement, significantly refactor `MockBaseModel` and `MockPydanticValidationError` to strictly adhere to Pydantic v2 core structures, handling deeply nested dictionaries and strict schema validation accurately. Provide robust dummy implementations for `numpy` matrices and shape properties.
- **Resolve Mypy Strictness**: Clean up the 61 reported mypy errors. Remove unused `type: ignore` directives, explicitly type `BaseModel` and `SqliteQueueDatabase` implementations if possible, and ensure functions returning `Any` are correctly strongly typed or ignored properly.

#### B. Frontend Modernization
- **Update Deprecated Dependencies**: Bump underlying packages (like `whatwg-url` or `tr46`) to their latest major versions, or implement userland alternatives, to resolve the `punycode` Node deprecation warnings and keep CI logs clean.
- **Test Isolation configuration**: Ensure `web/vitest.config.ts` explicitly scopes unit tests (e.g., excluding `e2e/**`) to permanently prevent matcher collisions with Playwright if users execute a generic `npm test` command.

#### C. Rust Optimizations
- **Clean Up Warnings**: Resolve the unused variables (e.g., `AF_STRIDES` in `frigate-yolo-rs`), unused functions (`not_a_test_debug_step_by_step` in `frigate-motion-rs`), and remove unnecessary `mut` bindings highlighted by the compiler during tests.

#### D. Database & Video Pipeline
- **Utilize Bulk Operations**: Given the high throughput demonstrated in SQLite batch benchmarks, refactor logic that loops over singular `select` or `insert` statements (e.g., in `frigate.record.export`) to utilize Peewee batch chunking for significant IO gains.
- **Quantized Model Loading**: For CPU-constrained or APU setups, implement dynamic loading for INT8/quantized models to reduce overhead in ONNX/YOLO pipelines (e.g., minimizing `np.transpose` contiguous copy bottlenecks).