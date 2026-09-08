## Testing Status and Next Steps Update (Current Iteration)

### Frontend Tests
- Executed `npm --prefix web ci` and `npm --prefix web run test -- --run src/`.
- All 137 unit tests across 13 files passed successfully.
- Deprecation warnings (`DEP0040`) for `punycode` continue to be logged.

### Rust Backend Tests
- Executed `cargo test` against `frigate-detector-rs`, `frigate-frame-rs`, `frigate-motion-rs`, and `frigate-yolo-rs`.
- All standard and pixel pipeline tests pass completely (47 tests total). No errors or failures.
- A minor compilation warning was observed in `frigate-detector-rs` (unused variant `Shutdown` in enum `Msg`).

### Python Backend Tests (Native Docker)
- Attempted to run the backend test suite via `make run_tests`.
- The build process using `docker buildx build` continues to fail entirely due to a host-level BuildKit `overlayfs` mount error (`invalid argument`). Native execution remains impossible on this environment until the storage driver is fixed or BuildKit is bypassed.

### Python Backend Tests (Local Fallback)
- Executed `python3 test_runner.py` outside of Docker.
- The script executes 757 tests but generates significant false negatives (89 failures, 200 errors).
- Notable mocked failures:
  - `AssertionError: <ModuleMock ...> is not None` in path validation tests (`test_util_path.py`), indicating that the mock for `sanitize_filename` does not behave as expected.
  - Test failures in `test_video.py` regarding object bounding boxes (e.g., overlapping objects not reduced) likely due to mocked multidimensional array comparisons.

### Code Quality Checks
- **Type Checks**: Executed `python3 -m mypy --config-file frigate/mypy.ini frigate` after installing `mypy` and typing stubs (`types-PyYAML`, `types-requests`, `types-peewee`). Found 139 errors, primarily unused `type: ignore` comments, an untyped decorator (`init_landmark_detector`), and some incorrect indexing and typing inside `frigate/data_processing/common/license_plate/mixin.py`.
- **Backend Linters**: Executed `ruff check frigate/`. Found 10 errors, mostly un-sorted import blocks and a few unused imports (`numpy`, `SqliteVecQueueDatabase`, `intersection_over_union_rust`).
- **Frontend Linters**: Executed `npm --prefix web run lint`. Found 8 Prettier formatting warnings.

### Actionable Roadmaps
1. **Fix Code Quality Issues**:
   - Address the 10 fixable linter errors in the backend by running `ruff check --fix frigate/`.
   - Address the 8 fixable linter warnings in the frontend by running `npm --prefix web run lint:fix` (or equivalent Prettier write command).
   - Clean up unused `type: ignore` comments flagged by MyPy and fix typing issues in `frigate/data_processing/common/license_plate/mixin.py`.
2. **Improve Local Testing Environment**:
   - Resolve the Docker BuildKit `overlayfs` issue to allow `make run_tests` to execute in a true environment, bypassing the brittle `test_runner.py` mocks.
   - If local execution is required, further refine `test_runner.py` to better mock `sanitize_filename` and NumPy array comparisons.
3. **Rust Warnings**:
   - Remove or implement the unused `Shutdown` variant in `frigate-detector-rs/src/main.rs`.
4. **Frontend Deprecations**:
   - Update underlying dependencies (like `whatwg-url` or `tr46`) to mitigate the `punycode` deprecation warnings.