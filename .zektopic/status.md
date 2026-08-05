# Test Results and Improvements

## Backend Test Failure Summary

Currently, running tests via `test_runner.py` outside of the Docker container fails on numerous imports, missing dependencies, and incorrect mocks (e.g. OpenCV, OpenVINO, Ruamel YAML, PyTorch, RKNN, etc.). There are 23 failures and 240 errors.

A major reason is that Frigate's codebase relies on a heavily populated Linux environment (Docker image `frigate:latest`), which includes a vast number of AI/ML libraries, database dependencies (peewee), hardware acceleration interfaces, and video encoding (ffmpeg) binaries. Mocks exist in `test_runner.py` but they are incomplete or fail to properly mock newer versions of the libraries, especially Pydantic v2 and `ruamel.yaml`.

## Suggested Improvements

1. **Improve the mock environment in `test_runner.py`**:
   The current test runner lacks accurate mocks for:
   - `ruamel.yaml`: Missing mock attributes like `indent` for YAML generation and saving, preventing config tests from running.
   - `pydantic`: Missing proper serialization features and attributes on mocked classes. Tests relying heavily on validation often fail when validating nested structures.
   - Pydantic models in `frigate/config/config.py` break easily if standard `typing` methods are mocked incorrectly.
   - Database operations: The `peewee` mock fails when calling `Recordings.select`, breaking `frigate.record.export`.

2. **Docker Build Issues**:
   - `make run_tests` fails locally on BuildKit `overlayfs` mounts when run without specific Docker configurations. Ensuring Docker can cleanly build and cache layers in CI and local setups is critical for natively running tests.

3. **Backend Optimization Opportunities**:
   - Database bulk insert benchmarks are relatively fast (115,000 r/s with batch=100), but `frigate/record/export.py` currently relies on un-batched `select` queries on large `Recordings` datasets. Batch fetching could be optimized.
   - GPU stats collection fails on platforms without DRM fdinfo. Add graceful fallback.
   - Detection pre-processing has high max latency (11.3ms vs avg 0.5ms) due to cold starts. We can warm up inference engines or pre-allocate memory buffers.


## Frontend Test Failure Summary

When running Vitest, the E2E Playwright tests in `web/e2e/specs/` are being incorrectly swept up by the `vitest` runner. Since Vitest cannot execute Playwright suites, 20 test files fail immediately with:
`Error: Playwright Test did not expect test.describe() to be called here.`

## Suggested Improvements

1. **Scope Vitest execution**:
   To prevent Vite from running Playwright E2E tests, `npm run test` in `web/package.json` should be updated to strictly target `src/` (e.g. `vitest run src/`).

