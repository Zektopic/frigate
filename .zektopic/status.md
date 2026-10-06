# Testing Status Update

## 1. Test Results Summary

- **Web Frontend Tests**: Ran using `npm ci && npm run test -- --run src/` in the `web/` directory. All 137 unit tests across 13 files passed successfully. The `DEP0040` deprecation warnings for the `punycode` module are present and need to be addressed in the future.
- **Rust Component Tests**: Ran `cargo test` in all four Rust component directories (`frigate-detector-rs`, `frigate-frame-rs`, `frigate-motion-rs`, `frigate-yolo-rs`). All 26 tests passed successfully without any errors or failures.
- **Python Backend Tests**: Ran using the local `test_runner.py` fallback (`python3 test_runner.py`). There are still ~180 errors and failures. Native execution via `make run_tests` is still blocked by the Docker BuildKit `overlayfs` mount error locally, requiring use of the brittle mock framework.

## 2. Testing Constraints Overview
- **Docker Mount Issue**: `make run_tests` fails early because of Docker BuildKit `overlayfs` mount restrictions on the host sandbox (`invalid argument` on overlay mount).
- **Mock Limits**: The ad-hoc unit test runner `test_runner.py` is brittle and fails native backend validation due to complex nested schema validations (e.g. `pydantic` V2 core limits, config parsing dictionary fallbacks) and multidimensional C-extension asserts (`cv2.cvtColor().shape`, `numpy.ndarray`).
