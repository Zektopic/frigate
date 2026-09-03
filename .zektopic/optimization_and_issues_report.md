# Optimization & Testing Report

## Frigate Backend Testing
I have run Python unittests locally using `python3 test_runner.py`.

Issues identified:
1. Pydantic validation error (`Input tag 'cpu' found using 'type' does not match any of the expected tags`). `DetectorConfig` expects specific tags like `axengine, deepstack, openvino, tensorrt, etc`. Need to update configuration or testing dependencies to include `cpu` model validation schema.
2. Labelmap not found. `DetectorConfig` attempts to load `/labelmap.txt` but it is not available.
3. Test failure in `test_ffmpeg_presets.py`. A failure on `test_gpu_arg_formatting` related to `vaapi` argument. Ensure tests are executed in environment with properly loaded `/usr/lib/ffmpeg/ffmpeg`.
4. Tests inside `test_runner.py` are extremely brittle because it uses mock implementations that lack full functionality, causing ~198 errors out of 710 test files. `test_runner.py` should be abandoned and unittests should be executed strictly on the docker environment.
5. The `make run_tests` command fails on local docker environments with an `overlayfs mount invalid argument` error in Buildkit. The storage driver for Docker might need to be reconfigured.

## Rust Module Testing
Ran `cargo test` in all subdirectories `frigate-detector-rs`, `frigate-frame-rs`, `frigate-motion-rs`, and `frigate-yolo-rs`. All passed completely.

## Frontend Testing
The Vitest tests were run correctly with `npm run test src/` to bypass Playwright e2e conflicts.
All 138 Vitest unit tests passed successfully. Node dependency deprecation for `punycode` is throwing warnings.

## Documentation
Documentation has been updated with detailed findings on errors, issues to test against, and vitest command formatting.
