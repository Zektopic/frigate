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

### Documentation
Documentation has been updated with detailed findings on errors, issues to test against, and vitest command formatting.

### Backend Testing Updates (test_runner.py mocks)
I have run Python unittests locally using `python3 test_runner.py` and identified further test mock failures.
The mocks for `BaseModel` and `unidecode` were incomplete.
- We fixed the `BaseModel` mock to include an `__init__` constructor that accepts `**kwargs`.
- We fixed the `unidecode` mock to map accented characters to non-accented ones.

### Final Testing Run Report
**Backend Tests:**
- Running `python3 test_runner.py` yields 28 failures and 198 errors out of 710 tests.
- Mocks like `UntrackedSharedMemory`, `MockPydanticValidationError`, and `cv2` lack necessary depth causing widespread failures (e.g., shape tests and bounding box NMS).
- The `make run_tests` Docker environment is still blocked by an `overlayfs` invalid argument during BuildKit execution. It is highly recommended to prioritize fixing the Docker environment to avoid these brittle python mocks in the future.

**Frontend Tests:**
- Scoped frontend tests (`cd web && npm ci && npx vitest run src/`) passed flawlessly with 138 passing tests.
- Punycode deprecation warnings still exist and should be addressed in future dependencies updates.
