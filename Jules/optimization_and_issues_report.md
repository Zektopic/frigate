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

### Backend Testing Updates (test_runner.py & make run_tests)
I have run Python unittests locally using `python3 test_runner.py` and attempted to run natively via `make run_tests`.

Issues identified:
1. Docker BuildKit `overlayfs` issue (`mount source: "overlay"... err: invalid argument`) prevents running `make run_tests` locally, blocking native backend test execution.
2. The `python3 test_runner.py` mock environment continues to fail with 23 failures and 240 errors (out of 682 tests). This is due to missing dependencies and incomplete mocks for `requests`, `peewee`, `numpy`, `cv2`, `filelock`, etc.
3. Complex nested Pydantic validation and serialization flows in tests (e.g. `test_profiles.py`) are virtually impossible to mock perfectly via `sys.modules`, as the actual Pydantic schema engines are missing.

### Frontend Testing (Vitest)
The Vitest tests were run correctly with `cd web && npm ci && npm run test src/`.
All 138 tests across 13 test files passed successfully in isolation without matching `e2e` Playwright test files.
