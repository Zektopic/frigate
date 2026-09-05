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

## Test Runner Mocks & Fallback Infrastructure Upgrades

The fallback native python test suite (`python3 test_runner.py`) uses deep `sys.modules` patching to bypass massive missing C-extensions (`cv2`, `numpy`) and application structures (`peewee`, `pydantic`). We have identified critical flaws in this environment that must be fixed to allow complete local execution:

1. **Pydantic Validation**:
   - `MockPydanticValidationError` fails to assert properly across model creation tests like `test_profiles.py`.
   - **Improvement**: Refactor `MockBaseModel` so that `setattr` and `__init__` parse fields explicitly and raise `MockPydanticValidationError` natively rather than bypassing standard python `super()` assignments.

2. **OpenCV NMSBoxes Mocking (`cv2.dnn.NMSBoxes`)**:
   - Tests in `test_video.py` involving object overlap and reduction (like `test_overlapping_objects_reduced`) fail.
   - **Improvement**: Inject a mock for `cv2.dnn.NMSBoxes` that accurately mimics non-maxima suppression output by returning indices as structured integers instead of a generic mock.

3. **Numpy Math Assertions (`numpy.prod`)**:
   - `test_shared_memory_frame_manager.py` fails when asserting frame dimensions because `numpy.prod` is unmocked or returns a `MagicMock`.
   - **Improvement**: Add an explicit side effect or static implementation for `numpy.prod` in the `numpy` module override.

4. **Shared Memory Testing Mismatches**:
   - Tests asserting caching behavior (e.g., `test_get_reopens_when_cached_segment_is_smaller_than_shape`) fail because the `UntrackedSharedMemory` mock deviates from the expected real-world API implementation (it sometimes incorrectly returns None or causes the `arr` to evaluate to None).
