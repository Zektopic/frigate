# Future Improvements

Based on the test runs and codebase review, here are several suggested improvements to implement in the future:

1. **Dependency Management**: Ensure consistent versions of libraries (`pandas`, `numpy`, `peewee`, etc.) are pinned in requirements or `pyproject.toml` to avoid the "ModuleNotFoundError" or "AttributeError" exceptions encountered during the test runs. The environment was missing multiple dependencies initially.
2. **Robust Error Handling for Media API**: Add broader safety checks when accessing `frame.shape` in `frigate/api/media.py`. Returning an empty black frame rather than a 500 error when `frame is None` might be a safer approach for live feeds.
3. **Pydantic Upgrades**: Several tests emitted warnings about `parse_obj_as` being deprecated in Pydantic v2.0 (`PydanticDeprecatedSince20`). Refactoring these to use `TypeAdapter.validate_python()` will future-proof the config loading system.
4. **FastAPI Lifespan Events**: There are deprecation warnings for using `@app.on_event("startup")` in `frigate/api/fastapi_app.py`. Migrating to the newer `lifespan` event handlers will ensure compatibility with future FastAPI versions.
5. **Data Downsampling Edge Cases**: In `frigate/api/review.py`, resampling Pandas `DatetimeIndex` structures back into UNIX timestamps manually required subtracting a specific epoch offset. Using a more native timestamp conversion method, or caching normalized timestamps could improve performance for larger summary queries.
# Future Improvements and Features
## Authentication and Security
- Implement password strength validation for new passwords.
- Consider adding rate limiting on login endpoints to prevent brute-force attacks.

## Optimizations
- Add incremental vacuuming configuration to SQLite instead of full to reduce wear on flash memory.
- Reduce frequency of background processes on low-core APUs.
- Investigate using zero-copy reading in the ffmpeg video ingestion loop.

## Frontend
- Verify input validation across all config forms.

## Testing Improvements
- Test suite fails to run because dependencies mock in `test_runner.py` is incomplete. Should mock additional modules such as `peewee`, `playhouse`, `unidecode`, `filelock`, `fastapi`, `httpx`, `peewee_migrate`, `pytz`, `scipy`, `sherpa_onnx`, `zeep`, `norfair.camera_motion`, `onvif`, and some missing `pydantic` fields to fix 65 failing tests.
- Backend tests also face TypeErrors during image processing. For example, `test_crop_yuv` throws `< not supported between instances of int and MagicMock` because it's using mocked cv2 methods where mock isn't sufficient.
- Python 3.12 compatibility issues inside `test_runner.py` mocks cause Pydantic to throw `TypeError: FrigateConfig() takes no arguments`.
- Fix python unit test mock errors. Improve `test_runner.py` to fully mock `peewee`, `pydantic` (with `AfterValidator`, `ValidationInfo`), `unidecode`, and `filelock`.
- Ensure frontend vitest tests use explicitly passed timezones when evaluating formatting, preventing timezone-dependent flakiness across platforms.

## Backend Testing Mocks and Fixes
- `test_runner.py` needs better Pydantic mock implementations to ensure that `ValidationError` is correctly raised and caught in `frigate.test.test_profiles`. Currently, test suite fails because `MockPydanticValidationError not raised`.
- `os.makedirs` should be mocked or the `CONFIG_DIR` should point to a writable temporary directory in `frigate.test.test_profiles` to avoid `PermissionError: [Errno 13] Permission denied: '/config'`.
- Add a mock for `norfair.drawing.draw_boxes` to fix the `ModuleNotFoundError` during `frigate.video` module import.
- Improve mock for `cv2.cvtColor().shape` to return an actual tuple of integers instead of a `MagicMock` so that `test_video.py` and `test_yuv_region_2_rgb.py` do not fail when comparing shapes/regions.
- Improve mock for `np.ndarray().shape` in `test_shared_memory_frame_manager.py` to return the expected dimensions.
- Improve mock for `unidecode.unidecode` so it returns correct strings for assertions in `test_video.py`.
- Improve the mock for `pydantic.ValidationError` in `test_runner.py` by making it correctly triggerable from within mocked Pydantic components.
- In `test_shared_memory_frame_manager.py`, `UntrackedSharedMemory` mock is being called when tests expect it to not have been called.
- In `test_proxy_auth.py`, `auth_secret` env variable substitution is not correctly mocked/functioning, leading to mismatched string assertions.
- Peewee chunked queries might need mocked responses rather than generic MagicMocks.
- Fix `peewee` mock in `test_runner.py` because currently `from peewee import *` or specific imports fail with `ModuleNotFoundError: No module named 'peewee'` which causes all `test_http_*.py` tests to error out due to import failure.

## Frontend Testing Fixes
- Playwright E2E tests (`e2e/specs/**/*.spec.ts`) fail when run with Vitest (`vitest`) because they contain `test.describe()`, which conflicts with Vitest's `describe`. Need to ensure `vitest` only runs on `src/` directory and ignores `e2e/` folder.

## Recent Test Run Results
- Tests were run, and some mocks in `test_runner.py` were identified to be missing or returning incorrect values (e.g. MagicMock instead of tuple for `.shape`).
- Backend tests were run (`python3 test_runner.py`), resulting in failures related to `MockPydanticValidationError`, `os.makedirs(MODEL_CACHE_DIR)` permission errors in `/config`, missing mock methods on `cv2`, `unidecode`, and more.
- Frontend tests were successfully run isolated (`cd web && npm run test src/`) passing all 115 tests.

## Final Improvements
- Enhanced testing environment by explicitly returning integer tuples for `cv2.cvtColor().shape` to allow downstream assertions to work correctly.
- Added a basic `cv2.dnn.NMSBoxes` implementation in mocks to support tests utilizing `reduce_detections`.
- Note: Tests involving `numpy` and `Pydantic` mock behaviors still need refinement. Mocking `numpy.array().shape` and `numpy.prod` is essential to prevent `MagicMock` instances from causing comparison assertion errors (e.g. `AssertionError: <MockNumpy name='mock.ndarray().shape'> != (1620, 1920)`).

### Backend Testing Updates (test_runner.py mocks)
I have run Python unittests locally using `python3 test_runner.py` and identified further test mock failures.
The mocks for `BaseModel` and `unidecode` were incomplete.
- We fixed the `BaseModel` mock to include an `__init__` constructor that accepts `**kwargs`.
- We fixed the `unidecode` mock to map accented characters to non-accented ones.
- However, there remain multiple tests failing due to Missing dependencies such as `filelock`, `numpy`, and `requests` preventing smooth imports, and `TypeError` when dealing with `cv2` properties being MagicMocks when integers were expected.
- We need to accurately mock `RootModel` on `pydantic` in `test_runner.py`.
- Shared Memory frame manager caching test assumptions are being violated due to `UntrackedSharedMemory` being called.

### Test Outcomes
- All frontend vitest components pass correctly.
- Ensure backend test runner correctly applies patches to `sys.modules` without throwing `ModuleNotFoundError`.

## Backend Unit Test Mock State
Extensive effort was put into resolving mocked dependencies (`numpy`, `cv2`, `pydantic_core.ValidationError`, `unidecode`) for `test_runner.py`.
Although some mock updates worked, `pydantic.ValidationError` still isn't fully mocked for `test_profiles.py` because `MockPydanticValidationError` isn't structurally equivalent to `pydantic_core.ValidationError` which is now the base exception in Pydantic v2. Similarly, mock numpy array indexing behaviors clash with backend logic.
Future action: It is highly recommended to run backend tests solely via the Docker environment (`make run_tests`) to avoid brittle sys.modules patching for complex C-extensions like `cv2` and `numpy`.
