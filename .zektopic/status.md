# Test Environment Setup and Code Review Request

All of the previous unit tests failures (except the network/requests errors related to go2rtc connection which shouldn't block the logic validation) have been resolved.

- Fixed Pandas version incompatibility.
- Fixed `AttributeError: 'NoneType' object has no attribute 'shape'` by properly checking if frame is `None` before referencing its `shape`.
- Fixed start_time discrepancy bug by converting Pandas DatetimeIndex properly back into UNIX timestamps in `motion_activity()`.
- Fixed the `test_gpu_arg_formatting` test failure by using `"preset-vaapi"` instead of `"hwaccel_vaapi"`.

Ready to create the execution plan.

## Test Environment Setup and Code Review Request (Update)

Tested backend functionality with unit tests using `python3 test_runner.py`. Identified multiple failures and errors primarily stemming from incomplete mocking in `test_runner.py`.
- **Permission Errors**: Tests fail trying to create `/config/model_cache` due to lack of permissions. Setting `CONFIG_DIR` to a temporary writable location (e.g. `/tmp/config`) resolves some but not all hardcoded `/config` issues.
- **Pydantic Validation Mocks**: Profile tests expecting `pydantic.ValidationError` fail because `MockPydanticValidationError` is not properly integrated into the `MockPydantic` validation logic.
- **Missing Module Mocks**: `norfair.drawing.draw_boxes` is missing from `test_runner.py`, causing `frigate.video` import failures.
- **Mock Return Values**: Several tests fail because they expect specific return types (integers, strings) but receive `MagicMock` objects (e.g. `unidecode()`, `ndarray.shape`, `cv2.cvtColor().shape`).
- **Logic Mismatches**: Mocked behavior deviates from expected execution in areas like shared memory management and go2rtc restricted source checks.

Tested frontend functionality using `npm run test` (Vitest).
- Running tests from root `web/` causes collisions with Playwright `e2e/` tests due to conflicting `test.describe()`.
- Successfully ran isolated tests with `npm run test src/` (115 passing tests across 10 suites).

Documented the necessary fixes and testing mock improvements in `Jules/improvements.md` for future implementation.

## Test Environment Setup and Code Review Request (Update)

Tested backend functionality with unit tests using `python3 test_runner.py`. Identified multiple failures and errors primarily stemming from incomplete mocking in `test_runner.py`.
- **Pydantic Validation Mocks**: Fixed `test_runner.py` to properly raise `MockPydanticValidationError` inside `MockPydantic.v1.BaseModel`. However, there are still some `MockPydanticValidationError not raised` failures in test_profiles.py.
- **Missing Module Mocks**: `norfair.drawing.draw_boxes` is missing from `test_runner.py`, causing `frigate.video` import failures.
- **Mock Return Values**: Several tests fail because they expect specific return types (integers, strings) but receive `MagicMock` objects (e.g. `unidecode()`, `ndarray.shape`, `cv2.cvtColor().shape`).
- **Logic Mismatches**: Mocked behavior deviates from expected execution in areas like shared memory management and go2rtc restricted source checks.

Tested frontend functionality using `npm run test` (Vitest).
- Running tests from root `web/` causes collisions with Playwright `e2e/` tests due to conflicting `test.describe()`.
- Successfully ran isolated tests with `npm run test src/` (115 passing tests across 10 suites).

Documented the necessary fixes and testing mock improvements in `Jules/improvements.md` for future implementation.
