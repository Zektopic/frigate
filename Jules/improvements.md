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

## Backend Testing
- Test suite fails to run because dependencies mock in `test_runner.py` is incomplete. Should mock additional modules such as `peewee`, `playhouse`, `unidecode`, `filelock`, `fastapi`, `httpx`, `peewee_migrate`, `pytz`, `scipy`, `sherpa_onnx`, `zeep`, `norfair.camera_motion`, `onvif`, and some missing `pydantic` fields to fix 65 failing tests.
- Backend tests also face TypeErrors during image processing. For example, `test_crop_yuv` throws `< not supported between instances of int and MagicMock` because it's using mocked cv2 methods where mock isn't sufficient.
- Python 3.12 compatibility issues inside `test_runner.py` mocks cause Pydantic to throw `TypeError: FrigateConfig() takes no arguments`.

## Backend Testing Additions
- `test_runner.py` needs an exhaustive list of dependencies mocked to run the backend test suite successfully, missing dependencies like `peewee`, `playhouse`, `unidecode`, `filelock`, `fastapi`, `httpx`, `peewee_migrate`, `pytz`, `scipy`, `sherpa_onnx`, `zeep`, `norfair`, `onvif`, `pydantic` fields, etc.
- Image processing TypeErrors during unittests. E.g. `test_copy_yuv_to_position` uses mocked cv2 which throws type errors when comparing integers with MagicMocks.
- The `test_runner.py` mocks for Pydantic lack methods/properties required by tests on Python 3.12, causing `TypeError: FrigateConfig() takes no arguments` or `ModuleNotFoundError` for submodules like `playhouse.sqliteq`.
