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

## Backend Testing - Extended Errors
- `test_zone_aggregate_blocked_for_restricted` and `test_zone_aggregate_visible_to_admin` fail checking WebSocket message permissions.
- `test_get_event_snapshot_bytes_reads_clean_webp` returns an `AssertionError` verifying parsed image properties.
- `test_skip_motion_threshold_default` expects motion boxes when skip threshold is disabled, returning none.
- OpenCV (`cv2`) mocked configurations within `test_mask_matches_scaled_dims_and_has_coverage` continue returning MagicMock types inside resizing functionalities.
- `test_non_overlapping_objects_not_reduced` bounding box length reduction mismatch.

## Further Action Required
- Fully populate API mock dependencies such as `httpx` to properly assert API integrations instead of `NoneType` responses.
- Fix tuple dimension shapes for OpenCV operations such as `cv2.resize`.
