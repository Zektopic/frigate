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
- Fix python unit test mock errors. Improve `test_runner.py` to fully mock `peewee`, `pydantic` (with `AfterValidator`, `ValidationInfo`), `unidecode`, and `filelock`.
- Ensure frontend vitest tests use explicitly passed timezones when evaluating formatting, preventing timezone-dependent flakiness across platforms.
