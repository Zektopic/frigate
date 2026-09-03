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
- Fix frontend `punycode` module deprecations node warning during test execution.
- Fix `make run_tests` overlay mount error during docker build kit compilation so the actual python modules can be tested properly.
- Abandon the use of the `test_runner.py` script as it attempts to mock extremely complex C-extensions (`cv2`, `numpy`) and deep schema validation dictionaries (`pydantic` v2) which results in dozens of false positive assertions and TypeErrors. `sys.modules` overriding is inadequate for tests of this magnitude. Tests should only be executed natively in Docker.

## Frontend Test Failure Summary

When running Vitest, the E2E Playwright tests in `web/e2e/specs/` are being incorrectly swept up by the `vitest` runner. Since Vitest cannot execute Playwright suites, 20 test files fail immediately with:
`Error: Playwright Test did not expect test.describe() to be called here.`

## Suggested Improvements

1. **Scope Vitest execution**: Run tests strictly by appending `src/` to prevent e2e conflicts (e.g. `npm run test src/`).
