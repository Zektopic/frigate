# Test Status and Future Improvements

## Current Status
- **Web tests:** Passing (137 tests over 13 suites)
- **Rust tests (detector, frame, motion, yolo):** Passing
- **Backend native (Docker):** Blocked due to BuildKit error ('invalid argument' on overlayfs mount).
- **Backend local fallback:** Blocked natively. Mocking Python's C-extensions and Pydantic V2 deeply is unfeasible without integration environment.

## Needed Features / Future Improvements
1. **Docker Environment Resolution**: Resolving the overlayfs mount error on BuildKit to unblock full backend integration tests.
2. **Front-End Maintenance**: Address punycode deprecation warnings in the web tests.
