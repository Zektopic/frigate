## Actionable Roadmap of Future Implementations and Improvements (Current Iteration)

### A. Code Quality and Typing Hygiene
- **Backend Formatting**: Automatically fix the 10 unsorted import and unused import errors across `frigate/test/` using `ruff check --fix frigate/`.
- **MyPy Typing Fixes**: Remove or update the 139 unused `type: ignore` comments flagged by MyPy. Add strong typing to the untyped `init_landmark_detector` decorator. Correct the index types (`dict[str, CameraConfig]` instead of expected `str`) and return types within `frigate/data_processing/common/license_plate/mixin.py`.
- **Frontend Formatting**: Automatically resolve the 8 Prettier formatting warnings by running `npm --prefix web run lint:fix`.

### B. Rust Component Maintenance
- **Clean Up Warnings**: Resolve the unused `Shutdown` variant within the `Msg` enum in `frigate-detector-rs/src/main.rs`. While minor, keeping the Rust codebase free of dead code warnings ensures CI remains strict. The components themselves (`frigate-detector-rs`, `frigate-frame-rs`, `frigate-motion-rs`, `frigate-yolo-rs`) boast complete test coverage with 47 passing tests.

### C. Testing Environment and Mock Infrastructure
- **Address Docker Mount Issues**: The primary blocker for testing the backend accurately natively is the host environment's Docker BuildKit `overlayfs` failure (`invalid argument`). Consider updating the host Docker storage driver or providing a fallback `make` target that completely disables BuildKit (`DOCKER_BUILDKIT=0`).
- **Refine the Fallback Mock Engine**: If local execution via `test_runner.py` remains necessary, the `MockBaseModel` and `MockPydanticValidationError` classes must be heavily refactored to support deep nested dictionary validation matching Pydantic v2 metadata requirements. Additionally, robust mock implementations for `sanitize_filename` and NumPy array multi-dimensional values and comparison operators must be created to resolve the 89 failures and 200 errors.

### D. Frontend Modernization
- **Update Deprecated Dependencies**: Bump underlying packages (like `whatwg-url` or `tr46`) to their latest major versions, or implement userland alternatives, to resolve the `DEP0040` `punycode` Node deprecation warnings and keep CI logs clean.
- **E2E Test Matcher Conflict**: Ensure unit testing relies strictly on executing against the source folder (e.g. `npm run test -- --run src/`) to avoid matcher collisions (`TypeError: Cannot redefine property: Symbol($$jest-matchers-object)`) caused by Playwright's integration suite.
