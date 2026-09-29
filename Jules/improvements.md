# Future Improvements

Based on test runs, here are future improvements and bugfixes that should be addressed in subsequent iterations:

### 1. Docker BuildKit/Testing Envs
- Ensure tests can be successfully executed with Docker. Currently it hits an `overlayfs` mount error: `mount source: "overlay"... err: invalid argument`.
- Once Docker testing is working, `test_runner.py` and its extremely brittle `sys.modules` patch for mock dependencies (like OpenCV, Numpy, and Pydantic) should be deprecated or scaled back, as they create major false positives/negatives without proper C-extension bounds checking and schema parsing.

### 2. Frontend Modernization
- Update dependencies generating Node `DEP0040` deprecation warnings for `punycode`. This includes bumping tools such as `whatwg-url` or `tr46`.

### 3. Backend Logic
- The `sanitize_path_component` function currently doesn't pass relative path validation perfectly, which is mocked over. Ensure path traversal vulnerabilities are strictly caught.
- `cv2.dnn.NMSBoxes` loop needs better index integer type matching, which fails in the mock environment but must be handled natively for robust detection.
- Utilize batch/chunk inserts (as shown by SQLite benchmarking 100-batch ~95,000 r/s vs unbatched) inside bulk database interactions to maximize speed.

