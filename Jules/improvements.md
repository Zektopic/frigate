# Future Improvements & Refactoring Roadmap

Based on the full-codebase testing evaluation, here are specific features and optimizations that should be implemented in future iterations:

## 1. Backend Testing & Environment Overhaul
- **Address Docker Mount Issues**: To properly test the backend natively without brittle mocks, the host environment's Docker storage driver (e.g., switching from `overlayfs` to `vfs` or properly configuring `containerd`) must be addressed, or the project needs a specific test target that strictly disables `buildx`/BuildKit.
- **Refine the Fallback Mock Engine**: If local `test_runner.py` execution remains a requirement, significantly refactor `MockBaseModel` and `MockPydanticValidationError` to strictly adhere to Pydantic v2 core structures, handling deeply nested dictionaries and strict schema validation accurately. Provide robust dummy implementations for `numpy` matrices and shape properties.
- **Dependencies Management**: Define explicit pip requirements to prevent `ModuleNotFoundError` errors when executing tests natively without containers.

## 2. Frontend Modernization
- **Update Deprecated Dependencies**: Bump underlying packages (like `whatwg-url` or `tr46`) to their latest major versions, or implement userland alternatives, to resolve the `punycode` Node deprecation warnings and keep CI logs clean.
- **Test Isolation Configuration**: Ensure `web/vitest.config.ts` explicitly scopes unit tests (e.g., excluding `e2e/**`) to permanently prevent matcher collisions with Playwright if users execute a generic `npm test` command.

## 3. Rust Code Health
- **Clean Up Warnings**: Resolve unused variables (e.g., `AF_STRIDES` in `frigate-yolo-rs`), unused functions (`not_a_test_debug_step_by_step` in `frigate-motion-rs`), and remove unnecessary `mut` bindings highlighted by the compiler during test pipelines.

## 4. Database Optimization
- **SQLite Batching**: The database bulk benchmarks demonstrate massive gains (`~90k-120k records/sec`) when using batched inserts. Review any singular write/export queries that iterate manually inside `frigate.record.export` and replace them with `peewee` batched executions.
- **Quantization Optimization**: Evaluate INT8/quantized models specifically inside limited hardware (e.g. CPU or APU) edge cases to minimize data transposition overhead.
