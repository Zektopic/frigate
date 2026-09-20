import re

def deduplicate_testing_updates(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # The block we appended:
    block = """
### Testing Update

All python and rust backend tests as well as web frontend tests have been successfully ran.

Backend Mock Architecture:
- To properly test the backend natively without brittle mocks, the host environment's Docker storage driver (e.g., switching from overlayfs to vfs or properly configuring containerd) must be addressed, or the project needs a specific test target that strictly disables buildx/BuildKit.
- If local test_runner.py execution remains a requirement, significantly refactor MockBaseModel and MockPydanticValidationError to strictly adhere to Pydantic v2 core structures, handling deeply nested dictionaries and strict schema validation accurately. Provide robust dummy implementations for numpy matrices and shape properties.

Frontend Modernization:
- Bump underlying packages (like whatwg-url or tr46) to their latest major versions, or implement userland alternatives, to resolve the punycode Node deprecation warnings and keep CI logs clean.
- Ensure web/vitest.config.ts explicitly scopes unit tests (e.g., excluding e2e/**) to permanently prevent matcher collisions with Playwright if users execute a generic npm test command.

Database & Video Pipeline:
- Refactor logic that loops over singular select or insert statements to utilize Peewee batch chunking for significant IO gains.
- Implement dynamic loading for INT8/quantized models to reduce overhead in ONNX/YOLO pipelines.
"""

    # Remove all instances of the exact block
    cleaned_content = content.replace(block, "")

    # Add it back exactly once at the end
    cleaned_content += block

    with open(filepath, 'w') as f:
        f.write(cleaned_content)

deduplicate_testing_updates(".zektopic/status.md")
deduplicate_testing_updates("Jules/improvements.md")
