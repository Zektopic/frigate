with open("test_runner.py", "r") as f:
    content = f.read()

replacement = """class MockModel:
    ValidationError = MockPydanticValidationError

    @classmethod
    def select(cls):
        return MagicMock()

    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return True
"""
# Replace duplicate class definition with single
import re
content = re.sub(r'class MockModel:.*?peewee_mock = ModuleMock\(\)', replacement + '\n\npeewee_mock = ModuleMock()', content, flags=re.DOTALL)

with open("test_runner.py", "w") as f:
    f.write(content)
