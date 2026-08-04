with open("test_runner.py", "r") as f:
    content = f.read()

replacement = """class MockModel:
    ValidationError = MockPydanticValidationError
    @classmethod
    def select(cls):
        return MagicMock()
"""

content = content.replace("class MockModel:\n    ValidationError = MockPydanticValidationError\n    select = MagicMock()\n", replacement)

with open("test_runner.py", "w") as f:
    f.write(content)
