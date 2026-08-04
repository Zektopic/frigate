with open("test_runner.py", "r") as f:
    content = f.read()

replacement = """class MockModel:
    ValidationError = MockPydanticValidationError
    select = MagicMock()
"""

content = content.replace("class MockModel:\n    ValidationError = MockPydanticValidationError", replacement)

with open("test_runner.py", "w") as f:
    f.write(content)
