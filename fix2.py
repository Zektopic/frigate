with open("test_runner.py", "r") as f:
    content = f.read()

replacement_peewee = """class MockModel:
    ValidationError = MockPydanticValidationError

    @classmethod
    def select(cls):
        return MagicMock()

    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return True
"""
content = content.replace("class MockModel:\n    ValidationError = MockPydanticValidationError\n    \n    @classmethod\n    def select(cls):\n        return MagicMock()\n\n    def __lt__(self, other):\n        return False\n        \n    def __gt__(self, other):\n        return True\n\n\n    @classmethod\n    def select(cls):\n", replacement_peewee)

with open("test_runner.py", "w") as f:
    f.write(content)
