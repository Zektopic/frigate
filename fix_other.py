with open("test_runner.py", "r") as f:
    content = f.read()

replacement = """class ModuleMock(MagicMock):
    def __getattr__(self, name):
        if name in (
            "__path__",
            "__file__",
            "__loader__",
            "__spec__",
            "__name__",
            "__mro_entries__",
            "__origin__",
            "__args__",
            "__parameters__",
        ):
            raise AttributeError(name)
        if name == "DEFAULT_VERSION":
            return 2 # To fix regex KeyError
        return super().__getattr__(name)"""

content = content.replace('class ModuleMock(MagicMock):\n    def __getattr__(self, name):\n        if name in (\n            "__path__",\n            "__file__",\n            "__loader__",\n            "__spec__",\n            "__name__",\n            "__mro_entries__",\n            "__origin__",\n            "__args__",\n            "__parameters__",\n        ):\n            raise AttributeError(name)\n        return super().__getattr__(name)', replacement)

replacement2 = """sys.modules["zeep"] = ModuleMock()
sys.modules["zeep"].__version__ = "1.0.0" """

content = content.replace('sys.modules["zeep"] = ModuleMock()', replacement2)

with open("test_runner.py", "w") as f:
    f.write(content)
