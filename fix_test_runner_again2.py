with open("test_runner.py", "r") as f:
    content = f.read()

replacement = """sys.modules["ruamel"] = ModuleMock()
sys.modules["ruamel.yaml"] = ModuleMock()
sys.modules["filelock"] = ModuleMock()"""

content = content.replace('sys.modules["filelock"] = ModuleMock()', replacement)

replacement2 = """class ModuleMock(MagicMock):
    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return True

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

content = content.replace("""class ModuleMock(MagicMock):
    def __lt__(self, other):
        return False

    def __gt__(self, other):
        return True

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
        return super().__getattr__(name)""", replacement2)

replacement3 = """sys.modules["zeep"] = ModuleMock()
sys.modules["zeep"].__version__ = "1.0.0" """

content = content.replace('sys.modules["zeep"] = ModuleMock()', replacement3)


with open("test_runner.py", "w") as f:
    f.write(content)
