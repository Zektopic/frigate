with open("test_runner.py", "r") as f:
    content = f.read()

replacement = """sys.modules["ruamel"] = ModuleMock()
sys.modules["ruamel.yaml"] = ModuleMock()
sys.modules["filelock"] = ModuleMock()"""

content = content.replace('sys.modules["filelock"] = ModuleMock()', replacement)

with open("test_runner.py", "w") as f:
    f.write(content)
