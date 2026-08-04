with open("test_runner.py", "r") as f:
    content = f.read()

replacement = """sys.modules["frigate.util.services"] = ModuleMock()
sys.modules["frigate.util.services"]._go2rtc_arbitrary_exec_allowed = True"""

content += "\n" + replacement + "\n"

with open("test_runner.py", "w") as f:
    f.write(content)
