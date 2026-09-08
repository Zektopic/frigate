with open("test_runner.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_lines.append(line)
    if "class MockModel:" in line:
        new_lines.append("    @classmethod\n")
        new_lines.append("    def select(cls):\n")
        new_lines.append("        return MagicMock()\n")
        new_lines.append("    def __lt__(self, other):\n")
        new_lines.append("        return False\n")
        new_lines.append("    def __gt__(self, other):\n")
        new_lines.append("        return True\n")
    if "class ModuleMock(MagicMock):" in line:
        new_lines.append("    def __lt__(self, other):\n")
        new_lines.append("        return False\n")
        new_lines.append("    def __gt__(self, other):\n")
        new_lines.append("        return True\n")
    if "sys.modules[\"zeep\"] = ModuleMock()" in line:
        new_lines.append("sys.modules[\"zeep\"].__version__ = \"1.0.0\"\n")
    if "sys.modules[\"frigate.util.services\"] = ModuleMock()" in line:
        # Check if next line is already what we want
        pass


with open("test_runner.py", "w") as f:
    f.writelines(new_lines)
