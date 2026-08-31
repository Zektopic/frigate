with open("test_runner.py", "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "sys.modules[\"unidecode\"] = ModuleMock()" in line:
        lines[i] = "class MockUnidecodeModule(ModuleMock):\n    def unidecode(self, text: str) -> str:\n        return (\n            text.replace(\"é\", \"e\")\n            .replace(\"è\", \"e\")\n            .replace(\"ê\", \"e\")\n            .replace(\"á\", \"a\")\n            .replace(\"í\", \"i\")\n            .replace(\"ó\", \"o\")\n            .replace(\"ú\", \"u\")\n            .replace(\"ñ\", \"n\")\n        )\nsys.modules[\"unidecode\"] = MockUnidecodeModule()\n"

with open("test_runner.py", "w") as f:
    f.writelines(lines)
