with open("test_runner.py", "r") as f:
    lines = f.readlines()

with open("test_runner.py", "w") as f:
    for line in lines:
        if "sys.modules[\"ruamel\"] = ModuleMock()" in line or "sys.modules[\"ruamel.yaml\"] = ModuleMock()" in line:
            continue
        f.write(line)
