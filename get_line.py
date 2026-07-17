with open("frigate/test/test_ws_outbound_filter.py", "r") as f:
    for i, line in enumerate(f.readlines()):
        if "proxy=" in line:
            print(f"{i}: {line}")
