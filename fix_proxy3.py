with open("frigate/test/test_ws_outbound_filter.py", "r") as f:
    lines = f.readlines()
lines[78] = '        proxy={"header_map": {"user": "Remote-User", "role": "Remote-Role"}, "separator": ","},\n'
with open("frigate/test/test_ws_outbound_filter.py", "w") as f:
    f.writelines(lines)
