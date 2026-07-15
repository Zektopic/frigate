with open("frigate/test/test_ws_outbound_filter.py", "r") as f:
    content = f.read()

content = content.replace(
    'proxy={"header_map": {"user": "Remote-User", "role": "Remote-Role"}, "separator": ",", "default_role": "viewer", "logout_url": None, "auth_secret": None}',
    'proxy={"header_map": {"user": "Remote-User", "role": "Remote-Role"}, "separator": ","}'
)

with open("frigate/test/test_ws_outbound_filter.py", "w") as f:
    f.write(content)
