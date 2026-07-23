from frigate.config import ProxyConfig

proxy = ProxyConfig(
    header_map={"user": "Remote-User", "role": "Remote-Role"}, separator=","
)
print(proxy.header_map)
