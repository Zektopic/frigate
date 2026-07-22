import sys
import unittest
from frigate.config import AuthConfig, ProxyConfig
from frigate.config.proxy import HeaderMappingConfig

proxy = ProxyConfig(
    header_map={"user": "Remote-User", "role": "Remote-Role"}, separator=","
)
print(proxy.header_map)
