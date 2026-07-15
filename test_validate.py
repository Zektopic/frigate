import sys
import unittest
from frigate.config import AuthConfig
auth = AuthConfig(roles={"house_only": ["front_door", "back_door"]})
print(auth.roles)
