import re

with open("frigate/genai/__init__.py", "r") as f:
    content = f.read()

if "from __future__ import annotations" not in content:
    content = "from __future__ import annotations\n" + content

with open("frigate/genai/__init__.py", "w") as f:
    f.write(content)
