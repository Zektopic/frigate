import re

with open("frigate/util/object.py", "r") as f:
    content = f.read()

# Make sure we have Sequence imported
if "from typing import " in content and "Sequence" not in content:
    content = content.replace("from typing import Any", "from typing import Any, Sequence")

content = content.replace("def average_boxes(boxes: list[list[int] | tuple[int, ...]]) -> list[float]:", "def average_boxes(boxes: Sequence[list[int] | tuple[int, ...]]) -> list[float]:")
content = content.replace("def median_of_boxes(boxes: list[list[int] | tuple[int, ...]]) -> list[int] | tuple[int, ...]:", "def median_of_boxes(boxes: Sequence[list[int] | tuple[int, ...]]) -> list[int] | tuple[int, ...]:")
content = content.replace("def median_of_boxes(\n    boxes: list[list[int] | tuple[int, ...]],\n) -> list[int] | tuple[int, ...]:", "def median_of_boxes(\n    boxes: Sequence[list[int] | tuple[int, ...]],\n) -> list[int] | tuple[int, ...]:")

with open("frigate/util/object.py", "w") as f:
    f.write(content)
