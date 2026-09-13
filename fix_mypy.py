import sys
import re

with open('frigate/track/norfair_tracker.py', 'r') as f:
    content = f.read()

# Fix types in average_boxes and median_of_boxes
# Since it was already modified in frigate/util/object.py we just need to fix here
content = content.replace("list[list[int] | tuple[int, ...]]", "typing.Sequence[list[int] | tuple[int, ...]]")

with open('frigate/track/norfair_tracker.py', 'w') as f:
    f.write(content)
