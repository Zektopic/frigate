import re

with open('frigate/track/norfair_tracker.py', 'r') as f:
    content = f.read()

content = re.sub(
    r'avg_box = average_boxes\(self\.stationary_box_history\[id\]\)',
    'avg_box = average_boxes([tuple(b) for b in self.stationary_box_history[id]])',
    content
)

content = re.sub(
    r'median_box = median_of_boxes\(self\.stationary_box_history\[id\]\)',
    'median_box = median_of_boxes([tuple(b) for b in self.stationary_box_history[id]])',
    content
)

with open('frigate/track/norfair_tracker.py', 'w') as f:
    f.write(content)
