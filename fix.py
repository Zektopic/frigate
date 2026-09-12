import sys

with open("frigate/track/norfair_tracker.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.strip() == "avg_box = average_boxes(self.stationary_box_history[id])":
        new_lines.append("        avg_box = average_boxes(self.stationary_box_history[id])  # type: ignore[arg-type]\n")
    elif line.strip() == "median_box = median_of_boxes(self.stationary_box_history[id])":
        new_lines.append("        median_box = median_of_boxes(self.stationary_box_history[id])  # type: ignore[arg-type]\n")
    else:
        new_lines.append(line)

with open("frigate/track/norfair_tracker.py", "w") as f:
    f.writelines(new_lines)
