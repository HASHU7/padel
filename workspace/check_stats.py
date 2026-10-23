import json
from pathlib import Path

res_dir = Path("workspace/results/the_wildest_match")
shots = json.loads((res_dir / "shots.json").read_text())
points = json.loads((res_dir / "points.json").read_text())
print(f"Total shots: {len(shots)}")
print(f"Total points: {len(points)}")

with_point = [s for s in shots if s.get("point") is not None]
print(f"Shots with point != None: {len(with_point)}")

for i, p in enumerate(points):
    print(f"Point {i}: start={p['start_s']}s end={p['end_s']}s shots={p['rally_length']}")
