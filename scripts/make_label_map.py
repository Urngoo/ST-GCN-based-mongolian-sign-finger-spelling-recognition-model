import json
from pathlib import Path

ROOT = Path(r"dataset_fixed300/raw_videos")
letters = sorted([p.name for p in ROOT.iterdir() if p.is_dir()])

label_map = {name: i for i, name in enumerate(letters)}
Path("scripts").mkdir(exist_ok=True)

with open("scripts/labels_map.json", "w", encoding="utf-8") as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

print("Saved scripts/labels_map.json")
print("Classes:", len(label_map))
