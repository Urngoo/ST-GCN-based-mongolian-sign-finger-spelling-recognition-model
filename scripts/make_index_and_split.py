import json
import random
import csv
from pathlib import Path

KEYPOINT_ROOT = Path(r"dataset/processed/keypoints")
LABEL_MAP_PATH = Path(r"scripts/labels_map.json")
OUT_CSV = Path(r"dataset/processed/labels.csv")

random.seed(42)

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

# Collect all npy files
rows = []
for npy in KEYPOINT_ROOT.rglob("*.npy"):
    # path like: keypoints/А-A/xxx.npy
    parts = npy.relative_to(KEYPOINT_ROOT).parts
    letter = parts[0]
    if letter not in label_map:
        continue
    rows.append([str(npy), int(label_map[letter]), letter])

print("Samples:", len(rows))

# Shuffle and split (seen-signer style)
random.shuffle(rows)
n = len(rows)
n_train = int(0.8 * n)
n_val = int(0.1 * n)

train = rows[:n_train]
val = rows[n_train:n_train + n_val]
test = rows[n_train + n_val:]

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["path", "label", "letter", "split"])
    for r in train: w.writerow([*r, "train"])
    for r in val:   w.writerow([*r, "val"])
    for r in test:  w.writerow([*r, "test"])

print("Saved:", OUT_CSV)
print("train/val/test:", len(train), len(val), len(test))
