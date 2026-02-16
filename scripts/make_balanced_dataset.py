import random
import shutil
from pathlib import Path

# ====== CONFIG ======
SOURCE_ROOT = Path(r"dataset/raw_videos")            # your current data
TARGET_ROOT = Path(r"dataset_fixed300/raw_videos")   # new output folder
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}
TARGET_PER_CLASS = 300
SEED = 42  # reproducible sampling
# ====================

random.seed(SEED)

def list_videos_recursive(folder: Path):
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]

if not SOURCE_ROOT.exists():
    raise SystemExit(f"Source folder not found: {SOURCE_ROOT}")

TARGET_ROOT.mkdir(parents=True, exist_ok=True)

summary = []

for letter_dir in sorted(SOURCE_ROOT.iterdir()):
    if not letter_dir.is_dir():
        continue

    vids = list_videos_recursive(letter_dir)
    n = len(vids)
    if n == 0:
        continue

    # choose up to 300
    chosen = random.sample(vids, TARGET_PER_CLASS) if n > TARGET_PER_CLASS else vids

    out_dir = TARGET_ROOT / letter_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in chosen:
        # avoid name collisions
        dst = out_dir / src.name
        if dst.exists():
            dst = out_dir / f"{src.parent.name}_{src.name}"

        shutil.copy2(src, dst)
        copied += 1

    summary.append((letter_dir.name, n, copied))

print("\n✅ Done. Created:", TARGET_ROOT)
print(f"Rule: copy min({TARGET_PER_CLASS}, available) per letter.\n")
print("Letter folder       available  copied")
print("-" * 36)
for letter, available, copied in summary:
    print(f"{letter:16s} {available:9d} {copied:6d}")

total_avail = sum(a for _, a, _ in summary)
total_copied = sum(c for _, _, c in summary)
print("-" * 36)
print("Total available:", total_avail)
print("Total copied   :", total_copied)
