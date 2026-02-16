import json
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

SOURCE_ROOT = Path(r"dataset_fixed300/raw_videos")
OUT_ROOT = Path(r"dataset/processed/keypoints")  # output
OUT_ROOT.mkdir(parents=True, exist_ok=True)

T_TARGET = 90  # 3 sec * 30 fps
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}

mp_hands = mp.solutions.hands

def resample_frames(arr, t_target):
    # arr: (T, 21, 3)
    t = arr.shape[0]
    if t == t_target:
        return arr
    idx = np.linspace(0, t - 1, t_target).astype(int)
    return arr[idx]

def fill_missing(frames):
    # frames list of (21,3) or None
    out = []
    last = None
    for f in frames:
        if f is None:
            out.append(last)
        else:
            out.append(f)
            last = f
    # if first frames None, forward fill
    first_valid = next((x for x in out if x is not None), None)
    if first_valid is None:
        return None
    out = [first_valid if x is None else x for x in out]
    return np.stack(out, axis=0)

def normalize(kp):
    # kp: (T,21,3)
    wrist = kp[:, 0:1, :]              # (T,1,3)
    kp = kp - wrist                    # center
    # scale by wrist->middle_mcp (id 9)
    scale = np.linalg.norm(kp[:, 9, :2], axis=1)  # use xy for scale
    scale = np.clip(scale, 1e-6, None)
    kp[:, :, 0] /= scale[:, None]
    kp[:, :, 1] /= scale[:, None]
    kp[:, :, 2] /= scale[:, None]
    return kp

def extract_one(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # (21,3)
                frames.append(pts)
            else:
                frames.append(None)

    cap.release()
    kp = fill_missing(frames)
    if kp is None:
        return None
    kp = resample_frames(kp, T_TARGET)
    kp = normalize(kp)
    return kp  # (90,21,3)

# Create index and extract
videos = []
for letter_dir in sorted(SOURCE_ROOT.iterdir()):
    if not letter_dir.is_dir():
        continue
    for p in letter_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)

print("Found videos:", len(videos))

bad = 0
for vp in tqdm(videos):
    rel = vp.relative_to(SOURCE_ROOT)  # LETTER/filename.mp4 (or deeper)
    out_path = OUT_ROOT / rel.with_suffix(".npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        continue

    kp = extract_one(vp)
    if kp is None:
        bad += 1
        continue

    np.save(out_path, kp)

print("Done. Failed videos:", bad)
print("Keypoints saved to:", OUT_ROOT)
