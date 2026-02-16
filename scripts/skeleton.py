import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

input_dir = "../dataset/raw_videos"
output_path = "../dataset/processed/skeleton_dataset.npy"

# MediaPipe setup
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    refine_face_landmarks=False
)

def extract_skeleton(video_path):
    cap = cv2.VideoCapture(video_path)
    skeletons = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        joints = []

        # Body (33)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                joints.append([lm.x, lm.y])
        else:
            joints.extend([[0, 0]] * 33)

        # Left hand (21)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                joints.append([lm.x, lm.y])
        else:
            joints.extend([[0, 0]] * 21)

        # Right hand (21)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                joints.append([lm.x, lm.y])
        else:
            joints.extend([[0, 0]] * 21)

        skeletons.append(joints)

    cap.release()
    return np.array(skeletons)  # (T, V, C)


# Collect all videos
all_data = []

for label in os.listdir(input_dir):
    label_path = os.path.join(input_dir, label)
    if not os.path.isdir(label_path):
        continue

    for video_file in tqdm(os.listdir(label_path), desc=f"Processing {label}"):
        input_path = os.path.join(label_path, video_file)
        skeleton = extract_skeleton(input_path)
        all_data.append(skeleton)

# Make sure all videos have same T (IMPORTANT)
min_T = min(sample.shape[0] for sample in all_data)

# Trim all to same length
all_data = [sample[:min_T] for sample in all_data]

# Stack into (N, T, V, C)
data = np.stack(all_data)

print("Final shape:", data.shape)  # (N, T, V, C)

np.save(output_path, data)
