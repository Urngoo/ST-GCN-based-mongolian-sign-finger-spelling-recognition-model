import cv2
import os
import time

# ===== CONFIG =====
LABEL = "T"              # Change per letter ctrl+s before running the code
FPS = 30
DURATION = 3             # seconds
REST_TIME = 0.1           # pause between clips
FRAME_COUNT = FPS * DURATION
CAMERA_ID = 0

SAVE_DIR = f"C:/Users/HP/Desktop/thesis_final/dataset/raw_videos/{LABEL}"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==================

cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FPS, FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

recording = False
clip_index = len(os.listdir(SAVE_DIR)) + 1

print("Press 'S' to start recording")
print("Press 'Q' to stop and exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, f"Label: {LABEL}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Recorder", frame)

    key = cv2.waitKey(1) & 0xFF

    # START RECORDING LOOP
    if key == ord('s') and not recording:
        recording = True
        print("Recording started...")

        while recording:
            video_name = f"{LABEL}_{clip_index:03d}.mp4"
            video_path = os.path.join(SAVE_DIR, video_name)

            out = cv2.VideoWriter(
                video_path, fourcc, FPS,
                (int(cap.get(3)), int(cap.get(4)))
            )

            print(f"Recording clip {video_name}")

            frames = 0
            while frames < FRAME_COUNT:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                cv2.imshow("Recorder", frame)
                cv2.waitKey(1)
                frames += 1

            out.release()
            print("Saved:", video_name)
            clip_index += 1

            time.sleep(REST_TIME)

            # Check for stop key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                recording = False
                break

    # EXIT
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
