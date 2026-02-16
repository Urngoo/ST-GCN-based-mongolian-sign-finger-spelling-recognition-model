import os

folder = r"C:\Users\HP\Desktop\thesis_final\dataset\raw_videos\YA"

video_exts = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv")

count = 0
for root, dirs, files in os.walk(folder):
    count += sum(1 for f in files if f.lower().endswith(video_exts))

print("Total videos (recursive):", count)
