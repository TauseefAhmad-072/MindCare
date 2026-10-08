"""
extract_frames.py — Pull every Nth frame from a video into ./dataset_raw/<class>/

Record separate clips per expression, or use one video with timestamps.
This script just dumps frames; you label them by moving/copying into folders.
"""
import cv2
import os
import sys

def extract(video_path: str, out_dir: str, every_n: int = 5):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            out = os.path.join(out_dir, f"frame_{idx:05d}.jpg")
            cv2.imwrite(out, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            saved += 1
        idx += 1
    cap.release()
    print(f"Saved {saved} frames from {video_path} → {out_dir} (fps={fps:.1f})")


if __name__ == "__main__":
    # Usage: python extract_frames.py path/to/video.mp4 output_folder
    if len(sys.argv) != 3:
        print("Usage: python extract_frames.py <video> <output_dir>")
        sys.exit(1)
    extract(sys.argv[1], sys.argv[2], every_n=5)