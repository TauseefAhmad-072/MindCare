"""
predict_face.py — CLI test harness for the image_emotion module.

FIX vs. the previous version: the old script tried to `urlopen`
"https://pexels.com" and save the response as a JPEG — that URL is a
homepage (HTML), not an image file, so it would either fail outright or
silently write an unusable .html-as-.jpg file. There is no reliable
"download a real face" one-liner without picking a specific, licensed
image URL, so this version does the honest thing instead: it looks for
a local image (or lets you pass one on the command line / grab a frame
from your webcam), and otherwise tells you exactly what to provide.

Usage:
    python predict_face.py path/to/image.jpg
    python predict_face.py path/to/video.mp4 --video
    python predict_face.py --webcam
"""
import argparse
import os
import sys

try:
    from .predictor import predict, process_video
except ImportError:
    from predictor import predict, process_video


def _print_image_result(result: dict):
    print("\n" + " IMAGE ANALYSIS OUTPUT ".center(80, "="))
    print(f"Faces detected      : {result.get('num_faces')}")
    if not result.get("face_detected"):
        print(f"Reason              : {result.get('reason')}")
        return

    print(f"Primary emotion     : {str(result.get('emotion')).upper()}")
    print(f"Confidence          : {result.get('confidence', 0.0) * 100:.2f}%")
    pose = result.get("pose") or {}
    print("\nPose / kinematic features (primary subject):")
    for k, v in pose.items():
        print(f"  - {k:32s}: {v}")

    print(f"\nAll {result.get('num_faces')} face(s):")
    for i, f in enumerate(result.get("faces", [])):
        e = f.get("emotion_raw")
        label = e["emotion"] if e else "SKIPPED (" + ",".join(f.get("quality", {}).get("reasons", [])) + ")"
        print(f"  Face {i}: bbox={f['bbox']} emotion={label}")

    print("\nClass probabilities (primary):")
    for mood, p in result.get("probabilities", {}).items():
        bar = "#" * int(p * 40)
        print(f"  {mood.ljust(10)}: {p * 100:6.2f}% {bar}")

    print(f"\nDisclaimer: {result.get('disclaimer')}")
    print("=" * 80)


def _print_video_result(result: dict):
    print("\n" + " VIDEO ANALYSIS OUTPUT ".center(80, "="))
    if result.get("status") != "success":
        print(f"Error: {result.get('error')}")
        return
    print(f"Duration            : {result['duration_sec']}s over {result['frames_processed']} sampled frames")
    print(f"Tracked people      : {len(result['tracks'])}")
    for tid, summary in result["tracks"].items():
        print(f"\n-- Track {tid} --")
        print(f"  Observations           : {summary['num_observations']}")
        print(f"  Emotion distribution   : {summary['emotion_distribution']}")
        print(f"  Avg movement speed     : {summary['avg_movement_speed_px_per_sec']} px/s")
        if summary["pacing_index_series"]:
            print(f"  Pacing index (latest)  : {summary['pacing_index_series'][-1]}")
    print(f"\nDisclaimer: {result.get('disclaimer')}")
    print("=" * 80)


def _grab_webcam_frame(save_path: str) -> bool:
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam (device 0).")
        return False
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Could not read a frame from the webcam.")
        return False
    cv2.imwrite(save_path, frame)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MindCare image/video emotion + kinematics test harness")
    parser.add_argument("path", nargs="?", help="Path to an image or video file")
    parser.add_argument("--video", action="store_true", help="Treat `path` as a video file")
    parser.add_argument("--webcam", action="store_true", help="Grab a single frame from the webcam instead")
    args = parser.parse_args()

    print("=" * 80)
    print("MINDCARE AI — IMAGE / VIDEO EMOTION + KINEMATICS TEST HARNESS")
    print("=" * 80)

    target = args.path
    if args.webcam:
        target = "webcam_capture.jpg"
        print("Capturing one frame from webcam...")
        if not _grab_webcam_frame(target):
            sys.exit(1)
        print(f"Saved frame to '{target}'.")

    if not target:
        print(
            "\nNo input provided. Pass a real image/video path, e.g.:\n"
            "  python predict_face.py my_photo.jpg\n"
            "  python predict_face.py my_clip.mp4 --video\n"
            "  python predict_face.py --webcam\n"
            "\n(There is no automatic image download here — the previous version's "
            "attempt to pull a face from a homepage URL never worked.)"
        )
        sys.exit(1)

    if not os.path.exists(target):
        print(f"File not found: {target}")
        sys.exit(1)

    if args.video:
        result = process_video(target)
        _print_video_result(result)
    else:
        result = predict(target)
        _print_image_result(result)