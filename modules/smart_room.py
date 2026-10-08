"""
smart_room.py — Multi-Modal Session Orchestrator (v4).

WHAT THIS DOES
---------------
Given an .mp4 (or .mov/.avi/.mkv) file, this module:
  1. Creates a per-session workspace directory to hold the extracted
     audio track and, optionally, saved keyframe images for later audit.
  2. Extracts the audio track to a 16kHz mono WAV via ffmpeg.
  3. Walks the video frame-by-frame, selecting DISTINCT frames:
       - Scene change: any GRID CELL of the frame's perceptual hash
         differs from the corresponding cell in the previous frame by
         more than `scene_change_hamming_threshold` bits.
       - Periodic safety net: `keep_alive_interval_sec` has elapsed
         since the last kept frame, so long static spans still get
         sampled.
       - Cooldown: a scene-change event cannot fire sooner than
         `min_gap_between_keeps_sec` after the previous keep, so a
         single expression change does not produce a burst of dozens
         of near-identical frames.
  4. For every kept frame, runs it through the REAL video pipeline
     (modules.image_emotion) and pulls the audio window centered on
     the same timestamp.
  5. Returns one structured observation per kept frame, plus per-person
     (video) and overall (audio) summaries.

CHANGES FROM v3
----------------
- Added `min_gap_between_keeps_sec` cooldown. Grid-based scene
  detection at threshold 3 is very sensitive; without a cooldown a
  single expression change can fire on dozens of consecutive frames.
  Default 0.3s = no more than ~3 scene-change keeps per second.

DEPENDENCIES
------------
- ffmpeg on PATH (audio extraction)
- opencv-python, numpy
- modules.image_emotion / modules.audio_emotion dependencies
"""

import datetime as _dt
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

sys.path.append(os.path.abspath("."))

from modules.image_emotion.predictor import predict_frame
from modules.image_emotion.tracker import PersonTracker
from modules.image_emotion.config import (
    TRACK_IOU_MATCH_THRESHOLD, TRACK_MAX_MISSED_FRAMES, TRACK_HISTORY_WINDOW,
    DIAGNOSTIC_DISCLAIMER as VIDEO_DISCLAIMER,
)
from modules.audio_emotion.predictor import get_window_at_timestamp
from modules.audio_emotion.config import AUDIO_WINDOW_SEC, DIAGNOSTIC_DISCLAIMER as AUDIO_DISCLAIMER

logger = logging.getLogger("mindcare.smart_room")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

SUPPORTED_VIDEO_FORMATS = {".mp4", ".mov", ".avi", ".mkv"}
SESSIONS_ROOT = os.environ.get("MINDCARE_SESSIONS_DIR", "./smart_room_sessions")


class SmartRoomError(RuntimeError):
    """Raised for unrecoverable setup failures (missing ffmpeg, unreadable video)."""


# ── Perceptual hashing for real scene-change detection ────────────────────────

def _perceptual_hash(frame_bgr: np.ndarray, hash_size: int = 8) -> int:
    """Single-block hash (building block for the grid version)."""
    small_gray = cv2.resize(
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (hash_size, hash_size)
    )
    avg = small_gray.mean()
    bits = (small_gray.flatten() > avg).astype(np.uint8)
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def _hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _grid_perceptual_hashes(
    frame_bgr: np.ndarray,
    rows: int = 4,
    cols: int = 4,
    hash_size: int = 8,
) -> List[int]:
    """Split the frame into a rows x cols grid and hash each cell."""
    h, w = frame_bgr.shape[:2]
    if h == 0 or w == 0:
        return []
    hashes: List[int] = []
    for r in range(rows):
        for c in range(cols):
            y0 = int(r * h / rows)
            y1 = int((r + 1) * h / rows)
            x0 = int(c * w / cols)
            x1 = int((c + 1) * w / cols)
            cell = frame_bgr[y0:y1, x0:x1]
            if cell.size == 0:
                hashes.append(0)
                continue
            hashes.append(_perceptual_hash(cell, hash_size))
    return hashes


def _max_cell_hamming(hashes_a: List[int], hashes_b: List[int]) -> int:
    """Largest per-cell Hamming distance between two grid hashes."""
    if not hashes_a or not hashes_b or len(hashes_a) != len(hashes_b):
        return 0
    return max(_hamming_distance(a, b) for a, b in zip(hashes_a, hashes_b))


# ── Session workspace ──────────────────────────────────────────────────────────

@dataclass
class SessionWorkspace:
    session_id: str
    root_dir: str
    audio_wav_path: str
    keep_files: bool = False

    def cleanup(self):
        if not self.keep_files and os.path.isdir(self.root_dir):
            shutil.rmtree(self.root_dir, ignore_errors=True)
            logger.info("Cleaned up session workspace: %s", self.root_dir)


def create_session_workspace(keep_files: bool = False) -> SessionWorkspace:
    os.makedirs(SESSIONS_ROOT, exist_ok=True)
    session_id = uuid.uuid4().hex[:12]
    root_dir = os.path.join(SESSIONS_ROOT, session_id)
    os.makedirs(root_dir, exist_ok=True)
    audio_wav_path = os.path.join(root_dir, "audio_track.wav")
    logger.info("Created session workspace: %s", root_dir)
    return SessionWorkspace(
        session_id=session_id,
        root_dir=root_dir,
        audio_wav_path=audio_wav_path,
        keep_files=keep_files,
    )


# ── Audio extraction ───────────────────────────────────────────────────────────

def extract_audio_track(video_path: str, out_wav_path: str, sample_rate: int = 16000) -> bool:
    if shutil.which("ffmpeg") is None:
        raise SmartRoomError(
            "ffmpeg is not installed or not on PATH. Install it — it's free, "
            "local, and required for audio extraction from video containers."
        )

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", "1",
        out_wav_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore")
        if "does not contain any stream" in stderr.lower() or "output file is empty" in stderr.lower():
            logger.info("No audio stream found in %s — proceeding video-only.", video_path)
            return False
        raise SmartRoomError(f"ffmpeg audio extraction failed:\n{stderr[-800:]}")

    if not os.path.exists(out_wav_path) or os.path.getsize(out_wav_path) < 44:
        logger.info("ffmpeg produced no usable audio for %s — proceeding video-only.", video_path)
        return False
    return True


# ── Main orchestrator ───────────────────────────────────────────────────────────

@dataclass
class SmartRoomSession:
    video_path: str
    scene_change_hamming_threshold: int = 3
    keep_alive_interval_sec: float = 2.0
    min_gap_between_keeps_sec: float = 0.3      # cooldown — prevents frame bursts
    grid_rows: int = 4
    grid_cols: int = 4
    audio_window_sec: float = AUDIO_WINDOW_SEC
    keep_workspace_files: bool = False

    def __post_init__(self):
        ext = os.path.splitext(self.video_path)[1].lower()
        if ext not in SUPPORTED_VIDEO_FORMATS:
            raise SmartRoomError(
                f"Unsupported video format '{ext}'. Supported: {sorted(SUPPORTED_VIDEO_FORMATS)}"
            )
        if not os.path.exists(self.video_path):
            raise SmartRoomError(f"Video file not found: {self.video_path}")

    def run(self) -> Dict:
        workspace = create_session_workspace(keep_files=self.keep_workspace_files)
        try:
            return self._run_in_workspace(workspace)
        finally:
            workspace.cleanup()

    def _run_in_workspace(self, workspace: SessionWorkspace) -> Dict:
        logger.info("Extracting audio track from %s ...", self.video_path)
        has_audio = extract_audio_track(self.video_path, workspace.audio_wav_path)

        audio_waveform, audio_sr = None, None
        if has_audio:
            import librosa
            audio_waveform, audio_sr = librosa.load(workspace.audio_wav_path, sr=16000, mono=True)
            logger.info(
                "Audio track loaded: %.2fs @ %dHz",
                len(audio_waveform) / audio_sr, audio_sr,
            )

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise SmartRoomError(f"Could not open video for reading: {self.video_path}")
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        tracker = PersonTracker(
            iou_threshold=TRACK_IOU_MATCH_THRESHOLD,
            max_missed_frames=TRACK_MAX_MISSED_FRAMES,
            history_window=TRACK_HISTORY_WINDOW,
        )

        observations: List[dict] = []
        prev_hashes: Optional[List[int]] = None
        last_kept_ts = -1e9
        frame_idx = 0

        logger.info(
            "Scanning video (grid scene changes, cooldown=%.2fs, periodic=%.2fs)...",
            self.min_gap_between_keeps_sec, self.keep_alive_interval_sec,
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp_sec = frame_idx / native_fps

            current_hashes = _grid_perceptual_hashes(
                frame, rows=self.grid_rows, cols=self.grid_cols
            )

            since_last_keep = timestamp_sec - last_kept_ts

            if prev_hashes is None:
                is_scene_change = False
            else:
                max_delta = _max_cell_hamming(current_hashes, prev_hashes)
                is_scene_change = (
                    max_delta > self.scene_change_hamming_threshold
                    and since_last_keep >= self.min_gap_between_keeps_sec
                )

            is_periodic_due = since_last_keep >= self.keep_alive_interval_sec
            keep_this_frame = is_scene_change or is_periodic_due or prev_hashes is None

            if keep_this_frame:
                if prev_hashes is None:
                    trigger = "first_frame"
                elif is_scene_change:
                    trigger = "scene_change"
                else:
                    trigger = "periodic"

                video_result = predict_frame(frame, tracker=tracker, timestamp=timestamp_sec)

                # In-memory base64 thumbnail
                try:
                    import base64
                    scale_ratio = 0.5
                    preview_w = int(frame.shape[1] * scale_ratio)
                    preview_h = int(frame.shape[0] * scale_ratio)
                    small_frame = cv2.resize(
                        frame, (preview_w, preview_h), interpolation=cv2.INTER_AREA
                    )
                    _, buffer = cv2.imencode(
                        '.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75]
                    )
                    base64_image_str = base64.b64encode(buffer).decode('utf-8')
                    frame_data_uri = f"data:image/jpeg;base64,{base64_image_str}"
                except Exception as b64_err:
                    logger.warning(f"In-memory thumbnail rendering bypassed: {str(b64_err)}")
                    frame_data_uri = None

                audio_result = None
                if audio_waveform is not None:
                    audio_result = get_window_at_timestamp(
                        audio_waveform, audio_sr,
                        center_sec=timestamp_sec, window_sec=self.audio_window_sec,
                    )

                observations.append({
                    "frame_index": frame_idx,
                    "frame_b64_uri": frame_data_uri,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "trigger": trigger,
                    "video": video_result,
                    "audio": audio_result,
                })
                last_kept_ts = timestamp_sec

            prev_hashes = current_hashes
            frame_idx += 1

        cap.release()
        duration_sec = frame_idx / native_fps if native_fps else 0.0
        logger.info(
            "Scan complete: %d frames total, %d kept as distinct observations.",
            frame_idx, len(observations),
        )

        summary = self._summarize(observations)

        result = {
            "status": "success",
            "session_id": workspace.session_id,
            "video_path": self.video_path,
            "duration_sec": round(duration_sec, 2),
            "native_fps": round(native_fps, 2),
            "has_audio": audio_waveform is not None,
            "num_frames_total": frame_idx,
            "num_observations": len(observations),
            "observations": observations,
            "summary": summary,
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "video_disclaimer": VIDEO_DISCLAIMER,
            "audio_disclaimer": AUDIO_DISCLAIMER,
            "is_diagnosis": False,
        }

        log_path = os.path.join(workspace.root_dir, "session_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        result["session_log_path"] = log_path if self.keep_workspace_files else None

        return result

    @staticmethod
    def _summarize(observations: List[dict]) -> Dict:
        video_by_track: Dict[int, Dict[str, int]] = {}
        audio_tone_counts: Dict[str, int] = {}
        audio_windows_seen = 0

        for obs in observations:
            v = obs.get("video") or {}
            if v.get("face_detected"):
                tid = v.get("track_id")
                emotion = v.get("emotion")
                if tid is not None and emotion:
                    video_by_track.setdefault(tid, {})
                    video_by_track[tid][emotion] = video_by_track[tid].get(emotion, 0) + 1

            a = obs.get("audio")
            if a and a.get("status") == "success":
                audio_windows_seen += 1
                tone = a.get("primary_tone")
                if tone:
                    audio_tone_counts[tone] = audio_tone_counts.get(tone, 0) + 1

        video_summary = {}
        for tid, counts in video_by_track.items():
            total = sum(counts.values()) or 1
            video_summary[tid] = {
                "num_observations": total,
                "emotion_distribution": {k: round(v / total, 3) for k, v in counts.items()},
                "dominant_emotion": max(counts, key=counts.get),
            }

        audio_summary = None
        if audio_windows_seen:
            total = sum(audio_tone_counts.values())
            dom = max(audio_tone_counts, key=audio_tone_counts.get) if audio_tone_counts else None
            audio_summary = {
                "num_windows_analyzed": audio_windows_seen,
                "tone_distribution": {k: round(v / total, 3) for k, v in audio_tone_counts.items()},
                "dominant_tone": dom,
                "dominant_emotion": dom.lower() if dom else None,
                "emotion_distribution": {k.lower(): round(v / total, 3) for k, v in audio_tone_counts.items()},
            }

        return {"video_by_track": video_summary, "audio_overall": audio_summary}


def analyze_video(
    video_path: str,
    scene_change_hamming_threshold: int = 3,
    keep_alive_interval_sec: float = 2.0,
    keep_workspace_files: bool = False,
) -> Dict:
    session = SmartRoomSession(
        video_path=video_path,
        scene_change_hamming_threshold=scene_change_hamming_threshold,
        keep_alive_interval_sec=keep_alive_interval_sec,
        keep_workspace_files=keep_workspace_files,
    )
    return session.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MindCare Smart-Room multi-modal video/audio session analyzer")
    parser.add_argument("video", help="Path to an .mp4/.mov/.avi/.mkv file")
    parser.add_argument("--scene-threshold", type=int, default=3, help="Grid-cell Hamming distance to count as a scene change")
    parser.add_argument("--keep-alive-sec", type=float, default=2.0, help="Max seconds between kept frames even with no scene change")
    parser.add_argument("--min-gap-sec", type=float, default=0.3, help="Cooldown: minimum seconds between two scene-change keeps")
    parser.add_argument("--keep-files", action="store_true", help="Keep the extracted audio + session log instead of deleting them")
    parser.add_argument("--out", default=None, help="Optional path to write the full JSON result to")
    args = parser.parse_args()

    result = analyze_video(
        args.video,
        scene_change_hamming_threshold=args.scene_threshold,
        keep_alive_interval_sec=args.keep_alive_sec,
        keep_workspace_files=args.keep_files,
    )

    print("\n" + " SMART-ROOM SESSION SUMMARY ".center(80, "="))
    print(f"Video               : {result['video_path']}")
    print(f"Duration            : {result['duration_sec']}s ({result['num_frames_total']} frames total)")
    print(f"Has audio           : {result['has_audio']}")
    print(f"Distinct frames kept: {result['num_observations']}")
    print("\nPer-person video summary:")
    for tid, s in result["summary"]["video_by_track"].items():
        print(f"  Track {tid}: {s['num_observations']} obs, dominant={s['dominant_emotion']}, dist={s['emotion_distribution']}")
    if result["summary"]["audio_overall"]:
        print(f"\nAudio summary: {result['summary']['audio_overall']}")
    else:
        print("\nAudio summary: none (no audio track or all windows failed quality gate)")
    print(f"\nVideo disclaimer: {result['video_disclaimer']}")
    print(f"Audio disclaimer: {result['audio_disclaimer']}")
    print("=" * 80)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nFull JSON result written to: {args.out}")