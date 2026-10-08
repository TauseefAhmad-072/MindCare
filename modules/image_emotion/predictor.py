"""
modules/image_emotion/predictor.py — Public interface for the upgraded
Multi-Person Video/Image Emotion & Kinematic module.

Two entry points:
  predict(image_path) / predict_bytes(raw) — single-image API, kept
      backward-compatible with the old single-face contract for callers
      (e.g. smart_room.py) that just want "the" primary face's emotion,
      but now returns ALL detected faces too.

  process_video(path) — NEW. Samples a video at VIDEO_SAMPLE_FPS, runs
      detection + tracking + pose + emotion per sampled frame, and
      returns a per-track time series plus a per-track summary. This
      replaces ad-hoc single-frame narrative generation with real,
      structured, per-person measurements over time.

Design principle carried through this whole module: every field in the
output is either (a) a raw model output (with its own confidence score
attached) or (b) a number computed directly from detected landmark
geometry. Nothing here fabricates descriptive text about a scene, and
nothing here outputs a diagnostic label.
"""
import datetime as _dt
import io
import logging
import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from .config import (
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
    MAX_IMAGE_SIZE_BYTES,
    DIAGNOSTIC_DISCLAIMER,
    VIDEO_SAMPLE_FPS,
    EMOTION_EMA_ALPHA,
    TRACK_IOU_MATCH_THRESHOLD,
    TRACK_MAX_MISSED_FRAMES,
    TRACK_HISTORY_WINDOW,
)
from .detector import FaceDetector, FaceBox
from .tracker import PersonTracker
from .model import classify_faces_batch
from .pose import extract_pose_landmarks, compute_static_pose_features, compute_temporal_features
from .quality import assess_crop_quality

logger = logging.getLogger("mindcare.image_emotion.predictor")

_detector_instance: Optional[FaceDetector] = None


class ImageValidationError(ValueError):
    pass


def _get_detector() -> FaceDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FaceDetector()
    return _detector_instance


def _utc_timestamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _load_image_bgr_from_path(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise ImageValidationError(f"Image file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_IMAGE_FORMATS:
        raise ImageValidationError(f"Unsupported image format '{ext}'. Supported: {sorted(SUPPORTED_IMAGE_FORMATS)}")
    if os.path.getsize(path) > MAX_IMAGE_SIZE_BYTES:
        raise ImageValidationError(f"Image exceeds max size of {MAX_IMAGE_SIZE_BYTES} bytes.")
    img = cv2.imread(path)
    if img is None:
        raise ImageValidationError(f"Could not decode image (corrupt or unsupported): {path}")
    return img


def _load_image_bgr_from_bytes(raw: bytes) -> np.ndarray:
    if raw is None or len(raw) == 0:
        raise ImageValidationError("Empty image data.")
    if len(raw) > MAX_IMAGE_SIZE_BYTES:
        raise ImageValidationError(f"Image exceeds max size of {MAX_IMAGE_SIZE_BYTES} bytes.")
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ImageValidationError(f"Could not decode image (corrupt or unsupported): {exc}") from exc
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _ema_update(prev: Optional[Dict[str, float]], new: Dict[str, float], alpha: float) -> Dict[str, float]:
    if prev is None:
        return dict(new)
    keys = set(prev) | set(new)
    return {k: alpha * new.get(k, 0.0) + (1 - alpha) * prev.get(k, 0.0) for k in keys}

def _analyze_faces_in_frame(
    frame_bgr: np.ndarray,
    tracker: Optional[PersonTracker] = None,
    timestamp: Optional[float] = None,
    compute_pose: bool = True,
) -> List[dict]:
    """
    Core routine shared by single-image and video paths.

    Pipeline order (v3):
        1. Enhance the FULL frame — improves face detection recall on
           dark / soft input (MSI webcam case).
        2. Detect faces on the enhanced frame.
        3. Extract each face crop from the enhanced frame.
        4. Enhance each crop a second time (upscale + CLAHE + light sharpen).
        5. Run the quality gate on the ENHANCED crop, so enhancement
           actually helps borderline frames pass.
        6. Classify emotion.
        7. Drop predictions below MIN_PREDICTION_CONFIDENCE to null —
           preserves the honesty contract in quality.py.
        8. Extract pose + temporal features, attach track_id, EMA smoothing.
    """
    from modules.image_emotion.enhance import enhance_frame, enhance_face_crop
    from modules.image_emotion.config import (
        ENHANCE_ENABLED,
        MIN_PREDICTION_CONFIDENCE,
    )

    # ── 1. Enhance full frame before detection ────────────────────────────
    if ENHANCE_ENABLED:
        try:
            frame_for_analysis = enhance_frame(frame_bgr)
        except Exception as e:
            logger.warning("Frame enhancement failed, using raw frame: %s", e)
            frame_for_analysis = frame_bgr
    else:
        frame_for_analysis = frame_bgr

    # ── 2. Detect faces on the enhanced frame ─────────────────────────────
    detector = _get_detector()
    boxes: List[FaceBox] = detector.detect(frame_for_analysis)

    if not boxes:
        return []

    # ── 3. Tracking ───────────────────────────────────────────────────────
    track_map = {}
    if tracker is not None:
        track_map = tracker.update(boxes, timestamp=timestamp)
        box_to_track = {id(b): tid for tid, b in track_map.items()}
    else:
        box_to_track = {}

    # ── 4. Crop, enhance crop, then gate ──────────────────────────────────
    frame_h, frame_w = frame_for_analysis.shape[:2]

    crops = []
    quality_reports = []
    valid_idx = []

    for i, b in enumerate(boxes):
        # Clamp bbox to frame bounds so cropping never returns an empty slice
        x0 = max(0, b.x)
        y0 = max(0, b.y)
        x1 = min(frame_w, b.x + b.w)
        y1 = min(frame_h, b.y + b.h)

        if x1 <= x0 or y1 <= y0:
            # Degenerate bbox — record a failed quality report and skip
            quality_reports.append(
                QualityReport(0.0, 0.0, 0, 0, False, ["degenerate_bbox"])
            )
            continue

        crop = frame_bgr[y0:y1, x0:x1]

        # Enhance the crop BEFORE the gate so the gate sees the quality we
        # will actually feed to the classifier.
                # Crop from the ORIGINAL frame, no per-crop enhancement.
        # The classifier was fine-tuned on raw AffectNet/FER2013 crops;
        # running CLAHE / sharpen / upscale on the input shifts it off
        # the training distribution and flips predictions
        # (verified: happy 0.308 on raw → disgust 0.29 on enhanced).
        crop_for_gate = crop

        q = assess_crop_quality(crop_for_gate)
        quality_reports.append(q)

        if q.is_acceptable and crop_for_gate.size > 0:
            crops.append(crop_for_gate)
            valid_idx.append(i)

    # ── 5. Classify ───────────────────────────────────────────────────────
    emotion_results = classify_faces_batch(crops) if crops else []

    # ── 6. Apply confidence floor: anything below MIN is treated as null ──
        # ── 6. Pass all predictions through; flag low-confidence ones ─────────
    # Previous behaviour nulled anything below MIN_PREDICTION_CONFIDENCE,
    # but FER models legitimately output 25–40% on real-world faces. That
    # contract hid correct predictions. We now mark them instead of
    # dropping them — downstream code can filter on the flag if it wants.
    filtered_results = []
    for r in emotion_results:
        if r is None:
            filtered_results.append(None)
            continue
        conf = r.get("confidence", 0.0) if isinstance(r, dict) else 0.0
        if conf < MIN_PREDICTION_CONFIDENCE:
            r = dict(r)                          # don't mutate the caller's dict
            r["low_confidence"] = True
            r["confidence_floor"] = MIN_PREDICTION_CONFIDENCE
        filtered_results.append(r)
    emotion_results = filtered_results

    emotion_by_box_idx = {
        valid_idx[j]: emotion_results[j] for j in range(len(emotion_results))
    }

    # ── 7. Assemble per-box results ───────────────────────────────────────
    results = []
    for i, b in enumerate(boxes):
        qr = quality_reports[i] if i < len(quality_reports) else None

        entry = {
            "bbox": {"x": b.x, "y": b.y, "w": b.w, "h": b.h},
            "detection_confidence": round(b.confidence, 4),
            "quality": {
                "acceptable": qr.is_acceptable if qr else False,
                "blur_variance": qr.blur_variance if qr else 0.0,
                "brightness_mean": qr.brightness_mean if qr else 0.0,
                "reasons": qr.reasons if qr else ["no_report"],
            },
        }

        emotion = emotion_by_box_idx.get(i)
        track_id = box_to_track.get(id(b))
        entry["track_id"] = track_id

        if emotion is not None:
            entry["emotion_raw"] = emotion
            if tracker is not None and track_id is not None:
                track = tracker.get_track(track_id)
                if track is not None:
                    track.emotion_ema = _ema_update(
                        track.emotion_ema,
                        emotion["probabilities"],
                        EMOTION_EMA_ALPHA,
                    )
                    smoothed_probs = track.emotion_ema
                    top_label = max(smoothed_probs, key=smoothed_probs.get)
                    entry["emotion_smoothed"] = {
                        "emotion": top_label,
                        "confidence": round(smoothed_probs[top_label], 4),
                        "probabilities": {
                            k: round(v, 4) for k, v in smoothed_probs.items()
                        },
                    }
        else:
            entry["emotion_raw"] = None
            # Distinguish "the gate rejected the crop" from "the model
            # returned a prediction below the confidence floor".
            if qr is None or not qr.is_acceptable:
                entry["skip_reason"] = "quality_gate_failed"
            else:
                entry["skip_reason"] = "low_confidence"

        # ── 8. Pose (from the enhanced frame for better landmarks) ────────
        if compute_pose:
            try:
                landmarks = extract_pose_landmarks(frame_for_analysis, b)
                entry["pose"] = compute_static_pose_features(landmarks)
                if tracker is not None and track_id is not None:
                    track = tracker.get_track(track_id)
                    if track is not None:
                        entry["pose"].update(compute_temporal_features(track))
            except Exception as e:
                logger.warning("Pose extraction failed for box %d: %s", i, e)
                entry["pose"] = None

        results.append(entry)

    return results

def predict(image_path: str) -> Dict:
    """Single-image entry point. Backward compatible in spirit with the
    old single-face API (`face_detected`, `emotion`, `confidence`,
    `probabilities`) but now also returns `faces` (all detected people)
    and `num_faces`, so multi-person frames aren't silently truncated."""
    try:
        frame_bgr = _load_image_bgr_from_path(image_path)
    except ImageValidationError as exc:
        return _empty_result(str(exc))
    return _predict_from_bgr(frame_bgr)


def predict_bytes(raw_bytes: bytes) -> Dict:
    try:
        frame_bgr = _load_image_bgr_from_bytes(raw_bytes)
    except ImageValidationError as exc:
        return _empty_result(str(exc))
    return _predict_from_bgr(frame_bgr)


def predict_frame(frame_bgr: np.ndarray, tracker: Optional[PersonTracker] = None,
                   timestamp: Optional[float] = None) -> Dict:
    """
    Public entry point for callers that already have a decoded frame in
    memory (e.g. smart_room.py reading frames straight from a video with
    cv2) and don't want a temp-file round trip through predict()/
    predict_bytes(). If a `tracker` is supplied, results include stable
    track_ids and temporally-smoothed emotion/pose — required for
    multi-frame video use. Without a tracker, behaves like a one-off
    single-frame snapshot (same shape as _predict_from_bgr).
    """
    faces = _analyze_faces_in_frame(frame_bgr, tracker=tracker, timestamp=timestamp, compute_pose=True)
    if not faces:
        return _empty_result("No face detected in the provided frame.")

    primary = max(faces, key=lambda f: f["bbox"]["w"] * f["bbox"]["h"])
    primary_emotion = primary.get("emotion_smoothed") or primary.get("emotion_raw")

    return {
        "face_detected": True,
        "num_faces": len(faces),
        "emotion": primary_emotion["emotion"] if primary_emotion else None,
        "confidence": primary_emotion["confidence"] if primary_emotion else 0.0,
        "probabilities": primary_emotion["probabilities"] if primary_emotion else {},
        "pose": primary.get("pose"),
        "track_id": primary.get("track_id"),
        "faces": faces,
        "timestamp": _utc_timestamp(),
        "disclaimer": DIAGNOSTIC_DISCLAIMER,
        "is_diagnosis": False,
    }


def _predict_from_bgr(frame_bgr: np.ndarray) -> Dict:
    faces = _analyze_faces_in_frame(frame_bgr, tracker=None, compute_pose=True)
    if not faces:
        return _empty_result("No face detected in the provided image.")

    primary = max(faces, key=lambda f: f["bbox"]["w"] * f["bbox"]["h"])  # largest face = primary subject
    primary_emotion = primary.get("emotion_raw")

    return {
        "face_detected": True,
        "num_faces": len(faces),
        "emotion": primary_emotion["emotion"] if primary_emotion else None,
        "confidence": primary_emotion["confidence"] if primary_emotion else 0.0,
        "probabilities": primary_emotion["probabilities"] if primary_emotion else {},
        "pose": primary.get("pose"),
        "faces": faces,
        "timestamp": _utc_timestamp(),
        "disclaimer": DIAGNOSTIC_DISCLAIMER,
        "is_diagnosis": False,
        "note": "Reflects OBSERVED facial expression(s) and body kinematics only, not a mental-health assessment.",
    }


def _empty_result(reason: str) -> Dict:
    return {
        "face_detected": False, "num_faces": 0, "emotion": None, "confidence": 0.0,
        "probabilities": {}, "pose": None, "faces": [], "reason": reason,
        "timestamp": _utc_timestamp(), "disclaimer": DIAGNOSTIC_DISCLAIMER, "is_diagnosis": False,
    }


# ── Video pipeline ────────────────────────────────────────────────────────────

def process_video(video_path: str, sample_fps: float = VIDEO_SAMPLE_FPS, max_frames: Optional[int] = None) -> Dict:
    """
    Samples a video at `sample_fps`, runs multi-person detection +
    tracking + emotion + pose on each sampled frame, and returns:
      {
        "status": "success",
        "duration_sec": float,
        "frames_processed": int,
        "tracks": {
            track_id: {
                "num_observations": int,
                "emotion_distribution": {label: fraction_of_frames_dominant},
                "final_smoothed_emotion": {...},
                "avg_movement_speed_px_per_sec": float | None,
                "pacing_index_series": [float, ...],
                "torso_flexion_series_deg": [float, ...],
            }, ...
        },
        "disclaimer": ...
      }
    This is real structured per-person data over time — no synthetic
    scene narration, no fabricated environment descriptions.
    """
    ext = os.path.splitext(video_path)[1].lower()
    if ext not in SUPPORTED_VIDEO_FORMATS:
        return {"status": "error", "error": f"Unsupported video format '{ext}'."}
    if not os.path.exists(video_path):
        return {"status": "error", "error": f"Video file not found: {video_path}"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "error", "error": "Could not open video file."}

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(native_fps / max(0.1, sample_fps))))

    tracker = PersonTracker(
        iou_threshold=TRACK_IOU_MATCH_THRESHOLD,
        max_missed_frames=TRACK_MAX_MISSED_FRAMES,
        history_window=TRACK_HISTORY_WINDOW,
    )

    per_track_emotion_counts: Dict[int, Dict[str, int]] = {}
    per_track_pacing: Dict[int, List[float]] = {}
    per_track_flexion: Dict[int, List[float]] = {}
    per_track_speeds: Dict[int, List[float]] = {}
    per_track_last_smoothed: Dict[int, dict] = {}

    frame_idx, processed = 0, 0
    start_t = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            ts = frame_idx / native_fps
            results = _analyze_faces_in_frame(frame, tracker=tracker, timestamp=ts, compute_pose=True)
            for r in results:
                tid = r.get("track_id")
                if tid is None:
                    continue
                smoothed = r.get("emotion_smoothed")
                if smoothed:
                    per_track_emotion_counts.setdefault(tid, {})
                    per_track_emotion_counts[tid][smoothed["emotion"]] = (
                        per_track_emotion_counts[tid].get(smoothed["emotion"], 0) + 1
                    )
                    per_track_last_smoothed[tid] = smoothed

                pose = r.get("pose") or {}
                if pose.get("pacing_index") is not None:
                    per_track_pacing.setdefault(tid, []).append(pose["pacing_index"])
                if pose.get("torso_flexion_deg") is not None:
                    per_track_flexion.setdefault(tid, []).append(pose["torso_flexion_deg"])
                if pose.get("movement_speed_px_per_sec") is not None:
                    per_track_speeds.setdefault(tid, []).append(pose["movement_speed_px_per_sec"])
            processed += 1
            if max_frames is not None and processed >= max_frames:
                break
        frame_idx += 1

    cap.release()
    duration_sec = frame_idx / native_fps if native_fps else 0.0

    tracks_summary = {}
    all_track_ids = set(per_track_emotion_counts) | set(per_track_pacing) | set(per_track_flexion) | set(per_track_speeds)
    for tid in all_track_ids:
        counts = per_track_emotion_counts.get(tid, {})
        total = sum(counts.values()) or 1
        distribution = {k: round(v / total, 3) for k, v in counts.items()}
        speeds = per_track_speeds.get(tid, [])
        tracks_summary[tid] = {
            "num_observations": total,
            "emotion_distribution": distribution,
            "final_smoothed_emotion": per_track_last_smoothed.get(tid),
            "avg_movement_speed_px_per_sec": round(float(np.mean(speeds)), 2) if speeds else None,
            "pacing_index_series": per_track_pacing.get(tid, []),
            "torso_flexion_series_deg": per_track_flexion.get(tid, []),
        }

    logger.info(
        "process_video: %d frames sampled (%d total), %d tracks, %.1fs wall time",
        processed, frame_idx, len(tracks_summary), time.time() - start_t
    )

    return {
        "status": "success",
        "duration_sec": round(duration_sec, 2),
        "native_fps": round(native_fps, 2),
        "frames_processed": processed,
        "tracks": tracks_summary,
        "timestamp": _utc_timestamp(),
        "disclaimer": DIAGNOSTIC_DISCLAIMER,
        "is_diagnosis": False,
    }


__all__ = ["predict", "predict_bytes", "predict_frame", "process_video", "ImageValidationError"]