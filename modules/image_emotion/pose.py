"""
modules/image_emotion/pose.py — Kinematic feature extraction via
MediaPipe Pose landmarks.

Every value returned here is a direct numeric computation from detected
2D landmark coordinates (angles, distances, and displacement-over-time).
Nothing here is invented text or a canned phrase — if the pose model
does not find a person, or a landmark is low-confidence, the
corresponding feature is returned as None rather than guessed.

These are KINEMATIC PROXIES, not clinical labels. E.g. `torso_flexion_deg`
is a real angle; whether a given angle is clinically meaningful is a
question for a professional, not something this code decides. Callers
should treat this as raw structured measurement, matching the module's
DIAGNOSTIC_DISCLAIMER.

NOTE ON MEDIAPIPE API: uses mp_compat.py rather than touching
`mediapipe.solutions.pose` directly, since that attribute doesn't exist
on newer mediapipe pip builds (confirmed: 0.10.33 exposes only the Tasks
API). If pose estimation is unavailable at all on a given machine (no
usable MediaPipe API, or no internet for the one-time Tasks model
download), every function below degrades to returning None fields
rather than raising — pose is an enhancement on top of the emotion
classifier, not a hard dependency of the module.
"""
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .mp_compat import *
from .config import (
    POSE_MODEL_COMPLEXITY,
    POSE_MIN_DETECTION_CONF,
    POSE_MIN_TRACKING_CONF,
    POSE_CROP_PADDING_RATIO,
)
from .tracker import Track

logger = logging.getLogger("mindcare.image_emotion.pose")

_pose_unavailable = False


def _get_pose_backend():
    global _pose_unavailable
    if _pose_unavailable:
        return None
    try:
        return mp_compat.get_pose_backend(POSE_MODEL_COMPLEXITY, POSE_MIN_DETECTION_CONF, POSE_MIN_TRACKING_CONF)
    except Exception:
        _pose_unavailable = True
        return None


def _expand_box(x, y, w, h, frame_w, frame_h, pad_ratio):
    """Expand a face box into an approximate upper-body region so the
    pose model has shoulders/wrists to work with, clamped to frame."""
    cx, cy = x + w / 2.0, y + h / 2.0
    new_w = w * (1 + 2 * pad_ratio)
    new_h = h * (1 + 5 * pad_ratio)  # extend further downward to include torso/wrists
    nx1 = max(0, int(cx - new_w / 2))
    ny1 = max(0, int(cy - new_h * 0.15))  # keep some margin above the face too
    nx2 = min(frame_w, int(cx + new_w / 2))
    ny2 = min(frame_h, int(cy + new_h))
    return nx1, ny1, nx2, ny2


def _angle_deg(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Angle at point b, formed by rays b->a and b->c, in degrees."""
    v1 = np.array([a[0] - b[0], a[1] - b[1]])
    v2 = np.array([c[0] - b[0], c[1] - b[1]])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(math.degrees(math.acos(cos_theta)))


def extract_pose_landmarks(frame_bgr: np.ndarray, box, pad_ratio: float = POSE_CROP_PADDING_RATIO) -> Optional[dict]:
    """Runs pose estimation on the upper-body region around a detected
    face box. Returns a dict of pixel-space landmark coordinates (in the
    ORIGINAL frame's coordinate system) or None if no person was found."""
    import cv2

    frame_h, frame_w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = _expand_box(box.x, box.y, box.w, box.h, frame_w, frame_h, pad_ratio)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    backend = _get_pose_backend()
    if backend is None:
        return None

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    try:
        landmarks: Optional[List] = backend.detect(rgb)
    except Exception as exc:
        logger.warning("Pose detection failed on this frame (%s); skipping pose for it.", exc)
        return None
    if not landmarks:
        return None

    ch, cw = crop.shape[:2]

    def px(landmark_idx):
        p = landmarks[landmark_idx]
        if p.visibility is not None and p.visibility < 0.4:
            return None
        return (x1 + p.x * cw, y1 + p.y * ch)

    return {
        "left_shoulder": px(mp_compat.POSE_LANDMARK_LEFT_SHOULDER),
        "right_shoulder": px(mp_compat.POSE_LANDMARK_RIGHT_SHOULDER),
        "left_hip": px(mp_compat.POSE_LANDMARK_LEFT_HIP),
        "right_hip": px(mp_compat.POSE_LANDMARK_RIGHT_HIP),
        "left_wrist": px(mp_compat.POSE_LANDMARK_LEFT_WRIST),
        "right_wrist": px(mp_compat.POSE_LANDMARK_RIGHT_WRIST),
        "nose": px(mp_compat.POSE_LANDMARK_NOSE),
    }


def compute_static_pose_features(landmarks: dict) -> Dict[str, Optional[float]]:
    """Features computable from a single frame's landmarks (no history needed)."""
    if landmarks is None:
        return {"torso_flexion_deg": None, "shoulder_symmetry_deg": None,
                "min_wrist_to_face_dist_norm": None}

    ls, rs = landmarks.get("left_shoulder"), landmarks.get("right_shoulder")
    lh, rh = landmarks.get("left_hip"), landmarks.get("right_hip")
    nose = landmarks.get("nose")
    lw, rw = landmarks.get("left_wrist"), landmarks.get("right_wrist")

    out: Dict[str, Optional[float]] = {
        "torso_flexion_deg": None, "shoulder_symmetry_deg": None,
        "min_wrist_to_face_dist_norm": None,
    }

    # Torso flexion: angle between the shoulder midpoint, hip midpoint, and
    # a point directly above the hip midpoint (i.e. deviation from upright).
    if ls and rs and lh and rh:
        shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        hip_mid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        vertical_ref = (hip_mid[0], hip_mid[1] - 100)  # point straight up from hip
        out["torso_flexion_deg"] = round(_angle_deg(shoulder_mid, hip_mid, vertical_ref), 1)

        shoulder_width = math.dist(ls, rs) or 1.0
        # Normalize wrist-to-nose distance by shoulder width so it's scale-invariant
        if nose:
            dists = []
            for w in (lw, rw):
                if w:
                    dists.append(math.dist(w, nose) / shoulder_width)
            if dists:
                out["min_wrist_to_face_dist_norm"] = round(min(dists), 3)

    if ls and rs:
        # Shoulder line tilt from horizontal — a crude asymmetry/slump indicator
        dy, dx = (rs[1] - ls[1]), (rs[0] - ls[0])
        out["shoulder_symmetry_deg"] = round(abs(math.degrees(math.atan2(dy, dx))), 1)

    return out


def compute_temporal_features(track: Track) -> Dict[str, Optional[float]]:
    """Features that require a position history (movement speed, pacing)."""
    history = list(track.history)
    if len(history) < 2:
        return {"movement_speed_px_per_sec": None, "pacing_index": None}

    # Instantaneous speed from the last two samples
    p_prev, p_last = history[-2], history[-1]
    dt = max(1e-3, p_last.t - p_prev.t)
    dist = math.dist((p_prev.cx, p_prev.cy), (p_last.cx, p_last.cy))
    speed = dist / dt

    # Pacing index: normalized std-dev of horizontal centroid position over
    # the tracked window. High value ~ oscillating side-to-side movement
    # (a real, measurable proxy for restless pacing); low value ~ mostly
    # stationary. Normalized by box width so it's roughly scale-invariant.
    xs = np.array([h.cx for h in history])
    avg_w = float(np.mean([h.box_w for h in history])) or 1.0
    pacing_index = float(np.std(xs) / avg_w)

    return {
        "movement_speed_px_per_sec": round(speed, 2),
        "pacing_index": round(pacing_index, 4),
    }