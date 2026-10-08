"""
modules/image_emotion/mp_compat.py — Compatibility shim across MediaPipe's
two incompatible public Python APIs.

WHY THIS EXISTS
-----------------
MediaPipe has shipped two generations of Python API:
  - "Solutions" API (`mp.solutions.face_detection`, `mp.solutions.pose`) —
    the classic API most existing tutorials/code (including the first
    draft of this module) are written against. Model weights are bundled
    directly in the pip package, so it works with zero network calls.
  - "Tasks" API (`mediapipe.tasks.python.vision`) — the current API in
    recent mediapipe releases. CONFIRMED IN THIS PROJECT'S ENVIRONMENT:
    `pip install mediapipe` (0.10.33) does NOT expose `mp.solutions` at
    all — `mp.solutions.face_detection` raises
    `AttributeError: module 'mediapipe' has no attribute 'solutions'`.
    The Tasks API requires a separately downloaded `.tflite`/`.task`
    model file (a few MB), fetched once from Google's own official
    model hosting at `storage.googleapis.com` — a real, working,
    documented URL (unlike the placeholder URLs in the pre-upgrade
    version of this codebase), then cached locally.

This module tries the Solutions API first (works immediately, no
network) and falls back to the Tasks API (one-time model download,
cached under MINDCARE_MP_MODEL_CACHE afterward) if Solutions isn't
available. Either way, callers get ONE stable interface:

    get_face_backend(...).detect(frame_rgb) -> List[RelFaceBox]
    get_pose_backend(...).detect(frame_rgb) -> Optional[List[RelLandmark]]

so detector.py / pose.py don't need to know or care which generation of
MediaPipe is actually installed.
"""
import logging
import os
import threading
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger("mindcare.image_emotion.mp_compat")

# ── Standard BlazePose 33-point topology indices ──────────────────────────────
# These indices are stable across BOTH MediaPipe API generations (they
# describe the same underlying pose model's output layout), so the rest of
# the codebase can reference them directly instead of relying on the
# Solutions-API-only `mp.solutions.pose.PoseLandmark` enum.
POSE_LANDMARK_NOSE = 0
POSE_LANDMARK_LEFT_SHOULDER = 11
POSE_LANDMARK_RIGHT_SHOULDER = 12
POSE_LANDMARK_LEFT_WRIST = 15
POSE_LANDMARK_RIGHT_WRIST = 16
POSE_LANDMARK_LEFT_HIP = 23
POSE_LANDMARK_RIGHT_HIP = 24

_MODEL_CACHE_DIR = os.environ.get("MINDCARE_MP_MODEL_CACHE", "./mp_models")

_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_FACE_MODEL_FILENAME = "blaze_face_short_range.tflite"

_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
_POSE_MODEL_FILENAME = "pose_landmarker_lite.task"


def _ensure_model_file(url: str, filename: str) -> str:
    os.makedirs(_MODEL_CACHE_DIR, exist_ok=True)
    local_path = os.path.join(_MODEL_CACHE_DIR, filename)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    logger.info("Downloading MediaPipe Tasks model '%s' (one-time, a few MB)...", filename)
    try:
        urllib.request.urlretrieve(url, local_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download the required MediaPipe model file from {url}: {exc}. "
            f"For a fully offline machine, download it once elsewhere and place it at "
            f"'{local_path}'."
        ) from exc
    return local_path


@dataclass
class RelFaceBox:
    xmin: float
    ymin: float
    width: float
    height: float
    score: float


@dataclass
class RelLandmark:
    x: float
    y: float
    visibility: float


# ── Face detection backends ────────────────────────────────────────────────────

class _SolutionsFaceBackend:
    def __init__(self, model_selection: int, min_conf: float):
        import mediapipe as mp
        self._detector = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection, min_detection_confidence=min_conf
        )

    def detect(self, frame_rgb) -> List[RelFaceBox]:
        result = self._detector.process(frame_rgb)
        out = []
        if result.detections:
            for det in result.detections:
                b = det.location_data.relative_bounding_box
                score = float(det.score[0]) if det.score else 0.0
                out.append(RelFaceBox(b.xmin, b.ymin, b.width, b.height, score))
        return out


class _TasksFaceBackend:
    def __init__(self, min_conf: float):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        model_path = _ensure_model_file(_FACE_MODEL_URL, _FACE_MODEL_FILENAME)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=min_conf)
        self._detector = mp_vision.FaceDetector.create_from_options(options)
        self._mp = mp

    def detect(self, frame_rgb) -> List[RelFaceBox]:
        h, w = frame_rgb.shape[:2]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect(mp_image)
        out = []
        for det in result.detections:
            bbox = det.bounding_box
            score = float(det.categories[0].score) if det.categories else 0.0
            out.append(RelFaceBox(bbox.origin_x / w, bbox.origin_y / h, bbox.width / w, bbox.height / h, score))
        return out


_face_backend = None
_face_lock = threading.Lock()


def get_face_backend(model_selection: int, min_conf: float):
    """Returns a face-detection backend, preferring the offline Solutions
    API and falling back to the Tasks API (which needs one internet call
    on first run to fetch its model file). Raises if neither works."""
    global _face_backend
    if _face_backend is not None:
        return _face_backend
    with _face_lock:
        if _face_backend is not None:
            return _face_backend
        try:
            _face_backend = _SolutionsFaceBackend(model_selection, min_conf)
            logger.info("MediaPipe face backend: legacy Solutions API (offline).")
        except Exception as exc:
            logger.info("MediaPipe Solutions API unavailable (%s); trying Tasks API.", exc)
            _face_backend = _TasksFaceBackend(min_conf)
            logger.info("MediaPipe face backend: Tasks API.")
        return _face_backend


# ── Pose backends ──────────────────────────────────────────────────────────────

class _SolutionsPoseBackend:
    def __init__(self, complexity: int, min_det: float, min_track: float):
        import mediapipe as mp
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True, model_complexity=complexity,
            min_detection_confidence=min_det, min_tracking_confidence=min_track,
        )

    def detect(self, frame_rgb) -> Optional[List[RelLandmark]]:
        result = self._pose.process(frame_rgb)
        if not result.pose_landmarks:
            return None
        return [RelLandmark(p.x, p.y, p.visibility) for p in result.pose_landmarks.landmark]


class _TasksPoseBackend:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        model_path = _ensure_model_file(_POSE_MODEL_URL, _POSE_MODEL_FILENAME)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(base_options=base_options, num_poses=1)
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._mp = mp

    def detect(self, frame_rgb) -> Optional[List[RelLandmark]]:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return None
        lm_list = result.pose_landmarks[0]  # first (only) requested pose
        return [RelLandmark(p.x, p.y, getattr(p, "visibility", 1.0)) for p in lm_list]


_pose_backend = None
_pose_lock = threading.Lock()
_pose_backend_failed = False


def get_pose_backend(complexity: int, min_det: float, min_track: float):
    """Returns a pose-estimation backend, or raises if neither MediaPipe
    API generation is usable (e.g. no internet for the Tasks model on a
    machine with an old mediapipe build). Callers should catch this and
    degrade to 'pose features unavailable' rather than crashing, since
    pose is an enhancement, not a required feature."""
    global _pose_backend, _pose_backend_failed
    if _pose_backend is not None:
        return _pose_backend
    if _pose_backend_failed:
        raise RuntimeError("Pose backend previously failed to initialize; see earlier log for details.")
    with _pose_lock:
        if _pose_backend is not None:
            return _pose_backend
        try:
            _pose_backend = _SolutionsPoseBackend(complexity, min_det, min_track)
            logger.info("MediaPipe pose backend: legacy Solutions API (offline).")
        except Exception as exc:
            logger.info("MediaPipe Solutions API unavailable (%s); trying Tasks API.", exc)
            try:
                _pose_backend = _TasksPoseBackend()
                logger.info("MediaPipe pose backend: Tasks API.")
            except Exception as exc2:
                _pose_backend_failed = True
                logger.warning("Pose estimation unavailable on this machine (%s). "
                                "Pose/kinematic features will be omitted.", exc2)
                raise
        return _pose_backend