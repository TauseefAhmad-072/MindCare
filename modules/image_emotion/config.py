"""
modules/image_emotion/config.py — Configuration for the Multi-Person
Video/Image Emotion & Kinematic Feature module (v3).

v3 additions:
- Enhancement config (denoise / CLAHE / brightness / sharpen) applied
  before the quality gate so borderline webcam frames can pass.
- MIN_PREDICTION_CONFIDENCE floor: predictions below this are dropped
  to null rather than reported as confident-looking output.
- Slightly loosened blur/brightness gate defaults when enhancement
  is enabled.

Prior versions: MediaPipe face detection + pose, quality gating,
EMA tracking.
"""
import os


# ── Frame enhancement (pre-gate quality lift) ────────────────────────────────
ENHANCE_ENABLED = os.environ.get("MINDCARE_ENHANCE", "0") == "1"
ENHANCE_DENOISE = os.environ.get("MINDCARE_ENHANCE_DENOISE", "1") == "1"
ENHANCE_CLAHE = os.environ.get("MINDCARE_ENHANCE_CLAHE", "1") == "1"
ENHANCE_SHARPEN = os.environ.get("MINDCARE_ENHANCE_SHARPEN", "1") == "1"
ENHANCE_TARGET_BRIGHTNESS = float(
    os.environ.get("MINDCARE_ENHANCE_TARGET_BRIGHTNESS", "110.0")
)
ENHANCE_CLAHE_CLIP = float(os.environ.get("MINDCARE_ENHANCE_CLAHE_CLIP", "2.0"))
ENHANCE_SHARPEN_ALPHA = float(os.environ.get("MINDCARE_ENHANCE_SHARPEN_ALPHA", "1.3"))
ENHANCE_SHARPEN_BETA = float(os.environ.get("MINDCARE_ENHANCE_SHARPEN_BETA", "-0.3"))


# ── Face detection (MediaPipe) ────────────────────────────────────────────────
FACE_DETECTOR_CONFIDENCE_THRESHOLD = float(
    os.environ.get("MINDCARE_FACE_DET_CONF", "0.5")
)
FACE_DETECTOR_MODEL_SELECTION = int(
    os.environ.get("MINDCARE_FACE_DET_MODEL", "1")
)  # 0=close-range, 1=full-range
MAX_PERSONS_PER_FRAME = int(os.environ.get("MINDCARE_MAX_PERSONS", "8"))
MIN_FACE_SIZE_PX = int(os.environ.get("MINDCARE_MIN_FACE_PX", "40"))


# ── Emotion classifier ────────────────────────────────────────────────────────
IMAGE_EMOTION_MODEL_ID = os.environ.get(
    "MINDCARE_IMAGE_EMOTION_MODEL", "trpakov/vit-face-expression"
)
USE_TTA_FLIP = os.environ.get("MINDCARE_USE_TTA", "1") == "1"
EMOTION_TEMPERATURE = float(os.environ.get("MINDCARE_EMOTION_TEMP", "1.0"))
MIN_PREDICTION_CONFIDENCE = float(
    os.environ.get("MINDCARE_CONF_FLOOR", "0.15")
)


# ── Pose / kinematic features (MediaPipe Pose) ────────────────────────────────
POSE_MODEL_COMPLEXITY = int(os.environ.get("MINDCARE_POSE_COMPLEXITY", "1"))
POSE_MIN_DETECTION_CONF = float(os.environ.get("MINDCARE_POSE_DET_CONF", "0.5"))
POSE_MIN_TRACKING_CONF = float(os.environ.get("MINDCARE_POSE_TRACK_CONF", "0.5"))
POSE_CROP_PADDING_RATIO = float(os.environ.get("MINDCARE_POSE_PAD_RATIO", "0.6"))


# ── Multi-object tracking ─────────────────────────────────────────────────────
TRACK_IOU_MATCH_THRESHOLD = float(os.environ.get("MINDCARE_TRACK_IOU", "0.3"))
TRACK_MAX_MISSED_FRAMES = int(os.environ.get("MINDCARE_TRACK_MAX_MISSED", "8"))
TRACK_HISTORY_WINDOW = int(os.environ.get("MINDCARE_TRACK_HISTORY", "45"))
EMOTION_EMA_ALPHA = float(os.environ.get("MINDCARE_EMOTION_EMA_ALPHA", "0.35"))


# ── Frame quality gating ──────────────────────────────────────────────────────
_enhance_on = os.environ.get("MINDCARE_ENHANCE", "1") == "1"

MIN_BLUR_VARIANCE = float(
    os.environ.get(
        "MINDCARE_MIN_BLUR_VAR",
        "55.0" if _enhance_on else "100.0",
    )
)
MIN_BRIGHTNESS = float(
    os.environ.get(
        "MINDCARE_MIN_BRIGHTNESS",
        "30.0" if _enhance_on else "40.0",
    )
)
MAX_BRIGHTNESS = float(os.environ.get("MINDCARE_MAX_BRIGHTNESS", "235.0"))


# ── Video sampling ─────────────────────────────────────────────────────────────
VIDEO_SAMPLE_FPS = float(os.environ.get("MINDCARE_VIDEO_SAMPLE_FPS", "4.0"))

SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VIDEO_FORMATS = {".mp4", ".mov", ".avi", ".mkv"}
MAX_IMAGE_SIZE_BYTES = int(
    os.environ.get("MINDCARE_MAX_IMAGE_BYTES", str(10 * 1024 * 1024))
)


DIAGNOSTIC_DISCLAIMER = (
    "This module reports AI-OBSERVED FACIAL EXPRESSIONS and BODY KINEMATICS "
    "(posture angle, movement speed, pacing index, hand-to-face proximity) "
    "computed directly from detected image/video landmarks. These are "
    "descriptive signal-processing outputs, not a mental-health diagnosis "
    "or clinical risk score, and none of them alone or combined should be "
    "presented to a clinician as a finding — only as raw observational data "
    "for a qualified professional to interpret."
)

HF_CACHE_DIR = os.environ.get("MINDCARE_HF_CACHE", "./hf_cache")