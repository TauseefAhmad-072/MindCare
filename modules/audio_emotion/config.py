"""
modules/audio_emotion/config.py — Configuration for the Speech Emotion Recognition module.
"""
import os

AUDIO_EMOTION_MODEL_ID = os.environ.get(
    "MINDCARE_AUDIO_EMOTION_MODEL",
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
)

TARGET_SAMPLE_RATE = 16000
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_AUDIO_SIZE_BYTES = int(os.environ.get("MINDCARE_MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))
MAX_AUDIO_DURATION_SEC = float(os.environ.get("MINDCARE_MAX_AUDIO_SEC", "600"))
MIN_AUDIO_DURATION_SEC = float(os.environ.get("MINDCARE_MIN_AUDIO_SEC", "0.5"))

# ── Quality gating parameters (DSP fallbacks)
LOW_ENERGY_RMS_THRESHOLD = float(os.environ.get("MINDCARE_LOW_ENERGY_RMS", "0.01"))
MAX_CLIPPING_RATIO = float(os.environ.get("MINDCARE_MAX_CLIPPING_RATIO", "0.01"))

# ── Windowed / time-synced analysis parameters
AUDIO_WINDOW_SEC = float(os.environ.get("MINDCARE_AUDIO_WINDOW_SEC", "3.0"))
AUDIO_WINDOW_STRIDE_SEC = float(os.environ.get("MINDCARE_AUDIO_WINDOW_STRIDE_SEC", "1.5"))
AUDIO_MIN_WINDOW_SEC = float(os.environ.get("MINDCARE_AUDIO_MIN_WINDOW_SEC", "0.75"))

# ── Calibration Temperature Parameter (FIXED: Resolves app.py initialization crash)
EMOTION_TEMPERATURE = float(os.environ.get("MINDCARE_AUDIO_TEMP", "1.0"))

DIAGNOSTIC_DISCLAIMER = (
    "This is an AI-observed VOCAL EMOTION indicator based on acoustic "
    "patterns, not a mental-health diagnosis. Speech emotion recognition "
    "has known accuracy limits across accents, languages, and recording "
    "conditions, and lower accuracy still when falling back to the "
    "handcrafted-feature heuristic. Do not use this output alone."
)

HF_CACHE_DIR = os.environ.get("MINDCARE_HF_CACHE", "./hf_cache")
