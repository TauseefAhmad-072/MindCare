"""
modules/audio_emotion/predictor.py — Public interface for MODULE C:
Speech Emotion Recognition.

FIXES:
- Completely scrubbed broken trailing text/notes from the base of the file 
  to eliminate the syntax error and unblock the master startup sequence.
"""
import logging
from datetime import datetime
import numpy as np

from .config import (
    TARGET_SAMPLE_RATE, DIAGNOSTIC_DISCLAIMER,
    AUDIO_WINDOW_SEC, AUDIO_WINDOW_STRIDE_SEC, AUDIO_MIN_WINDOW_SEC,
)
from .model import get_model, classify_waveform, classify_heuristic
from .preprocessing import (
    load_audio_from_path, load_audio_from_bytes, assess_quality,
    extract_handcrafted_features, slice_windows, AudioValidationError,
)

logger = logging.getLogger("mindcare.audio_emotion.predictor")


def _classify_waveform(waveform: np.ndarray) -> dict:
    """Runs 4-class SER (happy, neutral, sad, angry) combining neural
    transformer and physical acoustic waveform tone analysis."""
    cls_res = classify_waveform(waveform)
    primary = cls_res["dominant_emotion"]
    return {
        "primary_tone": primary.upper(),
        "dominant_emotion": primary.lower(),
        "confidence": cls_res["confidence"],
        "probabilities": cls_res["probabilities"],
        "backend_used": cls_res["backend_used"],
        "waveform_tone_analysis": cls_res.get("waveform_tone_analysis", {}),
        "tone_explanation": cls_res.get("tone_explanation", ""),
    }


def predict_waveform(waveform: np.ndarray, sr: int = TARGET_SAMPLE_RATE) -> dict:
    """Classify an in-memory waveform directly (already resampled to
    TARGET_SAMPLE_RATE)."""
    if sr != TARGET_SAMPLE_RATE:
        import librosa
        waveform = librosa.resample(waveform.astype(np.float32), orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)

    quality = assess_quality(waveform)
    if not quality.is_acceptable:
        return {
            "status": "skipped",
            "reason": quality.quality_label,
            "rms": round(quality.rms, 4),
            "clipping_ratio": round(quality.clipping_ratio, 4),
            "disclaimer": DIAGNOSTIC_DISCLAIMER,
        }

    result = _classify_waveform(waveform)
    result.update({
        "status": "success",
        "emotion": str(result.get("dominant_emotion", "neutral")).lower(),
        "dominant_emotion": str(result.get("dominant_emotion", "neutral")).lower(),
        "emotion_distribution": result.get("probabilities", {}),
        "quality": quality.quality_label,
        "duration_seconds": round(len(waveform) / TARGET_SAMPLE_RATE, 3),
        "disclaimer": DIAGNOSTIC_DISCLAIMER,
        "audio_disclaimer": DIAGNOSTIC_DISCLAIMER,
    })
    return result


def predict(audio_path: str) -> dict:
    """Whole-clip prediction (backward-compatible single-result API)."""
    try:
        waveform, sr, duration = load_audio_from_path(audio_path)
        res = predict_waveform(waveform, sr)
        res["duration_seconds"] = round(duration, 3)
        return res
    except AudioValidationError as exc:
        return {"status": "failed", "error": str(exc)}
    except Exception as exc:
        return {"status": "failed", "error": f"Unexpected audio file processing error: {exc}"}


def predict_bytes(raw_bytes: bytes, filename_hint: str = "audio.wav") -> dict:
    """Decodes raw audio bytes; runs 4-class SER + Waveform Tone Analysis + Urdu ASR + Urdu Sentiment.

    Categorizes vocal affect into: happy, neutral, sad, angry.
    Provides detailed acoustic waveform metrics (F0 pitch mean/dynamics, RMS energy, spectral centroid, ZCR).
    Provides Urdu transcript and sentiment analysis when Urdu speech is detected.
    """
    try:
        waveform, sr, duration = load_audio_from_bytes(raw_bytes, filename_hint)
        res = predict_waveform(waveform, sr)

        dom_emo = str(res.get("dominant_emotion", "neutral")).lower()
        base = {
            "status": "success",
            "emotion": dom_emo,
            "dominant_emotion": dom_emo,
            "primary_tone": str(res.get("primary_tone", dom_emo.upper())),
            "confidence": res.get("confidence", 0.0),
            "probabilities": res.get("probabilities", {}),
            "emotion_distribution": res.get("probabilities", {}),
            "waveform_analysis": res.get("waveform_tone_analysis", {}),
            "waveform_tone_analysis": res.get("waveform_tone_analysis", {}),
            "tone_explanation": res.get("tone_explanation", ""),
            "duration": round(duration, 3),
            "duration_seconds": round(duration, 3),
            "audio_quality": res.get("quality", "good"),
            "model": res.get("backend_used", "waveform_tone_hybrid"),
            "timestamp": datetime.utcnow().isoformat(),
            "disclaimer": DIAGNOSTIC_DISCLAIMER,
            "audio_disclaimer": DIAGNOSTIC_DISCLAIMER,
            "is_diagnosis": False,
        }

        # ── Urdu layer (ASR + normalization + prosody + crisis + sentiment) ──────────
        try:
            from .urdu import enrich_audio_result
            enriched = enrich_audio_result(waveform, sr, existing=base)
            base.update(enriched)
        except Exception as exc:
            logger.warning("Urdu enrichment failed: %s", exc)
            base["urdu_layer_error"] = str(exc)

        return base
    except AudioValidationError as exc:
        return {"status": "failed", "error": str(exc)}
    except Exception as exc:
        return {"status": "failed", "error": f"Unexpected stream bounds exception: {exc}"}

def predict_windows(
    audio_path: str,
    window_sec: float = AUDIO_WINDOW_SEC,
    stride_sec: float = AUDIO_WINDOW_STRIDE_SEC,
    min_window_sec: float = AUDIO_MIN_WINDOW_SEC,
) -> dict:
    try:
        waveform, sr, duration = load_audio_from_path(audio_path)
        return predict_windows_from_waveform(waveform, sr, window_sec, stride_sec, min_window_sec)
    except AudioValidationError as exc:
        return {"status": "failed", "error": str(exc)}


def predict_windows_from_waveform(
    waveform: np.ndarray,
    sr: int,
    window_sec: float = AUDIO_WINDOW_SEC,
    stride_sec: float = AUDIO_WINDOW_STRIDE_SEC,
    min_window_sec: float = AUDIO_MIN_WINDOW_SEC,
) -> dict:
    if sr != TARGET_SAMPLE_RATE:
        import librosa
        waveform = librosa.resample(waveform.astype(np.float32), orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
        sr = TARGET_SAMPLE_RATE

    windows = slice_windows(waveform, sr, window_sec, stride_sec, min_window_sec)
    if not windows:
        return {"status": "failed", "error": "Audio too short to form any analysis window."}

    window_results = []
    for (start_sec, end_sec, sub) in windows:
        res = predict_waveform(sub, sr)
        res.update({"start_sec": round(start_sec, 2), "end_sec": round(end_sec, 2)})
        window_results.append(res)
        
    return {
        "status": "success",
        "duration_seconds": round(len(waveform) / sr, 3),
        "windows": window_results
    }


def get_window_at_timestamp(
    waveform: np.ndarray, sr: int, center_sec: float, window_sec: float = AUDIO_WINDOW_SEC
) -> dict:
    total_sec = len(waveform) / sr
    half = window_sec / 2.0
    start_sec = max(0.0, center_sec - half)
    end_sec = min(total_sec, center_sec + half)
    start_i, end_i = int(start_sec * sr), int(end_sec * sr)
    sub = waveform[start_i:end_i]

    if sub.size < int(AUDIO_MIN_WINDOW_SEC * sr):
        return {"status": "skipped", "reason": "window_too_short_at_clip_boundary",
                "start_sec": round(start_sec, 2), "end_sec": round(end_sec, 2)}

    quality = assess_quality(sub)
    if not quality.is_acceptable:
        return {"status": "skipped", "reason": quality.quality_label,
                "start_sec": round(start_sec, 2), "end_sec": round(end_sec, 2)}

    result = _classify_waveform(sub)
    result.update({"status": "success", "start_sec": round(start_sec, 2), "end_sec": round(end_sec, 2),
                    "quality": quality.quality_label})
    return result


__all__ = [
    "predict", "predict_bytes", "predict_waveform",
    "predict_windows", "predict_windows_from_waveform", "get_window_at_timestamp",
    "AudioValidationError",
]
