"""
modules/audio_emotion/preprocessing.py — Audio loading, validation,
resampling, and hand-crafted feature extraction.

FIX: this file previously imported `LOW_ENERGY_RMS_THRESHOLD` from
`.config`, which did not define it — importing this module would raise
`ImportError: cannot import name 'LOW_ENERGY_RMS_THRESHOLD'`. It's now
defined in config.py, and `MAX_CLIPPING_RATIO` is also pulled from
config instead of being a bare literal, so both are tunable in one place.

This module is now actually wired into predictor.py's main path (v1 had
predictor.py bypass it entirely and call librosa.load directly, so none
of the validation/quality logic here ever ran).
"""
import io
import logging
import os
from dataclasses import dataclass
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf

from .config import (
    TARGET_SAMPLE_RATE,
    SUPPORTED_AUDIO_FORMATS,
    MAX_AUDIO_SIZE_BYTES,
    MAX_AUDIO_DURATION_SEC,
    MIN_AUDIO_DURATION_SEC,
    LOW_ENERGY_RMS_THRESHOLD,
    MAX_CLIPPING_RATIO,
)

logger = logging.getLogger("mindcare.audio_emotion.preprocessing")


class AudioValidationError(ValueError):
    """Raised for invalid, corrupt, or unsupported audio input."""


@dataclass
class AudioQualityReport:
    rms: float
    clipping_ratio: float
    quality_label: str  # "good" | "fair" | "poor (...)"
    is_acceptable: bool


def _validate_duration(duration_sec: float):
    if duration_sec < MIN_AUDIO_DURATION_SEC:
        raise AudioValidationError(
            f"Audio too short ({duration_sec:.2f}s); minimum is {MIN_AUDIO_DURATION_SEC}s."
        )
    if duration_sec > MAX_AUDIO_DURATION_SEC:
        raise AudioValidationError(
            f"Audio too long ({duration_sec:.1f}s); maximum is {MAX_AUDIO_DURATION_SEC}s."
        )


def load_audio_from_path(path: str) -> Tuple[np.ndarray, int, float]:
    """Returns (waveform_mono, sample_rate, duration_sec) resampled to TARGET_SAMPLE_RATE."""
    if not os.path.exists(path):
        raise AudioValidationError(f"Audio file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_AUDIO_FORMATS:
        raise AudioValidationError(
            f"Unsupported audio format '{ext}'. Supported: {sorted(SUPPORTED_AUDIO_FORMATS)}"
        )
    if os.path.getsize(path) > MAX_AUDIO_SIZE_BYTES:
        raise AudioValidationError(f"Audio exceeds max size of {MAX_AUDIO_SIZE_BYTES} bytes.")

    try:
        waveform, sr = librosa.load(path, sr=TARGET_SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise AudioValidationError(f"Could not decode audio (corrupt or unsupported): {exc}") from exc

    if waveform.size == 0:
        raise AudioValidationError("Decoded audio is empty.")

    duration_sec = len(waveform) / sr
    _validate_duration(duration_sec)
    return waveform, sr, duration_sec


def load_audio_from_bytes(raw: bytes, filename_hint: str = "audio.wav") -> Tuple[np.ndarray, int, float]:
    if raw is None or len(raw) == 0:
        raise AudioValidationError("Empty audio data.")
    if len(raw) > MAX_AUDIO_SIZE_BYTES:
        raise AudioValidationError(f"Audio exceeds max size of {MAX_AUDIO_SIZE_BYTES} bytes.")
    ext = os.path.splitext(filename_hint)[1].lower()
    if ext and ext not in SUPPORTED_AUDIO_FORMATS:
        raise AudioValidationError(
            f"Unsupported audio format '{ext}'. Supported: {sorted(SUPPORTED_AUDIO_FORMATS)}"
        )
    try:
        try:
            data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception:
            data, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != TARGET_SAMPLE_RATE:
            data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
            sr = TARGET_SAMPLE_RATE
    except Exception as exc:
        raise AudioValidationError(f"Could not decode audio (corrupt or unsupported): {exc}") from exc

    if data.size == 0:
        raise AudioValidationError("Decoded audio is empty.")

    duration_sec = len(data) / sr
    _validate_duration(duration_sec)
    return data.astype(np.float32), sr, duration_sec


def assess_quality(waveform: np.ndarray) -> AudioQualityReport:
    """RMS energy + clipping based quality heuristic. Used to decide whether
    a given window is worth trusting before it's sent to either backend."""
    if waveform.size == 0:
        return AudioQualityReport(0.0, 0.0, "poor (empty)", False)

    rms = float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2)))
    clipping_ratio = float(np.mean(np.abs(waveform) >= 0.99))

    if clipping_ratio > MAX_CLIPPING_RATIO:
        return AudioQualityReport(rms, clipping_ratio, "poor (clipping detected)", False)
    if rms < LOW_ENERGY_RMS_THRESHOLD:
        return AudioQualityReport(rms, clipping_ratio, "poor (very low signal energy)", False)
    if rms < LOW_ENERGY_RMS_THRESHOLD * 3:
        return AudioQualityReport(rms, clipping_ratio, "fair", True)
    return AudioQualityReport(rms, clipping_ratio, "good", True)


def extract_handcrafted_features(waveform: np.ndarray, sr: int) -> dict:
    """
    MFCC + pitch + energy statistics. Used by:
      (a) the heuristic fallback classifier in model.py when the neural
          model is unavailable, and
      (b) optionally alongside neural predictions as extra diagnostic
          metadata (e.g. for debugging why a clip got a given label).
    """
    if waveform.size < 512:
        # Too short for a meaningful MFCC/pitch estimate — return zeros
        # rather than letting librosa error out or fabricate noise.
        return {
            "mfcc_mean": np.zeros(13), "mfcc_std": np.zeros(13),
            "rms_mean": 0.0, "rms_std": 0.0, "zcr_mean": 0.0,
            "pitch_mean": 0.0, "pitch_std": 0.0,
        }

    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    rms = librosa.feature.rms(y=waveform)[0]
    zcr = librosa.feature.zero_crossing_rate(waveform)[0]

    try:
        f0, voiced_flag, _ = librosa.pyin(
            waveform, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
        pitch_mean = float(np.nanmean(f0_voiced)) if f0_voiced.size else 0.0
        pitch_std = float(np.nanstd(f0_voiced)) if f0_voiced.size else 0.0
    except Exception:  # noqa: BLE001 - pyin can fail on very short/silent clips
        pitch_mean, pitch_std = 0.0, 0.0

    return {
        "mfcc_mean": mfcc_mean,
        "mfcc_std": mfcc_std,
        "rms_mean": float(rms.mean()),
        "rms_std": float(rms.std()),
        "zcr_mean": float(zcr.mean()),
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
    }


def slice_windows(waveform: np.ndarray, sr: int, window_sec: float, stride_sec: float,
                   min_window_sec: float) -> List[Tuple[float, float, np.ndarray]]:
    """Splits a waveform into overlapping (start_sec, end_sec, sub_waveform)
    windows for time-series analysis. The final partial window is kept only
    if it's at least `min_window_sec` long, so a 0.05s sliver at the tail
    doesn't get force-fed to the model."""
    total_samples = len(waveform)
    window_samples = int(window_sec * sr)
    stride_samples = max(1, int(stride_sec * sr))
    min_samples = int(min_window_sec * sr)

    windows = []
    start = 0
    while start < total_samples:
        end = min(start + window_samples, total_samples)
        if end - start >= min_samples:
            windows.append((start / sr, end / sr, waveform[start:end]))
        if end >= total_samples:
            break
        start += stride_samples
    return windows