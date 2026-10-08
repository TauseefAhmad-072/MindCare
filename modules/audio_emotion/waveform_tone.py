"""
modules/audio_emotion/waveform_tone.py — Acoustic Waveform Tone Analysis for Speech Emotion Recognition.

Analyzes raw waveform tone acoustics and categorizes speech into four primary emotions:
  1. happy
  2. neutral
  3. sad
  4. angry

Acoustic Principles:
  - F0 (Pitch): Fundamental frequency mean, variability (std), and contour.
    * Happy: Elevated pitch, moderate-high pitch variability, animated contour.
    * Neutral: Moderate pitch around speaker baseline (100-220 Hz), low variability.
    * Sad: Low pitch, flat pitch contour (very low std), downward trend.
    * Angry: Elevated pitch, high pitch variability, sharp inflection peaks.

  - Energy (RMS): Vocal loudness and dynamic range.
    * Happy: Moderate-to-high RMS energy.
    * Neutral: Moderate, steady RMS energy.
    * Sad: Low RMS energy (quiet, subdued, withdrawn).
    * Angry: High RMS energy (forceful, loud, sudden spikes).

  - Spectral Timbre:
    * Spectral Centroid: Measures brightness/sharpness of voice.
      - Bright (>1800 Hz): Happy / Angry.
      - Dark/Muffled (<1400 Hz): Sad / Neutral.
    * Zero-Crossing Rate (ZCR): Measures breathiness, friction, harshness.
      - High ZCR: Angry / Agitated.
      - Balanced ZCR: Happy / Neutral.
      - Low ZCR: Sad.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Any, Tuple
import numpy as np
import librosa

logger = logging.getLogger("mindcare.audio_emotion.waveform_tone")

TARGET_4_CLASSES = ["happy", "neutral", "sad", "angry"]


def extract_waveform_tone_features(waveform: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
    """
    Extracts core physical acoustic features directly from the audio waveform:
      - Pitch (F0) mean, std, min, max, range via librosa.yin
      - RMS energy mean, std, dynamic range
      - Spectral Centroid (tone brightness)
      - Zero-Crossing Rate (tension / harshness)
      - Voiced speech ratio
    """
    y = np.asarray(waveform, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)

    duration = float(len(y) / sr) if sr > 0 else 0.0
    if len(y) < 512:
        return {
            "duration_sec": duration,
            "pitch_mean_hz": 0.0,
            "pitch_std_hz": 0.0,
            "pitch_range_hz": 0.0,
            "pitch_contour": "Silent / Insufficient Data",
            "energy_rms": 0.0,
            "energy_level": "None",
            "spectral_centroid_hz": 0.0,
            "tone_brightness": "Unknown",
            "zero_crossing_rate": 0.0,
            "voiced_ratio": 0.0,
            "pause_ratio": 1.0,
        }

    # 1. RMS Energy
    rms_frames = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    rms_mean = float(np.mean(rms_frames))
    rms_std = float(np.std(rms_frames))
    db_frames = 20 * np.log10(rms_frames + 1e-8)
    dynamic_range_db = float(np.percentile(db_frames, 95) - np.percentile(db_frames, 5))

    if rms_mean > 0.06:
        energy_level = "High Loudness / Intense"
    elif rms_mean < 0.025:
        energy_level = "Low / Subdued Vocal Energy"
    else:
        energy_level = "Moderate / Balanced Loudness"

    # 2. Fundamental Frequency (Pitch F0) using yin
    try:
        f0 = librosa.yin(y, fmin=65, fmax=500, sr=sr, frame_length=1024, hop_length=256)
        # Filter voiced frames by energy to eliminate silence artifacts
        med_rms = float(np.median(rms_frames)) if len(rms_frames) else 0.0
        f0_voiced = np.where(rms_frames[:len(f0)] > 0.4 * med_rms, f0, np.nan)
        valid = f0_voiced[np.isfinite(f0_voiced)]

        if len(valid) >= 3:
            pitch_mean = float(np.mean(valid))
            pitch_std = float(np.std(valid))
            p5, p95 = np.percentile(valid, [5, 95])
            pitch_range = float(p95 - p5)
            voiced_ratio = float(len(valid) / max(1, len(f0)))

            if pitch_std > 55.0:
                pitch_contour = "Highly Variable / Animated"
            elif pitch_std < 18.0:
                pitch_contour = "Flat / Monotone Contour"
            elif pitch_mean > 210.0:
                pitch_contour = "Elevated & Expressive"
            else:
                pitch_contour = "Steady & Measured"
        else:
            pitch_mean = 0.0
            pitch_std = 0.0
            pitch_range = 0.0
            voiced_ratio = 0.0
            pitch_contour = "Unvoiced / Flat"
    except Exception as exc:
        logger.debug("Pitch extraction exception: %s", exc)
        pitch_mean = 0.0
        pitch_std = 0.0
        pitch_range = 0.0
        voiced_ratio = 0.0
        pitch_contour = "Undetermined"

    # 3. Spectral Centroid (Tone Brightness)
    try:
        cent_frames = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=256)[0]
        cent_mean = float(np.mean(cent_frames))
        if cent_mean > 2100:
            tone_brightness = "Bright & Resonant (>2100 Hz)"
        elif cent_mean < 1400:
            tone_brightness = "Dark & Subdued (<1400 Hz)"
        else:
            tone_brightness = "Neutral & Balanced (1400-2100 Hz)"
    except Exception:
        cent_mean = 0.0
        tone_brightness = "Moderate"

    # 4. Zero-Crossing Rate (ZCR - breathiness / harshness)
    try:
        zcr_frames = librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256)[0]
        zcr_mean = float(np.mean(zcr_frames))
    except Exception:
        zcr_mean = 0.0

    # 5. Pauses
    try:
        intervals = librosa.effects.split(y, top_db=28, frame_length=1024, hop_length=256)
        speech_time = float(sum(e - s for s, e in intervals)) / sr
        pause_ratio = float(max(0.0, 1.0 - speech_time / max(duration, 1e-5)))
    except Exception:
        pause_ratio = 0.1

    return {
        "duration_sec": round(duration, 3),
        "pitch_mean_hz": round(pitch_mean, 2),
        "pitch_std_hz": round(pitch_std, 2),
        "pitch_range_hz": round(pitch_range, 2),
        "pitch_contour": pitch_contour,
        "energy_rms": round(rms_mean, 4),
        "energy_std": round(rms_std, 4),
        "dynamic_range_db": round(dynamic_range_db, 2),
        "energy_level": energy_level,
        "spectral_centroid_hz": round(cent_mean, 1),
        "tone_brightness": tone_brightness,
        "zero_crossing_rate": round(zcr_mean, 4),
        "voiced_ratio": round(voiced_ratio, 3),
        "pause_ratio": round(pause_ratio, 3),
    }


def classify_waveform_tone(waveform: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
    """
    Categorizes the speech waveform into 4 core emotions:
      - happy
      - neutral
      - sad
      - angry

    Returns:
      {
        "dominant_emotion": "happy",
        "confidence": 0.82,
        "probabilities": {"happy": 0.82, "neutral": 0.09, "sad": 0.03, "angry": 0.06},
        "waveform_tone_analysis": {...detailed acoustic metrics...},
        "tone_explanation": "..."
      }
    """
    feats = extract_waveform_tone_features(waveform, sr)

    rms = feats["energy_rms"]
    pitch_m = feats["pitch_mean_hz"]
    pitch_s = feats["pitch_std_hz"]
    cent = feats["spectral_centroid_hz"]
    zcr = feats["zero_crossing_rate"]
    pause = feats["pause_ratio"]

    # Scale into standard conversational baselines
    energy_norm = float(np.clip(rms / 0.07, 0.0, 2.5))
    pitch_mean_norm = float(np.clip(pitch_m / 200.0 if pitch_m > 0 else 0.5, 0.0, 2.5))
    pitch_var_norm = float(np.clip(pitch_s / 45.0, 0.0, 2.5))
    centroid_norm = float(np.clip(cent / 1800.0, 0.0, 2.5))
    zcr_norm = float(np.clip(zcr / 0.12, 0.0, 2.5))

    # Acoustic rule scorers:
    # 1. Angry: high energy, high pitch variability, sharp centroid, elevated ZCR
    s_angry = 2.4 * energy_norm + 1.8 * pitch_var_norm + 1.2 * zcr_norm + 0.8 * centroid_norm - 1.0 * pause

    # 2. Happy: elevated pitch, moderate-high energy, bright centroid, animated variability
    s_happy = 1.9 * pitch_mean_norm + 1.7 * energy_norm + 1.6 * centroid_norm + 0.9 * pitch_var_norm - 1.4 * zcr_norm

    # 3. Sad: low energy, flat pitch (low variability), dark centroid, high pause ratio
    s_sad = 2.4 * (1.2 - min(1.2, energy_norm)) + 1.8 * (1.2 - min(1.2, pitch_mean_norm)) + 1.5 * (1.0 - min(1.0, pitch_var_norm)) + 1.2 * pause

    # 4. Neutral: balanced energy, balanced pitch, steady contour, moderate centroid
    energy_diff = abs(energy_norm - 0.9)
    pitch_diff = abs(pitch_mean_norm - 0.9)
    s_neutral = 2.0 - 1.2 * energy_diff - 1.0 * pitch_diff - 0.8 * pitch_var_norm + 0.5 * (1.0 - abs(centroid_norm - 1.0))

    scores = {
        "happy": float(s_happy),
        "neutral": float(s_neutral),
        "sad": float(s_sad),
        "angry": float(s_angry),
    }

    # Softmax normalization
    arr = np.array([scores[c] for c in TARGET_4_CLASSES], dtype=np.float64)
    arr = arr - np.max(arr)
    exp_vals = np.exp(arr * 1.6)
    probs = exp_vals / np.sum(exp_vals)
    prob_dict = {c: round(float(p), 4) for c, p in zip(TARGET_4_CLASSES, probs)}

    dominant = max(prob_dict, key=prob_dict.get)
    confidence = prob_dict[dominant]

    # Clinical explanation
    if dominant == "happy":
        tone_exp = (
            f"Vocal tone displays elevated pitch ({feats['pitch_mean_hz']} Hz) with animated "
            f"modulation and bright spectral resonance ({feats['spectral_centroid_hz']} Hz), "
            "indicating positive, enthusiastic vocal affect."
        )
    elif dominant == "angry":
        tone_exp = (
            f"Acoustic waveform shows high energy volume (RMS {feats['energy_rms']:.3f}) and "
            f"wide pitch fluctuation ({feats['pitch_std_hz']} Hz variation) with elevated vocal tension, "
            "indicating heightened arousal and forceful vocal tone."
        )
    elif dominant == "sad":
        tone_exp = (
            f"Waveform exhibits subdued vocal energy (RMS {feats['energy_rms']:.3f}) with flat pitch "
            f"contour and prolonged pauses ({int(feats['pause_ratio']*100)}% silence), "
            "characterizing restrained, low-affect speech."
        )
    else:
        tone_exp = (
            f"Acoustic features demonstrate steady, conversational pitch ({feats['pitch_mean_hz']} Hz) "
            "and balanced dynamic range, characteristic of composed, neutral speech."
        )

    return {
        "dominant_emotion": dominant,
        "confidence": confidence,
        "probabilities": prob_dict,
        "waveform_tone_analysis": feats,
        "tone_explanation": tone_exp,
    }
