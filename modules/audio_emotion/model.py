"""
modules/audio_emotion/model.py — Speech Emotion Recognition model & waveform tone analysis.

Categorizes audio into four primary emotions:
  1. happy
  2. neutral
  3. sad
  4. angry

Combines:
  (a) Local Wav2Vec2 neural model with classification head (from ./models/audio)
  (b) Direct waveform tone acoustic analysis (F0/pitch contour, RMS energy, spectral centroid, ZCR)
"""
import logging
import os
import threading
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .config import (
    AUDIO_EMOTION_MODEL_ID,
    HF_CACHE_DIR,
    TARGET_SAMPLE_RATE,
    EMOTION_TEMPERATURE,
)
from .waveform_tone import (
    classify_waveform_tone,
    extract_waveform_tone_features,
    TARGET_4_CLASSES,
)

logger = logging.getLogger("mindcare.audio_emotion.model")

_lock = threading.Lock()
_model = None
_feature_extractor = None
_device = None
_use_fallback = False
_load_error: Optional[str] = None
_is_custom_local_head = False


def _detect_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomWav2Vec2EmotionModel:
    """Wrapper around Wav2Vec2 base model + trained classification head."""
    def __init__(self, base_model, head, id2label, device):
        self.base_model = base_model
        self.head = head
        self.id2label = id2label
        self.device = device

    def eval(self):
        self.base_model.eval()
        self.head.eval()

    def __call__(self, input_values):
        import torch
        with torch.no_grad():
            outputs = self.base_model(input_values=input_values)
            last_hidden = outputs.last_hidden_state
            pooled = torch.mean(last_hidden, dim=1)
            logits = self.head(pooled)
        return logits


def _try_load_neural_model() -> bool:
    global _model, _feature_extractor, _device, _is_custom_local_head
    try:
        import torch
        import torch.nn as nn
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

        _device = _detect_device()

        # Priority 1: Check ./models/audio
        models_audio_dir = os.path.abspath("./models/audio")
        safetensor_file = os.path.join(models_audio_dir, "model.safetensors")
        if os.path.exists(safetensor_file):
            logger.info("Loading speech emotion model from local offline path: %s", models_audio_dir)
            _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                models_audio_dir, local_files_only=True
            )
            base = Wav2Vec2Model.from_pretrained(models_audio_dir, local_files_only=True).to(_device)

            class EmotionHead(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.dense = nn.Linear(1024, 1024)
                    self.output = nn.Linear(1024, 8)
                def forward(self, x):
                    return self.output(torch.tanh(self.dense(x)))

            head = EmotionHead().to(_device)
            from safetensors import safe_open
            with safe_open(safetensor_file, framework="pt") as f:
                head.dense.weight.data.copy_(f.get_tensor("classifier.dense.weight"))
                head.dense.bias.data.copy_(f.get_tensor("classifier.dense.bias"))
                head.output.weight.data.copy_(f.get_tensor("classifier.output.weight"))
                head.output.bias.data.copy_(f.get_tensor("classifier.output.bias"))

            id2label = {
                0: "angry", 1: "calm", 2: "disgust", 3: "fearful",
                4: "happy", 5: "neutral", 6: "sad", 7: "surprised"
            }
            _model = CustomWav2Vec2EmotionModel(base, head, id2label, _device)
            _model.eval()
            _is_custom_local_head = True
            logger.info("Speech-emotion local neural model loaded successfully.")
            return True

        # Priority 2: Try HuggingFace offline cache
        from transformers import Wav2Vec2ForSequenceClassification
        try:
            _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                AUDIO_EMOTION_MODEL_ID, cache_dir=HF_CACHE_DIR, local_files_only=True
            )
            _model = Wav2Vec2ForSequenceClassification.from_pretrained(
                AUDIO_EMOTION_MODEL_ID, cache_dir=HF_CACHE_DIR, local_files_only=True
            ).to(_device)
            _model.eval()
            _is_custom_local_head = False
            logger.info("Speech-emotion model loaded from offline HF cache.")
            return True
        except Exception:
            pass

        return False
    except Exception as exc:
        logger.warning("Neural audio-emotion model unavailable (%s); using waveform tone analysis.", exc)
        return False


def get_model() -> bool:
    """Returns True if the neural model is active, False if running waveform tone fallback."""
    global _use_fallback, _load_error
    if _model is not None or _use_fallback:
        return not _use_fallback
    with _lock:
        if _model is not None or _use_fallback:
            return not _use_fallback
        if not _try_load_neural_model():
            _use_fallback = True
            _load_error = "Neural model offline — using direct waveform tone analysis."
    return not _use_fallback


def is_loaded() -> bool:
    return _model is not None


def model_status() -> dict:
    if _model is not None:
        return {
            "loaded": True,
            "backend": "neural_wav2vec2_hybrid",
            "model_path": "./models/audio",
            "device": str(_device),
            "target_classes": TARGET_4_CLASSES,
        }
    return {
        "loaded": False,
        "backend": "waveform_tone_dsp",
        "error": _load_error,
        "target_classes": TARGET_4_CLASSES,
    }


def map_8_to_4_classes(raw_8_probs: Dict[str, float]) -> Dict[str, float]:
    """
    Maps 8 standard RAVDESS/SER classes to the 4 target emotions:
      - happy
      - neutral
      - sad
      - angry
    """
    p_happy = raw_8_probs.get("happy", 0.0) + 0.6 * raw_8_probs.get("surprised", 0.0)
    p_neutral = raw_8_probs.get("neutral", 0.0) + raw_8_probs.get("calm", 0.0)
    p_sad = raw_8_probs.get("sad", 0.0) + 0.4 * raw_8_probs.get("fearful", 0.0)
    p_angry = raw_8_probs.get("angry", 0.0) + raw_8_probs.get("disgust", 0.0) + 0.6 * raw_8_probs.get("fearful", 0.0)

    total = p_happy + p_neutral + p_sad + p_angry
    if total <= 1e-6:
        return {c: 0.25 for c in TARGET_4_CLASSES}

    return {
        "happy": round(float(p_happy / total), 4),
        "neutral": round(float(p_neutral / total), 4),
        "sad": round(float(p_sad / total), 4),
        "angry": round(float(p_angry / total), 4),
    }


def classify_neural(waveform: np.ndarray) -> Dict[str, float]:
    """Runs neural model and returns 4-class probabilities."""
    import torch
    import torch.nn.functional as F

    inputs = _feature_extractor(waveform, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(_device)

    with torch.no_grad():
        if _is_custom_local_head:
            logits = _model(input_values)
            id2label = _model.id2label
        else:
            logits = _model(input_values).logits
            id2label = _model.config.id2label

        probs = F.softmax(logits / EMOTION_TEMPERATURE, dim=-1)[0].cpu().numpy()

    raw_8 = {id2label[i].lower(): float(p) for i, p in enumerate(probs)}
    return map_8_to_4_classes(raw_8)


def classify_waveform(waveform: np.ndarray) -> Dict[str, Any]:
    """
    Unified Classifier:
      1. Runs direct waveform tone acoustic analysis (F0 pitch, RMS energy, spectral centroid, ZCR).
      2. If neural Wav2Vec2 model is available, evaluates neural probabilities and fuses with waveform tone scores.
      3. Categorizes strictly into: happy, neutral, sad, angry.
    """
    # 1. Waveform Tone Analysis
    dsp_result = classify_waveform_tone(waveform, sr=TARGET_SAMPLE_RATE)
    dsp_probs = dsp_result["probabilities"]
    tone_analysis = dsp_result["waveform_tone_analysis"]
    tone_explanation = dsp_result["tone_explanation"]

    neural_active = get_model()
    if neural_active:
        try:
            neural_probs = classify_neural(waveform)
            # Fused ensemble: 55% neural transformer representation + 45% physical acoustic waveform tone
            fused_probs = {}
            for c in TARGET_4_CLASSES:
                fused_probs[c] = round(0.55 * neural_probs.get(c, 0.0) + 0.45 * dsp_probs.get(c, 0.0), 4)
            total = sum(fused_probs.values())
            if total > 0:
                fused_probs = {c: round(v / total, 4) for c, v in fused_probs.items()}
            primary = max(fused_probs, key=fused_probs.get)
            return {
                "dominant_emotion": primary,
                "confidence": fused_probs[primary],
                "probabilities": fused_probs,
                "backend_used": "wav2vec2_waveform_tone_hybrid",
                "waveform_tone_analysis": tone_analysis,
                "tone_explanation": tone_explanation,
            }
        except Exception as exc:
            logger.warning("Neural forward pass failed (%s); using direct waveform tone analysis.", exc)

    # Pure waveform tone classifier
    primary = dsp_result["dominant_emotion"]
    return {
        "dominant_emotion": primary,
        "confidence": dsp_result["confidence"],
        "probabilities": dsp_probs,
        "backend_used": "waveform_tone_acoustic_dsp",
        "waveform_tone_analysis": tone_analysis,
        "tone_explanation": tone_explanation,
    }


def classify_heuristic(features: dict) -> Dict[str, float]:
    """Backward compatibility alias for heuristic SER callers."""
    return classify_waveform_tone(np.zeros(16000, dtype=np.float32), sr=TARGET_SAMPLE_RATE)["probabilities"]