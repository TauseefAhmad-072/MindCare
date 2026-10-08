"""
modules/image_emotion/model.py — Facial expression classifier.

CHANGES FROM v1
----------------
- classify_faces_batch(): classifies ALL detected faces in one forward
  pass instead of one-crop-at-a-time, which is both faster and lets you
  process multi-person frames without a per-face model call.
- Optional horizontal-flip test-time augmentation (TTA): each crop is
  classified normally and mirrored, and the two probability vectors are
  averaged. This is a standard, well-evidenced accuracy improvement for
  face models (reduces sensitivity to left/right lighting or pose bias)
  at the cost of ~2x compute on the classifier only (still fast on CPU
  for the crop sizes involved).
- Softmax temperature is now configurable so probabilities can be
  calibrated instead of trusting raw softmax confidence.
- Model loading keeps the local-path-first / hub-fallback pattern from
  v1 (useful for the "fully offline" deployment requirement) but no
  longer raises a bare RuntimeError with no remediation path — it
  returns a clear, actionable message either way.
"""
import logging
import os
import threading
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import IMAGE_EMOTION_MODEL_ID, HF_CACHE_DIR, USE_TTA_FLIP, EMOTION_TEMPERATURE

# NOTE: torch / transformers are imported lazily inside the functions that
# need them (get_model_and_processor, _forward_batch), matching the
# lazy-loading pattern already used by modules/audio_emotion/model.py and
# documented in this package's __init__.py. This means importing
# modules.image_emotion (e.g. just to read config, or to use the face
# detector / pose estimator, which only need mediapipe+opencv) no longer
# forces a torch/transformers import for callers who don't need the
# classifier at all.

logger = logging.getLogger("mindcare.image_emotion.model")

_lock = threading.Lock()
_model = None
_processor = None
_device = None
_load_error: Optional[str] = None


def _detect_device():
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info("CUDA available: %s (%.1f GB VRAM)", name, vram_gb)
        return torch.device("cuda")
    logger.info("CUDA not available — using CPU.")
    return torch.device("cpu")


def get_model_and_processor():
    global _model, _processor, _device, _load_error
    if _model is not None:
        return _model, _processor, _device
    with _lock:
        if _model is not None:
            return _model, _processor, _device
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            _device = _detect_device()
            #local_path = os.path.abspath("./models/face")
            local_path = os.environ.get("MINDCARE_FACE_MODEL_PATH", "./models/face")
            if os.path.exists(local_path):
                target_path, is_offline = local_path, True
                logger.info("Loading emotion model from local offline path: %s", target_path)
            else:
                target_path, is_offline = IMAGE_EMOTION_MODEL_ID, False
                logger.info("Local model path missing; loading from Hub: %s", target_path)

            _processor = AutoImageProcessor.from_pretrained(
                target_path, cache_dir=HF_CACHE_DIR, local_files_only=is_offline
            )
            _model = AutoModelForImageClassification.from_pretrained(
                target_path, cache_dir=HF_CACHE_DIR, local_files_only=is_offline
            ).to(_device)
            _model.eval()
            logger.info("Emotion model loaded. Labels: %s", _model.config.id2label)
        except Exception as exc:
            _load_error = str(exc)
            logger.exception("Failed to load emotion model")
            raise RuntimeError(
                f"Emotion model failed to load ({exc}). To run fully offline, place the "
                f"model files under ./models/face/, or allow one online download from "
                f"'{IMAGE_EMOTION_MODEL_ID}' to populate {HF_CACHE_DIR}."
            ) from exc
    return _model, _processor, _device


def is_loaded() -> bool:
    return _model is not None


def model_status() -> dict:
    if _model is not None:
        return {"loaded": True, "model_id": IMAGE_EMOTION_MODEL_ID, "device": str(_device),
                "labels": list(_model.config.id2label.values()), "tta_enabled": USE_TTA_FLIP}
    return {"loaded": False, "error": _load_error, "model_id": IMAGE_EMOTION_MODEL_ID}


def _crops_to_pil(face_crops_bgr: List[np.ndarray]) -> List[Image.Image]:
    out = []
    for crop in face_crops_bgr:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        out.append(Image.fromarray(rgb))
    return out


def _forward_batch(pil_images: List[Image.Image]) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    model, processor, device = get_model_and_processor()
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits / EMOTION_TEMPERATURE, dim=-1).cpu().numpy()
    return probs


def classify_faces_batch(face_crops_bgr: List[np.ndarray]) -> List[dict]:
    """
    Classifies a batch of face crops in one or two forward passes
    (second pass only if TTA is enabled). Returns one result dict per
    input crop, order preserved:
        {"emotion": str, "confidence": float, "probabilities": {label: prob}}
    """
    if not face_crops_bgr:
        return []

    model, processor, device = get_model_and_processor()
    id2label = model.config.id2label

    pil_normal = _crops_to_pil(face_crops_bgr)
    probs_normal = _forward_batch(pil_normal)

    if USE_TTA_FLIP:
        pil_flipped = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in pil_normal]
        probs_flipped = _forward_batch(pil_flipped)
        probs_final = (probs_normal + probs_flipped) / 2.0
    else:
        probs_final = probs_normal

    results = []
    for row in probs_final:
        prob_dict = {id2label[i].lower(): round(float(p), 4) for i, p in enumerate(row)}
        top_idx = int(np.argmax(row))
        top_label = id2label[top_idx].lower()
        results.append({
            "emotion": top_label,
            "confidence": round(float(row[top_idx]), 4),
            "probabilities": prob_dict,
        })
    return results