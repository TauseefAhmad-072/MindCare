"""
modules/text/model.py — Lazy singleton loader for the existing, ALREADY
TRAINED MindCarePredictor (from mindcare_inference.py).

Why a singleton: loading SBERT + two PyTorch checkpoints + an LR model is
expensive (~seconds). We load once per process and reuse across requests
(single text, batch, and later the Flask layer all share this instance).
"""
import logging
import threading
from typing import Optional

from ._mindcare_core import MindCarePredictor
from .config import TEXT_MODELS_DIR

logger = logging.getLogger("mindcare.text.model")

_lock = threading.Lock()
_predictor: Optional[MindCarePredictor] = None
_load_error: Optional[str] = None


def get_predictor() -> MindCarePredictor:
    """
    Returns the shared MindCarePredictor instance, loading it on first call.
    Thread-safe. Raises RuntimeError with a clean message if loading fails
    (e.g. missing model files) rather than letting a raw exception surface.
    """
    global _predictor, _load_error

    if _predictor is not None:
        return _predictor

    with _lock:
        if _predictor is not None:
            return _predictor
        try:
            logger.info("Loading MindCarePredictor from %s ...", TEXT_MODELS_DIR)
            _predictor = MindCarePredictor(models_dir=TEXT_MODELS_DIR)
            logger.info("MindCarePredictor loaded successfully.")
        except Exception as exc:  # noqa: BLE001 - we intentionally wrap all load errors
            _load_error = str(exc)
            logger.exception("Failed to load MindCarePredictor")
            raise RuntimeError(
                f"Text model failed to load from '{TEXT_MODELS_DIR}': {exc}. "
                "Ensure metadata.json, vocab.json, and the .pt/.pkl checkpoints "
                "produced by mindcare_train.ipynb are present in that directory."
            ) from exc

    return _predictor


def is_loaded() -> bool:
    return _predictor is not None


def load_error() -> Optional[str]:
    return _load_error


def model_status() -> dict:
    """Lightweight status used by GET /api/health without forcing a load."""
    if _predictor is not None:
        return {
            "loaded": True,
            "device": str(_predictor.device),
            "models_available": list(_predictor.models.keys()),
        }
    return {"loaded": False, "error": _load_error}
