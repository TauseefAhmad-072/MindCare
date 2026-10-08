"""
modules/text/predictor.py — Public interface for MODULE A: Text
Mental-Health Analysis.

This is the ONLY file the future Flask/React app (or any other caller)
should import from. It wraps the existing trained MindCarePredictor and
reshapes its output into the contract defined in the project spec:

    {
      "predicted_class": "...",
      "confidence": 0.XX,
      "probabilities": {...},
      "risk_level": "...",
      "model": "...",
      "timestamp": "...",
      "disclaimer": "...",
      ...additional clinically-useful fields preserved...
    }

Usage (integration contract):
    from modules.text.predictor import predict, predict_batch
    result = predict("some text")
"""
import datetime as _dt
import logging
from typing import Dict, List

from .config import RISK_LEVEL_MAP, CRISIS_LEVEL_THRESHOLD, DIAGNOSTIC_DISCLAIMER
from .model import get_predictor
from .preprocessing import validate_text, validate_batch, TextValidationError

logger = logging.getLogger("mindcare.text.predictor")


def _utc_timestamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _shape_result(raw: dict) -> Dict:
    """Reshape MindCarePredictor.predict() output into the Module A contract."""
    level = raw["level"]
    risk_level = RISK_LEVEL_MAP.get(level, "unknown")

    return {
        # ── Module A required fields ──
        "predicted_class": raw["level_name"],
        "confidence": raw["confidence"],
        "probabilities": raw.get("class_probs", {}),
        "risk_level": risk_level,
        "model": raw.get("model_used", "unknown"),
        "timestamp": _utc_timestamp(),

        # ── Additional clinically-useful fields (kept for frontend compatibility
        #    with ChatPage.js / AssessmentPage.js, which already expect these) ──
        "level": level,
        "level_name": raw["level_name"],
        "icon": raw.get("icon"),
        "color": raw.get("color"),
        "intervention": raw.get("intervention"),
        "highlights": raw.get("highlights", []),
        "clean_text": raw.get("clean_text", ""),
        "urgent": raw.get("urgent", False),
        "crisis_flag": level >= CRISIS_LEVEL_THRESHOLD,
        "keyword_override_applied": raw.get("keyword_override", False),
        "matched_keywords": raw.get("matched_keywords", []),
        "negation_detected": raw.get("negation_detected", False),

        # ── Safety framing (see PHASE 1 clinical-safety requirements) ──
        "disclaimer": DIAGNOSTIC_DISCLAIMER,
        "is_diagnosis": False,
    }


def predict(text: str) -> Dict:
    """
    Single-text prediction.

    Raises:
        TextValidationError: if `text` is empty/invalid.
        RuntimeError: if the underlying model failed to load.
    """
    clean_text = validate_text(text)
    predictor = get_predictor()
    raw = predictor.predict(clean_text, include_all_scores=True)
    return _shape_result(raw)


def predict_batch(texts: List[str]) -> List[Dict]:
    """
    Batch prediction. Each item is processed independently; a failure on
    one item does not abort the batch — it is reported inline.
    """
    validated = validate_batch(texts)
    predictor = get_predictor()
    results = []
    for t in validated:
        try:
            raw = predictor.predict(t, include_all_scores=True)
            results.append(_shape_result(raw))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Batch item failed")
            results.append({
                "error": str(exc),
                "predicted_class": None,
                "confidence": 0.0,
                "timestamp": _utc_timestamp(),
            })
    return results


def explain(text: str) -> str:
    """Human-readable explanation string (delegates to MindCarePredictor.explain)."""
    clean_text = validate_text(text)
    predictor = get_predictor()
    return predictor.explain(clean_text)


__all__ = ["predict", "predict_batch", "explain", "TextValidationError"]
