"""
modules/text/config.py — Configuration for the Text Mental-Health Analysis module.

All paths are overridable via environment variables so this module can be
dropped into the future Flask app without code changes.
"""
import os

# Directory containing the ALREADY-TRAINED artifacts produced by
# mindcare_train.ipynb (metadata.json, vocab.json, sbert_lr_model.pkl,
# bilstm_attention_best.pt, sbert_bilstm_ensemble_best.pt).
# These are reused as-is — no retraining is performed by this module.
TEXT_MODELS_DIR = os.environ.get("MINDCARE_TEXT_MODELS_DIR", "./models")

# Risk-level bucketing used ONLY for the coarse "risk_level" field in the
# public API contract (Module A spec). The underlying 8-level clinical
# taxonomy from MindCarePredictor is preserved unchanged in "predicted_class".
RISK_LEVEL_MAP = {
    0: "minimal",
    1: "low",
    2: "low",
    3: "moderate",
    4: "moderate",
    5: "elevated",
    6: "high",
    7: "high",
}

# Levels >= this value are treated as requiring an explicit crisis flag
# in the API response (does NOT gate the ML prediction itself).
CRISIS_LEVEL_THRESHOLD = int(os.environ.get("MINDCARE_CRISIS_THRESHOLD", "5"))

# Safety/legal disclaimer attached to every response. Do not remove —
# this system is not a diagnostic tool (see PHASE 1 clinical-safety notes).
DIAGNOSTIC_DISCLAIMER = (
    "This output is an AI-generated indicator based on text patterns. "
    "It is NOT a medical diagnosis. For any concerning result, especially "
    "'elevated' or 'high' risk levels, consult a qualified mental-health "
    "professional."
)

MAX_BATCH_SIZE = int(os.environ.get("MINDCARE_TEXT_MAX_BATCH", "64"))
MAX_TEXT_LENGTH_CHARS = int(os.environ.get("MINDCARE_TEXT_MAX_CHARS", "8000"))
