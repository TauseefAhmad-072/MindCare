"""
MODULE A — Text Mental-Health Analysis.

Public integration contract:

    from modules.text import predict, predict_batch

    result = predict("I feel really overwhelmed lately")
    results = predict_batch(["text one", "text two"])

This module wraps the EXISTING, ALREADY-TRAINED MindCarePredictor
(SBERT + LR / BiLSTM+Attention / SBERT+BiLSTM ensemble). No retraining is
performed here — see modules/text/_mindcare_core.py for the reused model
code and mindcare_train.ipynb (project root docs) for how it was trained.

NOTE ON LAZY LOADING: symbols below are resolved via module __getattr__
(PEP 562) instead of eager `from .predictor import ...` at the top of this
file. This is deliberate: `modules.text.preprocessing` and
`modules.text.config` have zero heavy dependencies (no torch, no
sentence-transformers) and should stay importable/testable in isolation
(e.g. for pure input-validation unit tests) without forcing torch to load.
Only touching `predict`, `predict_batch`, `explain`, etc. triggers the
heavy import chain, and only at first use.
"""
__all__ = [
    "predict",
    "predict_batch",
    "explain",
    "TextValidationError",
    "get_predictor",
    "is_loaded",
    "model_status",
]

_LAZY_ATTRS = {
    "predict": (".predictor", "predict"),
    "predict_batch": (".predictor", "predict_batch"),
    "explain": (".predictor", "explain"),
    "TextValidationError": (".predictor", "TextValidationError"),
    "get_predictor": (".model", "get_predictor"),
    "is_loaded": (".model", "is_loaded"),
    "model_status": (".model", "model_status"),
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        import importlib
        module_path, attr_name = _LAZY_ATTRS[name]
        mod = importlib.import_module(module_path, __name__)
        value = getattr(mod, attr_name)
        globals()[name] = value  # cache for subsequent access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
