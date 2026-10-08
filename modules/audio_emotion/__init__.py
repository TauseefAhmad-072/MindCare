"""
MODULE C — Audio/Speech Emotion Recognition.

Public integration contract:
    from modules.audio_emotion import predict, predict_bytes
    from modules.audio_emotion import predict_waveform, predict_windows, get_window_at_timestamp
"""
__all__ = [
    "predict",
    "predict_bytes",
    "predict_waveform",
    "predict_windows",
    "predict_windows_from_waveform",
    "get_window_at_timestamp",
    "AudioValidationError",
    "get_model",
    "is_loaded",
    "model_status",
]

_LAZY_ATTRS = {
    "predict": (".predictor", "predict"),
    "predict_bytes": (".predictor", "predict_bytes"),
    "predict_waveform": (".predictor", "predict_waveform"),
    "predict_windows": (".predictor", "predict_windows"),
    "predict_windows_from_waveform": (".predictor", "predict_windows_from_waveform"),
    "get_window_at_timestamp": (".predictor", "get_window_at_timestamp"),
    "AudioValidationError": (".preprocessing", "AudioValidationError"),
    "get_model": (".model", "get_model"),
    "is_loaded": (".model", "is_loaded"),
    "model_status": (".model", "model_status"),
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        import importlib
        module_path, attr_name = _LAZY_ATTRS[name]
        mod = importlib.import_module(module_path, __name__)
        value = getattr(mod, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")