"""
MODULE B — Multi-Person Image/Video Emotion & Kinematic Feature module.

Public integration contract:

    from modules.image_emotion import predict, predict_bytes, predict_frame, process_video

    result = predict("path/to/face.jpg")          # single image, from disk
    result = predict_bytes(uploaded_file_bytes)     # single image, from bytes
    result = predict_frame(frame_bgr, tracker=t)    # single decoded frame (e.g. from cv2/smart_room)
    result = process_video("path/to/clip.mp4")      # full video, sampled + tracked

NOTE ON LAZY LOADING: `modules.image_emotion.config` stays importable
without pulling in torch/transformers/mediapipe; only touching `predict`
etc. triggers the heavy import chain.
"""
__all__ = [
    "predict",
    "predict_bytes",
    "predict_frame",
    "process_video",
    "ImageValidationError",
    "get_model_and_processor",
    "is_loaded",
    "model_status",
]

_LAZY_ATTRS = {
    "predict": (".predictor", "predict"),
    "predict_bytes": (".predictor", "predict_bytes"),
    "predict_frame": (".predictor", "predict_frame"),
    "process_video": (".predictor", "process_video"),
    "ImageValidationError": (".predictor", "ImageValidationError"),
    "get_model_and_processor": (".model", "get_model_and_processor"),
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