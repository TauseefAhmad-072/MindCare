"""
modules/image_emotion/detector.py — Multi-person face detection.

REPLACES the previous implementation, which tried to download a Caffe
model from `_DNN_MODEL_URL = "https://githubusercontent.com"` — that is
a bare homepage, not a model file, so the download always failed and the
code silently fell back to a single-face Haar cascade. That meant every
frame with more than one person only ever "saw" one face, and relied on
a network call for a supposedly offline clinical tool.

This version uses MediaPipe Face Detection via mp_compat.py, which
handles BOTH of MediaPipe's incompatible API generations automatically
(confirmed in testing: the mediapipe 0.10.33 pip build exposes only the
newer Tasks API, not the classic `mp.solutions` API most examples use —
see mp_compat.py's module docstring). Either way you get real,
multi-face detection with confidence scores, needed for room-level (not
just single-subject) monitoring. If NEITHER MediaPipe API is usable on a
given machine, this falls back to OpenCV's built-in Haar cascade (also
bundled, also multi-face) so the module still works with a
reduced-accuracy backend rather than crashing.
"""
import logging
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from . import mp_compat
from .config import (
    FACE_DETECTOR_CONFIDENCE_THRESHOLD,
    FACE_DETECTOR_MODEL_SELECTION,
    MAX_PERSONS_PER_FRAME,
    MIN_FACE_SIZE_PX,
)

logger = logging.getLogger("mindcare.image_emotion.detector")


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float


class FaceDetector:
    def __init__(self):
        self._backend = None
        self._mp_backend = None
        self._haar = None

    def _try_load_mediapipe(self) -> bool:
        try:
            self._mp_backend = mp_compat.get_face_backend(
                FACE_DETECTOR_MODEL_SELECTION, FACE_DETECTOR_CONFIDENCE_THRESHOLD
            )
            self._backend = "mediapipe"
            logger.info("Face detector: MediaPipe backend loaded (multi-face).")
            return True
        except Exception as exc:
            logger.warning("MediaPipe face detector unavailable (%s); falling back to Haar.", exc)
            return False

    def _load_haar(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(cascade_path)
        self._backend = "haar"
        logger.info("Face detector: Haar-cascade backend loaded (multi-face fallback).")

    def _ensure_loaded(self):
        if self._backend is not None:
            return
        if not self._try_load_mediapipe():
            self._load_haar()

    def detect(self, image_bgr: np.ndarray) -> List[FaceBox]:
        self._ensure_loaded()
        h, w = image_bgr.shape[:2]

        if self._backend == "mediapipe":
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            rel_boxes = self._mp_backend.detect(rgb)
            faces = []
            for rb in rel_boxes:
                x1 = max(0, int(rb.xmin * w))
                y1 = max(0, int(rb.ymin * h))
                fw = max(0, min(int(rb.width * w), w - x1))
                fh = max(0, min(int(rb.height * h), h - y1))
                if fw >= MIN_FACE_SIZE_PX and fh >= MIN_FACE_SIZE_PX:
                    faces.append(FaceBox(x1, y1, fw, fh, rb.score))
            faces = sorted(faces, key=lambda f: f.confidence, reverse=True)[:MAX_PERSONS_PER_FRAME]
            return faces

        # Haar fallback
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # improves detection under poor/uneven lighting
        rects = self._haar.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE_PX, MIN_FACE_SIZE_PX)
        )
        faces = [FaceBox(int(x), int(y), int(fw), int(fh), 1.0) for (x, y, fw, fh) in rects]
        return faces[:MAX_PERSONS_PER_FRAME]

    def backend_name(self) -> str:
        self._ensure_loaded()
        return self._backend