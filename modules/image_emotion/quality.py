"""
modules/image_emotion/quality.py — Frame/crop quality gating.

Emotion classifiers are unreliable on blurry, over/under-exposed, or
tiny face crops — and silently returning a confident-looking prediction
on garbage input is exactly the kind of "unfakable" overconfidence this
project needs to avoid. This module scores crop quality with real,
cheap OpenCV metrics and lets the caller decide whether to trust,
downweight, or skip a given frame's prediction.
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import MIN_BLUR_VARIANCE, MIN_BRIGHTNESS, MAX_BRIGHTNESS, MIN_FACE_SIZE_PX


@dataclass
class QualityReport:
    blur_variance: float
    brightness_mean: float
    width: int
    height: int
    is_acceptable: bool
    reasons: list


def assess_crop_quality(face_crop_bgr: np.ndarray) -> QualityReport:
    reasons = []
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return QualityReport(0.0, 0.0, 0, 0, False, ["empty_crop"])

    h, w = face_crop_bgr.shape[:2]
    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)

    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))

    if w < MIN_FACE_SIZE_PX or h < MIN_FACE_SIZE_PX:
        reasons.append("face_too_small")
    if blur_var < MIN_BLUR_VARIANCE:
        reasons.append("too_blurry")
    if brightness < MIN_BRIGHTNESS:
        reasons.append("too_dark")
    if brightness > MAX_BRIGHTNESS:
        reasons.append("overexposed")

    return QualityReport(
        blur_variance=round(blur_var, 1),
        brightness_mean=round(brightness, 1),
        width=w, height=h,
        is_acceptable=(len(reasons) == 0),
        reasons=reasons,
    )