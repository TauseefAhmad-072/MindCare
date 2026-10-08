"""
modules/image_emotion/enhance.py — Frame and crop enhancement.

Applied BEFORE the quality gate so borderline frames can pass.
Subtle by design — heavy processing pushes frames out of the
classifier's training distribution and drops accuracy.
"""
import cv2
import numpy as np


def enhance_frame(bgr: np.ndarray, target_brightness: float = 110.0) -> np.ndarray:
    """Full-frame pass: denoise, contrast-normalize, brightness lift, mild sharpen."""
    if bgr is None or bgr.size == 0:
        return bgr
    out = bgr.copy()

    # Bilateral denoise preserves edges (unlike Gaussian)
    out = cv2.bilateralFilter(out, d=5, sigmaColor=35, sigmaSpace=35)

    # CLAHE on L channel of LAB — contrast without color shift
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Global brightness nudge only if truly dark
    mean_b = float(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean())
    if mean_b < target_brightness * 0.7:
        gain = min(1.8, target_brightness / max(mean_b, 1.0))
        out = cv2.convertScaleAbs(out, alpha=gain, beta=0)

    # Mild unsharp mask
    blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=1.2)
    out = cv2.addWeighted(out, 1.3, blurred, -0.3, 0)

    return out


def enhance_face_crop(bgr: np.ndarray, target_size: int = 160) -> np.ndarray:
    """Crop pass: upscale if small, CLAHE, very mild sharpen."""
    if bgr is None or bgr.size == 0:
        return bgr
    out = bgr.copy()

    # Upscale small crops to the ViT's effective input size
    h, w = out.shape[:2]
    if min(h, w) < target_size:
        scale = target_size / min(h, w)
        out = cv2.resize(
            out,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    # CLAHE (lighter than frame pass)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Very light unsharp
    blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0)
    out = cv2.addWeighted(out, 1.2, blurred, -0.2, 0)

    return out