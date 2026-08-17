"""Portrait-skin mask: person region × skin chromaticity.

Used so the second residual LUT only edits faces/limbs, not background
colours that happen to sit near skin in RGB.
"""

from __future__ import annotations

import threading

import numpy as np
from PIL import Image, ImageFilter

_PERSON_LOCK = threading.Lock()
_PERSON_SEGMENTER = False  # False = not tried; None = unavailable; else model
_WARNED_NO_PERSON = False


def detector_name() -> str:
    if _ensure_person_segmenter() is None:
        return "ycbcr_skin"
    return "mediapipe_selfie+ycbcr_skin"


def _ensure_person_segmenter():
    global _PERSON_SEGMENTER, _WARNED_NO_PERSON
    if _PERSON_SEGMENTER is not False:
        return _PERSON_SEGMENTER
    with _PERSON_LOCK:
        if _PERSON_SEGMENTER is not False:
            return _PERSON_SEGMENTER
        try:
            import mediapipe as mp
            solutions = getattr(mp, "solutions", None)
            if solutions is None:
                raise ImportError("mediapipe.solutions is unavailable")
            _PERSON_SEGMENTER = solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=1
            )
        except Exception:
            _PERSON_SEGMENTER = None
            if not _WARNED_NO_PERSON:
                print(
                    "portrait mask: mediapipe not available, using skin chromaticity only "
                    "(beige backgrounds may receive skin corrections). "
                    "Install mediapipe to restrict edits to people."
                )
                _WARNED_NO_PERSON = True
    return _PERSON_SEGMENTER


def skin_probability(rgb_u8: np.ndarray) -> np.ndarray:
    """Soft skin likelihood in sRGB, values in [0, 1]."""
    rgb = np.asarray(rgb_u8, dtype=np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    cb_w = np.clip(1.0 - np.abs(cb - 102.0) / 38.0, 0.0, 1.0)
    cr_w = np.clip(1.0 - np.abs(cr - 153.0) / 32.0, 0.0, 1.0)
    y_w = np.clip((y - 45.0) / 35.0, 0.0, 1.0)
    rg = np.clip((r - g) / 18.0, 0.0, 1.0)
    rb = np.clip((r - b) / 12.0, 0.0, 1.0)
    return (cb_w * cr_w * y_w * np.maximum(rg * rb, 0.15)).astype(np.float32)


def person_probability(rgb_u8: np.ndarray) -> np.ndarray:
    """Person/selfie probability. Falls back to 1 (everywhere) without mediapipe."""
    segmenter = _ensure_person_segmenter()
    height, width = rgb_u8.shape[:2]
    if segmenter is None:
        return np.ones((height, width), dtype=np.float32)

    image = Image.fromarray(np.asarray(rgb_u8, dtype=np.uint8), "RGB")
    longest = max(width, height)
    if longest > 512:
        scale = 512 / longest
        small = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BILINEAR,
        )
    else:
        small = image
    with _PERSON_LOCK:
        result = segmenter.process(np.asarray(small, dtype=np.uint8))
    mask = np.clip(result.segmentation_mask.astype(np.float32), 0.0, 1.0)
    if mask.shape != (height, width):
        mask_u8 = Image.fromarray(np.rint(mask * 255.0).astype(np.uint8), "L")
        mask = np.asarray(
            mask_u8.resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
    return mask


def portrait_skin_mask(rgb_u8: np.ndarray, blur_radius: float = 4.0) -> np.ndarray:
    """Soft mask for skin that lies on a person. Shape [H, W], values in [0, 1]."""
    mask = person_probability(rgb_u8) * skin_probability(rgb_u8)
    if blur_radius > 0:
        image = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), "L")
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        mask = np.asarray(image, dtype=np.float32) / 255.0
    return mask
