"""Portrait mask for the second colour stage.

Default region is the full person silhouette (skin, hair, clothes).
`--region skin` keeps the older face/limb-only mask.
"""

from __future__ import annotations

import threading

import numpy as np
from PIL import Image, ImageFilter

_PERSON_LOCK = threading.Lock()
_PERSON_SEGMENTER = False  # False = not tried; None = unavailable; else model
_INSTALL_HINT = (
    "抠出人像需要 mediapipe 人像分割（含头发和服装）。请安装: python3 -m pip install mediapipe"
)


def has_person_segmenter() -> bool:
    return _ensure_person_segmenter() is not None


def require_person_segmenter() -> None:
    if not has_person_segmenter():
        raise RuntimeError(_INSTALL_HINT)


def detector_name(region: str = "person") -> str:
    if region == "skin" and not has_person_segmenter():
        return "ycbcr_skin"
    if has_person_segmenter():
        return "mediapipe_selfie" if region == "person" else "mediapipe_selfie+ycbcr_skin"
    return "unavailable"


def _ensure_person_segmenter():
    global _PERSON_SEGMENTER
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
    """Person silhouette probability, including hair and clothing."""
    require_person_segmenter()
    segmenter = _ensure_person_segmenter()
    height, width = rgb_u8.shape[:2]
    image = Image.fromarray(np.asarray(rgb_u8, dtype=np.uint8), "RGB")
    longest = max(width, height)
    if longest > 768:
        scale = 768 / longest
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


def _blur_mask(mask: np.ndarray, blur_radius: float) -> np.ndarray:
    if blur_radius <= 0:
        return mask
    image = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), "L")
    image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.asarray(image, dtype=np.float32) / 255.0


def portrait_mask(
    rgb_u8: np.ndarray, blur_radius: float = 4.0, region: str = "person",
) -> np.ndarray:
    """Soft portrait mask. ``region`` is ``person`` (default) or ``skin``."""
    if region not in {"person", "skin"}:
        raise ValueError("region 必须是 person 或 skin")
    if region == "person":
        mask = person_probability(rgb_u8)
    else:
        if has_person_segmenter():
            mask = person_probability(rgb_u8) * skin_probability(rgb_u8)
        else:
            mask = skin_probability(rgb_u8)
    return _blur_mask(mask, blur_radius)


def portrait_region_from_metadata(metadata: dict | None) -> str:
    meta = metadata or {}
    region = str(meta.get("portrait_region", "")).lower()
    if region in {"person", "skin"}:
        return region
    if meta.get("portrait_skin"):
        return "skin"
    return "person"


def portrait_skin_mask(rgb_u8: np.ndarray, blur_radius: float = 4.0) -> np.ndarray:
    """Backward-compatible alias for the skin-only portrait mask."""
    return portrait_mask(rgb_u8, blur_radius=blur_radius, region="skin")
