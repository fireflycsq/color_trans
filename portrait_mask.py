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
_FACE_DETECTOR = False
_SELFIE_MAX = 768
_MIN_PERSON_PIXELS = 32
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


def _ensure_face_detector():
    global _FACE_DETECTOR
    if _FACE_DETECTOR is not False:
        return _FACE_DETECTOR
    with _PERSON_LOCK:
        if _FACE_DETECTOR is not False:
            return _FACE_DETECTOR
        try:
            import mediapipe as mp
            solutions = getattr(mp, "solutions", None)
            if solutions is None or not hasattr(solutions, "face_detection"):
                raise ImportError("mediapipe face detection is unavailable")
            _FACE_DETECTOR = solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.4,
            )
        except Exception:
            _FACE_DETECTOR = None
    return _FACE_DETECTOR


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


def _enough_person(mask: np.ndarray) -> bool:
    return int(np.count_nonzero(mask >= 0.45)) >= _MIN_PERSON_PIXELS


def _run_selfie(rgb_u8: np.ndarray) -> np.ndarray:
    """Selfie segmentation on one RGB crop, returned at the crop's resolution."""
    segmenter = _ensure_person_segmenter()
    height, width = rgb_u8.shape[:2]
    image = Image.fromarray(np.asarray(rgb_u8, dtype=np.uint8), "RGB")
    longest = max(width, height)
    if longest > _SELFIE_MAX:
        scale = _SELFIE_MAX / longest
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


def _paste_crop(full: np.ndarray, crop_mask: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    region = full[y0:y1, x0:x1]
    if region.shape[:2] != crop_mask.shape[:2]:
        crop_mask = np.asarray(
            Image.fromarray(np.clip(crop_mask * 255.0, 0, 255).astype(np.uint8), "L").resize(
                (region.shape[1], region.shape[0]), Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        ) / 255.0
    np.maximum(region, crop_mask, out=region)


def _grid_boxes(height: int, width: int, grid_y: int, grid_x: int, overlap: float = 0.2):
    tile_h = max(1, height // grid_y)
    tile_w = max(1, width // grid_x)
    pad_y = int(tile_h * overlap)
    pad_x = int(tile_w * overlap)
    boxes = []
    for i in range(grid_y):
        for j in range(grid_x):
            y0 = max(0, i * tile_h - pad_y)
            x0 = max(0, j * tile_w - pad_x)
            y1 = min(height, (i + 1) * tile_h + pad_y)
            x1 = min(width, (j + 1) * tile_w + pad_x)
            if y1 - y0 >= 32 and x1 - x0 >= 32:
                boxes.append((x0, y0, x1, y1))
    return boxes


def _tiled_selfie(rgb_u8: np.ndarray, grid_y: int, grid_x: int) -> np.ndarray:
    height, width = rgb_u8.shape[:2]
    merged = np.zeros((height, width), dtype=np.float32)
    for x0, y0, x1, y1 in _grid_boxes(height, width, grid_y, grid_x):
        _paste_crop(merged, _run_selfie(rgb_u8[y0:y1, x0:x1]), (x0, y0, x1, y1))
    return merged


def _face_crop_boxes(rgb_u8: np.ndarray) -> list[tuple[int, int, int, int]]:
    detector = _ensure_face_detector()
    if detector is None:
        return []
    height, width = rgb_u8.shape[:2]
    image = Image.fromarray(np.asarray(rgb_u8, dtype=np.uint8), "RGB")
    longest = max(width, height)
    scale = min(1.0, 1280 / longest)
    small = image if scale >= 1 else image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    with _PERSON_LOCK:
        result = detector.process(np.asarray(small, dtype=np.uint8))
    if not result.detections:
        return []
    small_w, small_h = small.size
    boxes = []
    for det in result.detections:
        box = det.location_data.relative_bounding_box
        fx0 = box.xmin * small_w / scale
        fy0 = box.ymin * small_h / scale
        fw = box.width * small_w / scale
        fh = box.height * small_h / scale
        body_h = max(fh * 7.5, 96)
        body_w = max(fw * 3.5, body_h * 0.4, 96)
        cx = fx0 + fw * 0.5
        cy = fy0 + fh * 0.5
        x0 = int(max(0, cx - body_w * 0.5))
        y0 = int(max(0, fy0 - fh * 0.8))
        x1 = int(min(width, cx + body_w * 0.5))
        y1 = int(min(height, y0 + body_h))
        if x1 - x0 >= 32 and y1 - y0 >= 32:
            boxes.append((x0, y0, x1, y1))
    return boxes


def _face_crop_selfie(rgb_u8: np.ndarray) -> np.ndarray:
    height, width = rgb_u8.shape[:2]
    merged = np.zeros((height, width), dtype=np.float32)
    for x0, y0, x1, y1 in _face_crop_boxes(rgb_u8):
        _paste_crop(merged, _run_selfie(rgb_u8[y0:y1, x0:x1]), (x0, y0, x1, y1))
    return merged


def person_probability(rgb_u8: np.ndarray) -> np.ndarray:
    """Person silhouette, with extra passes when the subject is too small in-frame."""
    require_person_segmenter()
    mask = _run_selfie(rgb_u8)
    if _enough_person(mask):
        return mask
    height, width = rgb_u8.shape[:2]
    if max(height, width) > _SELFIE_MAX:
        mask = np.maximum(mask, _tiled_selfie(rgb_u8, 2, 2))
        if _enough_person(mask):
            return mask
    mask = np.maximum(mask, _face_crop_selfie(rgb_u8))
    return mask


def portrait_edge_weight(mask: np.ndarray) -> np.ndarray:
    """Soft silhouette ring in [0, 1], peaking where the mask is 0.5."""
    mask = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    return (4.0 * mask * (1.0 - mask)).astype(np.float32)


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
