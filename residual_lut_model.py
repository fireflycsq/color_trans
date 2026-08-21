"""ICC-baseline plus RGB 3D residual-LUT colour model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from color_model import image_to_srgb, profile_from_bytes
from portrait_mask import (
    has_person_segmenter,
    portrait_edge_weight,
    portrait_mask,
    portrait_region_from_metadata,
)

DEFAULT_EDGE_LIFT = 0.05
DEFAULT_EDGE_LIFT_C = 0.02


def trilinear_lookup(table: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Interpolate a cubic 3D table at RGB coordinates in [0, 1]."""
    values = np.asarray(rgb, dtype=np.float32)
    size = int(table.shape[0])
    if table.shape[:3] != (size, size, size) or size < 2:
        raise ValueError("LUT must have shape [size, size, size, ...], size >= 2")
    position = np.clip(values, 0.0, 1.0) * (size - 1)
    lower = np.floor(position).astype(np.intp)
    upper = np.minimum(lower + 1, size - 1)
    fraction = position - lower

    result_shape = values.shape[:-1] + table.shape[3:]
    result = np.zeros(result_shape, dtype=np.float32)
    for dr in (0, 1):
        ir = upper[..., 0] if dr else lower[..., 0]
        wr = fraction[..., 0] if dr else 1.0 - fraction[..., 0]
        for dg in (0, 1):
            ig = upper[..., 1] if dg else lower[..., 1]
            wg = fraction[..., 1] if dg else 1.0 - fraction[..., 1]
            for db in (0, 1):
                ib = upper[..., 2] if db else lower[..., 2]
                wb = fraction[..., 2] if db else 1.0 - fraction[..., 2]
                weight = wr * wg * wb
                if table.ndim > 3:
                    weight = weight[..., None]
                result += table[ir, ig, ib].astype(np.float32) * weight
    return result


def edge_lift_amounts(
    metadata: dict | None, override: float | None = None,
) -> tuple[float, float]:
    """K and C reductions at the silhouette peak, in CMYK units of 0..1."""
    meta = metadata or {}
    if override is not None:
        k_lift = float(override)
        scale = k_lift / DEFAULT_EDGE_LIFT if DEFAULT_EDGE_LIFT else 0.0
        c_lift = DEFAULT_EDGE_LIFT_C * scale
    else:
        k_lift = float(meta.get("edge_lift", DEFAULT_EDGE_LIFT))
        c_lift = float(meta.get("edge_lift_c", DEFAULT_EDGE_LIFT_C if k_lift > 0 else 0.0))
    return max(0.0, k_lift), max(0.0, c_lift)


def _validate_lut(lut: np.ndarray, confidence: np.ndarray, name: str) -> None:
    size = int(lut.shape[0])
    if lut.shape != (size, size, size, 4):
        raise ValueError(f"{name} lut must have shape [size, size, size, 4]")
    if confidence.shape != (size, size, size):
        raise ValueError(f"{name} confidence must match the first three LUT dimensions")


@dataclass
class ResidualLUTModel:
    """CMYK ICC conversion corrected by a confidence-gated RGB residual LUT."""

    lut: np.ndarray
    confidence: np.ndarray
    target_icc: bytes
    metadata: dict
    skin_lut: np.ndarray | None = None
    skin_confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        _validate_lut(self.lut, self.confidence, "global")
        if self.skin_lut is None and self.skin_confidence is None:
            return
        if self.skin_lut is None or self.skin_confidence is None:
            raise ValueError("skin_lut and skin_confidence must be provided together")
        _validate_lut(self.skin_lut, self.skin_confidence, "skin")

    def predict_image(
        self,
        image: Image.Image,
        max_hue_shift: float | None = None,
        min_saturation: float = 0.16,
        chunk_rows: int = 128,
        edge_lift: float | None = None,
    ) -> Image.Image:
        """Apply ICC baseline and confidence-weighted residual corrections.

        ``max_hue_shift`` and ``min_saturation`` are accepted for API
        compatibility with the polynomial model. The LUT already uses explicit
        confidence fallback and residual clipping learned at training time.
        ``edge_lift`` is extra K (and a little C) removed on the person
        silhouette; ``None`` uses the model metadata, default 0.05.
        """
        del max_hue_shift, min_saturation
        source = image_to_srgb(image)
        width, height = source.size
        output = np.empty((height, width, 4), dtype=np.uint8)
        transform = ImageCms.buildTransform(
            ImageCms.createProfile("sRGB"),
            profile_from_bytes(self.target_icc),
            "RGB", "CMYK", renderingIntent=1,
            flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
        )
        strength = float(self.metadata.get("residual_strength", 1.0))
        portrait_strength = float(self.metadata.get("skin_residual_strength", 1.0))
        k_lift, c_lift = edge_lift_amounts(self.metadata, edge_lift)
        portrait = None
        rgb_u8_full = np.asarray(source, dtype=np.uint8)
        if self.skin_lut is not None:
            region = portrait_region_from_metadata(self.metadata)
            portrait = portrait_mask(rgb_u8_full, region=region)
        elif (k_lift > 0 or c_lift > 0) and has_person_segmenter():
            portrait = portrait_mask(rgb_u8_full, region="person")
        for y in range(0, height, chunk_rows):
            chunk = source.crop((0, y, width, min(y + chunk_rows, height)))
            rgb_u8 = np.asarray(chunk, dtype=np.uint8)
            rgb = rgb_u8.astype(np.float32) / 255.0
            baseline = np.asarray(ImageCms.applyTransform(chunk, transform), dtype=np.float32) / 255.0
            residual = trilinear_lookup(self.lut, rgb)
            confidence = trilinear_lookup(self.confidence, rgb)
            predicted = baseline + strength * confidence[..., None] * residual
            if portrait is not None:
                mask = portrait[y:y + rgb.shape[0]]
                if self.skin_lut is not None:
                    extra = trilinear_lookup(self.skin_lut, rgb)
                    extra_confidence = trilinear_lookup(self.skin_confidence, rgb)
                    predicted = predicted + (
                        mask * portrait_strength * extra_confidence
                    )[..., None] * extra
                if k_lift > 0 or c_lift > 0:
                    edge = portrait_edge_weight(mask)
                    if c_lift > 0:
                        predicted[..., 0] -= edge * c_lift
                    if k_lift > 0:
                        predicted[..., 3] -= edge * k_lift
            output[y:y + rgb.shape[0]] = np.rint(np.clip(predicted, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(output, "CMYK")

    def save(self, path: str | Path) -> None:
        payload = {
            "model_type": np.array("icc_residual_lut_v1"),
            "lut": self.lut.astype(np.float32),
            "confidence": self.confidence.astype(np.float32),
            "target_icc": np.frombuffer(self.target_icc, dtype=np.uint8),
            "metadata": np.array(json.dumps(self.metadata, ensure_ascii=False)),
        }
        if self.skin_lut is not None and self.skin_confidence is not None:
            payload["skin_lut"] = self.skin_lut.astype(np.float32)
            payload["skin_confidence"] = self.skin_confidence.astype(np.float32)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "ResidualLUTModel":
        with np.load(path, allow_pickle=False) as z:
            skin_lut = z["skin_lut"].astype(np.float32) if "skin_lut" in z.files else None
            skin_confidence = (
                z["skin_confidence"].astype(np.float32) if "skin_confidence" in z.files else None
            )
            return cls(
                z["lut"].astype(np.float32),
                z["confidence"].astype(np.float32),
                z["target_icc"].tobytes(),
                json.loads(str(z["metadata"])),
                skin_lut,
                skin_confidence,
            )
