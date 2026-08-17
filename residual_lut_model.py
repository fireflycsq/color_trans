"""ICC-baseline plus RGB 3D residual-LUT colour model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from color_model import image_to_srgb, profile_from_bytes
from portrait_mask import portrait_skin_mask


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
    ) -> Image.Image:
        """Apply ICC baseline and confidence-weighted residual corrections.

        ``max_hue_shift`` and ``min_saturation`` are accepted for API
        compatibility with the polynomial model. The LUT already uses explicit
        confidence fallback and residual clipping learned at training time.
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
        skin_strength = float(self.metadata.get("skin_residual_strength", 1.0))
        skin_mask = None
        if self.skin_lut is not None:
            skin_mask = portrait_skin_mask(np.asarray(source, dtype=np.uint8))
        for y in range(0, height, chunk_rows):
            chunk = source.crop((0, y, width, min(y + chunk_rows, height)))
            rgb_u8 = np.asarray(chunk, dtype=np.uint8)
            rgb = rgb_u8.astype(np.float32) / 255.0
            baseline = np.asarray(ImageCms.applyTransform(chunk, transform), dtype=np.float32) / 255.0
            residual = trilinear_lookup(self.lut, rgb)
            confidence = trilinear_lookup(self.confidence, rgb)
            predicted = baseline + strength * confidence[..., None] * residual
            if skin_mask is not None:
                mask = skin_mask[y:y + rgb.shape[0]]
                skin_residual = trilinear_lookup(self.skin_lut, rgb)
                skin_confidence = trilinear_lookup(self.skin_confidence, rgb)
                predicted = predicted + (
                    mask * skin_strength * skin_confidence
                )[..., None] * skin_residual
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
