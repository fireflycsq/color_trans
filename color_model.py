"""Polynomial RGB-to-CMYK colour model with embedded target ICC profile."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms


def polynomial_features(rgb: np.ndarray) -> np.ndarray:
    """Return all RGB monomials up to degree 3 (20 features)."""
    x = np.asarray(rgb, dtype=np.float32)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    return np.stack(
        [
            np.ones_like(r), r, g, b,
            r*r, g*g, b*b, r*g, r*b, g*b,
            r*r*r, g*g*g, b*b*b,
            r*r*g, r*r*b, g*g*r, g*g*b, b*b*r, b*b*g, r*g*b,
        ],
        axis=-1,
    )


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an sRGB float array in [0, 1] to CIE Lab (D65)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    d = 6 / 29
    f = np.where(xyz > d**3, np.cbrt(xyz), xyz / (3*d*d) + 4/29)
    return np.stack([116*f[..., 1]-16, 500*(f[..., 0]-f[..., 1]), 200*(f[..., 1]-f[..., 2])], axis=-1)


def profile_from_bytes(data: bytes) -> ImageCms.ImageCmsProfile:
    return ImageCms.ImageCmsProfile(io.BytesIO(data))


def render_cmyk_to_srgb(image: Image.Image, icc: bytes) -> Image.Image:
    """Render CMYK through its output profile into display sRGB."""
    src = profile_from_bytes(icc)
    dst = ImageCms.createProfile("sRGB")
    transform = ImageCms.buildTransform(src, dst, "CMYK", "RGB", renderingIntent=1)
    return ImageCms.applyTransform(image.convert("CMYK"), transform)


@dataclass
class ColorModel:
    weights: np.ndarray
    target_icc: bytes
    metadata: dict

    def predict_array(self, rgb_u8: np.ndarray, chunk_rows: int = 128) -> np.ndarray:
        h, w, _ = rgb_u8.shape
        out = np.empty((h, w, 4), dtype=np.uint8)
        for y in range(0, h, chunk_rows):
            rgb = rgb_u8[y:y+chunk_rows].astype(np.float32) / 255.0
            cmyk = polynomial_features(rgb) @ self.weights
            out[y:y+chunk_rows] = np.rint(np.clip(cmyk, 0, 1) * 255).astype(np.uint8)
        return out

    def predict_image(self, image: Image.Image) -> Image.Image:
        arr = np.asarray(image.convert("RGB"))
        return Image.fromarray(self.predict_array(arr), "CMYK")

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            weights=self.weights.astype(np.float32),
            target_icc=np.frombuffer(self.target_icc, dtype=np.uint8),
            metadata=np.array(json.dumps(self.metadata, ensure_ascii=False)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ColorModel":
        with np.load(path, allow_pickle=False) as z:
            return cls(
                z["weights"].astype(np.float32),
                z["target_icc"].tobytes(),
                json.loads(str(z["metadata"])),
            )
