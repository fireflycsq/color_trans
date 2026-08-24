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


def apply_embedded_srgb(image: Image.Image, icc: bytes | None = None) -> Image.Image:
    """Convert an RGB-like image, or a 1×N sample strip, through an embedded ICC."""
    rgb = image.convert("RGB")
    profile = icc if icc is not None else image.info.get("icc_profile")
    if not profile:
        return rgb
    try:
        return ImageCms.profileToProfile(
            rgb, profile_from_bytes(profile), ImageCms.createProfile("sRGB"), outputMode="RGB",
        )
    except Exception:
        return rgb


def samples_to_srgb(rgb_u8: np.ndarray, icc: bytes | None) -> np.ndarray:
    """Convert (N, 3) source-encoded pixels to sRGB. No-op if there is no ICC."""
    values = np.asarray(rgb_u8, dtype=np.uint8)
    if not icc or values.size == 0:
        return values
    strip = Image.fromarray(values[None, ...], "RGB")
    return np.asarray(apply_embedded_srgb(strip, icc), dtype=np.uint8)[0]


def image_to_srgb(image: Image.Image) -> Image.Image:
    """Normalise an RGB-like image through its embedded ICC when available."""
    return apply_embedded_srgb(image)


def render_cmyk_to_srgb(image: Image.Image, icc: bytes) -> Image.Image:
    """Render CMYK through its output profile into display sRGB."""
    src = profile_from_bytes(icc)
    dst = ImageCms.createProfile("sRGB")
    transform = ImageCms.buildTransform(src, dst, "CMYK", "RGB", renderingIntent=1)
    return ImageCms.applyTransform(image.convert("CMYK"), transform)


def hue_and_saturation(rgb_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised HSV hue (degrees) and saturation for an RGB uint8 array."""
    rgb = np.asarray(rgb_u8, dtype=np.float32) / 255.0
    high = rgb.max(axis=-1)
    low = rgb.min(axis=-1)
    delta = high - low
    saturation = np.divide(delta, high, out=np.zeros_like(delta), where=high > 1e-6)
    hue = np.zeros_like(high)
    valid = delta > 1e-6
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mr = valid & (high == r)
    mg = valid & (high == g) & ~mr
    mb = valid & ~(mr | mg)
    hue[mr] = 60 * np.mod((g[mr] - b[mr]) / delta[mr], 6)
    hue[mg] = 60 * ((b[mg] - r[mg]) / delta[mg] + 2)
    hue[mb] = 60 * ((r[mb] - g[mb]) / delta[mb] + 4)
    return hue, saturation


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

    def predict_image(
        self,
        image: Image.Image,
        max_hue_shift: float | None = None,
        min_saturation: float = 0.16,
        chunk_rows: int = 128,
        edge_lift: float | None = None,
        shadow_lift: float | None = None,
    ) -> Image.Image:
        """Predict CMYK, optionally protecting hues with an ICC conversion baseline.

        Pixels whose learned output rotates farther than ``max_hue_shift`` from
        the target-profile ICC baseline are progressively blended back toward
        that baseline. Full fallback is reached at twice the configured shift.
        """
        del edge_lift, shadow_lift
        source = image_to_srgb(image)
        if max_hue_shift is None or max_hue_shift <= 0:
            return Image.fromarray(self.predict_array(np.asarray(source), chunk_rows), "CMYK")

        width, height = source.size
        output = np.empty((height, width, 4), dtype=np.uint8)
        srgb = ImageCms.createProfile("sRGB")
        cmyk_profile = profile_from_bytes(self.target_icc)
        to_cmyk = ImageCms.buildTransform(
            srgb, cmyk_profile, "RGB", "CMYK", renderingIntent=1,
            flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
        )
        to_rgb = ImageCms.buildTransform(cmyk_profile, srgb, "CMYK", "RGB", renderingIntent=1)

        for y in range(0, height, chunk_rows):
            chunk_image = source.crop((0, y, width, min(y + chunk_rows, height)))
            learned = self.predict_array(np.asarray(chunk_image), chunk_rows)
            baseline_image = ImageCms.applyTransform(chunk_image, to_cmyk)
            baseline = np.asarray(baseline_image, dtype=np.uint8)
            learned_rgb = np.asarray(ImageCms.applyTransform(Image.fromarray(learned, "CMYK"), to_rgb))
            baseline_rgb = np.asarray(ImageCms.applyTransform(baseline_image, to_rgb))

            learned_hue, learned_sat = hue_and_saturation(learned_rgb)
            baseline_hue, baseline_sat = hue_and_saturation(baseline_rgb)
            hue_diff = np.abs(learned_hue - baseline_hue)
            hue_diff = np.minimum(hue_diff, 360 - hue_diff)
            reliable = np.maximum(learned_sat, baseline_sat) >= min_saturation
            fallback = np.zeros_like(hue_diff, dtype=np.float32)
            fallback[reliable] = np.clip(
                (hue_diff[reliable] - max_hue_shift) / max_hue_shift, 0, 1
            )
            mixed = learned.astype(np.float32) * (1 - fallback[..., None])
            mixed += baseline.astype(np.float32) * fallback[..., None]
            output[y:y+len(learned)] = np.rint(mixed).astype(np.uint8)
        return Image.fromarray(output, "CMYK")

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


def load_color_model(path: str | Path):
    """Load a polynomial, residual-LUT, or adaptive RGB-LUT model."""
    path = Path(path)
    if path.suffix.lower() in {".pt", ".pth"}:
        from adaptive_lut_model import AdaptiveLUTModel
        return AdaptiveLUTModel.load(path)
    with np.load(path, allow_pickle=False) as z:
        is_lut = "lut" in z.files
    if is_lut:
        from residual_lut_model import ResidualLUTModel
        return ResidualLUTModel.load(path)
    return ColorModel.load(path)
