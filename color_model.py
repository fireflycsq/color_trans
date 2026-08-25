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


_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float64)
_XYZ_TO_SRGB = np.linalg.inv(_SRGB_TO_XYZ)
_XYZ_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an sRGB float array in [0, 1] to CIE Lab (D65)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _SRGB_TO_XYZ.T
    xyz /= _XYZ_WHITE
    d = 6 / 29
    f = np.where(xyz > d**3, np.cbrt(xyz), xyz / (3 * d * d) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIE Lab (D65) to sRGB float in [0, 1]."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = lab[..., 1] / 500.0 + fy
    fz = fy - lab[..., 2] / 200.0
    d = 6 / 29
    xyz = np.stack(
        [
            np.where(fx > d, fx ** 3, (fx - 4 / 29) * 3 * d * d),
            np.where(fy > d, fy ** 3, (fy - 4 / 29) * 3 * d * d),
            np.where(fz > d, fz ** 3, (fz - 4 / 29) * 3 * d * d),
        ],
        axis=-1,
    )
    xyz *= _XYZ_WHITE
    lin = xyz @ _XYZ_TO_SRGB.T
    rgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(np.clip(lin, 0, None), 1 / 2.4) - 0.055)
    return np.clip(rgb, 0.0, 1.0)


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


def render_cmyk_to_srgb(image: Image.Image, icc: bytes, chunk_rows: int = 256) -> Image.Image:
    """Render CMYK through its output profile into display sRGB."""
    src = profile_from_bytes(icc)
    dst = ImageCms.createProfile("sRGB")
    transform = ImageCms.buildTransform(src, dst, "CMYK", "RGB", renderingIntent=1)
    cmyk = image.convert("CMYK")
    return _apply_transform_rows(cmyk, transform, chunk_rows)


def srgb_to_cmyk(image: Image.Image, icc: bytes, chunk_rows: int = 256) -> Image.Image:
    """Convert display sRGB into the model's target CMYK profile."""
    src = ImageCms.createProfile("sRGB")
    dst = profile_from_bytes(icc)
    transform = ImageCms.buildTransform(
        src, dst, "RGB", "CMYK", renderingIntent=1,
        flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
    )
    return _apply_transform_rows(image.convert("RGB"), transform, chunk_rows)


def _apply_transform_rows(image: Image.Image, transform, chunk_rows: int) -> Image.Image:
    width, height = image.size
    if chunk_rows <= 0 or height <= chunk_rows:
        return ImageCms.applyTransform(image, transform)
    parts = []
    mode = None
    for y in range(0, height, chunk_rows):
        strip = image.crop((0, y, width, min(y + chunk_rows, height)))
        converted = ImageCms.applyTransform(strip, transform)
        mode = converted.mode
        parts.append(np.asarray(converted, dtype=np.uint8))
    return Image.fromarray(np.concatenate(parts, axis=0), mode)


def _soft_highlights(img_f, start: float = 0.78, ceiling: float = 0.94):
    """Roll values above ``start`` (including >1) into [start, ceiling]."""
    img_f = np.maximum(img_f, 0.0)
    above = np.maximum(img_f - start, 0.0)
    room = max(ceiling - start, 1e-6)
    rolled = start + room * (above / (above + 0.50))
    return np.where(img_f > start, np.minimum(rolled, ceiling), img_f)


def avoid_washout_adjust(
    image: Image.Image,
    shadow_lift: float = 0.18,
    strength: float = 0.6,
    black_pct: float = 1.0,
    white_pct: float = 99.0,
    black_max: float = 0.18,
    highlight_ceiling: float = 0.94,
    cool: float = 0.55,
) -> Image.Image:
    """Crush gray fog and lift mids, without blowing stage highlights.

    99th-percentile stretch used to hard-clip the brightest 1% to 1.0 (hands,
    white fabric). Those values are kept above 1 and soft-rolled at the end.
    ``cool`` pulls midtone Lab yellow/orange toward pale cyan-gray so a golden
    model grade can match a cooler human target.
    """
    rgb = image.convert("RGB")
    img_f = np.asarray(rgb, dtype=np.float32) / 255.0
    if img_f.size == 0:
        return rgb

    luma = np.clip(
        img_f[..., 0] * 0.299 + img_f[..., 1] * 0.587 + img_f[..., 2] * 0.114,
        0.0, 1.0,
    )
    black_point = min(max(float(np.percentile(luma, black_pct)), 0.0), black_max)
    white_point = float(np.percentile(luma, white_pct))
    white_point = min(max(white_point, black_point + 0.20), 1.0)
    img_f = (img_f - black_point) / (white_point - black_point + 1e-6)
    img_f = np.maximum(img_f, 0.0)
    over = np.maximum(img_f - 1.0, 0.0)
    img_f = np.minimum(img_f, 1.0)

    if shadow_lift > 0:
        gamma = 1.0 - np.clip(shadow_lift, 0.0, 1.0) * 0.35
        img_f = np.power(img_f, gamma)

    s_strength = np.clip(strength, 0.0, 1.0) * 0.40
    if s_strength > 0:
        s_curved = 1.0 / (1.0 + np.exp(-6.0 * (img_f - 0.5)))
        s_min = 1.0 / (1.0 + np.exp(3.0))
        s_max = 1.0 / (1.0 + np.exp(-3.0))
        s_curved = (s_curved - s_min) / (s_max - s_min)
        img_f = (1.0 - s_strength) * img_f + s_strength * s_curved
        img_f = np.clip(img_f, 0.0, 1.0)

    img_f = _soft_highlights(img_f + over, ceiling=highlight_ceiling)
    img_f = np.clip(img_f, 0.0, 1.0)

    if strength > 0 or cool > 0:
        sat_gain = 1.0 + np.clip(strength, 0.0, 1.0) * 0.22
        cool_amt = np.clip(cool, 0.0, 1.0)
        rows = []
        chunk = 256
        for y in range(0, img_f.shape[0], chunk):
            lab = srgb_to_lab(img_f[y:y + chunk])
            if strength > 0:
                lab[..., 1:] *= sat_gain
            if cool_amt > 0:
                L = lab[..., 0]
                gate = np.clip((L - 32.0) / 36.0, 0.0, 1.0) * np.clip((98.0 - L) / 28.0, 0.0, 1.0)
                gate = gate[..., None]
                lab[..., 1:2] = lab[..., 1:2] - 5.0 * cool_amt * gate
                lab[..., 2:3] = lab[..., 2:3] - 16.0 * cool_amt * gate
                lab[..., 1:] *= 1.0 - 0.10 * cool_amt * gate
            rows.append(lab_to_srgb(lab).astype(np.float32))
        img_f = np.concatenate(rows, axis=0)

    out = np.rint(np.clip(img_f, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(out, "RGB")


def de_gray_cmyk(
    image: Image.Image,
    icc: bytes,
    shadow_lift: float = 0.18,
    strength: float = 0.6,
    black_pct: float = 1.0,
    white_pct: float = 99.0,
    highlight_ceiling: float = 0.94,
    cool: float = 0.55,
) -> Image.Image:
    """Apply washout correction in sRGB and write it back to target CMYK."""
    rgb = render_cmyk_to_srgb(image, icc)
    rgb = avoid_washout_adjust(
        rgb,
        shadow_lift=shadow_lift,
        strength=strength,
        black_pct=black_pct,
        white_pct=white_pct,
        highlight_ceiling=highlight_ceiling,
        cool=cool,
    )
    return srgb_to_cmyk(rgb, icc)


def render_cmyk_preview(
    image: Image.Image,
    icc: bytes,
    de_gray: bool = False,
    shadow_lift: float = 0.18,
    strength: float = 0.6,
    black_pct: float = 1.0,
    white_pct: float = 99.0,
    highlight_ceiling: float = 0.94,
    cool: float = 0.55,
) -> Image.Image:
    """ICC-render CMYK to sRGB. Optional extra de-gray if the CMYK is not already adjusted."""
    preview = render_cmyk_to_srgb(image, icc)
    if not de_gray:
        return preview
    return avoid_washout_adjust(
        preview,
        shadow_lift=shadow_lift,
        strength=strength,
        black_pct=black_pct,
        white_pct=white_pct,
        highlight_ceiling=highlight_ceiling,
        cool=cool,
    )


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


def load_color_model(path: str | Path, device: str | None = None):
    """Load a polynomial, residual-LUT, or adaptive CMYK-LUT model."""
    path = Path(path)
    if path.suffix.lower() in {".pt", ".pth"}:
        from adaptive_lut_model import AdaptiveLUTModel
        return AdaptiveLUTModel.load(path, device=device)
    with np.load(path, allow_pickle=False) as z:
        is_lut = "lut" in z.files
    if is_lut:
        from residual_lut_model import ResidualLUTModel
        return ResidualLUTModel.load(path)
    return ColorModel.load(path)
