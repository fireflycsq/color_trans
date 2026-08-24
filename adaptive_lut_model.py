"""Image-adaptive CMYK residual 3D LUTs on a fixed ICC baseline.

Global and portrait encoders are small CNNs. They emit a 17³×4 residual LUT
and a confidence volume from a 256×256 thumbnail. Lookup is keyed by RGB;
the residual is added to ICC(sRGB). Loss is against the human CMYK target.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from color_model import image_to_srgb, profile_from_bytes
from portrait_mask import (
    has_person_segmenter,
    portrait_crop,
    portrait_edge_weight,
    portrait_mask,
    portrait_region_from_metadata,
)
from residual_lut_model import (
    edge_lift_amounts,
    shadow_lift_amounts,
    shadow_weight,
    trilinear_lookup,
)

THUMBNAIL = 256
DEFAULT_GRID = 17
DEFAULT_CHANNELS = 32
LUT_CHANNELS = 4
MODEL_TYPE = "adaptive_cmyk_lut_v1"


def _torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "自适应 LUT 需要 PyTorch：python3 -m pip install torch"
        ) from exc
    return torch, nn, F


def resize_square(rgb_u8: np.ndarray, size: int = THUMBNAIL) -> Image.Image:
    return Image.fromarray(np.asarray(rgb_u8, dtype=np.uint8), "RGB").resize(
        (size, size), Image.Resampling.BILINEAR,
    )


def image_to_tensor(image: Image.Image, device):
    torch, _, _ = _torch()
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


class SmallLutEncoder:
    """Built lazily so importing this module does not require torch."""

    @staticmethod
    def create(grid_size: int = DEFAULT_GRID, channels: int = DEFAULT_CHANNELS):
        torch, nn, _ = _torch()

        class _Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.grid_size = grid_size
                c = channels
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, c, 3, 2, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(c, c, 3, 2, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(c, c * 2, 3, 2, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(c * 2, c * 2, 3, 2, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(c * 2, c * 4, 3, 2, 1), nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                )
                feat = c * 4
                nodes = grid_size ** 3
                self.lut_head = nn.Linear(feat, LUT_CHANNELS * nodes)
                self.conf_head = nn.Linear(feat, nodes)
                nn.init.zeros_(self.lut_head.weight)
                nn.init.zeros_(self.lut_head.bias)
                nn.init.zeros_(self.conf_head.weight)
                nn.init.constant_(self.conf_head.bias, 1.0)

            def forward(self, image):
                features = self.backbone(image).flatten(1)
                n = self.grid_size
                lut = self.lut_head(features).view(-1, n, n, n, LUT_CHANNELS)
                confidence = torch.sigmoid(self.conf_head(features)).view(-1, n, n, n)
                return lut, confidence

        return _Encoder()


def torch_trilinear(table, rgb):
    """table (n,n,n[,C]), rgb (N,3) in [0,1] → (N[,C])."""
    torch, _, _ = _torch()
    size = table.shape[0]
    position = rgb.clamp(0, 1) * (size - 1)
    lower = position.floor().long()
    upper = (lower + 1).clamp(max=size - 1)
    frac = position - lower.float()
    extra = table.shape[3:]
    result = rgb.new_zeros((rgb.shape[0],) + extra)
    for dr in (0, 1):
        ir = upper[:, 0] if dr else lower[:, 0]
        wr = frac[:, 0] if dr else 1 - frac[:, 0]
        for dg in (0, 1):
            ig = upper[:, 1] if dg else lower[:, 1]
            wg = frac[:, 1] if dg else 1 - frac[:, 1]
            for db in (0, 1):
                ib = upper[:, 2] if db else lower[:, 2]
                wb = frac[:, 2] if db else 1 - frac[:, 2]
                weight = wr * wg * wb
                values = table[ir, ig, ib]
                if extra:
                    weight = weight.unsqueeze(-1)
                result = result + values * weight
    return result


def apply_lut_torch(rgb, baseline, lut, confidence):
    """RGB-keyed residual added to an ICC CMYK baseline."""
    residual = torch_trilinear(lut, rgb)
    gate = torch_trilinear(confidence.unsqueeze(-1), rgb)
    return (baseline + gate * residual).clamp(0, 1)


def apply_lut_numpy(
    rgb: np.ndarray, baseline: np.ndarray, lut: np.ndarray, confidence: np.ndarray,
) -> np.ndarray:
    residual = trilinear_lookup(lut, rgb)
    gate = trilinear_lookup(confidence, rgb)
    return np.clip(baseline + gate[..., None] * residual, 0.0, 1.0)


def lut_smoothness(lut):
    dx = (lut[1:] - lut[:-1]).abs().mean()
    dy = (lut[:, 1:] - lut[:, :-1]).abs().mean()
    dz = (lut[:, :, 1:] - lut[:, :, :-1]).abs().mean()
    return (dx + dy + dz) / 3.0


class AdaptiveLUTModel:
    """CNN-conditioned CMYK residual LUTs on a fixed ICC baseline."""

    def __init__(
        self,
        global_encoder,
        target_icc: bytes,
        metadata: dict,
        portrait_encoder=None,
        device: str | None = None,
    ):
        torch, _, _ = _torch()
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.global_encoder = global_encoder.to(self.device).eval()
        self.portrait_encoder = (
            None if portrait_encoder is None else portrait_encoder.to(self.device).eval()
        )
        self.target_icc = target_icc
        self.metadata = metadata

    @property
    def portrait_region(self) -> str:
        return portrait_region_from_metadata(self.metadata)

    def encode_lut(self, encoder, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        torch, _, _ = _torch()
        with torch.no_grad():
            lut, confidence = encoder(image_to_tensor(image, self.device))
        return (
            lut[0].detach().cpu().numpy().astype(np.float32),
            confidence[0].detach().cpu().numpy().astype(np.float32),
        )

    def icc_baseline(self, source: Image.Image) -> np.ndarray:
        transform = ImageCms.buildTransform(
            ImageCms.createProfile("sRGB"),
            profile_from_bytes(self.target_icc),
            "RGB", "CMYK", renderingIntent=1,
            flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
        )
        return np.asarray(ImageCms.applyTransform(source, transform), dtype=np.float32) / 255.0

    def correct_cmyk(self, rgb_u8: np.ndarray, baseline: np.ndarray) -> np.ndarray:
        thumb = resize_square(rgb_u8, int(self.metadata.get("thumbnail", THUMBNAIL)))
        lut, confidence = self.encode_lut(self.global_encoder, thumb)
        rgb = rgb_u8.astype(np.float32) / 255.0
        corrected = apply_lut_numpy(rgb, baseline, lut, confidence)
        if self.portrait_encoder is None:
            return corrected
        region = self.portrait_region
        mask = portrait_mask(rgb_u8, region=region)
        crop = portrait_crop(
            rgb_u8, mask, size=int(self.metadata.get("thumbnail", THUMBNAIL)),
        )
        if crop is None:
            return corrected
        p_lut, p_conf = self.encode_lut(self.portrait_encoder, crop)
        portrait = apply_lut_numpy(rgb, corrected, p_lut, p_conf)
        weight = mask[..., None]
        return np.clip((1.0 - weight) * corrected + weight * portrait, 0.0, 1.0)

    def predict_image(
        self,
        image: Image.Image,
        max_hue_shift: float | None = None,
        min_saturation: float = 0.16,
        chunk_rows: int = 128,
        edge_lift: float | None = None,
        shadow_lift: float | None = None,
    ) -> Image.Image:
        del max_hue_shift, min_saturation, chunk_rows
        source = image_to_srgb(image)
        rgb_u8 = np.asarray(source, dtype=np.uint8)
        baseline = self.icc_baseline(source)
        cmyk = self.correct_cmyk(rgb_u8, baseline)
        k_lift, c_lift = edge_lift_amounts(self.metadata, edge_lift)
        if k_lift > 0 or c_lift > 0:
            region = self.portrait_region if self.portrait_encoder is not None else "person"
            if self.portrait_encoder is not None or has_person_segmenter():
                mask = portrait_mask(rgb_u8, region=region)
                edge = mask if region == "contour" else portrait_edge_weight(mask)
                if c_lift > 0:
                    cmyk[..., 0] -= edge * c_lift
                if k_lift > 0:
                    cmyk[..., 3] -= edge * k_lift
        shadow_k, shadow_cmy = shadow_lift_amounts(self.metadata, shadow_lift)
        if shadow_k > 0 or shadow_cmy > 0:
            dark = shadow_weight(rgb_u8.astype(np.float32) / 255.0)
            if shadow_cmy > 0:
                cmyk[..., :3] -= dark[..., None] * shadow_cmy
            if shadow_k > 0:
                cmyk[..., 3] -= dark * shadow_k
        output = np.rint(np.clip(cmyk, 0.0, 1.0) * 255.0).astype(np.uint8)
        return Image.fromarray(output, "CMYK")

    def save(self, path: str | Path) -> None:
        torch, _, _ = _torch()
        payload = {
            "model_type": MODEL_TYPE,
            "grid_size": int(self.metadata.get("grid_size", DEFAULT_GRID)),
            "channels": int(self.metadata.get("encoder_channels", DEFAULT_CHANNELS)),
            "lut_channels": LUT_CHANNELS,
            "thumbnail": int(self.metadata.get("thumbnail", THUMBNAIL)),
            "global_state": self.global_encoder.state_dict(),
            "portrait_state": None if self.portrait_encoder is None else self.portrait_encoder.state_dict(),
            "target_icc": self.target_icc,
            "metadata": json.dumps(self.metadata, ensure_ascii=False),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "AdaptiveLUTModel":
        torch, _, _ = _torch()
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        model_type = payload.get("model_type")
        if model_type == "adaptive_rgb_lut_v1":
            raise ValueError(
                f"{path} 是旧的自适应 RGB LUT，无法加载。请用 CMYK 残差路径重训"
            )
        if model_type != MODEL_TYPE:
            raise ValueError(f"不是自适应 CMYK LUT 模型：{path}")
        metadata = json.loads(payload["metadata"])
        grid = int(payload.get("grid_size", metadata.get("grid_size", DEFAULT_GRID)))
        channels = int(payload.get("channels", metadata.get("encoder_channels", DEFAULT_CHANNELS)))
        global_encoder = SmallLutEncoder.create(grid, channels)
        global_encoder.load_state_dict(payload["global_state"])
        portrait_encoder = None
        if payload.get("portrait_state") is not None:
            portrait_encoder = SmallLutEncoder.create(grid, channels)
            portrait_encoder.load_state_dict(payload["portrait_state"])
        return cls(
            global_encoder, payload["target_icc"], metadata, portrait_encoder, device,
        )
