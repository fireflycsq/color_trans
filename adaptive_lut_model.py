"""Image-adaptive CMYK residual LUTs on a fixed ICC baseline.

Encoders emit a 1D luma S-curve (contrast/density), a 17³×4 RGB-keyed residual
for hue shifts, and a confidence volume. The 3D residual is mean-centred so
it cannot carry the S-curve. Loss is against the human CMYK target.
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
DEFAULT_TONE_BINS = 17
MODEL_TYPE = "adaptive_cmyk_lut_v2"
MODEL_TYPE_V1 = "adaptive_cmyk_lut_v1"


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


def resolve_device(name: str | None = None, *, prefer_mps: bool = False):
    torch, _, _ = _torch()
    requested = ("" if name is None else str(name)).strip().lower()
    if requested in {"", "auto"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if prefer_mps and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError(
            "MPS 不可用。需要本机 macOS + Apple Silicon 的 PyTorch；"
            "Linux / Docker 请用 --device cpu 或 --device cuda"
        )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用，请改用 --device cpu")
    return device


def _numpy_torch_dtype(arr, torch):
    if arr.dtype == np.bool_ or arr.dtype == np.dtype(bool):
        return torch.bool
    if arr.dtype == np.uint8:
        return torch.uint8
    if arr.dtype == np.int32:
        return torch.int32
    if arr.dtype == np.int64:
        return torch.int64
    return torch.float32


def numpy_to_torch(array, device, dtype=None):
    """Copy a NumPy array onto device without asking PyTorch to infer numpy.float32.

    Some NumPy 2 + MPS builds raise "Could not infer dtype of numpy.float32"
    or "Numpy is not available" on from_numpy / torch.tensor(ndarray).
    """
    torch, _, _ = _torch()
    arr = np.ascontiguousarray(array)
    torch_dtype = dtype or _numpy_torch_dtype(arr, torch)
    if torch_dtype == torch.bool:
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
        storage_dtype = torch.uint8
    elif torch_dtype == torch.float32:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        storage_dtype = torch.float32
    else:
        storage_dtype = torch_dtype
    try:
        tensor = torch.from_numpy(np.array(arr, copy=True, order="C"))
        if tensor.dtype != storage_dtype:
            tensor = tensor.to(dtype=storage_dtype)
    except (RuntimeError, TypeError):
        try:
            tensor = torch.frombuffer(bytearray(arr.tobytes()), dtype=storage_dtype).clone()
            tensor = tensor.reshape(arr.shape)
        except (RuntimeError, TypeError, ValueError):
            tensor = torch.tensor(arr.reshape(-1).tolist(), dtype=storage_dtype, device="cpu")
            tensor = tensor.reshape(arr.shape)
    if torch_dtype == torch.bool:
        tensor = tensor.bool()
    return tensor.to(device)


def tensor_to_numpy(tensor) -> np.ndarray:
    torch, _, _ = _torch()
    cpu = tensor.detach().contiguous().to("cpu")
    try:
        return np.array(cpu.numpy(), copy=True)
    except (RuntimeError, TypeError):
        dtypes = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.float16: np.float16,
            torch.uint8: np.uint8,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.bool: np.bool_,
        }
        return np.array(cpu.tolist(), dtype=dtypes.get(cpu.dtype, np.float32))


def resize_square(rgb_u8: np.ndarray, size: int = THUMBNAIL) -> Image.Image:
    return Image.fromarray(np.asarray(rgb_u8, dtype=np.uint8), "RGB").resize(
        (size, size), Image.Resampling.BILINEAR,
    )


def image_to_tensor(image: Image.Image, device):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return numpy_to_torch(rgb, device).permute(2, 0, 1).unsqueeze(0).contiguous()


class SmallLutEncoder:
    """Built lazily so importing this module does not require torch."""

    @staticmethod
    def create(
        grid_size: int = DEFAULT_GRID,
        channels: int = DEFAULT_CHANNELS,
        tone_bins: int = DEFAULT_TONE_BINS,
    ):
        torch, nn, _ = _torch()

        class _Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.grid_size = grid_size
                self.tone_bins = tone_bins
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
                self.tone_head = nn.Linear(feat, LUT_CHANNELS * tone_bins)
                nn.init.zeros_(self.lut_head.weight)
                nn.init.zeros_(self.lut_head.bias)
                nn.init.zeros_(self.conf_head.weight)
                nn.init.constant_(self.conf_head.bias, 1.0)
                nn.init.zeros_(self.tone_head.weight)
                nn.init.zeros_(self.tone_head.bias)

            def forward(self, image):
                features = self.backbone(image).flatten(1)
                n = self.grid_size
                lut = self.lut_head(features).view(-1, n, n, n, LUT_CHANNELS)
                confidence = torch.sigmoid(self.conf_head(features)).view(-1, n, n, n)
                tone = self.tone_head(features).view(-1, self.tone_bins, LUT_CHANNELS)
                return lut, confidence, tone

        return _Encoder()


def unpack_encoder_out(out):
    if len(out) == 3:
        return out[0], out[1], out[2]
    return out[0], out[1], None


def rgb_luma(rgb):
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).clamp(0, 1)


def torch_linear_1d(table, luma):
    """table (T,C), luma (N,) in [0,1] → (N,C). Uses index_select for MPS."""
    bins = table.shape[0]
    position = luma.clamp(0, 1) * (bins - 1)
    lower = position.floor().long()
    upper = (lower + 1).clamp(max=bins - 1)
    frac = (position - lower.float()).unsqueeze(-1)
    lo = table.index_select(0, lower)
    hi = table.index_select(0, upper)
    return lo * (1 - frac) + hi * frac


def torch_trilinear(table, rgb):
    """table (n,n,n[,C]), rgb (N,3) in [0,1] → (N[,C]).

    Uses index_select instead of advanced indexing so MPS does not fall back
    through NumPy (which raises "Numpy is not available" on some builds).
    """
    squeeze = table.dim() == 3
    if squeeze:
        table = table.unsqueeze(-1)
    n = table.shape[0]
    channels = table.shape[-1]
    position = rgb.clamp(0, 1) * (n - 1)
    lower = position.floor().long()
    upper = (lower + 1).clamp(max=n - 1)
    frac = position - lower.float()
    flat = table.reshape(n * n * n, channels)
    result = rgb.new_zeros(rgb.shape[0], channels)
    for dr in (0, 1):
        ir = upper[:, 0] if dr else lower[:, 0]
        wr = frac[:, 0] if dr else 1 - frac[:, 0]
        for dg in (0, 1):
            ig = upper[:, 1] if dg else lower[:, 1]
            wg = frac[:, 1] if dg else 1 - frac[:, 1]
            for db in (0, 1):
                ib = upper[:, 2] if db else lower[:, 2]
                wb = frac[:, 2] if db else 1 - frac[:, 2]
                idx = ir * (n * n) + ig * n + ib
                values = flat.index_select(0, idx)
                result = result + values * (wr * wg * wb).unsqueeze(-1)
    return result.squeeze(-1) if squeeze else result


def torch_trilinear_fast(table, rgb):
    """Inference 3D lookup via grid_sample. Faster on CPU than numpy or MPS gathers."""
    torch, _, F = _torch()
    squeeze = table.dim() == 3
    if squeeze:
        table = table.unsqueeze(-1)
    volume = table.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
    grid = torch.stack((rgb[:, 2], rgb[:, 1], rgb[:, 0]), dim=-1).clamp(0, 1)
    grid = grid.mul(2).sub(1).view(1, 1, 1, -1, 3)
    sampled = F.grid_sample(
        volume, grid, mode="bilinear", padding_mode="border", align_corners=True,
    )
    result = sampled.reshape(volume.shape[1], -1).transpose(0, 1).contiguous()
    return result.squeeze(-1) if squeeze else result


def apply_correction_torch(rgb, baseline, lut, confidence, tone=None, chroma_only=False, fast=False):
    """ICC baseline + optional 1D luma S-curve + gated 3D residual."""
    lookup = torch_trilinear_fast if fast else torch_trilinear
    residual = lookup(lut, rgb)
    if chroma_only:
        residual = residual - residual.mean(dim=-1, keepdim=True)
    gate = lookup(confidence.unsqueeze(-1), rgb)
    out = baseline + gate * residual
    if tone is not None:
        out = out + torch_linear_1d(tone, rgb_luma(rgb))
    return out.clamp(0, 1)


def apply_lut_torch(rgb, baseline, lut, confidence, tone=None, chroma_only=False):
    return apply_correction_torch(
        rgb, baseline, lut, confidence, tone, chroma_only, fast=False,
    )


def apply_lut_torch_fast(rgb, baseline, lut, confidence, tone=None, chroma_only=False):
    return apply_correction_torch(
        rgb, baseline, lut, confidence, tone, chroma_only, fast=True,
    )


def apply_lut_numpy(
    rgb: np.ndarray,
    baseline: np.ndarray,
    lut: np.ndarray,
    confidence: np.ndarray,
    chunk_rows: int = 128,
    tone: np.ndarray | None = None,
    chroma_only: bool = False,
) -> np.ndarray:
    if rgb.ndim == 3 and rgb.shape[0] > chunk_rows > 0:
        parts = [
            apply_lut_numpy(
                rgb[y:y + chunk_rows],
                baseline[y:y + chunk_rows],
                lut,
                confidence,
                chunk_rows=0,
                tone=tone,
                chroma_only=chroma_only,
            )
            for y in range(0, rgb.shape[0], chunk_rows)
        ]
        return np.concatenate(parts, axis=0)
    residual = trilinear_lookup(lut, rgb)
    if chroma_only:
        residual = residual - residual.mean(axis=-1, keepdims=True)
    gate = trilinear_lookup(confidence, rgb)
    out = baseline + gate[..., None] * residual
    if tone is not None:
        luma = np.clip(
            rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114, 0.0, 1.0,
        )
        bins = tone.shape[0]
        position = luma * (bins - 1)
        lower = np.floor(position).astype(np.int64)
        upper = np.minimum(lower + 1, bins - 1)
        frac = (position - lower)[..., None]
        out = out + tone[lower] * (1.0 - frac) + tone[upper] * frac
    return np.clip(out, 0.0, 1.0)


def apply_lut_on_device(
    rgb, baseline, lut, confidence, device, chunk_rows: int = 128,
    tone=None, chroma_only: bool = False,
):
    """Apply 1D tone + 3D residual. Interpolation runs on CPU."""
    del device, chunk_rows
    torch, _, _ = _torch()
    cpu = torch.device("cpu")
    if hasattr(lut, "detach"):
        lut = lut.detach().to(cpu)
        confidence = confidence.detach().to(cpu)
        tone = None if tone is None else tone.detach().to(cpu)
    else:
        lut = numpy_to_torch(lut, cpu)
        confidence = numpy_to_torch(confidence, cpu)
        tone = None if tone is None else numpy_to_torch(tone, cpu)
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    baseline = np.ascontiguousarray(baseline, dtype=np.float32)
    spatial = rgb.shape[:2] if rgb.ndim == 3 else None
    with torch.no_grad():
        pred = apply_lut_torch_fast(
            numpy_to_torch(rgb.reshape(-1, 3), cpu),
            numpy_to_torch(baseline.reshape(-1, 4), cpu),
            lut, confidence, tone, chroma_only,
        )
        out = tensor_to_numpy(pred)
    return out.reshape(spatial[0], spatial[1], 4) if spatial else out


def lut_smoothness(lut):
    dx = (lut[1:] - lut[:-1]).abs().mean()
    dy = (lut[:, 1:] - lut[:, :-1]).abs().mean()
    dz = (lut[:, :, 1:] - lut[:, :, :-1]).abs().mean()
    return (dx + dy + dz) / 3.0


def tone_smoothness(tone):
    return (tone[1:] - tone[:-1]).abs().mean()


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
        self.device = resolve_device(None if device is None else str(device))
        self.global_encoder = global_encoder.to(self.device).eval()
        self.portrait_encoder = (
            None if portrait_encoder is None else portrait_encoder.to(self.device).eval()
        )
        self.target_icc = target_icc
        self.metadata = metadata

    @property
    def tone_split(self) -> bool:
        return bool(self.metadata.get("tone_split", self.metadata.get("model_type") == MODEL_TYPE))

    @property
    def portrait_region(self) -> str:
        return portrait_region_from_metadata(self.metadata)

    def encode_lut_tensors(self, encoder, image: Image.Image):
        torch, _, _ = _torch()
        with torch.no_grad():
            out = encoder(image_to_tensor(image, self.device))
        lut, confidence, tone = unpack_encoder_out(out)
        return lut[0], confidence[0], None if tone is None else tone[0]

    def encode_lut(self, encoder, image: Image.Image):
        lut, confidence, tone = self.encode_lut_tensors(encoder, image)
        return (
            tensor_to_numpy(lut).astype(np.float32),
            tensor_to_numpy(confidence).astype(np.float32),
            None if tone is None else tensor_to_numpy(tone).astype(np.float32),
        )

    def icc_baseline(self, source: Image.Image) -> np.ndarray:
        transform = ImageCms.buildTransform(
            ImageCms.createProfile("sRGB"),
            profile_from_bytes(self.target_icc),
            "RGB", "CMYK", renderingIntent=1,
            flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
        )
        return np.asarray(ImageCms.applyTransform(source, transform), dtype=np.float32) / 255.0

    def correct_cmyk(
        self,
        rgb_u8: np.ndarray,
        baseline: np.ndarray,
        rgb: np.ndarray | None = None,
        sample_idx: np.ndarray | None = None,
        chunk_rows: int = 128,
        thumb: Image.Image | None = None,
    ) -> np.ndarray:
        if thumb is None:
            thumb = resize_square(rgb_u8, int(self.metadata.get("thumbnail", THUMBNAIL)))
        lut, confidence, tone = self.encode_lut_tensors(self.global_encoder, thumb)
        if rgb is None:
            rgb = rgb_u8.astype(np.float32) / 255.0
        corrected = apply_lut_on_device(
            rgb, baseline, lut, confidence, self.device, chunk_rows,
            tone=tone, chroma_only=self.tone_split,
        )
        if self.portrait_encoder is None:
            return corrected
        region = self.portrait_region
        mask = portrait_mask(rgb_u8, region=region)
        crop = portrait_crop(
            rgb_u8, mask, size=int(self.metadata.get("thumbnail", THUMBNAIL)),
        )
        if crop is None:
            return corrected
        p_lut, p_conf, p_tone = self.encode_lut_tensors(self.portrait_encoder, crop)
        portrait = apply_lut_on_device(
            rgb, corrected, p_lut, p_conf, self.device, chunk_rows,
            tone=p_tone, chroma_only=self.tone_split,
        )
        weight = mask.reshape(-1)[sample_idx][..., None] if sample_idx is not None else mask[..., None]
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
        del max_hue_shift, min_saturation
        source = image_to_srgb(image)
        rgb_u8 = np.asarray(source, dtype=np.uint8)
        baseline = self.icc_baseline(source)
        cmyk = self.correct_cmyk(rgb_u8, baseline, chunk_rows=chunk_rows)
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
            "tone_bins": int(self.metadata.get("tone_bins", DEFAULT_TONE_BINS)),
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
        if model_type not in {MODEL_TYPE, MODEL_TYPE_V1}:
            raise ValueError(f"不是自适应 CMYK LUT 模型：{path}")
        metadata = json.loads(payload["metadata"])
        grid = int(payload.get("grid_size", metadata.get("grid_size", DEFAULT_GRID)))
        channels = int(payload.get("channels", metadata.get("encoder_channels", DEFAULT_CHANNELS)))
        tone_bins = int(payload.get("tone_bins", metadata.get("tone_bins", DEFAULT_TONE_BINS)))
        global_encoder = SmallLutEncoder.create(grid, channels, tone_bins)
        strict = model_type == MODEL_TYPE
        global_encoder.load_state_dict(payload["global_state"], strict=strict)
        portrait_encoder = None
        if payload.get("portrait_state") is not None:
            portrait_encoder = SmallLutEncoder.create(grid, channels, tone_bins)
            portrait_encoder.load_state_dict(payload["portrait_state"], strict=strict)
        if model_type == MODEL_TYPE_V1:
            metadata.setdefault("tone_split", False)
        return cls(
            global_encoder, payload["target_icc"], metadata, portrait_encoder, device,
        )
