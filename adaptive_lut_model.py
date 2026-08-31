"""Image-adaptive CMYK residual LUTs on a fixed ICC baseline.

v3 encoders read a 256×256 thumbnail plus its luma histogram, emit a 1D curve
keyed by *relative* luma, a mean-centred 17³×4 hue residual, and a look head
(black/white stretch, midtone lift, S, highlight roll, cool). Look is baked
into CMYK in the forward pass. Loss is CMYK Huber plus Lab appearance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from color_model import (
    _SRGB_TO_XYZ,
    _XYZ_TO_SRGB,
    _XYZ_WHITE,
    image_to_srgb,
    srgb_to_cmyk_transform,
)
from portrait_mask import (
    calibrate_soft_mask,
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
)

THUMBNAIL = 256
DEFAULT_GRID = 17
DEFAULT_CHANNELS = 32
LUT_CHANNELS = 4
DEFAULT_TONE_BINS = 17
STAT_BINS = 32
STAT_BLACK = STAT_BINS + 6
STAT_WHITE = STAT_BINS + 7
STAT_DIM = STAT_BINS + 8
LOOK_DIM = 6
MODEL_TYPE = "adaptive_cmyk_lut_v3"
MODEL_TYPE_V2 = "adaptive_cmyk_lut_v2"
MODEL_TYPE_V1 = "adaptive_cmyk_lut_v1"
ADAPTIVE_CMYK_TYPES = {MODEL_TYPE, MODEL_TYPE_V2, MODEL_TYPE_V1}


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


def _logit(prob: float) -> float:
    p = min(max(float(prob), 1e-4), 1.0 - 1e-4)
    return float(np.log(p / (1.0 - p)))


class SmallLutEncoder:
    """Built lazily so importing this module does not require torch."""

    @staticmethod
    def create(
        grid_size: int = DEFAULT_GRID,
        channels: int = DEFAULT_CHANNELS,
        tone_bins: int = DEFAULT_TONE_BINS,
        stat_dim: int = 0,
        look_dim: int = 0,
    ):
        torch, nn, _ = _torch()

        class _Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.grid_size = grid_size
                self.tone_bins = tone_bins
                self.stat_dim = stat_dim
                self.look_dim = look_dim
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
                if stat_dim > 0:
                    self.stat_proj = nn.Sequential(
                        nn.Linear(stat_dim, feat),
                        nn.ReLU(inplace=True),
                        nn.Linear(feat, feat),
                    )
                    nn.init.zeros_(self.stat_proj[-1].weight)
                    nn.init.zeros_(self.stat_proj[-1].bias)
                else:
                    self.stat_proj = None
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
                if look_dim > 0:
                    self.look_head = nn.Linear(feat, look_dim)
                    nn.init.zeros_(self.look_head.weight)
                    self.look_head.bias.data.copy_(
                        torch.tensor(
                            [
                                _logit(0.01),
                                _logit(0.01),
                                0.0,
                                _logit(0.99),
                                0.0,
                                0.0,
                            ],
                            dtype=torch.float32,
                        )
                    )
                else:
                    self.look_head = None

            def forward(self, image, stats=None):
                features = self.backbone(image).flatten(1)
                if self.stat_proj is not None:
                    if stats is None:
                        stats = features.new_zeros(features.shape[0], self.stat_dim)
                    elif stats.dim() == 1:
                        stats = stats.unsqueeze(0)
                    features = features + self.stat_proj(stats)
                n = self.grid_size
                lut = self.lut_head(features).view(-1, n, n, n, LUT_CHANNELS)
                confidence = torch.sigmoid(self.conf_head(features)).view(-1, n, n, n)
                tone = self.tone_head(features).view(-1, self.tone_bins, LUT_CHANNELS)
                look = None if self.look_head is None else self.look_head(features)
                return lut, confidence, tone, look

        return _Encoder()


def create_adaptive_encoder(
    grid_size: int = DEFAULT_GRID,
    channels: int = DEFAULT_CHANNELS,
    tone_bins: int = DEFAULT_TONE_BINS,
    *,
    version: str = MODEL_TYPE,
):
    if version == MODEL_TYPE:
        return SmallLutEncoder.create(grid_size, channels, tone_bins, STAT_DIM, LOOK_DIM)
    return SmallLutEncoder.create(grid_size, channels, tone_bins, 0, 0)


def unpack_encoder_out(out):
    lut, confidence = out[0], out[1]
    tone = out[2] if len(out) > 2 else None
    look = out[3] if len(out) > 3 else None
    return lut, confidence, tone, look


def rgb_luma(rgb):
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).clamp(0, 1)


def thumbnail_stats(image: Image.Image) -> np.ndarray:
    """Luma histogram, percentiles, mean RGB, and stretch black/white from a thumb."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    luma = np.clip(
        rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114, 0.0, 1.0,
    )
    hist, _ = np.histogram(luma.reshape(-1), bins=STAT_BINS, range=(0.0, 1.0))
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-6
    p1, p50, p99 = np.percentile(luma, [1.0, 50.0, 99.0]).astype(np.float32)
    mean = rgb.reshape(-1, 3).mean(axis=0).astype(np.float32)
    black = np.float32(min(max(float(p1), 0.0), 0.18))
    white = np.float32(min(max(float(p99), float(black) + 0.20), 1.0))
    return np.concatenate([hist, np.array([p1, p50, p99], np.float32), mean, np.array([black, white], np.float32)])


def image_stats_tensor(image: Image.Image, device):
    return numpy_to_torch(thumbnail_stats(image)[None, :], device)


def stats_black_white(stats):
    if stats is None:
        return 0.0, 1.0
    if hasattr(stats, "detach"):
        vec = stats.reshape(-1)
        return vec[STAT_BLACK], vec[STAT_WHITE]
    vec = np.asarray(stats, dtype=np.float32).reshape(-1)
    return float(vec[STAT_BLACK]), float(vec[STAT_WHITE])


def relative_luma(rgb, black, white):
    luma = rgb_luma(rgb)
    if hasattr(white, "clamp"):
        span = (white - black).clamp(min=1e-6)
        return ((luma - black) / span).clamp(0, 1)
    span = max(float(white) - float(black), 1e-6)
    return ((luma - black) / span).clamp(0, 1)


def _as_look_vector(look):
    if look.dim() == 2:
        look = look.reshape(-1) if look.shape[0] == 1 else look[0]
    return look.reshape(-1)


def decode_look(look_raw):
    look = _as_look_vector(look_raw)
    shadow_lift = look[0].sigmoid()
    strength = look[1].sigmoid()
    cool = look[2].tanh()
    highlight_ceiling = 0.85 + 0.15 * look[3].sigmoid()
    black_delta = 0.08 * look[4].tanh()
    white_delta = 0.10 * look[5].tanh()
    return shadow_lift, strength, cool, highlight_ceiling, black_delta, white_delta


def srgb_to_lab_torch(rgb):
    torch, _, _ = _torch()
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).clamp(min=1e-8) ** 2.4)
    matrix = numpy_to_torch(_SRGB_TO_XYZ.astype(np.float32), rgb.device)
    white = numpy_to_torch(_XYZ_WHITE.astype(np.float32), rgb.device)
    xyz = (lin @ matrix.T) / white
    d = 6.0 / 29.0
    f = torch.where(
        xyz > d ** 3,
        xyz.clamp(min=1e-8).pow(1.0 / 3.0),
        xyz / (3.0 * d * d) + 4.0 / 29.0,
    )
    return torch.stack(
        [
            116.0 * f[..., 1] - 16.0,
            500.0 * (f[..., 0] - f[..., 1]),
            200.0 * (f[..., 1] - f[..., 2]),
        ],
        dim=-1,
    )


def lab_to_srgb_torch(lab):
    torch, _, _ = _torch()
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = lab[..., 1] / 500.0 + fy
    fz = fy - lab[..., 2] / 200.0
    d = 6.0 / 29.0

    def _f_inv(f):
        return torch.where(f > d, f ** 3, (f - 4.0 / 29.0) * 3.0 * d * d)

    xyz = torch.stack([_f_inv(fx), _f_inv(fy), _f_inv(fz)], dim=-1)
    xyz = xyz * numpy_to_torch(_XYZ_WHITE.astype(np.float32), lab.device)
    matrix = numpy_to_torch(_XYZ_TO_SRGB.astype(np.float32), lab.device)
    lin = xyz @ matrix.T
    rgb = torch.where(
        lin <= 0.0031308,
        lin * 12.92,
        1.055 * lin.clamp(min=1e-8).pow(1.0 / 2.4) - 0.055,
    )
    return rgb.clamp(0, 1)


def naive_cmyk_to_rgb(cmyk):
    torch, _, _ = _torch()
    k = cmyk[..., 3]
    return torch.stack(
        [
            (1.0 - cmyk[..., 0]) * (1.0 - k),
            (1.0 - cmyk[..., 1]) * (1.0 - k),
            (1.0 - cmyk[..., 2]) * (1.0 - k),
        ],
        dim=-1,
    )


def rgb_to_cmyk_keep_k(rgb, k):
    torch, _, _ = _torch()
    denom = (1.0 - k).clamp(min=1e-4)
    c = (1.0 - rgb[..., 0] / denom).clamp(0, 1)
    m = (1.0 - rgb[..., 1] / denom).clamp(0, 1)
    y = (1.0 - rgb[..., 2] / denom).clamp(0, 1)
    return torch.stack([c, m, y, k.clamp(0, 1)], dim=-1)


def apply_look_cmyk_torch(cmyk, look_raw, black, white):
    rgb = apply_washout_torch(naive_cmyk_to_rgb(cmyk), black, white, look_raw)
    return rgb_to_cmyk_keep_k(rgb, cmyk[..., 3]).clamp(0, 1)


def soft_highlights_torch(img, start: float = 0.78, ceiling=0.94):
    torch, _, _ = _torch()
    img = img.clamp(min=0.0)
    above = (img - start).clamp(min=0.0)
    if hasattr(ceiling, "clamp"):
        room = (ceiling - start).clamp(min=1e-6)
        ceiling_t = ceiling
    else:
        room = max(float(ceiling) - start, 1e-6)
        ceiling_t = img.new_tensor(float(ceiling))
    rolled = start + room * (above / (above + 0.50))
    return torch.where(img > start, torch.minimum(rolled, ceiling_t), img)


def apply_washout_torch(rgb, black, white, look_raw):
    """Look starts near identity: stretch/S/cool are blended by learned amounts.

    Histogram black/white are only mixed in by ``strength``. A full percentile
    stretch on every image is what made v3 lose to the ICC baseline on ΔE.
    """
    torch, _, _ = _torch()
    shadow_lift, strength, cool, highlight_ceiling, black_delta, white_delta = decode_look(look_raw)
    black_p = (black + black_delta).clamp(0.0, 0.18)
    white_p = (white + white_delta).clamp(min=black_p + 0.20).clamp(max=1.0)
    stretched = (rgb - black_p) / (white_p - black_p).clamp(min=1e-6)
    amt = strength.clamp(0, 1)
    img = (1.0 - amt) * rgb + amt * stretched
    over = amt * (stretched - 1.0).clamp(min=0.0)
    img = img.clamp(0, 1)
    gamma = 1.0 - shadow_lift.clamp(0, 1) * 0.35
    img = img.clamp(min=1e-8).pow(gamma)
    s_strength = amt * 0.40
    s_curved = torch.sigmoid(6.0 * (img - 0.5))
    s_min = 1.0 / (1.0 + torch.exp(img.new_tensor(3.0)))
    s_max = 1.0 / (1.0 + torch.exp(img.new_tensor(-3.0)))
    s_curved = (s_curved - s_min) / (s_max - s_min)
    img = (1.0 - s_strength) * img + s_strength * s_curved
    roll_amt = ((1.0 - highlight_ceiling) / 0.15).clamp(0, 1)
    rolled = soft_highlights_torch(img + over, ceiling=highlight_ceiling)
    img = ((1.0 - roll_amt) * img + roll_amt * rolled).clamp(0, 1)
    lab = srgb_to_lab_torch(img)
    sat_gain = 1.0 + amt * 0.22
    ab = lab[..., 1:] * sat_gain
    L = lab[..., 0]
    gate = ((L - 32.0) / 36.0).clamp(0, 1) * ((98.0 - L) / 28.0).clamp(0, 1)
    gate = gate.unsqueeze(-1)
    shift = rgb.new_tensor([5.0, 16.0])
    ab = ab - cool * gate * shift
    ab = ab * (1.0 - 0.10 * cool.abs() * gate)
    return lab_to_srgb_torch(torch.cat([L.unsqueeze(-1), ab], dim=-1))


def appearance_loss(pred_rgb, target_rgb, huber_fn, delta: float = 0.08):
    """Lab Huber with extra weight on L (fog/contrast) and less on b (cool/warm)."""
    scale = pred_rgb.new_tensor([28.0, 40.0, 55.0])
    pred = srgb_to_lab_torch(pred_rgb.clamp(0, 1)) / scale
    target = srgb_to_lab_torch(target_rgb.clamp(0, 1)) / scale
    return huber_fn(pred, target, delta=delta)


def shadow_punch_loss(pred_cmyk, target_cmyk, rgb, boost: float = 0.04):
    """Hinge: dark pixels should not be lighter/grayer than the target (+ boost)."""
    luma = rgb_luma(rgb)
    dark = ((0.22 - luma) / 0.22).clamp(0, 1)
    pred_d = 0.35 * pred_cmyk[..., :3].mean(dim=-1) + 0.65 * pred_cmyk[..., 3]
    tgt_d = 0.35 * target_cmyk[..., :3].mean(dim=-1) + 0.65 * target_cmyk[..., 3]
    gap = (tgt_d + boost - pred_d).clamp(min=0)
    return (gap * dark).sum() / dark.sum().clamp_min(1e-6)


def shadow_k_loss(pred_cmyk, target_cmyk, rgb, delta: float = 0.125):
    """Huber on K in shadows: bidirectional match to target K (not just penalise low K)."""
    torch, _, _ = _torch()
    luma = rgb_luma(rgb)
    dark = ((0.28 - luma) / 0.28).clamp(0, 1)
    diff = (pred_cmyk[..., 3] - target_cmyk[..., 3]).abs()
    quad = 0.5 * diff.square()
    lin = delta * (diff - 0.5 * delta)
    huber = torch.where(diff <= delta, quad, lin)
    return (huber * dark).sum() / dark.sum().clamp_min(1e-6)


def midtone_warmth_loss(pred_rgb, target_rgb, boost: float = 6.0):
    """Hinge: midtones should not be cooler (lower Lab b) than the target (+ boost)."""
    pred = srgb_to_lab_torch(pred_rgb.clamp(0, 1))
    tgt = srgb_to_lab_torch(target_rgb.clamp(0, 1))
    L = tgt[..., 0]
    gate = ((L - 32.0) / 36.0).clamp(0, 1) * ((98.0 - L) / 28.0).clamp(0, 1)
    gap = (tgt[..., 2] + boost - pred[..., 2]).clamp(min=0)
    return (gap * gate).sum() / gate.sum().clamp_min(1e-6)


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


def apply_correction_torch(
    rgb, baseline, lut, confidence, tone=None, chroma_only=False, fast=False,
    tone_coord=None, residual_limits=None,
):
    """ICC baseline + optional 1D luma S-curve + gated 3D residual."""
    torch, _, _ = _torch()
    lookup = torch_trilinear_fast if fast else torch_trilinear
    residual = lookup(lut, rgb)
    if chroma_only:
        cmy = residual[..., :3]
        residual = torch.cat(
            [cmy - cmy.mean(dim=-1, keepdim=True), residual[..., 3:4]],
            dim=-1,
        )
    if residual_limits is not None:
        if hasattr(residual_limits, "to"):
            limits = residual_limits.to(device=residual.device, dtype=residual.dtype)
        else:
            limits = numpy_to_torch(
                np.asarray(residual_limits, dtype=np.float32), residual.device,
            )
        limits = limits.reshape(-1)
        if limits.numel() != residual.shape[-1]:
            raise ValueError(
                f"residual_limits 需要 {residual.shape[-1]} 个通道，实际为 {limits.numel()}"
            )
        # Bound the effective interpolated and mean-centred residual, rather
        # than only the LUT nodes. tanh keeps the limiter differentiable.
        residual = limits * torch.tanh(residual / limits.clamp_min(1e-6))
    gate = lookup(confidence.unsqueeze(-1), rgb)
    out = baseline + gate * residual
    if tone is not None:
        coord = rgb_luma(rgb) if tone_coord is None else tone_coord
        out = out + torch_linear_1d(tone, coord)
    return out.clamp(0, 1)


def apply_lut_torch(
    rgb, baseline, lut, confidence, tone=None, chroma_only=False,
    tone_coord=None, residual_limits=None,
):
    return apply_correction_torch(
        rgb, baseline, lut, confidence, tone, chroma_only, fast=False,
        tone_coord=tone_coord, residual_limits=residual_limits,
    )


def apply_lut_torch_fast(
    rgb, baseline, lut, confidence, tone=None, chroma_only=False,
    tone_coord=None, residual_limits=None,
):
    return apply_correction_torch(
        rgb, baseline, lut, confidence, tone, chroma_only, fast=True,
        tone_coord=tone_coord, residual_limits=residual_limits,
    )


def apply_lut_on_device(
    rgb, baseline, lut, confidence,
    tone=None, chroma_only: bool = False, look=None, stats=None,
    relative_tone: bool = False, residual_limits=None,
):
    """Apply 1D tone + 3D residual + optional look. Interpolation runs on CPU."""
    torch, _, _ = _torch()
    cpu = torch.device("cpu")
    if hasattr(lut, "detach"):
        lut = lut.detach().to(cpu)
        confidence = confidence.detach().to(cpu)
        tone = None if tone is None else tone.detach().to(cpu)
        look = None if look is None else look.detach().to(cpu)
    else:
        lut = numpy_to_torch(lut, cpu)
        confidence = numpy_to_torch(confidence, cpu)
        tone = None if tone is None else numpy_to_torch(tone, cpu)
        look = None if look is None else numpy_to_torch(np.asarray(look, dtype=np.float32), cpu)
    rgb = np.ascontiguousarray(rgb, dtype=np.float32)
    baseline = np.ascontiguousarray(baseline, dtype=np.float32)
    spatial = rgb.shape[:2] if rgb.ndim == 3 else None
    rgb_t = numpy_to_torch(rgb.reshape(-1, 3), cpu)
    tone_coord = None
    if relative_tone and stats is not None:
        black, white = stats_black_white(stats)
        tone_coord = relative_luma(rgb_t, black, white)
    with torch.no_grad():
        pred = apply_lut_torch_fast(
            rgb_t,
            numpy_to_torch(baseline.reshape(-1, 4), cpu),
            lut, confidence, tone, chroma_only, tone_coord=tone_coord,
            residual_limits=residual_limits,
        )
        if look is not None and stats is not None:
            black, white = stats_black_white(stats)
            black_t = pred.new_tensor(float(black) if not hasattr(black, "detach") else black.detach().cpu())
            white_t = pred.new_tensor(float(white) if not hasattr(white, "detach") else white.detach().cpu())
            pred = apply_look_cmyk_torch(pred, look, black_t, white_t)
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
        if "tone_split" in self.metadata:
            return bool(self.metadata["tone_split"])
        return self.metadata.get("model_type") in {MODEL_TYPE, MODEL_TYPE_V2}

    @property
    def has_look(self) -> bool:
        return bool(self.metadata.get("look_head", self.metadata.get("model_type") == MODEL_TYPE))

    @property
    def has_stats(self) -> bool:
        return int(self.metadata.get("stat_dim", STAT_DIM if self.has_look else 0)) > 0

    @property
    def relative_tone(self) -> bool:
        return bool(self.metadata.get("relative_tone", self.has_look))

    @property
    def portrait_region(self) -> str:
        return portrait_region_from_metadata(self.metadata)

    @property
    def portrait_lut_only(self) -> bool:
        return bool(self.metadata.get("portrait_lut_only", False))

    @property
    def portrait_residual_limits(self):
        if not self.portrait_lut_only:
            return None
        values = self.metadata.get("portrait_residual_limits")
        if values is None:
            return (0.05, 0.05, 0.05, 0.04)
        if not isinstance(values, (list, tuple)) or len(values) != LUT_CHANNELS:
            raise ValueError("portrait_residual_limits 必须包含 C、M、Y、K 四个值")
        return tuple(float(x) for x in values)

    @property
    def portrait_dual_mask(self) -> bool:
        """CMY on calibrated skin, K on the full person silhouette."""
        return bool(self.metadata.get("portrait_dual_mask", False))

    @property
    def portrait_skin_gate_range(self) -> tuple[float, float]:
        low = float(self.metadata.get("portrait_skin_gate_low", 0.15))
        high = float(self.metadata.get("portrait_skin_gate_high", 0.50))
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("portrait skin gate 需要满足 0 <= low < high <= 1")
        return low, high

    def encode_lut_tensors(self, encoder, image: Image.Image, stats=None):
        torch, _, _ = _torch()
        if stats is None and self.has_stats:
            stats = image_stats_tensor(image, self.device)
        with torch.no_grad():
            out = encoder(image_to_tensor(image, self.device), stats)
        lut, confidence, tone, look = unpack_encoder_out(out)
        return (
            lut[0],
            confidence[0],
            None if tone is None else tone[0],
            None if look is None else look[0],
        )

    def encode_lut(self, encoder, image: Image.Image):
        lut, confidence, tone, look = self.encode_lut_tensors(encoder, image)
        return (
            tensor_to_numpy(lut).astype(np.float32),
            tensor_to_numpy(confidence).astype(np.float32),
            None if tone is None else tensor_to_numpy(tone).astype(np.float32),
            None if look is None else tensor_to_numpy(look).astype(np.float32),
        )

    def icc_baseline(self, source: Image.Image) -> np.ndarray:
        transform = srgb_to_cmyk_transform(self.target_icc)
        return np.asarray(ImageCms.applyTransform(source, transform), dtype=np.float32) / 255.0

    def correct_cmyk(
        self,
        rgb_u8: np.ndarray,
        baseline: np.ndarray,
        rgb: np.ndarray | None = None,
        sample_idx: np.ndarray | None = None,
        thumb: Image.Image | None = None,
        mask: np.ndarray | None = None,
        skin_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        thumbnail = int(self.metadata.get("thumbnail", THUMBNAIL))
        if thumb is None:
            thumb = resize_square(rgb_u8, thumbnail)
        stats = thumbnail_stats(thumb) if self.has_stats else None
        stats_t = numpy_to_torch(stats[None, :], self.device) if stats is not None else None
        lut, confidence, tone, look = self.encode_lut_tensors(
            self.global_encoder, thumb, stats_t,
        )
        if rgb is None:
            rgb = rgb_u8.astype(np.float32) / 255.0
        corrected = apply_lut_on_device(
            rgb, baseline, lut, confidence,
            tone=tone, chroma_only=self.tone_split,
            look=look if self.has_look else None,
            stats=stats,
            relative_tone=self.relative_tone,
        )
        if self.portrait_encoder is None:
            return corrected
        if self.portrait_dual_mask:
            if mask is None:
                mask = portrait_mask(rgb_u8, region="person")
            if skin_mask is None:
                skin_mask = portrait_mask(rgb_u8, region="skin")
            crop_mask = mask
        else:
            if mask is None:
                mask = portrait_mask(rgb_u8, region=self.portrait_region)
            crop_mask = mask
        crop = portrait_crop(rgb_u8, crop_mask, size=thumbnail)
        if crop is None:
            return corrected
        p_stats = thumbnail_stats(crop) if self.has_stats else None
        p_stats_t = (
            numpy_to_torch(p_stats[None, :], self.device) if p_stats is not None else None
        )
        p_lut, p_conf, p_tone, p_look = self.encode_lut_tensors(
            self.portrait_encoder, crop, p_stats_t,
        )
        if self.portrait_lut_only:
            p_tone = None
            p_look = None
        portrait = apply_lut_on_device(
            rgb, corrected, p_lut, p_conf,
            tone=p_tone, chroma_only=self.tone_split,
            look=p_look if self.has_look else None,
            stats=p_stats,
            relative_tone=self.relative_tone,
            residual_limits=self.portrait_residual_limits,
        )
        if self.portrait_dual_mask:
            low, high = self.portrait_skin_gate_range
            skin_gate = calibrate_soft_mask(skin_mask, low, high)
            if sample_idx is not None:
                person_weight = mask.reshape(-1)[sample_idx][..., None]
                skin_weight = skin_gate.reshape(-1)[sample_idx][..., None]
            else:
                person_weight = mask[..., None]
                skin_weight = skin_gate[..., None]
            weight = np.concatenate(
                [np.repeat(skin_weight, 3, axis=-1), person_weight], axis=-1,
            )
        else:
            weight = (
                mask.reshape(-1)[sample_idx][..., None]
                if sample_idx is not None else mask[..., None]
            )
        return np.clip(corrected + weight * (portrait - corrected), 0.0, 1.0)

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
        k_lift, c_lift = edge_lift_amounts(self.metadata, edge_lift)
        need_edge = (k_lift > 0 or c_lift > 0) and (
            self.portrait_encoder is not None or has_person_segmenter()
        )
        region = self.portrait_region if self.portrait_encoder is not None else "person"
        mask = None
        skin_mask = None
        if self.portrait_encoder is not None or need_edge:
            if self.portrait_encoder is not None and self.portrait_dual_mask:
                mask = portrait_mask(rgb_u8, region="person")
                skin_mask = portrait_mask(rgb_u8, region="skin")
                region = "person"
            else:
                mask = portrait_mask(rgb_u8, region=region)
        cmyk = self.correct_cmyk(
            rgb_u8, baseline, mask=mask, skin_mask=skin_mask,
        )
        if need_edge:
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
            "stat_dim": int(self.metadata.get("stat_dim", STAT_DIM if self.has_look else 0)),
            "look_dim": int(self.metadata.get("look_dim", LOOK_DIM if self.has_look else 0)),
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
        if model_type not in ADAPTIVE_CMYK_TYPES:
            raise ValueError(f"不是自适应 CMYK LUT 模型：{path}")
        metadata = json.loads(payload["metadata"])
        grid = int(payload.get("grid_size", metadata.get("grid_size", DEFAULT_GRID)))
        channels = int(payload.get("channels", metadata.get("encoder_channels", DEFAULT_CHANNELS)))
        tone_bins = int(payload.get("tone_bins", metadata.get("tone_bins", DEFAULT_TONE_BINS)))
        global_encoder = create_adaptive_encoder(grid, channels, tone_bins, version=model_type)
        strict = model_type != MODEL_TYPE_V1
        global_encoder.load_state_dict(payload["global_state"], strict=strict)
        portrait_encoder = None
        if payload.get("portrait_state") is not None:
            portrait_encoder = create_adaptive_encoder(grid, channels, tone_bins, version=model_type)
            portrait_encoder.load_state_dict(payload["portrait_state"], strict=strict)
        if model_type == MODEL_TYPE_V1:
            metadata.setdefault("tone_split", False)
        if model_type == MODEL_TYPE_V2:
            metadata.setdefault("tone_split", True)
            metadata.setdefault("look_head", False)
            metadata.setdefault("relative_tone", False)
            metadata.setdefault("stat_dim", 0)
        return cls(
            global_encoder, payload["target_icc"], metadata, portrait_encoder, device,
        )
