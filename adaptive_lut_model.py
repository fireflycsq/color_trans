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


def resolve_device(name: str | None = None):
    torch, _, _ = _torch()
    if name:
        device = torch.device(name)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError(
                "MPS 不可用。需要本机 macOS + Apple Silicon 的 PyTorch；"
                "Linux / Docker 请用 --device cpu 或 --device cuda"
            )
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA 不可用，请改用 --device cpu")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def apply_lut_torch(rgb, baseline, lut, confidence):
    """RGB-keyed residual added to an ICC CMYK baseline."""
    residual = torch_trilinear(lut, rgb)
    gate = torch_trilinear(confidence.unsqueeze(-1), rgb)
    return (baseline + gate * residual).clamp(0, 1)


def apply_lut_numpy(
    rgb: np.ndarray,
    baseline: np.ndarray,
    lut: np.ndarray,
    confidence: np.ndarray,
    chunk_rows: int = 128,
) -> np.ndarray:
    if rgb.ndim == 3 and rgb.shape[0] > chunk_rows > 0:
        parts = [
            apply_lut_numpy(
                rgb[y:y + chunk_rows],
                baseline[y:y + chunk_rows],
                lut,
                confidence,
                chunk_rows=0,
            )
            for y in range(0, rgb.shape[0], chunk_rows)
        ]
        return np.concatenate(parts, axis=0)
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
        self.device = resolve_device(None if device is None else str(device))
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
            tensor_to_numpy(lut[0]).astype(np.float32),
            tensor_to_numpy(confidence[0]).astype(np.float32),
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
        lut, confidence = self.encode_lut(self.global_encoder, thumb)
        if rgb is None:
            rgb = rgb_u8.astype(np.float32) / 255.0
        corrected = apply_lut_numpy(rgb, baseline, lut, confidence, chunk_rows)
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
        portrait = apply_lut_numpy(rgb, corrected, p_lut, p_conf, chunk_rows)
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
