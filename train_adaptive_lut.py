#!/usr/bin/env python3
"""Train image-adaptive CMYK residual LUTs on a fixed ICC baseline.

Stage ``global`` trains a small CNN on the full frame. Stage ``portrait``
freezes that encoder and trains a second CNN on a MediaPipe person/skin crop.
The CNN emits a 17³×4 residual; loss is Huber against the human CMYK target.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from adaptive_lut_model import (
    THUMBNAIL,
    AdaptiveLUTModel,
    SmallLutEncoder,
    apply_lut_torch,
    image_to_tensor,
    lut_smoothness,
    numpy_to_torch,
    resolve_device,
)
from color_model import apply_embedded_srgb, samples_to_srgb
from portrait_mask import portrait_crop, portrait_mask, require_person_segmenter
from train import (
    Pair,
    collect_pairs,
    icc_status_counts,
    load_fixed_target_icc,
    profile_details,
    sample_budget,
    stratified_indices,
    validate_pairs,
)
from train_residual_lut import build_to_cmyk, metric_summary


def _torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("自适应 LUT 需要 PyTorch：python3 -m pip install torch") from exc
    return torch, F


def srgb_thumbnail(rgb: Image.Image, icc: bytes | None, size: int) -> Image.Image:
    thumb = rgb.resize((size, size), Image.Resampling.BILINEAR)
    return apply_embedded_srgb(thumb, icc)


def load_pair_arrays(
    pair: Pair, thumbnail: int = THUMBNAIL,
) -> tuple[np.ndarray, np.ndarray, bytes | None, Image.Image]:
    with Image.open(pair.source) as source:
        src_icc = source.info.get("icc_profile")
        rgb_img = source.convert("RGB")
        thumb = srgb_thumbnail(rgb_img, src_icc, thumbnail)
        rgb_u8 = np.asarray(rgb_img, dtype=np.uint8)
    with Image.open(pair.target) as target:
        target_u8 = np.asarray(target.convert("CMYK"), dtype=np.uint8)
    if rgb_u8.shape[:2] != target_u8.shape[:2]:
        raise ValueError(f"尺寸不一致 {pair.name}: {rgb_u8.shape[1]}x{rgb_u8.shape[0]} vs {target_u8.shape[1]}x{target_u8.shape[0]}")
    return rgb_u8, target_u8, src_icc, thumb


def icc_samples(rgb_u8: np.ndarray, transform) -> np.ndarray:
    strip = Image.fromarray(np.asarray(rgb_u8, dtype=np.uint8)[None, ...], "RGB")
    return np.asarray(ImageCms.applyTransform(strip, transform), dtype=np.float32)[0] / 255.0


def sample_pixels(
    rgb_u8: np.ndarray,
    target_u8: np.ndarray,
    count: int,
    seed: int,
    transform,
    src_icc: bytes | None = None,
):
    idx = stratified_indices(rgb_u8, min(count, rgb_u8.shape[0] * rgb_u8.shape[1]), seed)
    rgb_s = samples_to_srgb(rgb_u8.reshape(-1, 3)[idx], src_icc)
    target = target_u8.reshape(-1, 4)[idx].astype(np.float32) / 255.0
    baseline = icc_samples(rgb_s, transform)
    rgb = rgb_s.astype(np.float32) / 255.0
    return rgb, baseline, target, idx


def encode_full(encoder, image: Image.Image, device, size: int | None = None):
    if size is not None and image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    return encoder(image_to_tensor(image, device))


def train_epoch(
    encoder, optimizer, pairs: list[Pair], transform, args, device, seed: int,
    global_encoder=None, region: str = "person",
) -> float:
    torch, F = _torch()
    encoder.train()
    if global_encoder is not None:
        global_encoder.eval()
    total = 0.0
    steps = 0
    budget = sample_budget(len(pairs), args.samples_per_image, args.max_samples)
    for i, pair in enumerate(pairs, 1):
        with Image.open(pair.source) as preview:
            width, height = preview.size
        print(
            f"[train {i:>4}/{len(pairs)}] {pair.name} {width}x{height}: loading",
            flush=True,
        )
        rgb_u8, target_u8, src_icc, thumb = load_pair_arrays(pair, args.thumbnail)
        rgb, baseline, target, idx = sample_pixels(
            rgb_u8, target_u8, budget, seed + i, transform, src_icc,
        )
        rgb_t = numpy_to_torch(rgb, device)
        base_t = numpy_to_torch(baseline, device)
        target_t = numpy_to_torch(target, device)
        if global_encoder is None:
            lut, conf = encode_full(encoder, thumb, device)
            pred = apply_lut_torch(rgb_t, base_t, lut[0], conf[0])
        else:
            mask_img = portrait_mask(rgb_u8, region=region)
            crop = portrait_crop(rgb_u8, mask_img, args.thumbnail, args.mask_threshold)
            mask = mask_img.reshape(-1)[idx]
            if crop is None or float(mask.max()) < args.mask_threshold:
                print(f"[train {i:>4}/{len(pairs)}] {pair.name}: no {region} crop", flush=True)
                continue
            crop = apply_embedded_srgb(crop, src_icc)
            with torch.no_grad():
                g_lut, g_conf = encode_full(global_encoder, thumb, device)
                base = apply_lut_torch(rgb_t, base_t, g_lut[0], g_conf[0])
            lut, conf = encoder(image_to_tensor(crop, device))
            portrait = apply_lut_torch(rgb_t, base, lut[0], conf[0])
            gate = numpy_to_torch(mask.astype(np.float32), device).unsqueeze(-1)
            pred = (1.0 - gate) * base + gate * portrait
            keep = mask >= args.mask_threshold
            if int(np.count_nonzero(keep)) < 32:
                print(f"[train {i:>4}/{len(pairs)}] {pair.name}: too few {region} pixels", flush=True)
                continue
            keep_t = numpy_to_torch(keep, device)
            pred = pred[keep_t]
            target_t = target_t[keep_t]
        loss = F.huber_loss(pred, target_t, delta=args.huber_delta)
        loss = loss + args.lut_l1 * lut[0].abs().mean() + args.smoothness * lut_smoothness(lut[0])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        steps += 1
        print(
            f"[train {i:>4}/{len(pairs)}] {pair.name}: loss={loss.item():.5f} pixels={pred.shape[0]:,}",
            flush=True,
        )
    if steps == 0:
        raise ValueError("这个 epoch 没有可用样本")
    return total / steps


def evaluate_pairs(
    model: AdaptiveLUTModel, pairs: list[Pair], icc: bytes, per_image: int,
    maximum: int, seed: int, label: str,
) -> dict | None:
    if not pairs:
        return None
    transform = build_to_cmyk(icc)
    budget = sample_budget(len(pairs), per_image, maximum)
    baselines, preds, targets = [], [], []
    for i, pair in enumerate(pairs, 1):
        with Image.open(pair.source) as preview:
            width, height = preview.size
        print(
            f"[{label} {i:>4}/{len(pairs)}] {pair.name} {width}x{height}: "
            f"loading / sampling {budget} pixels",
            flush=True,
        )
        rgb_u8, target_u8, src_icc, thumb = load_pair_arrays(pair, int(model.metadata.get("thumbnail", THUMBNAIL)))
        rgb, baseline, target, idx = sample_pixels(
            rgb_u8, target_u8, budget, seed + i, transform, src_icc,
        )
        pred = model.correct_cmyk(rgb_u8, baseline, rgb=rgb, sample_idx=idx, thumb=thumb)
        pair_icc = metric_summary(baseline, target, icc)
        pair_model = metric_summary(pred, target, icc)
        print(
            f"[{label} {i:>4}/{len(pairs)}] {pair.name}: "
            f"ICC ΔE={pair_icc['delta_e76']['mean']:.3f} "
            f"model ΔE={pair_model['delta_e76']['mean']:.3f} "
            f"CMYK MAE={np.round(pair_model['cmyk_mae'], 1).tolist()}",
            flush=True,
        )
        baselines.append(baseline)
        preds.append(pred)
        targets.append(target)
    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    base = np.concatenate(baselines)
    icc_metrics = metric_summary(base, target, icc)
    model_metrics = metric_summary(pred, target, icc)
    base_de, model_de = icc_metrics["delta_e76"]["mean"], model_metrics["delta_e76"]["mean"]
    return {
        "pairs": len(pairs),
        "samples": int(len(pred)),
        "icc_baseline": icc_metrics,
        "icc_plus_lut": model_metrics,
        "delta_e76_improvement_percent": float(100 * (base_de - model_de) / base_de) if base_de else 0.0,
        "cmyk_mae": model_metrics["cmyk_mae"],
        "delta_e76": model_metrics["delta_e76"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_argument_group("data source (choose one)")
    source.add_argument("--input"); source.add_argument("--target")
    source.add_argument("--input-dir"); source.add_argument("--target-dir")
    source.add_argument("--pair-dir")
    source.add_argument("--input-suffix", default="_input")
    source.add_argument("--target-suffix", default="_target")
    source.add_argument("--manifest")
    source.add_argument("--val-input-dir"); source.add_argument("--val-target-dir")
    source.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--target-icc", required=True, help="fixed CMYK output ICC")
    p.add_argument("--model", default="models/adaptive_lut.pt")
    p.add_argument("--output", help="portrait stage output .pt; default overwrites --model")
    p.add_argument("--report")
    p.add_argument("--stage", choices=("global", "portrait"), default="global")
    p.add_argument("--region", choices=("person", "skin"), default="person")
    p.add_argument("--grid-size", type=int, default=17)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--thumbnail", type=int, default=THUMBNAIL)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--samples-per-image", type=int, default=8_192)
    p.add_argument("--max-samples", type=int, default=1_500_000)
    p.add_argument("--eval-samples-per-image", type=int, default=4_096)
    p.add_argument("--max-eval-samples", type=int, default=250_000)
    p.add_argument("--huber-delta", type=float, default=0.125, help="Huber delta in CMYK 0..1 (~32/255)")
    p.add_argument("--lut-l1", type=float, default=0.01)
    p.add_argument("--smoothness", type=float, default=0.03)
    p.add_argument("--mask-threshold", type=float, default=0.45)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch, _ = _torch()
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio 必须在 [0, 1) 范围")
    if args.grid_size < 2:
        raise ValueError("--grid-size 必须至少为 2")
    if args.epochs < 1:
        raise ValueError("--epochs 必须至少为 1")
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_pairs, val_pairs = collect_pairs(args)
    print(f"pairs: train={len(train_pairs)}, validation={len(val_pairs)} | device={device}")
    icc = load_fixed_target_icc(Path(args.target_icc))
    _, train_records = validate_pairs(train_pairs, fixed_icc=icc)
    profile_name, profile_hash = profile_details(icc)
    val_records = []
    if val_pairs:
        _, val_records = validate_pairs(val_pairs, profile_hash, fixed_icc=icc)
    status = icc_status_counts(train_records + val_records)

    loaded = None
    if args.stage == "global":
        encoder = SmallLutEncoder.create(args.grid_size, args.channels).to(device)
        global_encoder = None
        region = "person"
    else:
        loaded = AdaptiveLUTModel.load(args.model, device=str(device))
        args.grid_size = int(loaded.metadata.get("grid_size", args.grid_size))
        args.channels = int(loaded.metadata.get("encoder_channels", args.channels))
        args.thumbnail = int(loaded.metadata.get("thumbnail", args.thumbnail))
        global_encoder = loaded.global_encoder
        encoder = SmallLutEncoder.create(args.grid_size, args.channels).to(device)
        if args.region == "person":
            require_person_segmenter()
        region = args.region
        icc = loaded.target_icc
        profile_name, profile_hash = profile_details(icc)

    transform = build_to_cmyk(icc)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    history = []
    print(
        f"objective: adaptive CMYK residual | stage={args.stage} | "
        f"lut={args.grid_size}³×4 keyed by RGB | huber={args.huber_delta:g}"
    )
    for epoch in range(1, args.epochs + 1):
        print(f"epoch {epoch}/{args.epochs} stage={args.stage}")
        mean_loss = train_epoch(
            encoder, optimizer, train_pairs, transform, args, device,
            args.seed + epoch * 17, global_encoder, region,
        )
        print(f"epoch {epoch} mean loss={mean_loss:.5f}")
        history.append({"epoch": epoch, "train_loss": mean_loss})

    if args.stage == "global":
        model = AdaptiveLUTModel(encoder, icc, {}, device=str(device))
    else:
        model = AdaptiveLUTModel(
            global_encoder, icc, dict(loaded.metadata), encoder, str(device),
        )

    metadata = {
        **(loaded.metadata if loaded is not None else {}),
        "model_type": "adaptive_cmyk_lut_v1",
        "stage": args.stage,
        "grid_size": args.grid_size,
        "encoder_channels": args.channels,
        "lut_channels": 4,
        "thumbnail": args.thumbnail,
        "epochs": args.epochs,
        "lr": args.lr,
        "huber_delta_cmyk": args.huber_delta,
        "lut_l1": args.lut_l1,
        "smoothness": args.smoothness,
        "portrait_region": region if args.stage == "portrait" else None,
        "portrait_mask_threshold": args.mask_threshold if args.stage == "portrait" else None,
        "train_pairs": len(train_pairs),
        "validation_pairs": len(val_pairs),
        "samples_per_image": sample_budget(len(train_pairs), args.samples_per_image, args.max_samples),
        "target_profile": profile_name,
        "target_icc_sha256": profile_hash,
        "embedded_target_icc_status": status,
        "edge_lift": 0.05,
        "edge_lift_c": 0.02,
        "shadow_lift": 0.06,
        "shadow_lift_cmy": 0.035,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "history": history,
    }
    model.metadata = metadata
    model_path = Path(args.output or args.model)
    report_path = Path(args.report) if args.report else model_path.with_suffix(".report.json")
    train_metrics = evaluate_pairs(
        model, train_pairs, icc, args.eval_samples_per_image, args.max_eval_samples,
        args.seed + 1_000_000, "train",
    )
    val_metrics = evaluate_pairs(
        model, val_pairs, icc, args.eval_samples_per_image, args.max_eval_samples,
        args.seed + 2_000_000, "val",
    )
    metadata["train_metrics"] = train_metrics
    metadata["validation_metrics"] = val_metrics
    model.metadata = metadata
    model.save(model_path)
    report = metadata | {
        "model": model_path.name,
        "train_images": train_records,
        "validation_images": val_records,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(f"saved model: {model_path.resolve()}")
    print(f"saved report: {report_path.resolve()}")
    if val_metrics:
        before = val_metrics["icc_baseline"]["delta_e76"]
        after = val_metrics["icc_plus_lut"]["delta_e76"]
        print(f"val ΔE76 ICC mean/p95: {before['mean']:.3f} / {before['p95']:.3f}")
        print(f"val ΔE76 +LUT mean/p95: {after['mean']:.3f} / {after['p95']:.3f}")
        print(f"val CMYK MAE [C M Y K]: {np.round(val_metrics['cmyk_mae'], 2).tolist()}")
        print(f"val mean improvement: {val_metrics['delta_e76_improvement_percent']:.2f}%")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
