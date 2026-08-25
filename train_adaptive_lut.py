#!/usr/bin/env python3
"""Train image-adaptive CMYK residual LUTs on a fixed ICC baseline.

Stage ``global`` trains a small CNN on the full frame. Stage ``portrait``
freezes that encoder and trains a second CNN on a MediaPipe person/skin crop.
v3 emits histogram-conditioned 1D relative-luma tone, a mean-centred 17³×4
hue residual, and a look head (stretch / midtone / S / cool) baked into CMYK.
Loss is Huber on CMYK plus Lab appearance vs the human target.
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
    LOOK_DIM,
    MODEL_TYPE,
    STAT_DIM,
    THUMBNAIL,
    AdaptiveLUTModel,
    appearance_loss,
    apply_look_cmyk_torch,
    apply_lut_torch,
    apply_washout_torch,
    create_adaptive_encoder,
    image_stats_tensor,
    image_to_tensor,
    lut_smoothness,
    naive_cmyk_to_rgb,
    numpy_to_torch,
    relative_luma,
    resolve_device,
    stats_black_white,
    tone_smoothness,
    unpack_encoder_out,
)
from color_model import apply_embedded_srgb, samples_to_srgb
from portrait_mask import portrait_crop, portrait_mask, require_person_segmenter
from train import (
    Pair,
    collect_pairs,
    icc_status_counts,
    load_fixed_target_icc,
    profile_details,
    render_samples,
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


def encode_full(encoder, image: Image.Image, device, stats=None, size: int | None = None):
    if size is not None and image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    if stats is None and getattr(encoder, "stat_proj", None) is not None:
        stats = image_stats_tensor(image, device)
    return encoder(image_to_tensor(image, device), stats)


def apply_split(out, rgb, baseline, stats=None, relative_tone: bool = True, apply_look: bool = True):
    lut, conf, tone, look = unpack_encoder_out(out)
    black, white = stats_black_white(stats)
    tone_coord = relative_luma(rgb, black, white) if relative_tone else None
    pred = apply_lut_torch(
        rgb, baseline, lut[0], conf[0],
        None if tone is None else tone[0],
        chroma_only=True,
        tone_coord=tone_coord,
    )
    if apply_look and look is not None:
        black_t = pred.new_tensor(float(black) if not hasattr(black, "item") else float(black.item()))
        white_t = pred.new_tensor(float(white) if not hasattr(white, "item") else float(white.item()))
        pred = apply_look_cmyk_torch(pred, look[0], black_t, white_t)
    return pred, lut, tone, look


def luma_pixel_weight(rgb, strength: float):
    """Heavier Huber weight at shadows and highlights (S-curve tones)."""
    if strength <= 0:
        return rgb.new_ones(rgb.shape[0])
    luma = (
        rgb[:, 0] * 0.299 + rgb[:, 1] * 0.587 + rgb[:, 2] * 0.114
    ).clamp(0, 1)
    return 1.0 + strength * (2.0 * (luma - 0.5).abs())


def train_epoch(
    encoder, optimizer, pairs: list[Pair], transform, args, device, seed: int,
    icc: bytes, global_encoder=None, region: str = "person",
) -> float:
    torch, F = _torch()
    encoder.train()
    if global_encoder is not None:
        global_encoder.eval()
    total = 0.0
    steps = 0
    budget = sample_budget(len(pairs), args.samples_per_image, args.max_samples)
    relative_tone = not args.absolute_tone
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
        thumb_stats = image_stats_tensor(thumb, device)
        rgb_loss = rgb_t
        look_pred = None
        if global_encoder is None:
            pred, lut, tone, look_pred = apply_split(
                encode_full(encoder, thumb, device, thumb_stats),
                rgb_t, base_t, thumb_stats, relative_tone,
            )
        else:
            mask_img = portrait_mask(rgb_u8, region=region)
            crop = portrait_crop(rgb_u8, mask_img, args.thumbnail, args.mask_threshold)
            mask = mask_img.reshape(-1)[idx]
            if crop is None or float(mask.max()) < args.mask_threshold:
                print(f"[train {i:>4}/{len(pairs)}] {pair.name}: no {region} crop", flush=True)
                continue
            crop = apply_embedded_srgb(crop, src_icc)
            crop_stats = image_stats_tensor(crop, device)
            with torch.no_grad():
                base, _, _, _ = apply_split(
                    encode_full(global_encoder, thumb, device, thumb_stats),
                    rgb_t, base_t, thumb_stats, relative_tone,
                )
            portrait, lut, tone, look_pred = apply_split(
                encode_full(encoder, crop, device, crop_stats),
                rgb_t, base, crop_stats, relative_tone,
            )
            gate = numpy_to_torch(mask.astype(np.float32), device).unsqueeze(-1)
            pred = (1.0 - gate) * base + gate * portrait
            keep = mask >= args.mask_threshold
            if int(np.count_nonzero(keep)) < 32:
                print(f"[train {i:>4}/{len(pairs)}] {pair.name}: too few {region} pixels", flush=True)
                continue
            keep_t = numpy_to_torch(keep, device)
            pred = pred[keep_t]
            target_t = target_t[keep_t]
            rgb_loss = rgb_t[keep_t]
            baseline = baseline[keep]
            target = target[keep]
        huber = F.huber_loss(pred, target_t, delta=args.huber_delta, reduction="none")
        weight = luma_pixel_weight(rgb_loss, args.luma_weight)
        cmyk_loss = (huber.mean(dim=-1) * weight).sum() / weight.sum().clamp_min(1e-6)
        loss = args.cmyk_weight * cmyk_loss
        if args.appearance_weight > 0:
            loss = loss + args.appearance_weight * appearance_loss(
                naive_cmyk_to_rgb(pred), naive_cmyk_to_rgb(target_t), F.huber_loss,
                delta=args.appearance_delta,
            )
        if args.icc_look_weight > 0 and look_pred is not None:
            base_srgb = numpy_to_torch(render_samples(baseline, icc), device)
            target_srgb = numpy_to_torch(render_samples(target, icc), device)
            black, white = stats_black_white(thumb_stats if global_encoder is None else crop_stats)
            black_t = pred.new_tensor(float(black) if not hasattr(black, "item") else float(black.item()))
            white_t = pred.new_tensor(float(white) if not hasattr(white, "item") else float(white.item()))
            looked = apply_washout_torch(base_srgb, black_t, white_t, look_pred[0])
            loss = loss + args.icc_look_weight * appearance_loss(
                looked, target_srgb, F.huber_loss, delta=args.appearance_delta,
            )
        loss = loss + args.lut_l1 * lut[0].abs().mean() + args.smoothness * lut_smoothness(lut[0])
        if tone is not None:
            loss = loss + args.tone_smoothness * tone_smoothness(tone[0])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        steps += 1
        print(
            f"[train {i:>4}/{len(pairs)}] {pair.name}: loss={loss.item():.5f} "
            f"cmyk={float(cmyk_loss.item()):.5f} pixels={pred.shape[0]:,}",
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
    p.add_argument(
        "--luma-weight", type=float, default=1.0,
        help="extra Huber weight at shadows/highlights vs midtones; 0 disables",
    )
    p.add_argument(
        "--cmyk-weight", type=float, default=1.0,
        help="weight on CMYK Huber vs the human target",
    )
    p.add_argument(
        "--appearance-weight", type=float, default=1.0,
        help="weight on Lab Huber of naive RGB(pred) vs RGB(target); 0 disables",
    )
    p.add_argument(
        "--appearance-delta", type=float, default=0.08,
        help="Huber delta in scaled Lab (L/50, a/25, b/25)",
    )
    p.add_argument(
        "--icc-look-weight", type=float, default=0.35,
        help="extra Lab loss: look(ICC baseline sRGB) vs target sRGB; 0 disables",
    )
    p.add_argument(
        "--absolute-tone", action="store_true",
        help="1D curve keyed by absolute luma (v2); default is histogram-relative",
    )
    p.add_argument("--lut-l1", type=float, default=0.01)
    p.add_argument("--smoothness", type=float, default=0.03)
    p.add_argument("--tone-bins", type=int, default=17)
    p.add_argument(
        "--tone-smoothness", type=float, default=0.03,
        help="1D S-curve adjacent-bin smoothness; L1 is not applied to tone",
    )
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
    if args.tone_bins < 2:
        raise ValueError("--tone-bins 必须至少为 2")
    if args.tone_smoothness < 0:
        raise ValueError("--tone-smoothness 不能为负数")
    if args.luma_weight < 0:
        raise ValueError("--luma-weight 不能为负数")
    if args.cmyk_weight < 0:
        raise ValueError("--cmyk-weight 不能为负数")
    if args.appearance_weight < 0:
        raise ValueError("--appearance-weight 不能为负数")
    if args.icc_look_weight < 0:
        raise ValueError("--icc-look-weight 不能为负数")
    if args.appearance_delta <= 0:
        raise ValueError("--appearance-delta 必须为正数")
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
        encoder = create_adaptive_encoder(
            args.grid_size, args.channels, args.tone_bins, version=MODEL_TYPE,
        ).to(device)
        global_encoder = None
        region = "person"
    else:
        loaded = AdaptiveLUTModel.load(args.model, device=str(device))
        if not loaded.has_look:
            raise ValueError(
                "人像阶段需要 adaptive_cmyk_lut_v3（直方图 + 相对 1D + look）。请先重训 --stage global"
            )
        args.grid_size = int(loaded.metadata.get("grid_size", args.grid_size))
        args.channels = int(loaded.metadata.get("encoder_channels", args.channels))
        args.thumbnail = int(loaded.metadata.get("thumbnail", args.thumbnail))
        args.tone_bins = int(loaded.metadata.get("tone_bins", args.tone_bins))
        global_encoder = loaded.global_encoder
        encoder = create_adaptive_encoder(
            args.grid_size, args.channels, args.tone_bins, version=MODEL_TYPE,
        ).to(device)
        if args.region == "person":
            require_person_segmenter()
        region = args.region
        icc = loaded.target_icc
        profile_name, profile_hash = profile_details(icc)

    transform = build_to_cmyk(icc)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    history = []
    print(
        f"objective: adaptive CMYK v3 | stage={args.stage} | "
        f"hist+rel-1D={args.tone_bins} + lut={args.grid_size}³×4 chroma + look | "
        f"huber={args.huber_delta:g} | luma-weight={args.luma_weight:g} | "
        f"cmyk={args.cmyk_weight:g} appearance={args.appearance_weight:g} "
        f"icc-look={args.icc_look_weight:g}"
    )
    for epoch in range(1, args.epochs + 1):
        print(f"epoch {epoch}/{args.epochs} stage={args.stage}")
        mean_loss = train_epoch(
            encoder, optimizer, train_pairs, transform, args, device,
            args.seed + epoch * 17, icc, global_encoder, region,
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
        "model_type": MODEL_TYPE,
        "tone_split": True,
        "look_head": True,
        "relative_tone": not args.absolute_tone,
        "stat_dim": STAT_DIM,
        "look_dim": LOOK_DIM,
        "tone_bins": args.tone_bins,
        "stage": args.stage,
        "grid_size": args.grid_size,
        "encoder_channels": args.channels,
        "lut_channels": 4,
        "thumbnail": args.thumbnail,
        "epochs": args.epochs,
        "lr": args.lr,
        "huber_delta_cmyk": args.huber_delta,
        "luma_weight": args.luma_weight,
        "cmyk_weight": args.cmyk_weight,
        "appearance_weight": args.appearance_weight,
        "appearance_delta": args.appearance_delta,
        "icc_look_weight": args.icc_look_weight,
        "lut_l1": args.lut_l1,
        "smoothness": args.smoothness,
        "tone_smoothness": args.tone_smoothness,
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
        "shadow_lift": 0.0,
        "shadow_lift_cmy": 0.0,
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
