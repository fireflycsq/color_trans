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
    midtone_warmth_loss,
    naive_cmyk_to_rgb,
    numpy_to_torch,
    relative_luma,
    resolve_device,
    shadow_punch_loss,
    shadow_k_loss,
    stats_black_white,
    tone_smoothness,
    unpack_encoder_out,
)
from color_model import apply_embedded_srgb, samples_to_srgb, srgb_to_lab
from portrait_mask import (
    calibrate_soft_mask,
    portrait_crop,
    portrait_mask,
    require_person_segmenter,
)
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


def full_srgb_array(rgb_u8: np.ndarray, icc: bytes | None) -> np.ndarray:
    """Normalize a full RGB array for portrait masking/cropping and inference parity."""
    image = Image.fromarray(np.asarray(rgb_u8, dtype=np.uint8), "RGB")
    return np.asarray(apply_embedded_srgb(image, icc), dtype=np.uint8)


def mean_rendered_delta_l(pred: np.ndarray, target: np.ndarray, icc: bytes) -> float:
    """Signed rendered Lab lightness error: positive means brighter than target."""
    pred_lab = srgb_to_lab(render_samples(pred, icc))
    target_lab = srgb_to_lab(render_samples(target, icc))
    return float(np.mean(pred_lab[..., 0] - target_lab[..., 0]))


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


def apply_split(
    out, rgb, baseline, stats=None, relative_tone: bool = True,
    apply_tone: bool = True, apply_look: bool = True, residual_limits=None,
):
    lut, conf, tone, look = unpack_encoder_out(out)
    active_tone = tone if apply_tone else None
    active_look = look if apply_look else None
    black, white = stats_black_white(stats)
    tone_coord = relative_luma(rgb, black, white) if relative_tone else None
    pred = apply_lut_torch(
        rgb, baseline, lut[0], conf[0],
        None if active_tone is None else active_tone[0],
        chroma_only=True,
        tone_coord=tone_coord,
        residual_limits=residual_limits,
    )
    if active_look is not None:
        black_t = pred.new_tensor(float(black) if not hasattr(black, "item") else float(black.item()))
        white_t = pred.new_tensor(float(white) if not hasattr(white, "item") else float(white.item()))
        pred = apply_look_cmyk_torch(pred, active_look[0], black_t, white_t)
    return pred, lut, active_tone, active_look


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
            dual_mask = args.portrait_dual_mask and region == "skin"
            portrait_limits = None
            if args.portrait_lut_only:
                portrait_limits = (
                    args.portrait_residual_limit_cmy,
                    args.portrait_residual_limit_cmy,
                    args.portrait_residual_limit_cmy,
                    args.portrait_residual_limit_k,
                )
            portrait_rgb_u8 = full_srgb_array(rgb_u8, src_icc)
            if dual_mask:
                person_mask_img = portrait_mask(portrait_rgb_u8, region="person")
                skin_mask_img = portrait_mask(portrait_rgb_u8, region="skin")
                mask_img = person_mask_img
            else:
                skin_mask_img = None
                mask_img = portrait_mask(portrait_rgb_u8, region=region)
            crop = portrait_crop(
                portrait_rgb_u8, mask_img, args.thumbnail, args.mask_threshold,
            )
            mask = mask_img.reshape(-1)[idx]
            if crop is None or float(mask.max()) < args.mask_threshold:
                print(f"[train {i:>4}/{len(pairs)}] {pair.name}: no {region} crop", flush=True)
                continue
            crop_stats = image_stats_tensor(crop, device)
            with torch.no_grad():
                base, _, _, _ = apply_split(
                    encode_full(global_encoder, thumb, device, thumb_stats),
                    rgb_t, base_t, thumb_stats, relative_tone,
                )
            portrait, lut, tone, look_pred = apply_split(
                encode_full(encoder, crop, device, crop_stats),
                rgb_t, base, crop_stats, relative_tone,
                apply_tone=not args.portrait_lut_only,
                apply_look=not args.portrait_lut_only,
                residual_limits=portrait_limits,
            )
            if dual_mask:
                skin_gate = calibrate_soft_mask(
                    skin_mask_img, args.portrait_skin_gate_low,
                    args.portrait_skin_gate_high,
                ).reshape(-1)[idx]
                channel_gate = np.stack(
                    [skin_gate, skin_gate, skin_gate, mask], axis=-1,
                ).astype(np.float32)
                gate = numpy_to_torch(channel_gate, device)
            else:
                gate = numpy_to_torch(mask.astype(np.float32), device).unsqueeze(-1)
            pred = base + gate * (portrait - base)
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
        if args.punch_weight > 0:
            loss = loss + args.punch_weight * shadow_punch_loss(
                pred, target_t, rgb_loss, boost=args.punch_boost,
            )
        if args.k_punch_weight > 0:
            loss = loss + args.k_punch_weight * shadow_k_loss(
                pred, target_t, rgb_loss, delta=args.k_punch_delta,
            )
        if args.warmth_weight > 0:
            loss = loss + args.warmth_weight * midtone_warmth_loss(
                naive_cmyk_to_rgb(pred), naive_cmyk_to_rgb(target_t),
                boost=args.warmth_boost,
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
    compare_global = model.portrait_encoder is not None
    global_model = None
    if compare_global:
        global_model = AdaptiveLUTModel(
            model.global_encoder, icc, dict(model.metadata), device=str(model.device),
        )
    threshold = float(model.metadata.get("portrait_mask_threshold", 0.45))
    region = model.portrait_region
    baselines, preds, targets = [], [], []
    global_preds = []
    masked_globals, masked_preds, masked_targets = [], [], []
    skin_globals, skin_preds, skin_targets = [], [], []
    neutral_globals, neutral_preds, neutral_targets = [], [], []
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
        model_rgb_u8 = full_srgb_array(rgb_u8, src_icc) if compare_global else rgb_u8
        if compare_global:
            person_mask = portrait_mask(model_rgb_u8, region="person")
            skin_mask = portrait_mask(model_rgb_u8, region="skin")
            mask = person_mask if model.portrait_dual_mask else (
                person_mask if region == "person" else skin_mask
            )
        else:
            person_mask = skin_mask = mask = None
        pred = model.correct_cmyk(
            model_rgb_u8, baseline, rgb=rgb, sample_idx=idx, thumb=thumb,
            mask=mask, skin_mask=skin_mask,
        )
        pair_icc = metric_summary(baseline, target, icc)
        pair_model = metric_summary(pred, target, icc)
        portrait_note = ""
        if compare_global:
            global_pred = global_model.correct_cmyk(
                model_rgb_u8, baseline, rgb=rgb, sample_idx=idx, thumb=thumb,
            )
            keep = mask.reshape(-1)[idx] >= threshold
            if model.portrait_dual_mask:
                skin_low, skin_high = model.portrait_skin_gate_range
                sampled_skin = calibrate_soft_mask(
                    skin_mask, skin_low, skin_high,
                ).reshape(-1)[idx]
                skin_keep = sampled_skin >= 0.5
            else:
                skin_keep = skin_mask.reshape(-1)[idx] >= threshold
            person_values = person_mask.reshape(-1)[idx]
            luma = rgb[:, 0] * 0.299 + rgb[:, 1] * 0.587 + rgb[:, 2] * 0.114
            chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
            neutral_keep = (
                (person_values >= threshold) & (luma > 0.55) & (chroma < 0.10)
            )
            global_preds.append(global_pred)
            if np.any(keep):
                masked_globals.append(global_pred[keep])
                masked_preds.append(pred[keep])
                masked_targets.append(target[keep])
                pair_mask_global = metric_summary(global_pred[keep], target[keep], icc)
                pair_mask_model = metric_summary(pred[keep], target[keep], icc)
                portrait_note = (
                    f" portrait global/model ΔE="
                    f"{pair_mask_global['delta_e76']['mean']:.3f}/"
                    f"{pair_mask_model['delta_e76']['mean']:.3f}"
                )
            if np.any(skin_keep):
                skin_globals.append(global_pred[skin_keep])
                skin_preds.append(pred[skin_keep])
                skin_targets.append(target[skin_keep])
            if np.any(neutral_keep):
                neutral_globals.append(global_pred[neutral_keep])
                neutral_preds.append(pred[neutral_keep])
                neutral_targets.append(target[neutral_keep])
        print(
            f"[{label} {i:>4}/{len(pairs)}] {pair.name}: "
            f"ICC ΔE={pair_icc['delta_e76']['mean']:.3f} "
            f"model ΔE={pair_model['delta_e76']['mean']:.3f} "
            f"CMYK MAE={np.round(pair_model['cmyk_mae'], 1).tolist()}"
            f"{portrait_note}",
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
    result = {
        "pairs": len(pairs),
        "samples": int(len(pred)),
        "icc_baseline": icc_metrics,
        "icc_plus_lut": model_metrics,
        "delta_e76_improvement_percent": float(100 * (base_de - model_de) / base_de) if base_de else 0.0,
        "cmyk_mae": model_metrics["cmyk_mae"],
        "delta_e76": model_metrics["delta_e76"],
    }
    if compare_global:
        if not masked_preds:
            raise ValueError(
                f"{label} 验证集中没有达到 mask threshold {threshold:g} 的 {region} 像素"
            )
        global_pred = np.concatenate(global_preds)
        masked_global = np.concatenate(masked_globals)
        masked_pred = np.concatenate(masked_preds)
        masked_target = np.concatenate(masked_targets)
        global_metrics = metric_summary(global_pred, target, icc)
        mask_global_metrics = metric_summary(masked_global, masked_target, icc)
        mask_model_metrics = metric_summary(masked_pred, masked_target, icc)
        full_global_de = global_metrics["delta_e76"]["mean"]
        mask_global_de = mask_global_metrics["delta_e76"]["mean"]
        mask_model_de = mask_model_metrics["delta_e76"]["mean"]
        result["global_only"] = global_metrics
        result["global_plus_portrait"] = model_metrics
        result["global_to_portrait_delta_e76_improvement_percent"] = (
            float(100 * (full_global_de - model_de) / full_global_de)
            if full_global_de else 0.0
        )
        result["portrait_mask"] = {
            "region": "person" if model.portrait_dual_mask else region,
            "threshold": threshold,
            "samples": int(len(masked_pred)),
            "global_only": mask_global_metrics,
            "global_plus_portrait": mask_model_metrics,
            "delta_e76_improvement_percent": (
                float(100 * (mask_global_de - mask_model_de) / mask_global_de)
                if mask_global_de else 0.0
            ),
        }
        if model.portrait_dual_mask and not skin_preds:
            raise ValueError(f"{label} 验证集中没有可评估的校准皮肤像素")
        if skin_preds:
            skin_global_metrics = metric_summary(
                np.concatenate(skin_globals), np.concatenate(skin_targets), icc,
            )
            skin_model_metrics = metric_summary(
                np.concatenate(skin_preds), np.concatenate(skin_targets), icc,
            )
            result["skin_mask"] = {
                "threshold": 0.5 if model.portrait_dual_mask else threshold,
                "calibrated": model.portrait_dual_mask,
                "samples": int(sum(len(x) for x in skin_preds)),
                "global_only": skin_global_metrics,
                "global_plus_portrait": skin_model_metrics,
                "global_only_mean_delta_l": mean_rendered_delta_l(
                    np.concatenate(skin_globals), np.concatenate(skin_targets), icc,
                ),
                "global_plus_portrait_mean_delta_l": mean_rendered_delta_l(
                    np.concatenate(skin_preds), np.concatenate(skin_targets), icc,
                ),
            }
        if neutral_preds:
            neutral_global_metrics = metric_summary(
                np.concatenate(neutral_globals), np.concatenate(neutral_targets), icc,
            )
            neutral_model_metrics = metric_summary(
                np.concatenate(neutral_preds), np.concatenate(neutral_targets), icc,
            )
            result["neutral_white_mask"] = {
                "definition": "person>=threshold, luma>0.55, rgb_chroma<0.10",
                "samples": int(sum(len(x) for x in neutral_preds)),
                "global_only": neutral_global_metrics,
                "global_plus_portrait": neutral_model_metrics,
                "global_only_mean_delta_l": mean_rendered_delta_l(
                    np.concatenate(neutral_globals), np.concatenate(neutral_targets), icc,
                ),
                "global_plus_portrait_mean_delta_l": mean_rendered_delta_l(
                    np.concatenate(neutral_preds), np.concatenate(neutral_targets), icc,
                ),
            }
    return result


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
    p.add_argument(
        "--save-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write stem_epochNN.pt after each epoch; --no-save-every-epoch disables",
    )
    p.add_argument("--stage", choices=("global", "portrait"), default="global")
    p.add_argument("--region", choices=("person", "skin"), default="person")
    p.add_argument(
        "--portrait-lut-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="portrait stage trains only a bounded 3D CMYK residual; default on",
    )
    p.add_argument(
        "--portrait-residual-limit-cmy", type=float, default=0.05,
        help="absolute effective C/M/Y residual limit in CMYK 0..1",
    )
    p.add_argument(
        "--portrait-residual-limit-k", type=float, default=0.04,
        help="absolute effective K residual limit in CMYK 0..1",
    )
    p.add_argument(
        "--portrait-dual-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="with --region skin, apply CMY to calibrated skin and K to the full person; default on",
    )
    p.add_argument("--portrait-skin-gate-low", type=float, default=0.15)
    p.add_argument("--portrait-skin-gate-high", type=float, default=0.50)
    p.add_argument(
        "--portrait-neutral-max-regression", type=float, default=0.02,
        help="maximum allowed neutral/white DeltaE regression fraction vs global-only",
    )
    p.add_argument(
        "--portrait-skin-max-regression", type=float, default=0.01,
        help="maximum allowed skin-mask DeltaE regression fraction vs global-only",
    )
    p.add_argument("--grid-size", type=int, default=17)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--thumbnail", type=int, default=THUMBNAIL)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--early-stopping-patience", type=int, default=3,
        help="stop after this many consecutive epochs without lower validation mean DeltaE; 0 disables",
    )
    p.add_argument(
        "--lr-patience", type=int, default=1,
        help="ReduceLROnPlateau patience measured in validation epochs",
    )
    p.add_argument(
        "--lr-factor", type=float, default=0.5,
        help="ReduceLROnPlateau learning-rate multiplier",
    )
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--samples-per-image", type=int, default=8_192)
    p.add_argument("--max-samples", type=int, default=1_500_000)
    p.add_argument("--eval-samples-per-image", type=int, default=4_096)
    p.add_argument("--max-eval-samples", type=int, default=250_000)
    p.add_argument("--huber-delta", type=float, default=0.125, help="Huber delta in CMYK 0..1 (~32/255)")
    p.add_argument(
        "--luma-weight", type=float, default=1.5,
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
        help="Huber delta in scaled Lab (L/28, a/40, b/55)",
    )
    p.add_argument(
        "--punch-weight", type=float, default=0.35,
        help="hinge so dark pixels are at least as dense as the target; 0 disables",
    )
    p.add_argument(
        "--punch-boost", type=float, default=0.04,
        help="extra CMYK density asked of shadows beyond the target 0..1",
    )
    p.add_argument(
        "--k-punch-weight", type=float, default=0.35,
        help="shadow Huber weight on K vs target; 0 disables",
    )
    p.add_argument(
        "--k-punch-delta", type=float, default=0.125,
        help="Huber delta for shadow K term in CMYK 0..1 (~32/255)",
    )
    p.add_argument(
        "--warmth-weight", type=float, default=0.25,
        help="hinge so midtones are not cooler than the target; 0 disables",
    )
    p.add_argument(
        "--warmth-boost", type=float, default=6.0,
        help="extra Lab b (yellow) asked of midtones beyond the target",
    )
    p.add_argument(
        "--icc-look-weight", type=float, default=0.0,
        help="extra Lab loss: look(ICC baseline sRGB) vs target sRGB; 0 disables "
        "(default 0: this term pushes a display de-gray onto ICC and often worsens CMYK ΔE)",
    )
    p.add_argument(
        "--absolute-tone", action="store_true",
        help="1D curve keyed by absolute luma (v2); default is histogram-relative",
    )
    p.add_argument("--lut-l1", type=float, default=0.01)
    p.add_argument("--smoothness", type=float, default=0.03)
    p.add_argument("--tone-bins", type=int, default=17)
    p.add_argument(
        "--tone-smoothness", type=float, default=0.01,
        help="1D S-curve adjacent-bin smoothness; L1 is not applied to tone",
    )
    p.add_argument("--mask-threshold", type=float, default=0.45)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def wrap_model(args, encoder, icc, loaded, global_encoder, device):
    if args.stage == "global":
        return AdaptiveLUTModel(encoder, icc, {}, device=str(device))
    return AdaptiveLUTModel(
        global_encoder, icc, dict(loaded.metadata), encoder, str(device),
    )


def checkpoint_path(model_path: Path, epoch: int) -> Path:
    return model_path.with_name(f"{model_path.stem}_epoch{epoch:02d}{model_path.suffix}")


def model_metadata(
    args, loaded, region, train_pairs, val_pairs, profile_name, profile_hash,
    status, history, epoch: int,
):
    return {
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
        "epoch": epoch,
        "lr": args.lr,
        "early_stopping_patience": args.early_stopping_patience,
        "lr_scheduler": "ReduceLROnPlateau",
        "lr_patience": args.lr_patience,
        "lr_factor": args.lr_factor,
        "min_lr": args.min_lr,
        "validation_selection_metric": (
            "portrait_mask_delta_e76_mean"
            if args.stage == "portrait" else "full_frame_delta_e76_mean"
        ),
        "huber_delta_cmyk": args.huber_delta,
        "luma_weight": args.luma_weight,
        "cmyk_weight": args.cmyk_weight,
        "appearance_weight": args.appearance_weight,
        "appearance_delta": args.appearance_delta,
        "icc_look_weight": args.icc_look_weight,
        "punch_weight": args.punch_weight,
        "punch_boost": args.punch_boost,
        "k_punch_weight": args.k_punch_weight,
        "k_punch_delta": args.k_punch_delta,
        "warmth_weight": args.warmth_weight,
        "warmth_boost": args.warmth_boost,
        "lut_l1": args.lut_l1,
        "smoothness": args.smoothness,
        "tone_smoothness": args.tone_smoothness,
        "portrait_region": region if args.stage == "portrait" else None,
        "portrait_mask_threshold": args.mask_threshold if args.stage == "portrait" else None,
        "portrait_lut_only": args.portrait_lut_only if args.stage == "portrait" else False,
        "portrait_residual_limits": (
            [
                args.portrait_residual_limit_cmy,
                args.portrait_residual_limit_cmy,
                args.portrait_residual_limit_cmy,
                args.portrait_residual_limit_k,
            ]
            if args.stage == "portrait" and args.portrait_lut_only else None
        ),
        "portrait_dual_mask": (
            args.portrait_dual_mask
            if args.stage == "portrait" and region == "skin" else False
        ),
        "portrait_skin_gate_low": args.portrait_skin_gate_low,
        "portrait_skin_gate_high": args.portrait_skin_gate_high,
        "portrait_neutral_max_regression": args.portrait_neutral_max_regression,
        "portrait_skin_max_regression": args.portrait_skin_max_regression,
        "train_pairs": len(train_pairs),
        "validation_pairs": len(val_pairs),
        "samples_per_image": sample_budget(len(train_pairs), args.samples_per_image, args.max_samples),
        "target_profile": profile_name,
        "target_icc_sha256": profile_hash,
        "embedded_target_icc_status": status,
        "edge_lift": 0.0,
        "edge_lift_c": 0.0,
        "shadow_lift": 0.0,
        "shadow_lift_cmy": 0.0,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "history": history,
    }


def main() -> None:
    args = parse_args()
    torch, _ = _torch()
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio 必须在 [0, 1) 范围")
    if args.grid_size < 2:
        raise ValueError("--grid-size 必须至少为 2")
    if args.epochs < 1:
        raise ValueError("--epochs 必须至少为 1")
    if not 0 < args.portrait_residual_limit_cmy <= 1:
        raise ValueError("--portrait-residual-limit-cmy 必须在 (0, 1] 范围")
    if not 0 < args.portrait_residual_limit_k <= 1:
        raise ValueError("--portrait-residual-limit-k 必须在 (0, 1] 范围")
    if not 0 <= args.portrait_skin_gate_low < args.portrait_skin_gate_high <= 1:
        raise ValueError("portrait skin gate 需要满足 0 <= low < high <= 1")
    if args.portrait_neutral_max_regression < 0:
        raise ValueError("--portrait-neutral-max-regression 不能为负数")
    if args.portrait_skin_max_regression < 0:
        raise ValueError("--portrait-skin-max-regression 不能为负数")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience 不能为负数")
    if args.lr_patience < 0:
        raise ValueError("--lr-patience 不能为负数")
    if not 0 < args.lr_factor < 1:
        raise ValueError("--lr-factor 必须在 (0, 1) 范围")
    if args.min_lr < 0:
        raise ValueError("--min-lr 不能为负数")
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
    if args.punch_weight < 0:
        raise ValueError("--punch-weight 不能为负数")
    if args.punch_boost < 0:
        raise ValueError("--punch-boost 不能为负数")
    if args.k_punch_weight < 0:
        raise ValueError("--k-punch-weight 不能为负数")
    if args.k_punch_delta <= 0:
        raise ValueError("--k-punch-delta 必须为正数")
    if args.warmth_weight < 0:
        raise ValueError("--warmth-weight 不能为负数")
    if args.warmth_boost < 0:
        raise ValueError("--warmth-boost 不能为负数")
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
        if args.portrait_lut_only:
            for head_name in ("tone_head", "look_head"):
                head = getattr(encoder, head_name, None)
                if head is not None:
                    for parameter in head.parameters():
                        parameter.requires_grad_(False)
        if args.region == "person" or args.portrait_dual_mask:
            require_person_segmenter()
        region = args.region
        icc = loaded.target_icc
        profile_name, profile_hash = profile_details(icc)

    transform = build_to_cmyk(icc)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    history = []
    model_path = Path(args.output or args.model)
    best_state = None
    best_val_metrics = None
    best_val_delta_e = float("inf")
    best_epoch = 0
    portrait_global_val_delta_e = None
    last_val_metrics = None
    epochs_without_improvement = 0
    stopped_early = False
    completed_epochs = 0
    if not val_pairs:
        print(
            "warning: no validation pairs; validation checkpoint selection and early stopping are disabled",
            flush=True,
        )
    print(
        f"objective: adaptive CMYK v3 | stage={args.stage} | "
        f"hist+rel-1D={args.tone_bins} + lut={args.grid_size}³×4 chroma + look | "
        f"huber={args.huber_delta:g} | luma-weight={args.luma_weight:g} | "
        f"cmyk={args.cmyk_weight:g} appearance={args.appearance_weight:g} "
        f"punch={args.punch_weight:g} k-punch={args.k_punch_weight:g} warmth={args.warmth_weight:g} "
        f"icc-look={args.icc_look_weight:g}"
        + (
            f" | portrait LUT-only limits="
            f"[{args.portrait_residual_limit_cmy:g}×CMY, "
            f"{args.portrait_residual_limit_k:g}×K]"
            if args.stage == "portrait" and args.portrait_lut_only else ""
        )
        + (
            f" | dual-mask CMY=skin[{args.portrait_skin_gate_low:g},"
            f"{args.portrait_skin_gate_high:g}] K=person"
            if args.stage == "portrait" and args.region == "skin"
            and args.portrait_dual_mask else ""
        )
    )
    for epoch in range(1, args.epochs + 1):
        completed_epochs = epoch
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        print(f"epoch {epoch}/{args.epochs} stage={args.stage}")
        mean_loss = train_epoch(
            encoder, optimizer, train_pairs, transform, args, device,
            args.seed + epoch * 17, icc, global_encoder, region,
        )
        print(f"epoch {epoch} mean loss={mean_loss:.5f}")
        epoch_model = wrap_model(args, encoder, icc, loaded, global_encoder, device)
        epoch_metadata = model_metadata(
            args, loaded, region, train_pairs, val_pairs,
            profile_name, profile_hash, status, history, epoch,
        )
        epoch_model.metadata = epoch_metadata
        epoch_val_metrics = evaluate_pairs(
            epoch_model, val_pairs, icc,
            args.eval_samples_per_image, args.max_eval_samples,
            args.seed + 2_000_000, f"val epoch {epoch}",
        )
        last_val_metrics = epoch_val_metrics
        history_item = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "lr": epoch_lr,
        }
        improved = False
        if epoch_val_metrics is not None:
            if args.stage == "portrait":
                portrait_metrics = epoch_val_metrics["portrait_mask"]
                val_delta_e = float(
                    portrait_metrics["global_plus_portrait"]["delta_e76"]["mean"]
                )
                portrait_global_val_delta_e = float(
                    portrait_metrics["global_only"]["delta_e76"]["mean"]
                )
                history_item["portrait_mask_global_only_delta_e76_mean"] = (
                    portrait_global_val_delta_e
                )
                history_item["portrait_mask_delta_e76_mean"] = val_delta_e
                beats_global = val_delta_e < portrait_global_val_delta_e
                history_item["beats_global_only"] = beats_global
                skin_guard = True
                if "skin_mask" in epoch_val_metrics:
                    skin_metrics = epoch_val_metrics["skin_mask"]
                    skin_global_de = float(
                        skin_metrics["global_only"]["delta_e76"]["mean"]
                    )
                    skin_model_de = float(
                        skin_metrics["global_plus_portrait"]["delta_e76"]["mean"]
                    )
                    skin_guard = skin_model_de <= skin_global_de * (
                        1.0 + args.portrait_skin_max_regression
                    )
                    history_item["skin_mask_global_only_delta_e76_mean"] = skin_global_de
                    history_item["skin_mask_delta_e76_mean"] = skin_model_de
                neutral_guard = True
                if "neutral_white_mask" in epoch_val_metrics:
                    neutral_metrics = epoch_val_metrics["neutral_white_mask"]
                    neutral_global_de = float(
                        neutral_metrics["global_only"]["delta_e76"]["mean"]
                    )
                    neutral_model_de = float(
                        neutral_metrics["global_plus_portrait"]["delta_e76"]["mean"]
                    )
                    neutral_guard = neutral_model_de <= neutral_global_de * (
                        1.0 + args.portrait_neutral_max_regression
                    )
                    history_item["neutral_white_global_only_delta_e76_mean"] = (
                        neutral_global_de
                    )
                    history_item["neutral_white_delta_e76_mean"] = neutral_model_de
                history_item["skin_guard_passed"] = skin_guard
                history_item["neutral_white_guard_passed"] = neutral_guard
                improved = (
                    beats_global and skin_guard and neutral_guard
                    and val_delta_e < best_val_delta_e
                )
            else:
                val_delta_e = float(epoch_val_metrics["icc_plus_lut"]["delta_e76"]["mean"])
                improved = val_delta_e < best_val_delta_e
            history_item["validation_delta_e76_mean"] = val_delta_e
            history_item["improved"] = improved
            scheduler.step(val_delta_e)
            history_item["next_lr"] = float(optimizer.param_groups[0]["lr"])
            if improved:
                best_val_delta_e = val_delta_e
                best_epoch = epoch
                best_val_metrics = epoch_val_metrics
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in encoder.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            scheduler.step(mean_loss)
            history_item["next_lr"] = float(optimizer.param_groups[0]["lr"])
        history.append(history_item)
        epoch_metadata["history"] = history
        epoch_metadata["validation_metrics"] = epoch_val_metrics
        epoch_metadata["best_epoch"] = best_epoch or None
        epoch_metadata["best_validation_delta_e76_mean"] = (
            best_val_delta_e if best_epoch else None
        )
        epoch_model.metadata = epoch_metadata
        if improved:
            epoch_model.save(model_path)
            print(
                f"saved best model: {model_path.resolve()} (epoch={epoch}, "
                f"{'portrait-mask ' if args.stage == 'portrait' else ''}"
                f"val DeltaE76={best_val_delta_e:.4f})",
                flush=True,
            )
        if args.save_every_epoch:
            ckpt = checkpoint_path(model_path, epoch)
            epoch_model.save(ckpt)
            print(f"saved checkpoint: {ckpt.resolve()}", flush=True)
        if (
            epoch_val_metrics is not None
            and args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            stopped_early = True
            print(
                f"early stopping: validation DeltaE76 did not improve for "
                f"{epochs_without_improvement} consecutive epochs; "
                f"best epoch={best_epoch or 'global-only baseline'}",
                flush=True,
            )
            break

    if best_state is not None:
        encoder.load_state_dict(best_state)
        print(
            f"restored best weights from epoch {best_epoch} "
            f"(val DeltaE76={best_val_delta_e:.4f})",
            flush=True,
        )

    portrait_rejected = (
        args.stage == "portrait" and bool(val_pairs) and best_state is None
    )
    if portrait_rejected:
        print(
            "portrait branch rejected: no epoch improved validation DeltaE76 "
            "inside the portrait mask; saving global-only fallback",
            flush=True,
        )
        model = AdaptiveLUTModel(
            global_encoder, icc, dict(loaded.metadata), device=str(device),
        )
    else:
        model = wrap_model(args, encoder, icc, loaded, global_encoder, device)
    metadata = model_metadata(
        args, loaded, region, train_pairs, val_pairs,
        profile_name, profile_hash, status, history, best_epoch or completed_epochs,
    )
    metadata["completed_epochs"] = completed_epochs
    metadata["best_epoch"] = best_epoch or None
    metadata["best_validation_delta_e76_mean"] = best_val_delta_e if best_epoch else None
    metadata["stopped_early"] = stopped_early
    metadata["portrait_accepted"] = (
        None if args.stage != "portrait" or not val_pairs else not portrait_rejected
    )
    metadata["portrait_global_only_validation_delta_e76_mean"] = (
        portrait_global_val_delta_e
    )
    model.metadata = metadata
    report_path = Path(args.report) if args.report else model_path.with_suffix(".report.json")
    train_metrics = evaluate_pairs(
        model, train_pairs, icc, args.eval_samples_per_image, args.max_eval_samples,
        args.seed + 1_000_000, "train",
    )
    # The same deterministic validation sample was already evaluated when the
    # best checkpoint was selected. A rejected portrait branch is evaluated
    # once more because the saved output is the global-only fallback.
    if portrait_rejected:
        metadata["rejected_portrait_validation_metrics"] = last_val_metrics
        val_metrics = evaluate_pairs(
            model, val_pairs, icc,
            args.eval_samples_per_image, args.max_eval_samples,
            args.seed + 2_000_000, "val global-only fallback",
        )
    else:
        val_metrics = best_val_metrics
    metadata["train_metrics"] = train_metrics
    metadata["validation_metrics"] = val_metrics
    model.metadata = metadata
    model.save(model_path)
    # Per-image ICC records can dominate the report when a fixed target ICC is
    # intentionally assigned to many files carrying a different embedded
    # profile. Keep only the compact status counts in
    # ``embedded_target_icc_status``; validation/training metrics stay easy to
    # inspect near the top-level report.
    report = metadata | {"model": model_path.name}
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
        if args.stage == "portrait" and not portrait_rejected:
            portrait_metrics = val_metrics["portrait_mask"]
            portrait_before = portrait_metrics["global_only"]["delta_e76"]
            portrait_after = portrait_metrics["global_plus_portrait"]["delta_e76"]
            print(
                f"val portrait-mask ΔE76 global/model mean: "
                f"{portrait_before['mean']:.3f} / {portrait_after['mean']:.3f}"
            )
            print(
                f"val portrait-mask improvement: "
                f"{portrait_metrics['delta_e76_improvement_percent']:.2f}%"
            )
            if "skin_mask" in val_metrics:
                skin_metrics = val_metrics["skin_mask"]
                print(
                    f"val skin-mask ΔE76 global/model mean: "
                    f"{skin_metrics['global_only']['delta_e76']['mean']:.3f} / "
                    f"{skin_metrics['global_plus_portrait']['delta_e76']['mean']:.3f} "
                    f"| model mean ΔL={skin_metrics['global_plus_portrait_mean_delta_l']:.3f}"
                )
            if "neutral_white_mask" in val_metrics:
                neutral_metrics = val_metrics["neutral_white_mask"]
                print(
                    f"val neutral-white ΔE76 global/model mean: "
                    f"{neutral_metrics['global_only']['delta_e76']['mean']:.3f} / "
                    f"{neutral_metrics['global_plus_portrait']['delta_e76']['mean']:.3f} "
                    f"| model mean ΔL="
                    f"{neutral_metrics['global_plus_portrait_mean_delta_l']:.3f}"
                )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
