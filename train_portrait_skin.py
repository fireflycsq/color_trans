#!/usr/bin/env python3
"""Train a portrait residual LUT on top of an existing global residual LUT.

Stage 1 grades the whole image. Stage 2 learns target_CMYK − global_CMYK
inside a cut-out person mask (skin, hair, and clothing by default), then
inference applies that correction only on the person.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from color_model import image_to_srgb, load_color_model
from portrait_mask import detector_name, portrait_mask, require_person_segmenter
from residual_lut_model import ResidualLUTModel, trilinear_lookup
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
from train_residual_lut import (
    build_to_cmyk,
    finish_lut,
    metric_summary,
    predict_samples,
    splat,
)


def node_agreement(
    sum_sq: np.ndarray, weights: np.ndarray, mean: np.ndarray, sigma_255: float,
) -> np.ndarray:
    """Per-node agreement in [0, 1]; high residual variance across photos lowers it."""
    sigma = max(float(sigma_255) / 255.0, 1e-8)
    var = np.divide(
        sum_sq, weights[..., None], out=np.zeros_like(sum_sq), where=weights[..., None] > 0,
    )
    var -= np.square(mean)
    std = np.sqrt(np.maximum(var, 0.0)).max(axis=-1)
    return (1.0 / (1.0 + std / sigma)).astype(np.float32)


def sample_portrait_residual(
    pair: Pair, count: int, seed: int, transform, lut: np.ndarray, confidence: np.ndarray,
    mask_threshold: float = 0.45, region: str = "person",
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """RGB and target-minus-global-CMYK residual on portrait pixels."""
    with Image.open(pair.source) as source:
        rgb_u8 = np.asarray(image_to_srgb(source), dtype=np.uint8)
    with Image.open(pair.target) as target:
        target_u8 = np.asarray(target.convert("CMYK"), dtype=np.uint8)
    mask = portrait_mask(rgb_u8, region=region)
    eligible = np.flatnonzero(mask.reshape(-1) >= mask_threshold)
    if eligible.size < 32:
        return None
    take = min(count, int(eligible.size))
    person_rgb = rgb_u8.reshape(-1, 3)[eligible][:, None, :]
    local = stratified_indices(person_rgb, take, seed)
    idx = eligible[local]
    rgb = rgb_u8.reshape(-1, 3)[idx].astype(np.float32) / 255.0
    sampled_target = target_u8.reshape(-1, 4)[idx].astype(np.float32) / 255.0
    _, global_pred = predict_samples(rgb, transform, lut, confidence)
    return rgb, sampled_target - global_pred, float(mask.mean())


def accumulate_portrait(
    pairs: list[Pair], size: int, per_image: int, maximum: int, seed: int,
    transform, lut: np.ndarray, confidence: np.ndarray, clip: np.ndarray,
    reference: np.ndarray | None = None, huber_delta: float = 32 / 255,
    mask_threshold: float = 0.45, region: str = "person",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, dict]:
    sums = np.zeros((size, size, size, 4), dtype=np.float64)
    weights = np.zeros((size, size, size), dtype=np.float64)
    sum_sq = np.zeros((size, size, size, 4), dtype=np.float64)
    budget = sample_budget(len(pairs), per_image, maximum)
    total = 0
    used = 0
    abs_sum = np.zeros(4, dtype=np.float64)
    max_abs = np.zeros(4, dtype=np.float64)
    exceeding_clip = 0
    for i, pair in enumerate(pairs, 1):
        sampled = sample_portrait_residual(
            pair, budget, seed + i * 104729, transform, lut, confidence,
            mask_threshold, region,
        )
        if sampled is None:
            print(f"[portrait pass {'2' if reference is not None else '1'} {i:>4}/{len(pairs)}] "
                  f"{pair.name}: no {region} pixels")
            continue
        rgb, residual, mask_mean = sampled
        abs_residual = np.abs(residual)
        abs_sum += abs_residual.sum(axis=0)
        max_abs = np.maximum(max_abs, abs_residual.max(axis=0))
        exceeding_clip += int(np.any(abs_residual > clip + 1e-8, axis=1).sum())
        residual = np.clip(residual, -clip, clip)
        robust = np.ones(len(rgb), dtype=np.float32)
        if reference is not None:
            deviation = np.max(np.abs(residual - trilinear_lookup(reference, rgb)), axis=1)
            robust = np.minimum(1.0, huber_delta / np.maximum(deviation, 1e-8))
        splat(sums, weights, rgb, residual, robust, sum_sq if reference is None else None)
        total += len(rgb)
        used += 1
        print(
            f"[portrait pass {'2' if reference is not None else '1'} {i:>4}/{len(pairs)}] "
            f"{pair.name}: {len(rgb):,} {region} samples, mask_mean={mask_mean:.3f}, "
            f"mean weight={robust.mean():.3f}"
        )
    stats = {
        "mean_abs_255": [float(x) for x in abs_sum / max(total, 1) * 255],
        "max_abs_255": [float(x) for x in max_abs * 255],
        "samples_exceeding_clip": exceeding_clip,
        "samples": total,
        "pairs_with_portrait": used,
    }
    return sums, weights, sum_sq, total, used, stats


def apply_portrait_defaults(args: argparse.Namespace) -> None:
    """Fill unspecified hyperparameters; default is to follow the human portrait grade."""
    human = args.fit_human
    if args.confidence_samples is None:
        args.confidence_samples = 4.0 if human else 8.0
    if args.smoothness is None:
        args.smoothness = 0.02 if human else 0.06
    if args.baseline_regularization is None:
        args.baseline_regularization = 0.0 if human else 0.02
    if args.huber_delta is None:
        args.huber_delta = 64.0 if human else 32.0
    if args.agreement_sigma is None:
        args.agreement_sigma = 48.0 if human else 8.0
    if args.max_cmy_residual is None:
        args.max_cmy_residual = 255.0
    if args.max_k_residual is None:
        args.max_k_residual = 255.0


def evaluate_portrait(
    pairs: list[Pair], global_lut: np.ndarray, global_confidence: np.ndarray,
    skin_lut: np.ndarray, skin_confidence: np.ndarray, icc: bytes,
    per_image: int, maximum: int, seed: int, label: str, mask_threshold: float,
    region: str = "person",
) -> dict | None:
    if not pairs:
        return None
    transform = build_to_cmyk(icc)
    budget = sample_budget(len(pairs), per_image, maximum)
    globals_, predictions, targets = [], [], []
    per_pair = []
    for i, pair in enumerate(pairs, 1):
        sampled = sample_portrait_residual(
            pair, budget, seed + i * 130363, transform, global_lut, global_confidence,
            mask_threshold, region,
        )
        if sampled is None:
            print(f"[{label} {i:>4}/{len(pairs)}] {pair.name}: no {region} pixels")
            continue
        rgb, residual, _ = sampled
        _, global_pred = predict_samples(rgb, transform, global_lut, global_confidence)
        target = np.clip(global_pred + residual, 0, 1)
        skin_corr = (
            trilinear_lookup(skin_lut, rgb)
            * trilinear_lookup(skin_confidence, rgb)[..., None]
        )
        prediction = np.clip(global_pred + skin_corr, 0, 1)
        pair_global = metric_summary(global_pred, target, icc)
        pair_model = metric_summary(prediction, target, icc)
        delta = pair_model["delta_e76"]["mean"] - pair_global["delta_e76"]["mean"]
        per_pair.append({
            "name": pair.name, "samples": len(rgb),
            "global_delta_e76_mean": pair_global["delta_e76"]["mean"],
            "portrait_delta_e76_mean": pair_model["delta_e76"]["mean"],
            "delta_e76_change": delta,
            "portrait_cmyk_mae": pair_model["cmyk_mae"],
        })
        globals_.append(global_pred)
        predictions.append(prediction)
        targets.append(target)
        flag = " WORSE" if delta > 0.02 else ""
        print(
            f"[{label} {i:>4}/{len(pairs)}] {pair.name}: "
            f"global ΔE={pair_global['delta_e76']['mean']:.3f}, "
            f"portrait ΔE={pair_model['delta_e76']['mean']:.3f}, "
            f"Δ={delta:+.3f}{flag}"
        )
    if not targets:
        return None
    global_metrics = metric_summary(np.concatenate(globals_), np.concatenate(targets), icc)
    model_metrics = metric_summary(np.concatenate(predictions), np.concatenate(targets), icc)
    base_de, model_de = global_metrics["delta_e76"]["mean"], model_metrics["delta_e76"]["mean"]
    n_worse = sum(1 for x in per_pair if x["delta_e76_change"] > 0.02)
    n_better = sum(1 for x in per_pair if x["delta_e76_change"] < -0.02)
    n_same = len(per_pair) - n_worse - n_better
    print(f"[{label}] better={n_better} worse={n_worse} same={n_same}")
    per_pair.sort(key=lambda x: x["delta_e76_change"], reverse=True)
    return {
        "pairs": len(per_pair), "samples": sum(len(x) for x in targets),
        "global_only": global_metrics, "global_plus_portrait": model_metrics,
        "delta_e76_improvement_percent": float(100 * (base_de - model_de) / base_de) if base_de else 0.0,
        "pairs_better": n_better, "pairs_worse": n_worse, "pairs_same": n_same,
        "worst_pairs": per_pair[:min(20, len(per_pair))],
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
    p.add_argument("--model", required=True, help="existing global residual-LUT .npz")
    p.add_argument("--output", help="output .npz; default overwrites --model")
    p.add_argument("--report")
    p.add_argument("--target-icc", help="defaults to the ICC embedded in --model")
    p.add_argument("--grid-size", type=int, default=17)
    p.add_argument("--samples-per-image", type=int, default=20_000)
    p.add_argument("--max-samples", type=int, default=1_500_000)
    p.add_argument("--eval-samples-per-image", type=int, default=8_000)
    p.add_argument("--max-eval-samples", type=int, default=250_000)
    p.add_argument("--confidence-samples", type=float, default=None)
    p.add_argument("--smoothness", type=float, default=None)
    p.add_argument("--baseline-regularization", type=float, default=None)
    p.add_argument("--smooth-iterations", type=int, default=40)
    p.add_argument("--huber-delta", type=float, default=None)
    p.add_argument("--max-cmy-residual", type=float, default=None)
    p.add_argument("--max-k-residual", type=float, default=None)
    p.add_argument("--mask-threshold", type=float, default=0.45)
    p.add_argument("--agreement-sigma", type=float, default=None)
    p.add_argument(
        "--fit-human", action=argparse.BooleanOptionalAction, default=True,
        help="follow target portrait CMYK closely (default). Use --no-fit-human for safer fallback",
    )
    p.add_argument(
        "--region", choices=("person", "skin"), default="person",
        help="person: 整个人像含头发和服装（默认）；skin: 仅皮肤",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_portrait_defaults(args)
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio 必须在 [0, 1) 范围")
    if not 0 < args.mask_threshold <= 1:
        raise ValueError("--mask-threshold 必须在 (0, 1] 范围")
    if args.grid_size < 2:
        raise ValueError("--grid-size 必须至少为 2")
    if args.agreement_sigma <= 0:
        raise ValueError("--agreement-sigma 必须大于 0")
    loaded = load_color_model(args.model)
    if not isinstance(loaded, ResidualLUTModel):
        raise ValueError("人像阶段需要残差 LUT 模型，不能用旧多项式模型")
    if args.region == "person":
        require_person_segmenter()

    train_pairs, val_pairs = collect_pairs(args)
    print(f"pairs: train={len(train_pairs)}, validation={len(val_pairs)}")
    print(
        f"portrait objective: {'fit_human' if args.fit_human else 'safe_fallback'} | "
        f"region={args.region} | detector={detector_name(args.region)} | "
        f"agreement_sigma={args.agreement_sigma:g} huber={args.huber_delta:g} "
        f"smoothness={args.smoothness:g} baseline_reg={args.baseline_regularization:g}"
    )
    if args.target_icc:
        fixed_icc = load_fixed_target_icc(Path(args.target_icc))
    else:
        fixed_icc = loaded.target_icc
    target_icc, train_records = validate_pairs(train_pairs, fixed_icc=fixed_icc)
    profile_name, profile_hash = profile_details(target_icc)
    val_records = []
    if val_pairs:
        _, val_records = validate_pairs(val_pairs, profile_hash, fixed_icc=fixed_icc)
    status = icc_status_counts(train_records + val_records)

    transform = build_to_cmyk(target_icc)
    clip = np.array([args.max_cmy_residual] * 3 + [args.max_k_residual], dtype=np.float32) / 255
    sums, weights, sum_sq, train_samples, used, residual_stats = accumulate_portrait(
        train_pairs, args.grid_size, args.samples_per_image, args.max_samples,
        args.seed, transform, loaded.lut, loaded.confidence, clip,
        mask_threshold=args.mask_threshold, region=args.region,
    )
    if train_samples == 0:
        raise ValueError("训练集里没有检测人人像像素，无法训练第二阶段")
    print(
        f"{args.region} |ΔCMYK vs global| mean: "
        + ", ".join(f"{x:.1f}" for x in residual_stats["mean_abs_255"])
        + f" | pairs with portrait: {used}/{len(train_pairs)}"
    )
    pass1_mean = np.divide(
        sums, weights[..., None], out=np.zeros_like(sums), where=weights[..., None] > 0,
    )
    agreement = node_agreement(sum_sq, weights, pass1_mean, args.agreement_sigma)
    initial_lut, _ = finish_lut(
        sums, weights, args.confidence_samples, args.smoothness,
        args.baseline_regularization, args.smooth_iterations, clip,
    )
    sums, weights, _, train_samples, used, _ = accumulate_portrait(
        train_pairs, args.grid_size, args.samples_per_image, args.max_samples,
        args.seed, transform, loaded.lut, loaded.confidence, clip,
        initial_lut, args.huber_delta / 255, args.mask_threshold, args.region,
    )
    skin_lut, coverage = finish_lut(
        sums, weights, args.confidence_samples, args.smoothness,
        args.baseline_regularization, args.smooth_iterations, clip,
    )
    gate = np.sqrt(np.clip(agreement, 0.0, 1.0)) if args.fit_human else agreement
    skin_confidence = (coverage * gate).astype(np.float32)
    covered = weights > 0
    print(
        f"portrait node agreement mean={float(agreement[covered].mean()) if np.any(covered) else 0:.3f} "
        f"| coverage mean={float(coverage.mean()):.3f} "
        f"| gated confidence mean={float(skin_confidence.mean()):.3f}"
    )

    train_metrics = evaluate_portrait(
        train_pairs, loaded.lut, loaded.confidence, skin_lut, skin_confidence,
        target_icc, args.eval_samples_per_image, args.max_eval_samples,
        args.seed + 1_000_000, "train-portrait", args.mask_threshold, args.region,
    )
    val_metrics = evaluate_portrait(
        val_pairs, loaded.lut, loaded.confidence, skin_lut, skin_confidence,
        target_icc, args.eval_samples_per_image, args.max_eval_samples,
        args.seed + 2_000_000, "val-portrait", args.mask_threshold, args.region,
    )

    metadata = dict(loaded.metadata)
    metadata.update({
        "portrait_fit_human": args.fit_human,
        "portrait_skin": args.region == "skin",
        "portrait_region": args.region,
        "portrait_detector": detector_name(args.region),
        "portrait_grid_size": args.grid_size,
        "portrait_train_pairs": len(train_pairs),
        "portrait_validation_pairs": len(val_pairs),
        "portrait_training_samples": train_samples,
        "portrait_pairs_with_subject": used,
        "portrait_confidence_samples": args.confidence_samples,
        "portrait_smoothness": args.smoothness,
        "portrait_baseline_regularization": args.baseline_regularization,
        "portrait_huber_delta_255": args.huber_delta,
        "portrait_agreement_sigma_255": args.agreement_sigma,
        "portrait_mean_node_agreement": float(agreement[covered].mean()) if np.any(covered) else 0.0,
        "portrait_residual_limits_255": [args.max_cmy_residual] * 3 + [args.max_k_residual],
        "portrait_mask_threshold": args.mask_threshold,
        "portrait_residual_stats": residual_stats,
        "portrait_embedded_target_icc_status": status,
        "portrait_created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "skin_residual_strength": 1.0,
        "portrait_lut_nodes_with_samples": int(np.count_nonzero(weights)),
        "portrait_mean_node_confidence": float(skin_confidence.mean()),
        "portrait_train_metrics": train_metrics,
        "portrait_validation_metrics": val_metrics,
        "target_profile": profile_name,
        "target_icc_sha256": profile_hash,
    })
    model = ResidualLUTModel(
        loaded.lut, loaded.confidence, target_icc, metadata, skin_lut, skin_confidence,
    )
    model_path = Path(args.output or args.model)
    report_path = Path(args.report) if args.report else model_path.with_name(
        model_path.stem + ".portrait.report.json"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    report = metadata | {
        "model": model_path.name,
        "base_model": Path(args.model).name,
        "train_images": train_records,
        "validation_images": val_records,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(f"saved model: {model_path.resolve()}")
    print(f"saved report: {report_path.resolve()}")
    if val_metrics:
        before = val_metrics["global_only"]["delta_e76"]
        after = val_metrics["global_plus_portrait"]["delta_e76"]
        print(f"portrait val ΔE76 global mean/p95: {before['mean']:.3f} / {before['p95']:.3f}")
        print(f"portrait val ΔE76 +skin mean/p95: {after['mean']:.3f} / {after['p95']:.3f}")
        print(f"portrait val mean improvement: {val_metrics['delta_e76_improvement_percent']:.2f}%")
        print(
            f"portrait val pairs better/worse/same: "
            f"{val_metrics['pairs_better']}/{val_metrics['pairs_worse']}/{val_metrics['pairs_same']}"
        )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
