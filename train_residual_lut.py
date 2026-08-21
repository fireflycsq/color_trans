#!/usr/bin/env python3
"""Train a fixed-ICC baseline plus confidence-gated 3D CMYK residual LUT."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from color_model import image_to_srgb, profile_from_bytes, srgb_to_lab
from residual_lut_model import ResidualLUTModel, trilinear_lookup
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


def build_to_cmyk(icc: bytes) -> ImageCms.ImageCmsTransform:
    return ImageCms.buildTransform(
        ImageCms.createProfile("sRGB"), profile_from_bytes(icc), "RGB", "CMYK",
        renderingIntent=1, flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
    )


def sample_residual(
    pair: Pair, count: int, seed: int, transform: ImageCms.ImageCmsTransform,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalised input RGB and target-minus-ICC CMYK residual."""
    with Image.open(pair.source) as source:
        rgb_u8 = np.asarray(image_to_srgb(source), dtype=np.uint8)
    with Image.open(pair.target) as target:
        target_u8 = np.asarray(target.convert("CMYK"), dtype=np.uint8)
    idx = stratified_indices(rgb_u8, min(count, rgb_u8.shape[0] * rgb_u8.shape[1]), seed)
    sampled_rgb = rgb_u8.reshape(-1, 3)[idx]
    sampled_target = target_u8.reshape(-1, 4)[idx].astype(np.float32) / 255.0
    strip = Image.fromarray(sampled_rgb[None, ...], "RGB")
    baseline = np.asarray(ImageCms.applyTransform(strip, transform), dtype=np.float32)[0] / 255.0
    return sampled_rgb.astype(np.float32) / 255.0, sampled_target - baseline


def splat(
    sums: np.ndarray, weights: np.ndarray, rgb: np.ndarray, residual: np.ndarray,
    sample_weights: np.ndarray, sum_sq: np.ndarray | None = None,
) -> None:
    """Trilinearly distribute samples into their eight neighbouring nodes."""
    size = sums.shape[0]
    position = np.clip(rgb, 0, 1) * (size - 1)
    lower = np.floor(position).astype(np.intp)
    upper = np.minimum(lower + 1, size - 1)
    fraction = position - lower
    for dr in (0, 1):
        ir = upper[:, 0] if dr else lower[:, 0]
        wr = fraction[:, 0] if dr else 1 - fraction[:, 0]
        for dg in (0, 1):
            ig = upper[:, 1] if dg else lower[:, 1]
            wg = fraction[:, 1] if dg else 1 - fraction[:, 1]
            for db in (0, 1):
                ib = upper[:, 2] if db else lower[:, 2]
                wb = fraction[:, 2] if db else 1 - fraction[:, 2]
                w = wr * wg * wb * sample_weights
                np.add.at(weights, (ir, ig, ib), w)
                np.add.at(sums, (ir, ig, ib), residual * w[:, None])
                if sum_sq is not None:
                    np.add.at(sum_sq, (ir, ig, ib), np.square(residual) * w[:, None])


def accumulate(
    pairs: list[Pair], size: int, per_image: int, maximum: int, seed: int,
    transform: ImageCms.ImageCmsTransform, clip: np.ndarray,
    reference: np.ndarray | None = None, huber_delta: float = 8 / 255,
) -> tuple[np.ndarray, np.ndarray, int, dict]:
    sums = np.zeros((size, size, size, 4), dtype=np.float64)
    weights = np.zeros((size, size, size), dtype=np.float64)
    budget = sample_budget(len(pairs), per_image, maximum)
    total = 0
    abs_sum = np.zeros(4, dtype=np.float64)
    max_abs = np.zeros(4, dtype=np.float64)
    exceeding_clip = 0
    for i, pair in enumerate(pairs, 1):
        rgb, residual = sample_residual(pair, budget, seed + i * 104729, transform)
        abs_residual = np.abs(residual)
        abs_sum += abs_residual.sum(axis=0)
        if len(abs_residual):
            max_abs = np.maximum(max_abs, abs_residual.max(axis=0))
        exceeding_clip += int(np.any(abs_residual > clip + 1e-8, axis=1).sum())
        residual = np.clip(residual, -clip, clip)
        robust = np.ones(len(rgb), dtype=np.float32)
        if reference is not None:
            deviation = np.max(np.abs(residual - trilinear_lookup(reference, rgb)), axis=1)
            robust = np.minimum(1.0, huber_delta / np.maximum(deviation, 1e-8))
        splat(sums, weights, rgb, residual, robust)
        total += len(rgb)
        print(
            f"[LUT pass {'2' if reference is not None else '1'} {i:>4}/{len(pairs)}] "
            f"{pair.name}: {len(rgb):,} samples, mean weight={robust.mean():.3f}"
        )
    stats = {
        "mean_abs_255": [float(x) for x in abs_sum / max(total, 1) * 255],
        "max_abs_255": [float(x) for x in max_abs * 255],
        "samples_exceeding_clip": exceeding_clip,
        "samples": total,
    }
    return sums, weights, total, stats


def apply_objective_defaults(args: argparse.Namespace) -> None:
    """Fill unspecified hyperparameters; --fit-human prefers matching the target CMYK."""
    human = args.fit_human
    if args.max_cmy_residual is None:
        args.max_cmy_residual = 255.0 if human else 20.0
    if args.max_k_residual is None:
        args.max_k_residual = 255.0 if human else 15.0
    if args.baseline_regularization is None:
        args.baseline_regularization = 0.02 if human else 0.25
    if args.smoothness is None:
        args.smoothness = 0.06 if human else 0.12
    if args.huber_delta is None:
        args.huber_delta = 32.0 if human else 8.0
    if args.confidence_samples is None:
        args.confidence_samples = 8.0 if human else 32.0


def neighbour_sum(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros_like(values)
    count = np.zeros(values.shape[:3], dtype=np.float32)
    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis], right[axis] = slice(1, None), slice(None, -1)
        left_t, right_t = tuple(left), tuple(right)
        total[left_t] += values[right_t]
        total[right_t] += values[left_t]
        count[left_t] += 1
        count[right_t] += 1
    return total, count


def finish_lut(
    sums: np.ndarray, weights: np.ndarray, confidence_samples: float,
    smoothness: float, baseline_regularization: float, iterations: int,
    clip: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.divide(
        sums, weights[..., None], out=np.zeros_like(sums), where=weights[..., None] > 0
    ).astype(np.float32)
    confidence = (weights / (weights + confidence_samples)).astype(np.float32)
    lut = mean.copy()
    data = confidence
    for _ in range(iterations):
        adjacent, adjacent_count = neighbour_sum(lut)
        numerator = data[..., None] * mean + smoothness * adjacent
        denominator = (
            data + smoothness * adjacent_count
            + baseline_regularization * (1.0 - data)
        )
        lut = numerator / np.maximum(denominator[..., None], 1e-8)
        lut = np.clip(lut, -clip, clip)
    return lut.astype(np.float32), confidence


def predict_samples(
    rgb: np.ndarray, transform: ImageCms.ImageCmsTransform,
    lut: np.ndarray, confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rgb_u8 = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    strip = Image.fromarray(rgb_u8[None, ...], "RGB")
    baseline = np.asarray(ImageCms.applyTransform(strip, transform), dtype=np.float32)[0] / 255.0
    correction = trilinear_lookup(lut, rgb)
    trust = trilinear_lookup(confidence, rgb)
    prediction = np.clip(baseline + trust[:, None] * correction, 0, 1)
    return baseline, prediction


def metric_summary(pred: np.ndarray, target: np.ndarray, icc: bytes) -> dict:
    error = pred - target
    mae = np.mean(np.abs(error), axis=0) * 255
    rmse = float(np.sqrt(np.mean(np.square(error)))) * 255
    pred_rgb, target_rgb = render_samples(pred, icc), render_samples(target, icc)
    de = np.linalg.norm(srgb_to_lab(pred_rgb) - srgb_to_lab(target_rgb), axis=-1)
    return {
        "cmyk_mae": [float(x) for x in mae],
        "cmyk_psnr": float(20 * np.log10(255 / rmse)) if rmse else None,
        "delta_e76": {
            "mean": float(de.mean()), "p50": float(np.percentile(de, 50)),
            "p95": float(np.percentile(de, 95)), "max": float(de.max()),
        },
    }


def evaluate(
    pairs: list[Pair], lut: np.ndarray, confidence: np.ndarray, icc: bytes,
    per_image: int, maximum: int, seed: int, label: str,
) -> dict | None:
    if not pairs:
        return None
    transform = build_to_cmyk(icc)
    budget = sample_budget(len(pairs), per_image, maximum)
    baselines, predictions, targets = [], [], []
    per_pair = []
    for i, pair in enumerate(pairs, 1):
        rgb, residual = sample_residual(pair, budget, seed + i * 130363, transform)
        rgb_u8 = np.rint(rgb * 255).astype(np.uint8)
        strip = Image.fromarray(rgb_u8[None, ...], "RGB")
        baseline = np.asarray(ImageCms.applyTransform(strip, transform), dtype=np.float32)[0] / 255
        target = np.clip(baseline + residual, 0, 1)
        baseline, prediction = predict_samples(rgb, transform, lut, confidence)
        pair_baseline = metric_summary(baseline, target, icc)
        pair_model = metric_summary(prediction, target, icc)
        per_pair.append({
            "name": pair.name, "samples": len(rgb),
            "baseline_delta_e76_mean": pair_baseline["delta_e76"]["mean"],
            "model_delta_e76_mean": pair_model["delta_e76"]["mean"],
            "model_cmyk_mae": pair_model["cmyk_mae"],
        })
        baselines.append(baseline); predictions.append(prediction); targets.append(target)
        print(
            f"[{label} {i:>4}/{len(pairs)}] {pair.name}: "
            f"baseline ΔE={pair_baseline['delta_e76']['mean']:.3f}, "
            f"LUT ΔE={pair_model['delta_e76']['mean']:.3f}"
        )
    baseline_metrics = metric_summary(np.concatenate(baselines), np.concatenate(targets), icc)
    model_metrics = metric_summary(np.concatenate(predictions), np.concatenate(targets), icc)
    base_de, model_de = baseline_metrics["delta_e76"]["mean"], model_metrics["delta_e76"]["mean"]
    per_pair.sort(key=lambda x: x["model_delta_e76_mean"], reverse=True)
    return {
        "pairs": len(pairs), "samples": sum(len(x) for x in targets),
        "icc_baseline": baseline_metrics, "icc_plus_lut": model_metrics,
        "delta_e76_improvement_percent": float(100 * (base_de - model_de) / base_de) if base_de else 0.0,
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
    p.add_argument("--target-icc", required=True, help="fixed CMYK output ICC")
    p.add_argument("--model", default="residual_lut_model.npz")
    p.add_argument("--report")
    p.add_argument("--grid-size", type=int, default=17)
    p.add_argument("--samples-per-image", type=int, default=40_000)
    p.add_argument("--max-samples", type=int, default=3_000_000)
    p.add_argument("--eval-samples-per-image", type=int, default=10_000)
    p.add_argument("--max-eval-samples", type=int, default=500_000)
    p.add_argument("--confidence-samples", type=float, default=None)
    p.add_argument("--smoothness", type=float, default=None)
    p.add_argument("--baseline-regularization", type=float, default=None)
    p.add_argument("--smooth-iterations", type=int, default=40)
    p.add_argument("--huber-delta", type=float, default=None, help="robust threshold in CMYK 0..255")
    p.add_argument("--max-cmy-residual", type=float, default=None, help="C/M/Y correction limit")
    p.add_argument("--max-k-residual", type=float, default=None, help="K correction limit")
    p.add_argument(
        "--fit-human", action="store_true",
        help="fit target CMYK closely: full-range residuals, weaker ICC pull-back",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    apply_objective_defaults(args)
    if not 0 <= args.val_ratio < 1: raise ValueError("--val-ratio 必须在 [0, 1) 范围")
    if args.grid_size < 2: raise ValueError("--grid-size 必须至少为 2")
    positive = ("samples_per_image", "max_samples", "eval_samples_per_image", "max_eval_samples",
                "confidence_samples", "smooth_iterations", "huber_delta", "max_cmy_residual", "max_k_residual")
    for name in positive:
        if getattr(args, name) <= 0: raise ValueError(f"--{name.replace('_', '-')} 必须大于 0")
    if args.smoothness < 0 or args.baseline_regularization < 0:
        raise ValueError("平滑和基线正则参数不能小于 0")

    train_pairs, val_pairs = collect_pairs(args)
    print(f"pairs: train={len(train_pairs)}, validation={len(val_pairs)}")
    print(
        f"objective: {'fit_human' if args.fit_human else 'safe_print'} | "
        f"residual limits C/M/Y ±{args.max_cmy_residual:g}, K ±{args.max_k_residual:g} | "
        f"huber={args.huber_delta:g} smoothness={args.smoothness:g} "
        f"baseline_reg={args.baseline_regularization:g} confidence_samples={args.confidence_samples:g}"
    )
    fixed_icc = load_fixed_target_icc(Path(args.target_icc))
    target_icc, train_records = validate_pairs(train_pairs, fixed_icc=fixed_icc)
    profile_name, profile_hash = profile_details(target_icc)
    val_records = []
    if val_pairs:
        _, val_records = validate_pairs(val_pairs, profile_hash, fixed_icc=fixed_icc)
    status = icc_status_counts(train_records + val_records)
    print(f"target profile: {profile_name} ({profile_hash[:12]})")

    transform = build_to_cmyk(target_icc)
    clip = np.array([args.max_cmy_residual] * 3 + [args.max_k_residual], dtype=np.float32) / 255
    sums, weights, train_samples, residual_stats = accumulate(
        train_pairs, args.grid_size, args.samples_per_image, args.max_samples,
        args.seed, transform, clip,
    )
    print(
        "unclipped |ΔCMYK| mean: "
        + ", ".join(f"{x:.1f}" for x in residual_stats["mean_abs_255"])
        + f" | max: {max(residual_stats['max_abs_255']):.1f} | "
        f"samples beyond clip: {residual_stats['samples_exceeding_clip']:,}/{train_samples:,}"
    )
    initial_lut, _ = finish_lut(
        sums, weights, args.confidence_samples, args.smoothness,
        args.baseline_regularization, args.smooth_iterations, clip,
    )
    sums, weights, _, _ = accumulate(
        train_pairs, args.grid_size, args.samples_per_image, args.max_samples,
        args.seed, transform, clip, initial_lut, args.huber_delta / 255,
    )
    lut, confidence = finish_lut(
        sums, weights, args.confidence_samples, args.smoothness,
        args.baseline_regularization, args.smooth_iterations, clip,
    )

    train_metrics = evaluate(
        train_pairs, lut, confidence, target_icc, args.eval_samples_per_image,
        args.max_eval_samples, args.seed + 1_000_000, "train-eval",
    )
    val_metrics = evaluate(
        val_pairs, lut, confidence, target_icc, args.eval_samples_per_image,
        args.max_eval_samples, args.seed + 2_000_000, "val",
    )
    metadata = {
        "model_type": "icc_residual_lut_v1", "grid_size": args.grid_size,
        "target_profile": profile_name, "target_icc_sha256": profile_hash,
        "target_icc_source": Path(args.target_icc).name,
        "target_icc_mode": "fixed_assignment_no_pixel_conversion",
        "embedded_target_icc_status": status,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_pairs": len(train_pairs), "validation_pairs": len(val_pairs),
        "training_samples": train_samples,
        "samples_per_image": sample_budget(len(train_pairs), args.samples_per_image, args.max_samples),
        "confidence_samples": args.confidence_samples, "smoothness": args.smoothness,
        "baseline_regularization": args.baseline_regularization,
        "smooth_iterations": args.smooth_iterations, "huber_delta_255": args.huber_delta,
        "residual_limits_255": [args.max_cmy_residual] * 3 + [args.max_k_residual],
        "fit_human": args.fit_human,
        "residual_stats_unclipped": residual_stats,
        "residual_strength": 1.0,
        "shadow_lift": 0.06,
        "shadow_lift_cmy": 0.035,
        "lut_nodes_with_samples": int(np.count_nonzero(weights)),
        "lut_nodes_total": int(weights.size),
        "mean_node_confidence": float(confidence.mean()),
        "train_metrics": train_metrics, "validation_metrics": val_metrics,
    }
    model_path = Path(args.model)
    report_path = Path(args.report) if args.report else model_path.with_suffix(".report.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ResidualLUTModel(lut, confidence, target_icc, metadata).save(model_path)
    report = metadata | {"model": model_path.name, "train_images": train_records, "validation_images": val_records}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(f"saved model: {model_path.resolve()}")
    print(f"saved report: {report_path.resolve()}")
    if val_metrics:
        before = val_metrics["icc_baseline"]["delta_e76"]
        after = val_metrics["icc_plus_lut"]["delta_e76"]
        print(f"validation ΔE76 baseline mean/p95: {before['mean']:.3f} / {before['p95']:.3f}")
        print(f"validation ΔE76 ICC+LUT mean/p95: {after['mean']:.3f} / {after['p95']:.3f}")
        print(f"validation mean improvement: {val_metrics['delta_e76_improvement_percent']:.2f}%")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
