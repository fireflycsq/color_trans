#!/usr/bin/env python3
"""Train image-adaptive RGB LUTs, then print through a fixed ICC.

Stage ``global`` trains a small CNN on the full frame. Stage ``portrait``
freezes that encoder and trains a second CNN on a MediaPipe person/skin crop.
Loss is on soft-proofed RGB because ICC is not differentiable; inference still
does corrected RGB → ICC → CMYK, then edge-lift / shadow-lift.
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
    resize_square,
)
from color_model import image_to_srgb, render_cmyk_to_srgb, srgb_to_lab
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
from train_residual_lut import build_to_cmyk


def _torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("自适应 LUT 需要 PyTorch：python3 -m pip install torch") from exc
    return torch, F


def load_pair_arrays(pair: Pair, icc: bytes) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(pair.source) as source:
        rgb_u8 = np.asarray(image_to_srgb(source), dtype=np.uint8)
    with Image.open(pair.target) as target:
        proof = np.asarray(render_cmyk_to_srgb(target.convert("CMYK"), icc), dtype=np.uint8)
    return rgb_u8, proof


def sample_pixels(rgb_u8: np.ndarray, proof_u8: np.ndarray, count: int, seed: int):
    idx = stratified_indices(rgb_u8, min(count, rgb_u8.shape[0] * rgb_u8.shape[1]), seed)
    rgb = rgb_u8.reshape(-1, 3)[idx].astype(np.float32) / 255.0
    proof = proof_u8.reshape(-1, 3)[idx].astype(np.float32) / 255.0
    return rgb, proof, idx


def metric_rgb(pred: np.ndarray, target: np.ndarray) -> dict:
    mae = np.mean(np.abs(pred - target), axis=0) * 255
    de = np.linalg.norm(srgb_to_lab(pred) - srgb_to_lab(target), axis=-1)
    return {
        "rgb_mae": [float(x) for x in mae],
        "delta_e76": {
            "mean": float(de.mean()),
            "p50": float(np.percentile(de, 50)),
            "p95": float(np.percentile(de, 95)),
        },
    }


def encode_full(encoder, rgb_u8: np.ndarray, device, size: int):
    return encoder(image_to_tensor(resize_square(rgb_u8, size), device))


def train_epoch(
    encoder, optimizer, pairs: list[Pair], icc: bytes, args, device, seed: int,
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
        rgb_u8, proof_u8 = load_pair_arrays(pair, icc)
        rgb, proof, idx = sample_pixels(rgb_u8, proof_u8, budget, seed + i)
        rgb_t = torch.from_numpy(rgb).to(device)
        proof_t = torch.from_numpy(proof).to(device)
        if global_encoder is None:
            lut, conf = encode_full(encoder, rgb_u8, device, args.thumbnail)
            pred = apply_lut_torch(rgb_t, lut[0], conf[0])
        else:
            mask_img = portrait_mask(rgb_u8, region=region)
            crop = portrait_crop(rgb_u8, mask_img, args.thumbnail, args.mask_threshold)
            mask = mask_img.reshape(-1)[idx]
            if crop is None or float(mask.max()) < args.mask_threshold:
                print(f"[train {i:>4}/{len(pairs)}] {pair.name}: no {region} crop")
                continue
            with torch.no_grad():
                g_lut, g_conf = encode_full(global_encoder, rgb_u8, device, args.thumbnail)
                base = apply_lut_torch(rgb_t, g_lut[0], g_conf[0])
            lut, conf = encoder(image_to_tensor(crop, device))
            portrait = apply_lut_torch(base, lut[0], conf[0])
            gate = torch.from_numpy(mask.astype(np.float32)).to(device).unsqueeze(-1)
            pred = (1.0 - gate) * base + gate * portrait
            keep = mask >= args.mask_threshold
            if int(np.count_nonzero(keep)) < 32:
                print(f"[train {i:>4}/{len(pairs)}] {pair.name}: too few {region} pixels")
                continue
            keep_t = torch.from_numpy(keep).to(device)
            pred = pred[keep_t]
            proof_t = proof_t[keep_t]
        loss = F.huber_loss(pred, proof_t, delta=args.huber_delta)
        loss = loss + args.lut_l1 * lut[0].abs().mean() + args.smoothness * lut_smoothness(lut[0])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        steps += 1
        print(f"[train {i:>4}/{len(pairs)}] {pair.name}: loss={loss.item():.5f} pixels={pred.shape[0]:,}")
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
    preds, proofs, cmyk_preds, cmyk_targets = [], [], [], []
    for i, pair in enumerate(pairs, 1):
        rgb_u8, proof_u8 = load_pair_arrays(pair, icc)
        _, proof, idx = sample_pixels(rgb_u8, proof_u8, budget, seed + i)
        corrected = model.correct_rgb(rgb_u8).reshape(-1, 3)[idx]
        with Image.open(pair.target) as target:
            target_cmyk = np.asarray(target.convert("CMYK"), dtype=np.float32).reshape(-1, 4)[idx] / 255.0
        strip = Image.fromarray(
            np.rint(np.clip(corrected, 0, 1) * 255).astype(np.uint8)[None, ...], "RGB",
        )
        pred_cmyk = np.asarray(ImageCms.applyTransform(strip, transform), dtype=np.float32)[0] / 255.0
        rgb_m = metric_rgb(corrected, proof)
        print(
            f"[{label} {i:>4}/{len(pairs)}] {pair.name}: "
            f"RGB MAE={np.mean(rgb_m['rgb_mae']):.2f} ΔE={rgb_m['delta_e76']['mean']:.3f}"
        )
        preds.append(corrected)
        proofs.append(proof)
        cmyk_preds.append(pred_cmyk)
        cmyk_targets.append(target_cmyk)
    pred = np.concatenate(preds)
    proof = np.concatenate(proofs)
    cmyk_err = np.mean(np.abs(np.concatenate(cmyk_preds) - np.concatenate(cmyk_targets)), axis=0) * 255
    metrics = metric_rgb(pred, proof)
    metrics["cmyk_mae"] = [float(x) for x in cmyk_err]
    metrics["pairs"] = len(pairs)
    metrics["samples"] = int(len(pred))
    return metrics


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
    p.add_argument("--huber-delta", type=float, default=0.05, help="Huber delta in RGB 0..1")
    p.add_argument("--lut-l1", type=float, default=0.02)
    p.add_argument("--smoothness", type=float, default=0.05)
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
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
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

    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.epochs + 1):
        print(f"epoch {epoch}/{args.epochs} stage={args.stage}")
        mean_loss = train_epoch(
            encoder, optimizer, train_pairs, icc, args, device,
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
        "model_type": "adaptive_rgb_lut_v1",
        "stage": args.stage,
        "grid_size": args.grid_size,
        "encoder_channels": args.channels,
        "thumbnail": args.thumbnail,
        "epochs": args.epochs,
        "lr": args.lr,
        "huber_delta_rgb": args.huber_delta,
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
        print(
            f"val RGB MAE mean={np.mean(val_metrics['rgb_mae']):.2f} "
            f"ΔE={val_metrics['delta_e76']['mean']:.3f} "
            f"CMYK MAE={val_metrics['cmyk_mae']}"
        )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
