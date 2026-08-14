#!/usr/bin/env python3
"""Train a compact RGB-to-CMYK colour transform from aligned image pairs."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms

from color_model import ColorModel, polynomial_features


def stratified_indices(rgb: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Sample approximately uniformly over a coarse RGB cube."""
    rng = np.random.default_rng(seed)
    flat = rgb.reshape(-1, 3)
    # First make a bounded random pool. This keeps stratification fast even for
    # 24+ megapixel photographs while retaining broad spatial coverage.
    pool_size = min(len(flat), max(count * 6, 500_000))
    pool = rng.choice(len(flat), pool_size, replace=False)
    pool_rgb = flat[pool]
    bins = np.minimum(pool_rgb.astype(np.int16) // 32, 7)
    ids = bins[:, 0] * 64 + bins[:, 1] * 8 + bins[:, 2]
    occupied = np.unique(ids)
    per_bin = max(1, count // len(occupied))
    chosen = []
    for cell in occupied:
        candidates = np.flatnonzero(ids == cell)
        n = min(per_bin, len(candidates))
        chosen.append(pool[rng.choice(candidates, n, replace=False)])
    result = np.concatenate(chosen)
    if len(result) < count:
        remaining = rng.choice(len(flat), min(count-len(result), len(flat)), replace=False)
        result = np.concatenate([result, remaining])
    rng.shuffle(result)
    return result[:count]


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    phi = polynomial_features(x).astype(np.float64)
    target = y.astype(np.float64)
    reg = np.eye(phi.shape[1]) * ridge
    reg[0, 0] = ridge * 0.01  # weakly regularise the intercept
    return np.linalg.solve(phi.T @ phi + reg, phi.T @ target).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="aligned RGB input image")
    p.add_argument("--target", required=True, help="aligned CMYK target image")
    p.add_argument("--model", default="color_model.npz")
    p.add_argument("--samples", type=int, default=250_000)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src = Image.open(args.input)
    target = Image.open(args.target)
    if src.size != target.size:
        raise ValueError(f"Images must be aligned and equal-sized: {src.size} != {target.size}")
    if target.mode != "CMYK":
        raise ValueError(f"Expected a CMYK target, got {target.mode}")
    target_icc = target.info.get("icc_profile")
    if not target_icc:
        raise ValueError("Target image has no embedded CMYK ICC profile")

    rgb = np.asarray(src.convert("RGB"), dtype=np.uint8)
    cmyk = np.asarray(target, dtype=np.uint8)
    idx = stratified_indices(rgb, min(args.samples, rgb.shape[0]*rgb.shape[1]), args.seed)
    x = rgb.reshape(-1, 3)[idx].astype(np.float32) / 255.0
    y = cmyk.reshape(-1, 4)[idx].astype(np.float32) / 255.0
    weights = fit_ridge(x, y, args.ridge)

    pred = np.clip(polynomial_features(x) @ weights, 0, 1)
    mae = np.mean(np.abs(pred-y), axis=0) * 255
    profile = ImageCms.ImageCmsProfile(io.BytesIO(target_icc))
    metadata = {
        "feature_model": "RGB polynomial, total degree <= 3",
        "source": str(Path(args.input).resolve()),
        "target": str(Path(args.target).resolve()),
        "target_profile": ImageCms.getProfileName(profile).strip(),
        "samples": int(len(idx)),
        "ridge": args.ridge,
        "sample_train_mae_cmyk_8bit": [float(v) for v in mae],
    }
    ColorModel(weights, target_icc, metadata).save(args.model)
    print(f"saved: {Path(args.model).resolve()}")
    print(f"target profile: {metadata['target_profile']}")
    print("sample CMYK MAE [C M Y K]:", np.round(mae, 3).tolist())


if __name__ == "__main__":
    main()
