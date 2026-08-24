#!/usr/bin/env python3
"""Train an RGB-to-CMYK model from one or many aligned image pairs.

The trainer streams image pairs and accumulates small normal-equation matrices,
so memory use does not grow with the number of photographs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageCms

from color_model import (
    ColorModel,
    image_to_srgb,
    polynomial_features,
    render_cmyk_to_srgb,
    srgb_to_lab,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class Pair:
    source: Path
    target: Path
    split: str = ""

    @property
    def name(self) -> str:
        return self.source.stem


def stratified_indices(rgb: np.ndarray, count: int, seed: int, bins: int = 8) -> np.ndarray:
    """Sample approximately uniformly over an RGB cube from a bounded pool."""
    rng = np.random.default_rng(seed)
    flat = rgb.reshape(-1, 3)
    count = min(count, len(flat))
    pool_size = min(len(flat), max(count * 5, 250_000))
    pool = rng.choice(len(flat), pool_size, replace=False)
    quantised = np.minimum(flat[pool].astype(np.int16) * bins // 256, bins - 1)
    ids = quantised[:, 0] * bins * bins + quantised[:, 1] * bins + quantised[:, 2]
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    cells, starts, sizes = np.unique(sorted_ids, return_index=True, return_counts=True)
    per_cell = max(1, count // len(cells))
    chosen: list[np.ndarray] = []
    for start, size in zip(starts, sizes):
        n = min(per_cell, int(size))
        local = rng.choice(int(size), n, replace=False)
        chosen.append(pool[order[start + local]])
    result = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
    if len(result) < count:
        extra = pool[rng.choice(pool_size, min(count - len(result), pool_size), replace=False)]
        result = np.concatenate([result, extra])
    rng.shuffle(result)
    return result[:count]


def fit_from_normal_equations(a: np.ndarray, b: np.ndarray, ridge: float) -> np.ndarray:
    reg = np.eye(a.shape[0], dtype=np.float64) * ridge
    reg[0, 0] = ridge * 0.01
    return np.linalg.solve(a + reg, b).astype(np.float32)


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Backward-compatible in-memory helper used by external callers/tests."""
    phi = polynomial_features(x).astype(np.float64)
    return fit_from_normal_equations(phi.T @ phi, phi.T @ y.astype(np.float64), ridge)


def directory_pairs(input_dir: Path, target_dir: Path, recursive: bool) -> list[Pair]:
    glob = "**/*" if recursive else "*"

    def index(root: Path) -> dict[str, Path]:
        result: dict[str, Path] = {}
        for path in root.glob(glob):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = path.relative_to(root).with_suffix("").as_posix()
            if relative in result:
                raise ValueError(f"重复的配对键 {relative}: {result[relative]} / {path}")
            result[relative] = path
        return result

    sources, targets = index(input_dir), index(target_dir)
    missing_targets = sorted(set(sources) - set(targets))
    missing_sources = sorted(set(targets) - set(sources))
    if missing_targets or missing_sources:
        details = []
        if missing_targets:
            details.append(f"缺少目标图 {len(missing_targets)} 个，例如 {missing_targets[:3]}")
        if missing_sources:
            details.append(f"缺少输入图 {len(missing_sources)} 个，例如 {missing_sources[:3]}")
        raise ValueError("；".join(details))
    return [Pair(sources[key], targets[key]) for key in sorted(sources)]


def paired_directory_pairs(
    root: Path, input_suffix: str, target_suffix: str, recursive: bool
) -> list[Pair]:
    """Pair ``xxx_input.*`` and ``xxx_target.*`` files inside one directory."""
    if not root.is_dir():
        raise ValueError(f"配对目录不存在：{root}")
    if not input_suffix or not target_suffix or input_suffix == target_suffix:
        raise ValueError("输入和目标后缀必须非空且互不相同")
    glob = "**/*" if recursive else "*"
    sources: dict[str, Path] = {}
    targets: dict[str, Path] = {}

    for path in root.glob(glob):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        stem = path.stem
        kind: str | None = None
        if stem.endswith(input_suffix):
            base_stem = stem[:-len(input_suffix)]
            kind = "input"
        elif stem.endswith(target_suffix):
            base_stem = stem[:-len(target_suffix)]
            kind = "target"
        else:
            continue
        if not base_stem:
            raise ValueError(f"文件名缺少配对主名：{path}")
        relative_parent = path.parent.relative_to(root).as_posix()
        key = f"{relative_parent}/{base_stem}" if relative_parent != "." else base_stem
        index = sources if kind == "input" else targets
        if key in index:
            raise ValueError(f"重复的 {kind} 配对键 {key}: {index[key]} / {path}")
        index[key] = path

    missing_targets = sorted(set(sources) - set(targets))
    missing_sources = sorted(set(targets) - set(sources))
    if missing_targets or missing_sources:
        details = []
        if missing_targets:
            details.append(f"缺少 {target_suffix} 文件 {len(missing_targets)} 个，例如 {missing_targets[:3]}")
        if missing_sources:
            details.append(f"缺少 {input_suffix} 文件 {len(missing_sources)} 个，例如 {missing_sources[:3]}")
        raise ValueError("；".join(details))
    if not sources:
        raise ValueError(
            f"目录中没有找到 {input_suffix} / {target_suffix} 图片对：{root}"
        )
    return [Pair(sources[key], targets[key]) for key in sorted(sources)]


def manifest_pairs(path: Path) -> list[Pair]:
    base = path.parent
    rows: list[dict]
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    result = []
    for line_no, row in enumerate(rows, 2):
        if not row.get("input") or not row.get("target"):
            raise ValueError(f"清单第 {line_no} 行缺少 input 或 target")
        source, target = Path(row["input"]), Path(row["target"])
        if not source.is_absolute():
            source = base / source
        if not target.is_absolute():
            target = base / target
        result.append(Pair(source, target, str(row.get("split", "")).lower()))
    return result


def collect_pairs(args: argparse.Namespace) -> tuple[list[Pair], list[Pair]]:
    modes = sum(bool(x) for x in (
        args.manifest,
        args.pair_dir,
        args.input_dir or args.target_dir,
        args.input or args.target,
    ))
    if modes > 1:
        raise ValueError("--manifest、--pair-dir、双目录模式和单图模式只能选择一种")
    if args.manifest:
        pairs = manifest_pairs(Path(args.manifest))
    elif args.pair_dir:
        pairs = paired_directory_pairs(
            Path(args.pair_dir), args.input_suffix, args.target_suffix, args.recursive
        )
    elif args.input_dir or args.target_dir:
        if not args.input_dir or not args.target_dir:
            raise ValueError("--input-dir 和 --target-dir 必须同时提供")
        pairs = directory_pairs(Path(args.input_dir), Path(args.target_dir), args.recursive)
    elif args.input or args.target:
        if not args.input or not args.target:
            raise ValueError("--input 和 --target 必须同时提供")
        pairs = [Pair(Path(args.input), Path(args.target), "train")]
    else:
        raise ValueError("请提供单图参数、目录参数、--pair-dir 或 --manifest")
    if not pairs:
        raise ValueError("没有找到可训练的图片对")

    explicit_train = [x for x in pairs if x.split in {"train", "training"}]
    explicit_val = [x for x in pairs if x.split in {"val", "valid", "validation", "test"}]
    unspecified = [x for x in pairs if not x.split]
    if explicit_train or explicit_val:
        train = explicit_train + unspecified
        val = explicit_val
    else:
        shuffled = list(pairs)
        np.random.default_rng(args.seed).shuffle(shuffled)
        val_count = int(round(len(shuffled) * args.val_ratio))
        if args.val_ratio > 0 and len(shuffled) >= 5:
            val_count = max(1, val_count)
        val_count = min(val_count, max(0, len(shuffled) - 1))
        val, train = shuffled[:val_count], shuffled[val_count:]

    if args.val_input_dir or args.val_target_dir:
        if not args.val_input_dir or not args.val_target_dir:
            raise ValueError("--val-input-dir 和 --val-target-dir 必须同时提供")
        val.extend(directory_pairs(Path(args.val_input_dir), Path(args.val_target_dir), args.recursive))
    if not train:
        raise ValueError("训练集为空")
    return train, val


def profile_details(icc: bytes) -> tuple[str, str]:
    profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
    return ImageCms.getProfileName(profile).strip(), hashlib.sha256(icc).hexdigest()


def load_fixed_target_icc(path: Path) -> bytes:
    if not path.is_file():
        raise FileNotFoundError(f"固定 ICC 文件不存在：{path}")
    data = path.read_bytes()
    try:
        profile = ImageCms.ImageCmsProfile(io.BytesIO(data))
        ImageCms.buildTransform(
            profile, ImageCms.createProfile("sRGB"), "CMYK", "RGB", renderingIntent=1
        )
    except Exception as exc:
        raise ValueError(f"固定 ICC 不是可用的 CMYK 输出配置文件：{path}: {exc}") from exc
    return data


def validate_pairs(
    pairs: Iterable[Pair], expected_icc_hash: str | None = None,
    fixed_icc: bytes | None = None,
) -> tuple[bytes, list[dict]]:
    target_icc: bytes | None = fixed_icc
    fixed_hash = profile_details(fixed_icc)[1] if fixed_icc else None
    records = []
    for pair in pairs:
        if not pair.source.exists() or not pair.target.exists():
            raise FileNotFoundError(f"图片对不存在：{pair.source} / {pair.target}")
        with Image.open(pair.source) as src, Image.open(pair.target) as target:
            if src.size != target.size:
                raise ValueError(f"尺寸不一致 {pair.name}: {src.size} != {target.size}")
            if target.mode != "CMYK":
                raise ValueError(f"目标图必须是 CMYK，{pair.target} 当前为 {target.mode}")
            icc = target.info.get("icc_profile")
            embedded_name = None
            embedded_hash = None
            if icc:
                embedded_name, embedded_hash = profile_details(icc)

            if fixed_icc:
                icc_status = "matching" if embedded_hash == fixed_hash else (
                    "different" if embedded_hash else "missing"
                )
            else:
                if not icc:
                    raise ValueError(f"目标图没有内嵌 ICC：{pair.target}")
                if expected_icc_hash and embedded_hash != expected_icc_hash:
                    raise ValueError(
                        f"目标 ICC 不一致：{pair.target} ({embedded_name}, {embedded_hash[:12]})"
                    )
                if target_icc is None:
                    target_icc = icc
                    expected_icc_hash = embedded_hash
                elif embedded_hash != expected_icc_hash:
                    raise ValueError(
                        f"目标 ICC 不一致：{pair.target} ({embedded_name}, {embedded_hash[:12]})"
                    )
                icc_status = "matching"
            records.append({
                "name": pair.name,
                "width": src.width,
                "height": src.height,
                "embedded_icc_status": icc_status,
                "embedded_icc_name": embedded_name,
                "embedded_icc_sha256": embedded_hash,
            })
    assert target_icc is not None
    return target_icc, records


def icc_status_counts(records: Iterable[dict]) -> dict[str, int]:
    counts = {"matching": 0, "different": 0, "missing": 0}
    for record in records:
        counts[record["embedded_icc_status"]] += 1
    return counts


def sample_pair(pair: Pair, count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Image.open(pair.source) as source:
        rgb = np.asarray(image_to_srgb(source), dtype=np.uint8)
    with Image.open(pair.target) as target:
        cmyk = np.asarray(target, dtype=np.uint8)
    idx = stratified_indices(rgb, min(count, rgb.shape[0] * rgb.shape[1]), seed)
    x_u8 = rgb.reshape(-1, 3)[idx]
    y_u8 = cmyk.reshape(-1, 4)[idx]
    return x_u8.astype(np.float32) / 255.0, y_u8.astype(np.float32) / 255.0, x_u8


def sample_budget(pair_count: int, per_image: int, maximum: int) -> int:
    return max(1, min(per_image, maximum // max(1, pair_count)))


def accumulate_training(
    pairs: list[Pair], per_image: int, maximum: int, seed: int
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    a = np.zeros((20, 20), dtype=np.float64)
    b = np.zeros((20, 4), dtype=np.float64)
    coverage = np.zeros(512, dtype=np.int64)
    budget = sample_budget(len(pairs), per_image, maximum)
    total = 0
    for i, pair in enumerate(pairs, 1):
        x, y, rgb_u8 = sample_pair(pair, budget, seed + i * 104729)
        phi = polynomial_features(x).astype(np.float64)
        a += phi.T @ phi
        b += phi.T @ y.astype(np.float64)
        q = np.minimum(rgb_u8.astype(np.int16) // 32, 7)
        np.add.at(coverage, q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2], 1)
        total += len(x)
        print(f"[train {i:>4}/{len(pairs)}] {pair.name}: {len(x):,} samples")
    return a, b, total, coverage


def render_samples(cmyk: np.ndarray, icc: bytes, chunk: int = 4096) -> np.ndarray:
    values = np.asarray(cmyk, dtype=np.float32).reshape(-1, 4)
    if len(values) > chunk:
        parts = [render_samples(values[i:i + chunk], icc, chunk) for i in range(0, len(values), chunk)]
        return np.concatenate(parts, axis=0)
    image = Image.fromarray(np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)[None, ...], "CMYK")
    return np.asarray(render_cmyk_to_srgb(image, icc), dtype=np.float32)[0] / 255.0


def evaluate(
    pairs: list[Pair], weights: np.ndarray, icc: bytes, per_image: int,
    maximum: int, seed: int, label: str,
) -> dict | None:
    if not pairs:
        return None
    budget = sample_budget(len(pairs), per_image, maximum)
    abs_sum = np.zeros(4, dtype=np.float64)
    squared_sum = 0.0
    total = 0
    delta_es: list[np.ndarray] = []
    per_pair = []
    for i, pair in enumerate(pairs, 1):
        x, y, _ = sample_pair(pair, budget, seed + i * 130363)
        pred = np.clip(polynomial_features(x) @ weights, 0, 1)
        error = pred - y
        mae = np.mean(np.abs(error), axis=0) * 255
        abs_sum += np.abs(error).sum(axis=0)
        squared_sum += float(np.square(error).sum())
        total += len(x)
        pred_rgb = render_samples(pred, icc)
        target_rgb = render_samples(y, icc)
        de = np.linalg.norm(srgb_to_lab(pred_rgb) - srgb_to_lab(target_rgb), axis=-1).astype(np.float32)
        delta_es.append(de)
        per_pair.append({
            "name": pair.name,
            "samples": len(x),
            "cmyk_mae": [float(v) for v in mae],
            "delta_e76_mean": float(de.mean()),
            "delta_e76_p95": float(np.percentile(de, 95)),
        })
        print(f"[{label} {i:>4}/{len(pairs)}] {pair.name}: ΔE76={de.mean():.3f}, CMYK MAE={mae.mean():.3f}")
    all_de = np.concatenate(delta_es)
    rmse = np.sqrt(squared_sum / (total * 4)) * 255
    per_pair.sort(key=lambda x: x["delta_e76_mean"], reverse=True)
    return {
        "pairs": len(pairs),
        "samples": total,
        "cmyk_mae": [float(v) for v in abs_sum / total * 255],
        "cmyk_psnr": float(20 * np.log10(255 / rmse)) if rmse else None,
        "delta_e76": {
            "mean": float(all_de.mean()),
            "p50": float(np.percentile(all_de, 50)),
            "p95": float(np.percentile(all_de, 95)),
            "max": float(all_de.max()),
        },
        "worst_pairs": per_pair[: min(20, len(per_pair))],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_argument_group("data source (choose one)")
    source.add_argument("--input", help="single aligned RGB input image")
    source.add_argument("--target", help="single aligned CMYK target image")
    source.add_argument("--input-dir", help="training RGB directory")
    source.add_argument("--target-dir", help="training CMYK directory")
    source.add_argument(
        "--pair-dir", help="directory containing xxx_input.* / xxx_target.* pairs"
    )
    source.add_argument("--input-suffix", default="_input")
    source.add_argument("--target-suffix", default="_target")
    source.add_argument("--manifest", help="CSV/JSONL with input,target[,split]")
    source.add_argument("--val-input-dir", help="optional validation RGB directory")
    source.add_argument("--val-target-dir", help="optional validation CMYK directory")
    source.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--val-ratio", type=float, default=0.1, help="automatic validation fraction")
    p.add_argument(
        "--target-icc",
        help="fixed CMYK ICC assigned to all target values and embedded in the model",
    )
    p.add_argument("--model", default="color_model.npz")
    p.add_argument("--report", help="JSON report path; defaults next to model")
    p.add_argument("--samples", type=int, help="legacy single-pair sample count")
    p.add_argument("--samples-per-image", type=int, default=40_000)
    p.add_argument("--max-samples", type=int, default=3_000_000)
    p.add_argument("--eval-samples-per-image", type=int, default=10_000)
    p.add_argument("--max-eval-samples", type=int, default=500_000)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio 必须在 [0, 1) 范围")
    for name in ("samples_per_image", "max_samples", "eval_samples_per_image", "max_eval_samples"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} 必须大于 0")
    if args.ridge < 0:
        raise ValueError("--ridge 不能小于 0")
    if args.samples:
        args.samples_per_image = args.samples
        args.max_samples = args.samples
    train_pairs, val_pairs = collect_pairs(args)
    print(f"pairs: train={len(train_pairs)}, validation={len(val_pairs)}")

    fixed_icc = load_fixed_target_icc(Path(args.target_icc)) if args.target_icc else None
    target_icc, train_records = validate_pairs(train_pairs, fixed_icc=fixed_icc)
    profile_name, profile_hash = profile_details(target_icc)
    val_records = []
    if val_pairs:
        _, val_records = validate_pairs(val_pairs, profile_hash, fixed_icc=fixed_icc)
    print(f"target profile: {profile_name} ({profile_hash[:12]})")
    all_icc_status = icc_status_counts(train_records + val_records)
    if fixed_icc:
        print(
            "embedded target ICC (informational only): "
            f"matching={all_icc_status['matching']}, "
            f"different={all_icc_status['different']}, missing={all_icc_status['missing']}"
        )

    a, b, train_samples, coverage = accumulate_training(
        train_pairs, args.samples_per_image, args.max_samples, args.seed
    )
    weights = fit_from_normal_equations(a, b, args.ridge)
    train_metrics = evaluate(
        train_pairs, weights, target_icc, args.eval_samples_per_image,
        args.max_eval_samples, args.seed + 1_000_000, "train-eval",
    )
    val_metrics = evaluate(
        val_pairs, weights, target_icc, args.eval_samples_per_image,
        args.max_eval_samples, args.seed + 2_000_000, "val",
    )

    report_path = Path(args.report) if args.report else Path(args.model).with_suffix(".report.json")
    metadata = {
        "feature_model": "RGB polynomial, total degree <= 3",
        "target_profile": profile_name,
        "target_icc_sha256": profile_hash,
        "target_icc_source": Path(args.target_icc).name if args.target_icc else "embedded_in_targets",
        "target_icc_mode": "fixed_assignment_no_pixel_conversion" if fixed_icc else "strict_embedded",
        "embedded_target_icc_status": all_icc_status,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(val_pairs),
        "training_samples": train_samples,
        "samples_per_image": sample_budget(len(train_pairs), args.samples_per_image, args.max_samples),
        "ridge": args.ridge,
        "seed": args.seed,
        "rgb_bins_occupied": int(np.count_nonzero(coverage)),
        "rgb_bins_total": int(len(coverage)),
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
    }
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ColorModel(weights, target_icc, metadata).save(model_path)
    report = metadata | {
        "model": model_path.name,
        "train_images": train_records,
        "validation_images": val_records,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), "utf-8")
    print(f"saved model: {model_path.resolve()}")
    print(f"saved report: {report_path.resolve()}")
    if val_metrics:
        print("validation CMYK MAE [C M Y K]:", np.round(val_metrics["cmyk_mae"], 3).tolist())
        print("validation ΔE76 mean/p95:",
              round(val_metrics["delta_e76"]["mean"], 3),
              round(val_metrics["delta_e76"]["p95"], 3))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
