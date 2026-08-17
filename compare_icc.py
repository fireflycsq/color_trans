#!/usr/bin/env python3
"""Compare how image pixels look with their embedded ICC and an assigned ICC."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageOps

from color_model import profile_from_bytes, srgb_to_lab


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}


def profile_name(profile: ImageCms.ImageCmsProfile) -> str:
    try:
        return ImageCms.getProfileName(profile).strip()
    except Exception:
        return "unknown"


def profile_space(profile: ImageCms.ImageCmsProfile) -> str:
    # ``getProfileColorSpace`` is absent in some Pillow releases; the wrapped
    # LittleCMS profile exposes the same four-character signature.
    getter = getattr(ImageCms, "getProfileColorSpace", None)
    value = getter(profile) if getter else profile.profile.xcolor_space
    return str(value).strip().upper()


def render(pixels: Image.Image, source: ImageCms.ImageCmsProfile, intent: int) -> Image.Image:
    transform = ImageCms.buildTransform(
        source,
        ImageCms.createProfile("sRGB"),
        pixels.mode,
        "RGB",
        renderingIntent=intent,
        flags=ImageCms.Flags.BLACKPOINTCOMPENSATION,
    )
    return ImageCms.applyTransform(pixels, transform)


def fit_within(image: Image.Image, maximum: int) -> Image.Image:
    if max(image.size) <= maximum:
        return image.copy()
    ratio = maximum / max(image.size)
    size = tuple(max(1, round(v * ratio)) for v in image.size)
    return image.resize(size, Image.Resampling.LANCZOS)


def delta_e76(first: Image.Image, second: Image.Image) -> np.ndarray:
    a = np.asarray(first, dtype=np.float32) / 255.0
    b = np.asarray(second, dtype=np.float32) / 255.0
    return np.linalg.norm(srgb_to_lab(a) - srgb_to_lab(b), axis=-1)


def heatmap(delta_e: np.ndarray, ceiling: float) -> Image.Image:
    # Black -> red -> yellow: values at or above ceiling are yellow.
    value = np.clip(delta_e / ceiling, 0.0, 1.0)
    red = np.clip(value * 2.0, 0.0, 1.0)
    green = np.clip(value * 2.0 - 1.0, 0.0, 1.0)
    rgb = np.stack([red, green, np.zeros_like(value)], axis=-1)
    return Image.fromarray(np.rint(rgb * 255).astype(np.uint8), "RGB")


def labelled_panel(image: Image.Image, label: str, top: int = 34) -> Image.Image:
    panel = Image.new("RGB", (image.width, image.height + top), "white")
    panel.paste(image, (0, top))
    draw = ImageDraw.Draw(panel)
    draw.text((10, 10), label, fill="black")
    return panel


def save_comparison(
    embedded: Image.Image,
    assigned: Image.Image,
    delta_e: np.ndarray,
    destination: Path,
    quality: int,
    heatmap_ceiling: float,
) -> None:
    panels = [
        labelled_panel(embedded, "Embedded ICC"),
        labelled_panel(assigned, "Assigned ICC"),
        labelled_panel(heatmap(delta_e, heatmap_ceiling), f"DeltaE76 heatmap (yellow >= {heatmap_ceiling:g})"),
    ]
    canvas = Image.new("RGB", (sum(x.width for x in panels), max(x.height for x in panels)), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=quality, subsampling=0)


def compare_one(
    path: Path,
    input_root: Path,
    output_root: Path,
    assigned_profile: ImageCms.ImageCmsProfile,
    assigned_name: str,
    assigned_hash: str,
    intent: int,
    max_dimension: int,
    quality: int,
    heatmap_ceiling: float,
) -> dict:
    relative = path.relative_to(input_root)
    record: dict = {"file": relative.as_posix(), "status": "error"}
    try:
        with Image.open(path) as opened:
            embedded_bytes = opened.info.get("icc_profile")
            if not embedded_bytes:
                return record | {"status": "missing_embedded_icc", "error": "image has no embedded ICC"}
            embedded_profile = profile_from_bytes(embedded_bytes)
            pixels = ImageOps.exif_transpose(opened).copy()

        embedded_space = profile_space(embedded_profile)
        assigned_space = profile_space(assigned_profile)
        if pixels.mode not in {"RGB", "CMYK"}:
            return record | {"status": "unsupported_mode", "mode": pixels.mode}
        if embedded_space != pixels.mode or assigned_space != pixels.mode:
            return record | {
                "status": "profile_mode_mismatch",
                "mode": pixels.mode,
                "embedded_space": embedded_space,
                "assigned_space": assigned_space,
            }

        preview_pixels = fit_within(pixels, max_dimension)
        embedded_rgb = render(preview_pixels, embedded_profile, intent)
        assigned_rgb = render(preview_pixels, assigned_profile, intent)
        de = delta_e76(embedded_rgb, assigned_rgb)
        preview_path = output_root / relative.parent / f"{relative.stem}_icc_compare.jpg"
        save_comparison(embedded_rgb, assigned_rgb, de, preview_path, quality, heatmap_ceiling)
        return record | {
            "status": "ok",
            "mode": pixels.mode,
            "width": pixels.width,
            "height": pixels.height,
            "evaluated_width": preview_pixels.width,
            "evaluated_height": preview_pixels.height,
            "embedded_profile": profile_name(embedded_profile),
            "embedded_icc_sha256": hashlib.sha256(embedded_bytes).hexdigest(),
            "assigned_profile": assigned_name,
            "assigned_icc_sha256": assigned_hash,
            "same_icc_bytes": hashlib.sha256(embedded_bytes).hexdigest() == assigned_hash,
            "delta_e76_mean": float(de.mean()),
            "delta_e76_p50": float(np.percentile(de, 50)),
            "delta_e76_p95": float(np.percentile(de, 95)),
            "delta_e76_max": float(de.max()),
            "pixels_delta_e_gt_2_percent": float(np.mean(de > 2) * 100),
            "pixels_delta_e_gt_5_percent": float(np.mean(de > 5) * 100),
            "preview": preview_path.relative_to(output_root).as_posix(),
        }
    except Exception as exc:
        return record | {"status": "error", "error": str(exc)}


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="directory containing images")
    parser.add_argument("--assigned-icc", required=True, help="ICC used to reinterpret unchanged pixels")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-dimension", type=int, default=1600, help="long edge used for preview and metrics")
    parser.add_argument("--intent", type=int, choices=range(4), default=1, help="ICC intent: 0/1/2/3")
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--heatmap-ceiling", type=float, default=10.0)
    args = parser.parse_args()

    input_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    assigned_path = Path(args.assigned_icc).resolve()
    if not input_root.is_dir():
        raise ValueError(f"input directory does not exist: {input_root}")
    if args.max_dimension <= 0 or args.heatmap_ceiling <= 0:
        raise ValueError("--max-dimension and --heatmap-ceiling must be positive")
    assigned_bytes = assigned_path.read_bytes()
    assigned_profile = profile_from_bytes(assigned_bytes)
    assigned_name = profile_name(assigned_profile)
    assigned_hash = hashlib.sha256(assigned_bytes).hexdigest()
    pattern = "**/*" if args.recursive else "*"
    images = sorted(p for p in input_root.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError(f"no supported images found under: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(images, 1):
        row = compare_one(
            path, input_root, output_root, assigned_profile, assigned_name, assigned_hash,
            args.intent, args.max_dimension, args.quality, args.heatmap_ceiling,
        )
        rows.append(row)
        metric = f"mean={row['delta_e76_mean']:.3f}, p95={row['delta_e76_p95']:.3f}" if row["status"] == "ok" else row["status"]
        print(f"[{index:>4}/{len(images)}] {row['file']}: {metric}")

    successful = sorted(
        (row for row in rows if row["status"] == "ok"),
        key=lambda row: row["delta_e76_mean"], reverse=True,
    )
    failures = [row for row in rows if row["status"] != "ok"]
    summary = {
        "assigned_icc": str(assigned_path),
        "assigned_profile": assigned_name,
        "assigned_icc_sha256": assigned_hash,
        "rendering_intent": args.intent,
        "images_found": len(images),
        "images_compared": len(successful),
        "images_skipped_or_failed": len(failures),
        "dataset_delta_e76_mean": float(np.mean([x["delta_e76_mean"] for x in successful])) if successful else None,
        "worst_images": successful[:20],
        "results": successful + failures,
    }
    (output_root / "icc_comparison_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), "utf-8"
    )
    write_csv(output_root / "icc_comparison_report.csv", successful + failures)
    print(f"report: {output_root / 'icc_comparison_report.json'}")
    print(f"csv:    {output_root / 'icc_comparison_report.csv'}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
