#!/usr/bin/env python3
"""Visualize the portrait cut-out used by the second colour stage.

Saves a four-panel JPEG: original, cyan overlay, person cut-out, and
soft mask with the training threshold outline. Use this to check whether
hair and clothing are inside the mask before training.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from color_model import image_to_srgb
from portrait_mask import detector_name, portrait_mask

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
PANEL_TITLES = ("原图", "叠加", "抠出", "遮罩 + 轮廓")


def collect_images(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    glob = "**/*" if recursive else "*"
    files = [
        p for p in sorted(path.glob(glob))
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not files:
        raise ValueError(f"目录中没有图片：{path}")
    return files


def fit_within(rgb: np.ndarray, mask: np.ndarray, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = rgb.shape[:2]
    longest = max(height, width)
    if longest <= maximum:
        return rgb, mask
    scale = maximum / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    rgb_image = Image.fromarray(rgb, "RGB").resize(size, Image.Resampling.LANCZOS)
    mask_image = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8), "L")
    mask_image = mask_image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(rgb_image), np.asarray(mask_image, dtype=np.float32) / 255.0


def overlay(rgb: np.ndarray, mask: np.ndarray, color=(0, 210, 255), alpha: float = 0.45) -> np.ndarray:
    weight = (np.clip(mask, 0, 1) * alpha)[..., None]
    tint = np.asarray(color, dtype=np.float32)
    mixed = rgb.astype(np.float32) * (1.0 - weight) + tint * weight
    return np.clip(mixed, 0, 255).astype(np.uint8)


def cutout(rgb: np.ndarray, mask: np.ndarray, cell: int = 16) -> np.ndarray:
    height, width = mask.shape
    yy, xx = np.ogrid[:height, :width]
    checker = ((((yy // cell) + (xx // cell)) % 2) * 55 + 42).astype(np.float32)
    background = np.stack([checker, checker, checker], axis=-1)
    weight = np.clip(mask, 0, 1)[..., None]
    mixed = rgb.astype(np.float32) * weight + background * (1.0 - weight)
    return np.clip(mixed, 0, 255).astype(np.uint8)


def heatmap(mask: np.ndarray) -> np.ndarray:
    m = np.clip(mask, 0, 1)
    red = np.clip(40 + 215 * m, 0, 255)
    green = np.clip(20 + 80 * (1.0 - m) + 200 * np.clip(m - 0.5, 0, 1) * 2, 0, 255)
    blue = np.clip(40 + 180 * (1.0 - m), 0, 255)
    return np.stack([red, green, blue], axis=-1).astype(np.uint8)


def threshold_outline(mask: np.ndarray, threshold: float, width: int = 3) -> np.ndarray:
    hard = mask >= threshold
    shifted = [
        np.pad(hard, ((1, 0), (0, 0)), mode="edge")[:-1],
        np.pad(hard, ((0, 1), (0, 0)), mode="edge")[1:],
        np.pad(hard, ((0, 0), (1, 0)), mode="edge")[:, :-1],
        np.pad(hard, ((0, 0), (0, 1)), mode="edge")[:, 1:],
    ]
    edge = np.zeros_like(hard)
    for neighbour in shifted:
        edge |= hard != neighbour
    if width <= 1:
        return edge
    padded = np.pad(edge, width, mode="constant")
    thick = np.zeros_like(edge)
    span = range(-width, width + 1)
    for dy in span:
        for dx in span:
            if dy * dy + dx * dx <= width * width:
                thick |= padded[width + dy:width + dy + edge.shape[0], width + dx:width + dx + edge.shape[1]]
    return thick


def load_font(size: int):
    candidates = (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size, index=0)
            except OSError:
                continue
    return ImageFont.load_default()


def caption(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, font) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(12, 14, 18))
    draw.text((x0 + 12, y0 + 6), text, fill=(240, 244, 248), font=font)


def compose_preview(
    rgb: np.ndarray, mask: np.ndarray, *, region: str, threshold: float, name: str,
) -> Image.Image:
    panels = [
        rgb,
        overlay(rgb, mask),
        cutout(rgb, mask),
        heatmap(mask),
    ]
    outline = threshold_outline(mask, threshold)
    panels[3][outline] = (255, 230, 0)
    height, width = rgb.shape[:2]
    bar = 36
    gap = 8
    canvas = Image.new("RGB", (width * 4 + gap * 3, height + bar), (18, 20, 24))
    draw = ImageDraw.Draw(canvas)
    font = load_font(18)
    coverage = float((mask >= threshold).mean())
    mean = float(mask.mean())
    stats = (
        f"{name}  |  {region}  |  {detector_name(region)}  |  "
        f"遮罩均值 {mean:.3f}  |  ≥{threshold:.2f} 占比 {100 * coverage:.1f}%"
    )
    caption(draw, (0, 0, canvas.width, bar), stats, font)
    title_font = load_font(16)
    for i, panel in enumerate(panels):
        x = i * (width + gap)
        canvas.paste(Image.fromarray(panel, "RGB"), (x, bar))
        label = PANEL_TITLES[i]
        tw = draw.textlength(label, font=title_font)
        draw.rectangle((x, bar, x + int(tw) + 16, bar + 22), fill=(12, 14, 18))
        draw.text((x + 8, bar + 2), label, fill=(230, 236, 242), font=title_font)
    return canvas


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="single image")
    source.add_argument("--input-dir", help="directory of images")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--region", choices=("person", "skin", "contour"), default="person")
    p.add_argument("--threshold", type=float, default=None, help="same cutoff used in training")
    p.add_argument("--blur-radius", type=float, default=4.0)
    p.add_argument("--max-dimension", type=int, default=1600)
    p.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-raw-mask", action="store_true", help="also write a grayscale mask PNG")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.threshold is None:
        args.threshold = 0.5 if args.region == "contour" else 0.45
    if not 0 < args.threshold <= 1:
        raise ValueError("--threshold 必须在 (0, 1] 范围")
    if args.max_dimension < 64:
        raise ValueError("--max-dimension 太小")
    root = Path(args.input or args.input_dir)
    images = collect_images(root, args.recursive)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, path in enumerate(images, 1):
        with Image.open(path) as image:
            rgb = np.asarray(image_to_srgb(image), dtype=np.uint8)
        mask = portrait_mask(rgb, blur_radius=args.blur_radius, region=args.region)
        coverage = float((mask >= args.threshold).mean())
        mean = float(mask.mean())
        preview_rgb, preview_mask = fit_within(rgb, mask, args.max_dimension)
        preview = compose_preview(
            preview_rgb, preview_mask, region=args.region,
            threshold=args.threshold, name=path.name,
        )
        dest = out_dir / f"{path.stem}_mask_preview.jpg"
        preview.save(dest, quality=92, subsampling=0)
        if args.save_raw_mask:
            raw = out_dir / f"{path.stem}_mask.png"
            Image.fromarray(np.rint(np.clip(mask, 0, 1) * 255).astype(np.uint8), "L").save(raw)
        rows.append({
            "file": path.name,
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "region": args.region,
            "detector": detector_name(args.region),
            "mask_mean": mean,
            "coverage_at_threshold": coverage,
            "preview": dest.name,
        })
        print(
            f"[{i:>4}/{len(images)}] {path.name}: mean={mean:.3f}, "
            f"≥{args.threshold:.2f}={100 * coverage:.1f}% -> {dest.name}"
        )

    report = {
        "region": args.region,
        "detector": detector_name(args.region),
        "threshold": args.threshold,
        "images": len(rows),
        "mean_coverage": float(np.mean([x["coverage_at_threshold"] for x in rows])),
        "rows": rows,
    }
    (out_dir / "mask_preview_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), "utf-8",
    )
    with (out_dir / "mask_preview_report.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} previews in {out_dir.resolve()}")
    print(f"mean coverage ≥{args.threshold:.2f}: {100 * report['mean_coverage']:.1f}%")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
