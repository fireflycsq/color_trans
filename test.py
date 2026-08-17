#!/usr/bin/env python3
"""Apply a trained colour model and optionally evaluate against a CMYK target."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from color_model import image_to_srgb, load_color_model, render_cmyk_to_srgb, srgb_to_lab
from portrait_mask import portrait_skin_mask


def metrics(pred: Image.Image, target: Image.Image, icc: bytes) -> None:
    p = np.asarray(pred, dtype=np.float32)
    t = np.asarray(target.convert("CMYK"), dtype=np.float32)
    err = p - t
    mae = np.mean(np.abs(err), axis=(0, 1))
    rmse = float(np.sqrt(np.mean(err**2)))
    psnr_cmyk = float("inf") if rmse == 0 else 20*np.log10(255/rmse)

    # Downsample only for perceptual reporting; full-resolution prediction is saved.
    size = (min(1200, pred.width), round(pred.height * min(1200, pred.width) / pred.width))
    pr = np.asarray(render_cmyk_to_srgb(pred.resize(size), icc), dtype=np.float32) / 255
    tr = np.asarray(render_cmyk_to_srgb(target.resize(size), icc), dtype=np.float32) / 255
    rgb_rmse = float(np.sqrt(np.mean((pr-tr)**2)))
    lab_p, lab_t = srgb_to_lab(pr), srgb_to_lab(tr)
    de76 = np.sqrt(np.sum((lab_p-lab_t)**2, axis=-1))
    print("CMYK MAE [C M Y K]:", np.round(mae, 3).tolist())
    print(f"CMYK PSNR: {psnr_cmyk:.3f} dB")
    print(f"rendered RGB PSNR: {20*np.log10(1/rgb_rmse):.3f} dB")
    print(f"rendered DeltaE76 mean/p50/p95/max: {de76.mean():.3f} / "
          f"{np.percentile(de76,50):.3f} / {np.percentile(de76,95):.3f} / {de76.max():.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True, help="output .jpg or lossless .tif")
    p.add_argument("--target", help="optional aligned CMYK target for evaluation")
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--save-portrait-mask", help="optional PNG of the person-skin mask")
    args = p.parse_args()

    model = load_color_model(args.model)
    src = Image.open(args.input)
    pred = model.predict_image(src)
    out = Path(args.output)
    save_args = {"icc_profile": model.target_icc}
    if out.suffix.lower() in {".jpg", ".jpeg"}:
        save_args.update(quality=args.quality, subsampling=0)
    pred.save(out, **save_args)
    extras = []
    if getattr(model, "skin_lut", None) is not None:
        extras.append("portrait-skin on")
    extra = f", {', '.join(extras)}" if extras else ""
    print(f"saved: {out.resolve()} ({pred.mode}, embedded {model.metadata['target_profile']}{extra})")
    if args.save_portrait_mask:
        mask = portrait_skin_mask(np.asarray(image_to_srgb(src), dtype=np.uint8))
        Image.fromarray(np.rint(mask * 255).astype(np.uint8), "L").save(args.save_portrait_mask)
        print(f"saved mask: {Path(args.save_portrait_mask).resolve()}")
    if args.target:
        target = Image.open(args.target)
        if target.size != pred.size:
            raise ValueError("Target and prediction sizes differ")
        metrics(pred, target, model.target_icc)


if __name__ == "__main__":
    main()
