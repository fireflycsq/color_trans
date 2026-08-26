#!/usr/bin/env python3
"""Apply a trained colour model and optionally evaluate against a CMYK target."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from color_model import (
    de_gray_cmyk,
    image_to_srgb,
    load_color_model,
    render_cmyk_to_srgb,
    resolve_de_gray_params,
    srgb_to_lab,
)
from residual_lut_model import ResidualLUTModel, edge_lift_amounts, shadow_lift_amounts
from adaptive_lut_model import AdaptiveLUTModel
from portrait_mask import portrait_mask, portrait_region_from_metadata


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
    p.add_argument(
        "--edge-lift", type=float, default=0.0,
        help="silhouette K lift 0..1; default 0 (off). e.g. 0.05 to restore",
    )
    p.add_argument(
        "--shadow-lift", type=float, default=0.0,
        help="dark-tone K lift 0..1; default 0 (off). Use e.g. 0.06 to restore",
    )
    p.add_argument(
        "--device", default="auto",
        help="PyTorch device for .pt models: auto, cpu, mps, or cuda",
    )
    p.add_argument(
        "--de-gray", action=argparse.BooleanOptionalAction, default=False,
        help="optional black crush / S / saturation into the saved CMYK; default off",
    )
    p.add_argument(
        "--de-gray-shadow-lift", type=float, default=None,
        help="midtone lift 0..1 after black crush; v3 default 0.22, otherwise 0.18",
    )
    p.add_argument("--de-gray-strength", type=float, default=None)
    p.add_argument(
        "--de-gray-highlight-ceiling", type=float, default=None,
        help="soft-roll highlights to this 0..1 ceiling so hands/white fabric do not clip",
    )
    p.add_argument(
        "--de-gray-cool", type=float, default=None,
        help="midtone Lab shift [-1, 1]; negative warms skin. v3 default -0.30, v1/v2 0.55",
    )
    args = p.parse_args()
    if args.edge_lift is not None and args.edge_lift < 0:
        raise ValueError("--edge-lift 不能为负数")
    if args.shadow_lift is not None and args.shadow_lift < 0:
        raise ValueError("--shadow-lift 不能为负数")
    if args.de_gray_shadow_lift is not None and not 0 <= args.de_gray_shadow_lift <= 1:
        raise ValueError("--de-gray-shadow-lift 必须在 [0, 1] 范围")
    if args.de_gray_strength is not None and not 0 <= args.de_gray_strength <= 1:
        raise ValueError("--de-gray-strength 必须在 [0, 1] 范围")
    if args.de_gray_highlight_ceiling is not None and not 0.5 <= args.de_gray_highlight_ceiling <= 1:
        raise ValueError("--de-gray-highlight-ceiling 必须在 [0.5, 1] 范围")
    if args.de_gray_cool is not None and not -1 <= args.de_gray_cool <= 1:
        raise ValueError("--de-gray-cool 必须在 [-1, 1] 范围")

    model = load_color_model(args.model, device=args.device)
    src = Image.open(args.input)
    pred = model.predict_image(src, edge_lift=args.edge_lift, shadow_lift=args.shadow_lift)
    out = Path(args.output)
    is_jpeg = out.suffix.lower() in {".jpg", ".jpeg"}
    baked_look = bool(getattr(model, "has_look", False))
    shadow_lift, strength, ceiling, cool = resolve_de_gray_params(
        baked_look,
        args.de_gray_shadow_lift, args.de_gray_strength,
        args.de_gray_highlight_ceiling, args.de_gray_cool,
    )
    if args.de_gray:
        pred = de_gray_cmyk(
            pred, model.target_icc,
            shadow_lift=shadow_lift,
            strength=strength,
            highlight_ceiling=ceiling,
            cool=cool,
        )
    if is_jpeg:
        preview = render_cmyk_to_srgb(pred, model.target_icc)
        preview.save(out, quality=args.quality, subsampling=0, optimize=True)
        saved_mode = preview.mode
    else:
        pred.save(out, icc_profile=model.target_icc)
        saved_mode = pred.mode
    de_gray_note = (
        f"grade punch+warm cool={cool:g}" if args.de_gray and baked_look
        else ("de-gray on" if args.de_gray else "de-gray off")
    )
    extras = []
    if isinstance(model, AdaptiveLUTModel):
        extras.append("adaptive CMYK LUT")
        if model.portrait_encoder is not None:
            extras.append(f"portrait-{model.portrait_region} on")
    if getattr(model, "skin_lut", None) is not None:
        extras.append(f"portrait-{portrait_region_from_metadata(model.metadata)} on")
    if isinstance(model, (ResidualLUTModel, AdaptiveLUTModel)):
        k_lift, c_lift = edge_lift_amounts(model.metadata, args.edge_lift)
        if k_lift > 0 or c_lift > 0:
            extras.append(f"edge-lift K={k_lift:g} C={c_lift:g}")
        shadow_k, shadow_cmy = shadow_lift_amounts(model.metadata, args.shadow_lift)
        if shadow_k > 0 or shadow_cmy > 0:
            extras.append(f"shadow-lift K={shadow_k:g} CMY={shadow_cmy:g}")
    extra = f", {', '.join(extras)}" if extras else ""
    device_note = ""
    if isinstance(model, AdaptiveLUTModel):
        device_note = f", device={model.device}"
    print(
        f"saved: {out.resolve()} ({saved_mode}, {de_gray_note}, "
        f"{model.metadata['target_profile']}{extra}{device_note})"
    )
    if args.save_portrait_mask:
        region = portrait_region_from_metadata(getattr(model, "metadata", {}))
        mask = portrait_mask(np.asarray(image_to_srgb(src), dtype=np.uint8), region=region)
        Image.fromarray(np.rint(mask * 255).astype(np.uint8), "L").save(args.save_portrait_mask)
        print(f"saved mask ({region}): {Path(args.save_portrait_mask).resolve()}")
    if args.target:
        target = Image.open(args.target)
        if target.size != pred.size:
            raise ValueError("Target and prediction sizes differ")
        metrics(pred, target, model.target_icc)


if __name__ == "__main__":
    main()
