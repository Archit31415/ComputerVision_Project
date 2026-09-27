#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np
import imageio.v2 as imageio

from src.data.nifti_io import load_volume, apply_hu_window, to_rgb
from src.data.volume_to_frames import write_frames_png
from src.data.prompts import bbox_from_mask, best_start_slice


def draw_rect(img, x0, y0, x1, y1, color=(0, 255, 0), thickness=2):
    img[y0:y0 + thickness, x0:x1] = color
    img[y1 - thickness:y1, x0:x1] = color
    img[y0:y1, x0:x0 + thickness] = color
    img[y0:y1, x1 - thickness:x1] = color


def main():
    parser = argparse.ArgumentParser(description="Preprocess one BTCV case")
    parser.add_argument("--case", required=True, help="e.g. img0001")
    parser.add_argument("--organ", type=int, default=6, help="BTCV organ label (6=liver, 1=spleen, 11=pancreas)")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    img_path = Path(cfg["paths"]["raw_images"]) / f"{args.case}.nii"
    lbl_path = Path(cfg["paths"]["raw_labels"]) / f"{args.case.replace('img', 'label')}.nii"
    out_dir = Path(cfg["paths"]["processed"]) / args.case

    print(f"Loading {img_path}")
    vol, affine, spacing = load_volume(str(img_path))
    print(f"  shape={vol.shape}  spacing={tuple(f'{s:.2f}' for s in spacing)} mm")

    do_resample = cfg.get("preprocess", {}).get("resample_isotropic", False)
    target_spacing = cfg.get("preprocess", {}).get("target_spacing_mm", 1.5)

    # Convert volume slices into sequential JPEG frames for SAM 2
    n = write_frames_png(
        str(img_path),
        str(out_dir),
        do_resample=do_resample,
        target_spacing=target_spacing,
    )
    print(f"  Wrote {n} JPEG frames -> {out_dir}")

    if not lbl_path.exists():
        print("  No label file found — skipping overlay.")
        return

    lbl, _, _ = load_volume(str(lbl_path))
    try:
        # Locate optimal start slice and draw bounding box overlay for visual verification
        start_z = best_start_slice(lbl, args.organ)
    except ValueError as e:
        print(f"  {e}")
        return

    vol_u8 = apply_hu_window(vol)
    slice_u8 = vol_u8[:, :, start_z]
    rgb = to_rgb(slice_u8).copy()
    gt_slice = (lbl[:, :, start_z] == args.organ).astype(np.uint8)

    rgb[gt_slice == 1] = (
        rgb[gt_slice == 1] * 0.5 + np.array([220, 60, 60]) * 0.5
    ).astype(np.uint8)

    bbox = bbox_from_mask(gt_slice, pad=4)
    if bbox is not None:
        x0, y0, x1, y1 = bbox.astype(int)
        draw_rect(rgb, x0, y0, x1, y1, color=(0, 255, 0), thickness=2)

    overlay_path = out_dir / "overlay.png"
    imageio.imwrite(str(overlay_path), rgb)
    print(f"  Best start slice: z={start_z}  bbox={bbox}")
    print(f"  Saved overlay -> {overlay_path}")


if __name__ == "__main__":
    main()
