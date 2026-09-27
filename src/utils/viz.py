from pathlib import Path
import imageio.v2 as imageio
import numpy as np


# Generates an animated GIF overlaying GT (green tint) and prediction (red tint) masks onto CT slices.
def save_overlay_gif(vol_u8, pred3d, out_path, gt3d=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    z_depth = vol_u8.shape[2]
    frames = []

    alpha = 0.5
    red_tint = np.array([220, 60, 60], dtype=np.float32)
    green_tint = np.array([60, 220, 60], dtype=np.float32)

    for z in range(z_depth):
        slice_2d = vol_u8[:, :, z]
        frame_rgb = np.repeat(slice_2d[..., None], 3, axis=2).astype(np.float32)

        if gt3d is not None:
            gt_mask = gt3d[:, :, z].astype(bool)
            frame_rgb[gt_mask] = (1 - alpha) * frame_rgb[gt_mask] + alpha * green_tint

        pred_mask = pred3d[:, :, z].astype(bool)
        frame_rgb[pred_mask] = (1 - alpha) * frame_rgb[pred_mask] + alpha * red_tint

        frames.append(frame_rgb.astype(np.uint8))

    imageio.mimsave(str(out_path), frames, duration=0.08, loop=0)
