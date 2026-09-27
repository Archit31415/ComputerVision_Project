from pathlib import Path
import numpy as np
from PIL import Image

from src.data.nifti_io import load_volume, apply_hu_window, to_rgb, resample_isotropic


# Slices 3D volumes into sequential JPEG frames to match SAM 2's video predictor input specification.
def write_frames_png(nifti_path, out_dir, do_resample=False, target_spacing=1.5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vol, _, spacing = load_volume(nifti_path)

    if do_resample:
        vol, _ = resample_isotropic(vol, spacing, target=target_spacing, order=1)

    vol_u8 = apply_hu_window(vol, lo=-150, hi=250, as_uint8=True)

    z_depth = vol_u8.shape[2]
    for z in range(z_depth):
        slice_2d = vol_u8[:, :, z]
        slice_rgb = to_rgb(slice_2d)
        frame_name = f"{z:05d}.jpg"
        frame_path = out_dir / frame_name
        Image.fromarray(slice_rgb).save(frame_path, quality=95)

    return z_depth


def frames_to_arrays(frames_dir):
    frames_dir = Path(frames_dir)
    frame_paths = sorted(frames_dir.glob("*.jpg"))

    if not frame_paths:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)

    slices_list = []
    for path in frame_paths:
        with Image.open(path) as img:
            slices_list.append(np.array(img, dtype=np.uint8))

    return np.stack(slices_list, axis=0)
