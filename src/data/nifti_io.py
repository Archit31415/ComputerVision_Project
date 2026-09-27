import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


def load_volume_sitk(path):
    try:
        import SimpleITK as sitk

        sitk_img = sitk.ReadImage(str(path))
        vol = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        # SimpleITK uses (Z, Y, X) order -> transpose to standard (X, Y, Z) / (H, W, Z).
        vol = np.transpose(vol, (2, 1, 0))
        spacing = tuple(float(s) for s in sitk_img.GetSpacing()[:3])
        affine = np.eye(4, dtype=np.float32)
        return vol, affine, spacing
    except Exception as e:
        raise RuntimeError(f"SimpleITK failed to load {path}: {e}")


def load_volume(path):
    try:
        img = nib.load(path)
        img = nib.as_closest_canonical(img)
        vol = img.get_fdata().astype(np.float32)
        spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
        affine = img.affine
    except Exception as err:
        print(f"nibabel failed to load {path} ({err}). Falling back to SimpleITK...")
        vol, affine, spacing = load_volume_sitk(path)

    print(f"Loaded volume: {path} | Spacing: {spacing}")
    return vol, affine, spacing


# Soft-tissue HU windowing isolates organ structures from surrounding bone and contrast.
def apply_hu_window(vol, lo=-150, hi=250, as_uint8=True):
    v = np.clip(vol, lo, hi)
    v = (v - lo) / (hi - lo)
    return (v * 255).astype(np.uint8) if as_uint8 else v.astype(np.float32)


# SAM 2 image encoder requires 3-channel input; single-channel CT slices are duplicated across RGB.
def to_rgb(slice2d_u8):
    return np.repeat(slice2d_u8[..., None], 3, axis=2)


# Isotropic resampling normalizes physical voxel scales across scans.
def resample_isotropic(vol, spacing, target=1.5, order=1):
    factors = [s / target for s in spacing]
    resampled = zoom(vol, factors, order=order)
    return resampled, (target, target, target)
