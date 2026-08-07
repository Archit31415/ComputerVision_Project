# [Week 2] src/data/dataset.py
# Dependency: src/data/nifti_io.py (Week 1), src/data/prompts.py (Week 2)
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

from src.data.nifti_io import load_volume, apply_hu_window, to_rgb, resample_isotropic
from src.data.prompts import bbox_from_mask


class BTCVSliceDataset(Dataset):
    """2-D prompted dataset of organ-containing axial slices for LoRA training."""

    def __init__(
        self,
        cases,
        organ_id,
        image_dir,
        label_dir,
        image_size=1024,
        do_resample=False,
        target_spacing=1.5,
    ):
        self.samples = []
        image_dir = Path(image_dir)
        label_dir = Path(label_dir)

        for case in cases:
            clean_case = case.replace("img", "")
            img_path = image_dir / f"img{clean_case}.nii"
            if not img_path.exists():
                img_path = image_dir / f"img{clean_case}.nii.gz"

            label_path = label_dir / f"label{clean_case}.nii"
            if not label_path.exists():
                label_path = label_dir / f"label{clean_case}.nii.gz"

            vol_img, _, spacing = load_volume(img_path)
            vol_lbl, _, _ = load_volume(label_path)

            if do_resample:
                vol_img, spacing = resample_isotropic(
                    vol_img, spacing, target=target_spacing, order=1
                )
                vol_lbl, _ = resample_isotropic(
                    vol_lbl, spacing, target=target_spacing, order=0
                )

            vol_img_u8 = apply_hu_window(vol_img, lo=-150, hi=250, as_uint8=True)

            z_depth = vol_img_u8.shape[2]
            for z in range(z_depth):
                lbl_slice = vol_lbl[:, :, z]

                if np.any(lbl_slice == organ_id):
                    gt_mask = (lbl_slice == organ_id).astype(np.uint8)

                    img_slice = to_rgb(vol_img_u8[:, :, z])

                    img_pil = Image.fromarray(img_slice).resize(
                        (image_size, image_size), resample=Image.BILINEAR
                    )
                    resized_img = np.array(img_pil, dtype=np.uint8)

                    gt_pil = Image.fromarray(gt_mask).resize(
                        (image_size, image_size), resample=Image.NEAREST
                    )
                    resized_gt = np.array(gt_pil, dtype=np.uint8)

                    bbox = bbox_from_mask(resized_gt, pad=4)

                    if bbox is None:
                        continue

                    self.samples.append((resized_img, resized_gt, bbox))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_arr, gt_arr, box_arr = self.samples[idx]

        img_tensor = torch.from_numpy(img_arr).to(torch.float32) / 255.0

        gt_tensor = torch.from_numpy(gt_arr).to(torch.float32)

        box_tensor = torch.from_numpy(box_arr).to(torch.float32)

        return img_tensor, gt_tensor, box_tensor