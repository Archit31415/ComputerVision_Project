import numpy as np
import torch
from monai.metrics import compute_hausdorff_distance


# 3D Dice Similarity Coefficient measures overall volumetric region overlap (0 -> 1).
def dice_score(pred3d, gt3d, eps=1e-6):
    pred_bool = pred3d.astype(bool)
    gt_bool = gt3d.astype(bool)

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    pred_sum = pred_bool.sum()
    gt_sum = gt_bool.sum()

    return float((2.0 * intersection + eps) / (pred_sum + gt_sum + eps))


# 95th Percentile Hausdorff Distance measures maximum 3D boundary distance error in millimeters.
def hd95(pred3d, gt3d, spacing):
    p = torch.tensor(pred3d.astype(np.uint8), dtype=torch.float32)[None, None]
    g = torch.tensor(gt3d.astype(np.uint8), dtype=torch.float32)[None, None]
    sp_zyx = (float(spacing[2]), float(spacing[1]), float(spacing[0]))

    dist = compute_hausdorff_distance(p, g, percentile=95, spacing=sp_zyx)
    return float(dist.squeeze())


# Volumetric Prediction Error measures absolute 3D organ volume discrepancy in cm³.
def vpe(pred3d, gt3d, spacing):
    voxel_vol_cm3 = (float(spacing[0]) * float(spacing[1]) * float(spacing[2])) / 1000.0
    pred_vol = float(pred3d.astype(bool).sum()) * voxel_vol_cm3
    gt_vol = float(gt3d.astype(bool).sum()) * voxel_vol_cm3
    return abs(pred_vol - gt_vol)