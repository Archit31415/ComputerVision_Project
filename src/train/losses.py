import torch
import torch.nn.functional as F


def dice_loss(logits, targets, eps=1e-6):
    p = torch.sigmoid(logits)
    inter = (p * targets).sum()
    union = p.sum() + targets.sum()
    dice_coeff = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice_coeff


# Combined loss: Dice handles region overlap & class imbalance while BCE refines pixel-level boundaries.
def total_loss(logits, targets):
    d_loss = dice_loss(logits, targets)
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets)
    return d_loss + bce_loss