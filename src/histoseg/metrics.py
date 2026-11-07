"""Binary segmentation metrics; report per-image Dice and IoU."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def segmentation_loss(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Binary cross entropy plus soft Dice loss for training."""
    if logits.shape != masks.shape:
        raise ValueError(f"logits {logits.shape} and masks {masks.shape} shape mismatch")
    bce = F.binary_cross_entropy_with_logits(logits, masks)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * masks).sum(dims)
    dice = (2.0 * intersection + 1e-6) / (probs.sum(dims) + masks.sum(dims) + 1e-6)
    return bce + (1.0 - dice.mean())


@torch.no_grad()
def batch_dice_iou(logits: torch.Tensor, masks: torch.Tensor, threshold: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Dice and IoU for *each image* in the batch.

    Empty prediction + empty ground truth scores 1.0 by convention.
    """
    pred = torch.sigmoid(logits) >= threshold
    true = masks >= 0.5
    dims = (1, 2, 3)
    intersection = (pred & true).sum(dims).float()
    n_pred = pred.sum(dims).float()
    n_true = true.sum(dims).float()
    dice = torch.where(n_pred + n_true > 0, 2 * intersection / (n_pred + n_true).clamp_min(1), 1.0)
    union = (pred | true).sum(dims).float()
    iou = torch.where(union > 0, intersection / union.clamp_min(1), 1.0)
    return dice, iou
