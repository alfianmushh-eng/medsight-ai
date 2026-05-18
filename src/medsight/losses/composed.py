Combined segmentation loss.
from __future__ import annotations
import torch.nn as nn
from .dice import DiceLoss

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight; self.bce_weight = bce_weight

    def forward(self, pred, target):
        target_f = target.float().unsqueeze(1) if target.dim() == 3 else target.float()
        return self.dice_weight * self.dice(pred, target) + self.bce_weight * self.bce(pred, target_f)
