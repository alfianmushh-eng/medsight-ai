Dice loss for segmentation.
from __future__ import annotations
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred) if pred.size(1) == 1 else torch.softmax(pred, dim=1)
        target = target.float().unsqueeze(1) if target.dim() == 3 else target.float()
        inter = (pred * target).sum(dim=(2, 3)) if pred.dim() == 4 else (pred * target).sum()
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) if pred.dim() == 4 else pred.sum() + target.sum()
        dice = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()
