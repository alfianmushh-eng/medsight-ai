import torch
from medsight.losses import DiceLoss, FocalLoss, DiceBCELoss

def test_dice_loss():
    pred = torch.randn(2, 1, 16, 16)
    target = torch.randint(0, 2, (2, 16, 16))
    loss = DiceLoss()
    val = loss(pred, target)
    assert val.item() >= 0

def test_focal_loss():
    pred = torch.randn(2, 3)
    target = torch.randint(0, 3, (2,))
    loss = FocalLoss()
    val = loss(pred, target)
    assert val.item() >= 0
