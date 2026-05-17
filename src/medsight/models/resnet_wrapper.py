ResNet wrapper with configurable backbone.
from __future__ import annotations
import torch.nn as nn
from torchvision import models

class ResNetClassifier(nn.Module):
    def __init__(self, *, depth: int = 18, num_classes: int = 2, pretrained: bool = True, in_channels: int = 3):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        depths = {18: models.resnet18, 34: models.resnet34, 50: models.resnet50}
        if depth not in depths:
            raise ValueError(f"unsupported depth: {depth}")
        self.backbone = depths[depth](weights=weights)
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(feat, num_classes)

    def forward(self, x):
        return self.backbone(x)
