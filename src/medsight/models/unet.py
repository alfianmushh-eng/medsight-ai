Basic U-Net for segmentation.
from __future__ import annotations
import torch.nn as nn

class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(), nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU())
    def forward(self, x): return self.seq(x)

class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 1, features: tuple = (64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        ch = in_channels
        for f in features:
            self.downs.append(_DoubleConv(ch, f))
            ch = f
        self.bottleneck = _DoubleConv(ch, ch * 2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(ch * 2, f, 2, stride=2))
            self.ups.append(_DoubleConv(ch * 2, f))
            ch = f
        self.final = nn.Conv2d(features[0], num_classes, 1)

    def forward(self, x):
        skip, h = [], x
        for down in self.downs:
            h = down(h); skip.append(h); h = self.pool(h)
        h = self.bottleneck(h)
        for i in range(0, len(self.ups), 2):
            h = self.ups[i](h)
            h = torch.cat([h, skip.pop()], dim=1)
            h = self.ups[i + 1](h)
        return self.final(h)
