"""Compact U-Net implementation for binary segmentation."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet(nn.Module):
    """Small U-Net; returns raw logits of shape (N, 1, H, W)."""

    def __init__(self, in_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        if base_channels < 1:
            raise ValueError("base_channels must be positive")
        b = base_channels
        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = ConvBlock(b, b * 2)
        self.bottleneck = ConvBlock(b * 2, b * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(b * 2, b)
        self.head = nn.Conv2d(b, 1, kernel_size=1)

    @staticmethod
    def _align(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        # Works for odd image sizes too.
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if min(x.shape[-2:]) < 4:
            raise ValueError("Images must be at least 4x4 pixels")
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool(skip1))
        x = self.bottleneck(self.pool(skip2))
        x = self.dec2(torch.cat((self._align(self.up2(x), skip2), skip2), dim=1))
        x = self.dec1(torch.cat((self._align(self.up1(x), skip1), skip1), dim=1))
        return self.head(x)
