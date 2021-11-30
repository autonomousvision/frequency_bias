"""Optimize image directly."""

import torch
import torch.nn as nn

__all__ = ['DirectGenerator']


class DirectGenerator(nn.Module):
    def __init__(self, resolution, z, image_channels=3):
        super().__init__()
        N = len(z)
        imgs = torch.empty((N, image_channels, resolution, resolution))
        nn.init.kaiming_normal_(imgs)
        self.imgs = nn.Parameter(imgs)

    def forward(self, idx):
        assert idx.ndim == 1
        return self.imgs[idx]

    def extra_repr(self):
        N, image_channels, resolution = self.imgs.shape[:3]
        s = f'{N}, {image_channels}, {resolution}, {resolution}'
        return s