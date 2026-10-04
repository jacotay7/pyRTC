"""Reconstructor networks."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1), nn.BatchNorm2d(c), nn.GELU(),
            nn.Conv2d(c, c, 3, 1, 1), nn.BatchNorm2d(c),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.body(x))


class FPNet(nn.Module):
    """Image(s) -> modal coefficients (nm).

    ``in_ch`` image channels (frames), optional ``cond`` vector (e.g. the
    previous DM increments in modal space) injected after the conv trunk.
    """

    def __init__(self, in_ch: int, n_modes: int, npix: int = 64, width: int = 48, cond: int = 0):
        super().__init__()
        w = width
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, 1, 1), nn.BatchNorm2d(w), nn.GELU(), ResBlock(w),
            nn.Conv2d(w, 2 * w, 3, 2, 1), nn.BatchNorm2d(2 * w), nn.GELU(), ResBlock(2 * w),
            nn.Conv2d(2 * w, 4 * w, 3, 2, 1), nn.BatchNorm2d(4 * w), nn.GELU(), ResBlock(4 * w),
            nn.Conv2d(4 * w, 4 * w, 3, 2, 1), nn.BatchNorm2d(4 * w), nn.GELU(), ResBlock(4 * w),
        )
        feat = 4 * w * (npix // 8) ** 2
        self.cond = cond
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(feat + cond, 1024), nn.GELU(), nn.Linear(1024, n_modes)
        )
        # cond is concatenated before the head's first layer
        self.flatten = nn.Flatten()

    def forward(self, x, c=None):
        f = self.flatten(self.trunk(x))
        if self.cond:
            f = torch.cat([f, c], dim=-1)
        return self.head[1:](f)
