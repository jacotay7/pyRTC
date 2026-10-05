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

    def __init__(self, in_ch: int, n_modes: int, npix: int = 64, width: int = 48, cond: int = 0,
                 stem_stride: int = 1, head: int = 1024):
        super().__init__()
        w = width
        layers = [nn.Conv2d(in_ch, w, 3, stem_stride, 1), nn.BatchNorm2d(w), nn.GELU(), ResBlock(w)]
        c, size = w, npix // stem_stride
        while size > 8:  # stride-2 stages down to an 8x8 map, whatever the frame size
            c_out = min(2 * c, 4 * w)
            layers += [nn.Conv2d(c, c_out, 3, 2, 1), nn.BatchNorm2d(c_out), nn.GELU(), ResBlock(c_out)]
            c, size = c_out, (size + 1) // 2  # stride-2, padding-1 conv: ceil(n / 2)
        self.trunk = nn.Sequential(*layers)
        feat = c * size**2
        self.cond = cond
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(feat + cond, head), nn.GELU(), nn.Linear(head, n_modes)
        )
        # cond is concatenated before the head's first layer
        self.flatten = nn.Flatten()

    def forward(self, x, c=None):
        f = self.flatten(self.trunk(x))
        if self.cond:
            f = torch.cat([f, c], dim=-1)
        return self.head[1:](f)
