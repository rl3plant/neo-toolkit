"""Small CNN that reads protein-position logits off a 4x5 grid crop.

Base structure (conv/relu/maxpool trunk + adaptive pool + 1x1 head) ported
from old_stuff/NEO_ECC/model.py; normalization, dropout, and kernel size are
real architectural search dimensions here, not just depth/width
(n_layers/base_channels).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

GRID_ROWS, GRID_COLS = 4, 5
Norm = Literal["none", "batch", "group"]


class Detector(nn.Module):
    """Convolutional trunk + adaptive pool to a 4x5 grid + 1x1 head -> 20 logits.

    Defaults (n_layers, base_channels, kernel_size, norm, dropout) are from
    scripts/tune_detector_architecture.py (Optuna TPE, 30 trials + a 3-seed
    confirmation run against the real leakage-free ECC setup): the original
    defaults (n_layers=2, base_channels=32, ~10K params, no dropout) scored
    best group WER 0.41-0.61 across the three D3.7 objectives; this
    architecture (~5.4K params -- fewer, but deeper with a larger receptive
    field and real dropout regularization) scored 0.0000 (neurosymbolic and
    hybrid, exactly matching the oracle, zero variance across seeds) and
    0.026 (supervised). See out/detector_arch_tuning.json.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 8,
        n_layers: int = 4,
        *,
        kernel_size: int = 5,
        norm: Norm = "none",
        dropout: float = 0.25,
    ):
        super().__init__()
        if n_layers not in (1, 2, 3, 4):
            raise ValueError("n_layers must be 1, 2, 3, or 4.")
        if norm not in ("none", "batch", "group"):
            raise ValueError("norm must be 'none', 'batch', or 'group'.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        layers: list[nn.Module] = []
        in_ch = in_channels
        for _ in range(n_layers):
            layers.append(nn.Conv2d(in_ch, base_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            if norm == "batch":
                layers.append(nn.BatchNorm2d(base_channels))
            elif norm == "group":
                layers.append(nn.GroupNorm(min(8, base_channels), base_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(2))
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = base_channels

        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((GRID_ROWS, GRID_COLS))
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, H, W) -> (B, 20) logits, one per grid cell."""
        h = self.body(x)
        h = self.pool(h)
        h = self.head(h)
        return h.view(h.size(0), -1)
