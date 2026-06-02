"""ResNet1D-Tiny: 25 % of the original ResNet1D layer count.

Original ResNet1D: 4 stages × 2 BasicBlock1D = 8 blocks.
Tiny             : 2 stages × 1 BasicBlock1D = 2 blocks.
Channel progression is truncated at stage 2 (32 → 64).
"""

import torch
from torch import nn

from bpe.models.blocks import ConvBnAct1d, RegressionHead, ensure_3d
from bpe.models.registry import register_model
from bpe.models.resnet1d import BasicBlock1D


@register_model("resnet1d_tiny")
class ResNet1DTiny(nn.Module):
    """Quarter-depth 1D ResNet — 2 residual blocks across 2 stages."""

    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 2,
        base_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._ch = base_channels
        self.stem = nn.Sequential(
            ConvBnAct1d(in_channels, base_channels, 15, stride=2),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = self._make_stage(BasicBlock1D, base_channels,     stride=1)
        self.stage2 = self._make_stage(BasicBlock1D, base_channels * 2, stride=2)
        self.head = RegressionHead(self._ch, out_features, dropout)

    def _make_stage(
        self,
        block: type[nn.Module],
        out_channels: int,
        stride: int,
    ) -> nn.Sequential:
        layer = block(self._ch, out_channels, stride)
        self._ch = out_channels * block.expansion
        return nn.Sequential(layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return self.head(x)
