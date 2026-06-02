"""ResNet1D-Micro: ~10 % of the original ResNet1D layer count.

Original ResNet1D: 4 stages × 2 BasicBlock1D = 8 blocks.
Micro            : 1 stage  × 1 BasicBlock1D = 1 block.
Channel progression is limited to a single stage (32 only).
"""

import torch
from torch import nn

from bpe.models.blocks import ConvBnAct1d, RegressionHead, ensure_3d
from bpe.models.registry import register_model
from bpe.models.resnet1d import BasicBlock1D


@register_model("resnet1d_micro")
class ResNet1DMicro(nn.Module):
    """Minimal 1D ResNet — 1 residual block in a single stage."""

    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 2,
        base_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBnAct1d(in_channels, base_channels, 15, stride=2),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = nn.Sequential(BasicBlock1D(base_channels, base_channels, stride=1))
        self.head = RegressionHead(base_channels, out_features, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)
        x = self.stem(x)
        x = self.stage1(x)
        return self.head(x)
