"""Multi-scale Inception-style 1D CNN for direct BP regression."""

import torch
from torch import nn

from bpe.models.blocks import ConvBnAct1d, RegressionHead, ensure_3d, validate_kernel_sizes
from bpe.models.registry import register_model


class InceptionBlock1D(nn.Module):
    """Parallel temporal convolutions with different receptive fields."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int, ...] = (9, 19, 39),
        bottleneck_channels: int | None = None,
    ):
        super().__init__()
        kernel_sizes = validate_kernel_sizes(kernel_sizes)
        bottleneck_channels = bottleneck_channels or max(out_channels // 2, 8)
        branch_channels = out_channels // len(kernel_sizes)
        remainder = out_channels - branch_channels * len(kernel_sizes)

        self.bottleneck = ConvBnAct1d(in_channels, bottleneck_channels, 1)
        branches = []
        for i, kernel_size in enumerate(kernel_sizes):
            channels = branch_channels + (1 if i < remainder else 0)
            branches.append(ConvBnAct1d(bottleneck_channels, channels, kernel_size))
        self.branches = nn.ModuleList(branches)
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            ConvBnAct1d(in_channels, branch_channels, 1),
        )
        merged_channels = out_channels + branch_channels
        self.project = ConvBnAct1d(merged_channels, out_channels, 1)
        self.shortcut = self._shortcut(in_channels, out_channels)

    @staticmethod
    def _shortcut(in_channels: int, out_channels: int) -> nn.Module:
        if in_channels == out_channels:
            return nn.Identity()
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        branches = [branch(z) for branch in self.branches]
        branches.append(self.pool_branch(x))
        y = self.project(torch.cat(branches, dim=1))
        return torch.relu(y + self.shortcut(x))


class MInceptionBackbone(nn.Module):
    """Reusable MInception feature extractor."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] = (48, 64, 96, 128),
        kernel_sizes: tuple[int, ...] = (9, 19, 39),
    ):
        super().__init__()
        layers: list[nn.Module] = []
        current_channels = in_channels
        for out_channels in channels:
            layers.append(InceptionBlock1D(current_channels, out_channels, kernel_sizes))
            layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
            current_channels = out_channels
        self.net = nn.Sequential(*layers)
        self.out_channels = current_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(ensure_3d(x))


@register_model("minception")
class MInception(nn.Module):
    """Multi-scale direct SBP/DBP regression model."""

    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 2,
        channels: tuple[int, ...] = (48, 64, 96, 128),
        kernel_sizes: tuple[int, ...] = (9, 19, 39),
        dropout: float = 0.15,
    ):
        super().__init__()
        self.backbone = MInceptionBackbone(in_channels, channels, kernel_sizes)
        self.head = RegressionHead(self.backbone.out_channels, out_features, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

