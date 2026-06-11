"""Hybrid NAS variant of ConvReg for SBP/DBP estimation.

This model keeps the six-stage ConvReg data flow, but learns two kinds of
architecture choices:

1. Stage-local kernel search over {3, 5, 7}
2. Backbone-level search over:
   - conv type: standard or depthwise separable
   - channel multiplier: 1.0x, 1.5x, 2.0x

Training runs the full supernet with softmax-weighted choices. Evaluation runs
only the best backbone and best kernel per stage for faster inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from bpe.models.blocks import ensure_3d, validate_kernel_sizes
from bpe.models.registry import register_model


BASE_CHANNELS = (8, 16, 32, 64, 64, 64)
KERNEL_SIZES = (3, 5, 7)
CHANNEL_MULTIPLIERS = (1.0, 1.5, 2.0)
BACKBONE_SPECS = (
    ("standard", 1.0),
    ("standard", 1.5),
    ("standard", 2.0),
    ("depthwise_separable", 1.0),
    ("depthwise_separable", 1.5),
    ("depthwise_separable", 2.0),
)


def _scaled_channels(base_channels: tuple[int, ...], multiplier: float) -> tuple[int, ...]:
    return tuple(max(1, int(round(ch * multiplier))) for ch in base_channels)


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-separable 1-D convolution without activation."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class KernelChoiceConvBlock(nn.Module):
    """One stage with learnable kernel selection over fixed candidate sizes."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        conv_type: str,
        kernel_sizes: tuple[int, ...] = KERNEL_SIZES,
    ):
        super().__init__()
        self.kernel_sizes = validate_kernel_sizes(kernel_sizes)
        self.conv_type = conv_type
        self.kernel_logits = nn.Parameter(torch.zeros(len(self.kernel_sizes)))

        if conv_type == "standard":
            self.ops = nn.ModuleList([
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=k,
                    padding=k // 2,
                    bias=False,
                )
                for k in self.kernel_sizes
            ])
        elif conv_type == "depthwise_separable":
            self.ops = nn.ModuleList([
                DepthwiseSeparableConv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=k,
                )
                for k in self.kernel_sizes
            ])
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")

        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.pool = nn.AvgPool1d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weights = torch.softmax(self.kernel_logits, dim=0)
            mixed = None
            for weight, op in zip(weights, self.ops):
                out = op(x)
                mixed = out.mul(weight) if mixed is None else mixed + out.mul(weight)
            x = mixed
        else:
            idx = int(self.kernel_logits.argmax().item())
            x = self.ops[idx](x)

        x = self.norm(x)
        x = self.act(x)
        x = self.pool(x)
        return x

    def best_kernel_size(self) -> int:
        idx = int(self.kernel_logits.argmax().item())
        return self.kernel_sizes[idx]


class ConvRegNasBackbone(nn.Module):
    """Fixed-width backbone with stage-local kernel search."""

    def __init__(
        self,
        *,
        conv_type: str,
        channel_multiplier: float,
        base_channels: tuple[int, ...] = BASE_CHANNELS,
        kernel_sizes: tuple[int, ...] = KERNEL_SIZES,
    ):
        super().__init__()
        self.conv_type = conv_type
        self.channel_multiplier = channel_multiplier
        self.channels = _scaled_channels(base_channels, channel_multiplier)

        stages = []
        in_channels = 1
        for out_channels in self.channels:
            stages.append(
                KernelChoiceConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    conv_type=conv_type,
                    kernel_sizes=kernel_sizes,
                )
            )
            in_channels = out_channels
        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool1d(1)

    @property
    def out_channels(self) -> int:
        return self.channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return self.pool(x)

    def best_kernel_sizes(self) -> tuple[int, ...]:
        return tuple(stage.best_kernel_size() for stage in self.stages)


@register_model("conv_reg_nas")
class ConvRegNas(nn.Module):
    """ConvReg supernet with architecture-level backbone search."""

    def __init__(self):
        super().__init__()
        self.backbone_specs = BACKBONE_SPECS
        self.backbone_logits = nn.Parameter(torch.zeros(len(self.backbone_specs)))

        self.backbones = nn.ModuleList([
            ConvRegNasBackbone(conv_type=conv_type, channel_multiplier=multiplier)
            for conv_type, multiplier in self.backbone_specs
        ])
        self.projections = nn.ModuleList([
            nn.Linear(backbone.out_channels, 64)
            for backbone in self.backbones
        ])
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),
        )

    def _project_backbone(self, backbone_idx: int, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbones[backbone_idx](x).flatten(1)
        return self.projections[backbone_idx](feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)

        if self.training:
            weights = torch.softmax(self.backbone_logits, dim=0)
            mixed = None
            for idx, weight in enumerate(weights):
                feat = self._project_backbone(idx, x)
                mixed = feat.mul(weight) if mixed is None else mixed + feat.mul(weight)
            return self.regressor(mixed)

        idx = self.best_backbone_index()
        feat = self._project_backbone(idx, x)
        return self.regressor(feat)

    def best_backbone_index(self) -> int:
        return int(self.backbone_logits.argmax().item())

    def architecture_summary(self) -> dict:
        idx = self.best_backbone_index()
        conv_type, multiplier = self.backbone_specs[idx]
        backbone = self.backbones[idx]
        return {
            "backbone_index": idx,
            "conv_type": conv_type,
            "channel_multiplier": multiplier,
            "channels": backbone.channels,
            "kernel_sizes": backbone.best_kernel_sizes(),
        }
