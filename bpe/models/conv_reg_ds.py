"""Depthwise-separable CNN regression baseline for SBP/DBP estimation.

Six depthwise/pointwise Conv1d stages with BatchNorm + ReLU + AvgPool halve
the time dimension at each step (1000 → 500 → 250 → 125 → 62 → 31 → 15),
followed by AdaptiveAvgPool1d(1). A two-layer MLP regressor maps the 64-d
feature vector to [SBP, DBP].
"""

import torch.nn as nn

from bpe.models.blocks import ensure_3d
from bpe.models.registry import register_model


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1-D convolution.

    Depthwise Conv1d extracts temporal features independently per channel.
    Pointwise Conv1d mixes information across channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        bias: bool = False,
    ):
        super().__init__()

        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


@register_model("conv_reg_ds")
class ConvRegDs(nn.Module):
    """Lightweight depthwise-separable 1-D CNN regressor.

    Input shape:  (batch, 1, 1000)  — 8 s PPG @ 125 Hz
    Output shape: (batch, 2)        — [SBP, DBP] in mmHg
    """

    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            DepthwiseSeparableConv1d(
                in_channels=1,
                out_channels=8,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),    # 1000 -> 500

            DepthwiseSeparableConv1d(
                in_channels=8,
                out_channels=16,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),    # 500 -> 250

            DepthwiseSeparableConv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),    # 250 -> 125

            DepthwiseSeparableConv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),    # 125 -> 62

            DepthwiseSeparableConv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),    # 62 -> 31

            DepthwiseSeparableConv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),    # 31 -> 15

            nn.AdaptiveAvgPool1d(1),        # (B, 64, 15) -> (B, 64, 1)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),                   # (B, 64, 1) -> (B, 64)
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),               # output = [SBP, DBP]
        )

    def forward(self, x):
        x = ensure_3d(x)                    # (B, L) -> (B, 1, L)
        x = self.feature_extractor(x)
        x = self.regressor(x)
        return x