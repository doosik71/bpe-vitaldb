"""MInception model with optional demographic side-channel features."""

import torch
from torch import nn

from bpe.models.minception import MInceptionBackbone
from bpe.models.registry import register_model


@register_model("minception_demographic")
class MInceptionDemographic(nn.Module):
    """Fuse PPG features with age/sex/BMI-style tabular features."""

    def __init__(
        self,
        in_channels: int = 1,
        demographic_features: int = 3,
        out_features: int = 2,
        channels: tuple[int, ...] = (48, 64, 96, 128),
        kernel_sizes: tuple[int, ...] = (9, 19, 39),
        demographic_hidden: int = 32,
        dropout: float = 0.15,
    ):
        super().__init__()
        if demographic_features < 1:
            raise ValueError("demographic_features must be >= 1")

        self.backbone = MInceptionBackbone(in_channels, channels, kernel_sizes)
        self.signal_pool = nn.AdaptiveAvgPool1d(1)
        self.demographic_net = nn.Sequential(
            nn.Linear(demographic_features, demographic_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(demographic_hidden, demographic_hidden),
            nn.ReLU(inplace=True),
        )
        fused_features = self.backbone.out_channels + demographic_hidden
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_features, fused_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fused_features // 2, out_features),
        )

    def forward(
        self,
        x: torch.Tensor,
        demographics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if demographics is None:
            raise ValueError("demographics tensor is required for MInceptionDemographic")
        signal = self.backbone(x)
        signal = self.signal_pool(signal).flatten(1)
        demo = self.demographic_net(demographics.float())
        return self.head(torch.cat([signal, demo], dim=1))

