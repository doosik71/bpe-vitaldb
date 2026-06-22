"""PulseResNet1D: segment-averaged 1D ResNet for SBP/DBP regression."""

import torch
import torch.nn.functional as F
from torch import nn

from bpe.models.blocks import ConvBnAct1d, RegressionHead, ensure_3d
from bpe.models.registry import register_model
from bpe.models.resnet1d import BasicBlock1D


class PulseBackbone(nn.Module):
    """Compact 3-stage ResNet for short PPG segments (~125 samples)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 2,
        base_channels: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        C = base_channels
        self.stem = nn.Sequential(
            ConvBnAct1d(in_channels, C, 7, stride=2),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = nn.Sequential(BasicBlock1D(C, C, stride=1))
        self.stage2 = nn.Sequential(BasicBlock1D(C, C * 2, stride=2))
        self.stage3 = nn.Sequential(BasicBlock1D(C * 2, C * 4, stride=2))
        self.head = RegressionHead(C * 4, out_features, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


@register_model("pulse_resnet1d")
class PulseResNet1D(nn.Module):
    """Segment-averaged 1D ResNet for SBP/DBP regression.

    Splits the input into num_segments non-overlapping windows, estimates
    SBP/DBP independently for each via a shared PulseBackbone, and returns
    the average of the num_segments predictions.

    Input:  (B, L) or (B, 1, L)   PPG waveform, default L=1000 (8 s @ 125 Hz)
    Output: (B, 2)                 [SBP, DBP] in mmHg
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 2,
        base_channels: int = 16,
        num_segments: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_segments = num_segments
        self.backbone = PulseBackbone(in_channels, out_features, base_channels, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)                                        # (B, C, L)
        B, C, L = x.shape
        seg_len = L // self.num_segments
        x = x[:, :, : seg_len * self.num_segments]              # trim if not divisible
        # (B, C, L) → (B*S, C, seg_len)
        x = x.reshape(B, C, self.num_segments, seg_len)         # (B, C, S, seg_len)
        x = x.permute(0, 2, 1, 3).contiguous()                  # (B, S, C, seg_len)
        x = x.view(B * self.num_segments, C, seg_len)           # (B*S, C, seg_len)
        # shared backbone
        x = self.backbone(x)                                    # (B*S, F)
        # average over segments via AvgPool1d
        x = x.view(B, self.num_segments, -1).permute(0, 2, 1)  # (B, F, S)
        return F.avg_pool1d(x, self.num_segments).squeeze(-1)   # (B, F)
