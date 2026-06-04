"""PulseWOResNet1D: quality-weighted overlapping-segment 1D ResNet for SBP/DBP regression."""

import torch
import torch.nn.functional as F
from torch import nn

from bpe.models.blocks import ensure_3d
from bpe.models.pulse_resnet1d import PulseBackbone
from bpe.models.registry import register_model


@register_model("pulsewo_resnet1d")
class PulseWOResNet1D(nn.Module):
    """Quality-weighted overlapping-segment 1D ResNet for SBP/DBP regression.

    Extends PulseWResNet1D with 50 % overlapping windows. Non-overlapping
    extraction may place a systolic peak at a segment boundary; overlapping
    windows ensure that every pulse appears near the centre of at least one
    segment, yielding more stable feature extraction.

    Default configuration (L=1000, seg_len=125, stride=62):
      - 15 overlapping segments, each 125 samples (1 s @ 125 Hz)
      - ~50 % overlap between adjacent segments

    Each segment independently predicts [SBP, DBP, quality]. Quality scores
    are normalised with softmax across all windows and used as weights for a
    quality-weighted average of the SBP/DBP estimates.

    Input:  (B, L) or (B, 1, L)   PPG waveform, default L=1000 (8 s @ 125 Hz)
    Output: (B, 2)                 [SBP, DBP] in mmHg
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 2,
        base_channels: int = 16,
        seg_len: int = 125,
        stride: int = 62,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seg_len = seg_len
        self.stride = stride
        self.out_features = out_features
        # backbone outputs [SBP, DBP, quality_score] — one extra output per segment
        self.backbone = PulseBackbone(
            in_channels, out_features + 1, base_channels, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)                                           # (B, C, L)
        B, C, _ = x.shape
        # overlapping windows via unfold: (B, C, L) → (B, C, S, seg_len)
        x = x.unfold(2, self.seg_len, self.stride)                 # (B, C, S, seg_len)
        S = x.size(2)
        x = x.permute(0, 2, 1, 3).contiguous()                    # (B, S, C, seg_len)
        x = x.view(B * S, C, self.seg_len)                         # (B*S, C, seg_len)
        # shared backbone → [SBP, DBP, quality] per segment
        x = self.backbone(x)                                       # (B*S, F+1)
        x = x.view(B, S, self.out_features + 1)                    # (B, S, F+1)
        bp = x[:, :, : self.out_features]                          # (B, S, F)
        q  = x[:, :,   self.out_features]                          # (B, S)
        # softmax-normalised quality weights over all S windows
        w = F.softmax(q, dim=1).unsqueeze(-1)                      # (B, S, 1)
        return (w * bp).sum(dim=1)                                  # (B, F)
