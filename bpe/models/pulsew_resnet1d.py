"""PulseWResNet1D: quality-weighted segment-averaged 1D ResNet for SBP/DBP regression."""

import torch
import torch.nn.functional as F
from torch import nn

from bpe.models.blocks import ensure_3d
from bpe.models.pulse_resnet1d import PulseBackbone
from bpe.models.registry import register_model


@register_model("pulsew_resnet1d")
class PulseWResNet1D(nn.Module):
    """Quality-weighted segment-averaged 1D ResNet for SBP/DBP regression.

    Extends PulseResNet1D: each segment additionally predicts a signal-quality
    score. Scores are normalised with softmax across the num_segments segments
    and used as weights for a weighted average of the per-segment SBP/DBP
    estimates. Low-quality segments are suppressed automatically.

    The quality scores are not supervised directly — the model learns to
    down-weight segments where its BP predictions are uncertain, driven solely
    by the BP regression loss.

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
        self.out_features = out_features
        # backbone outputs [SBP, DBP, quality_score] — one extra output per segment
        self.backbone = PulseBackbone(
            in_channels, out_features + 1, base_channels, dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)                                          # (B, C, L)
        B, C, L = x.shape
        seg_len = L // self.num_segments
        x = x[:, :, : seg_len * self.num_segments]                # trim if not divisible
        # split into segments: (B, C, L) → (B*S, C, seg_len)
        x = x.reshape(B, C, self.num_segments, seg_len)           # (B, C, S, seg_len)
        x = x.permute(0, 2, 1, 3).contiguous()                    # (B, S, C, seg_len)
        x = x.view(B * self.num_segments, C, seg_len)             # (B*S, C, seg_len)
        # shared backbone → [SBP, DBP, quality] per segment
        x = self.backbone(x)                                      # (B*S, F+1)
        x = x.view(B, self.num_segments, self.out_features + 1)   # (B, S, F+1)
        bp = x[:, :, : self.out_features]                         # (B, S, F)
        q  = x[:, :,   self.out_features]                         # (B, S)
        # softmax-normalised quality weights over segments
        w = F.softmax(q, dim=1).unsqueeze(-1)                     # (B, S, 1)
        return (w * bp).sum(dim=1)                                 # (B, F)
