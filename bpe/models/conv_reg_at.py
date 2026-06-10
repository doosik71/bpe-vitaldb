"""1-D CNN regression model with temporal attention pooling for SBP/DBP estimation.

Same six Conv1d stages as ConvReg (1000 → 500 → 250 → 125 → 62 → 31 → 15),
but replaces the global AvgPool with a learned temporal attention mechanism:
a two-layer 1×1 conv head produces per-timestep weights (softmax-normalised),
which are used to compute a weighted sum over the 15 time positions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from bpe.models.blocks import ensure_3d
from bpe.models.registry import register_model


@register_model("conv_reg_at")
class ConvRegAt(nn.Module):
    """1-D CNN with temporal attention pooling for blood-pressure estimation.

    Input shape:  (batch, 1, 1000)  — 8 s PPG @ 125 Hz
    Output shape: (batch, 2)        — [SBP, DBP] in mmHg

    Args:
        return_attention: When ``True``, ``forward`` returns a tuple
            ``(predictions, attention_weights)`` where attention_weights has
            shape ``(batch, 15)``.  Default: ``False``.
    """

    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, padding=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),    # 1000 -> 500

            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),    # 500 -> 250

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),    # 250 -> 125

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),    # 125 -> 62

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),    # 62 -> 31

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2),    # 31 -> 15
        )

        # Temporal attention: (B, 64, 15) -> (B, 1, 15)
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),                   # (B, 64, 1) -> (B, 64)
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2),               # output = [SBP, DBP]
        )

    def forward(self, x, return_attention: bool = False):
        x = ensure_3d(x)                            # (B, L) -> (B, 1, L)

        x = self.feature_extractor(x)               # (B, 64, 15)

        attn_score = self.attention(x)              # (B, 1, 15)
        attn_weight = F.softmax(attn_score, dim=-1) # (B, 1, 15)

        x = torch.sum(x * attn_weight, dim=-1, keepdim=True)  # (B, 64, 1)

        out = self.regressor(x)                     # (B, 2)

        if return_attention:
            return out, attn_weight.squeeze(1)      # (B, 2), (B, 15)
        return out
