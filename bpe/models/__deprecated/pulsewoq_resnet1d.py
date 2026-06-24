"""PulseWOQResNet1D: quality-supervised overlapping-segment 1D ResNet."""

import torch
import torch.nn.functional as F
from torch import nn

from bpe.models.blocks import ensure_3d
from bpe.models.pulse_resnet1d import PulseBackbone
from bpe.models.registry import register_model


@register_model("pulsewoq_resnet1d")
class PulseWOQResNet1D(nn.Module):
    """Quality-supervised overlapping-segment 1D ResNet for SBP/DBP regression.

    Extends PulseWOResNet1D by explicitly supervising the per-segment quality
    score during training.  Quality targets are derived on-the-fly from
    per-segment BP prediction errors: a segment that predicts BP accurately
    receives a quality target near 1; a segment with large error receives a
    target near 0.

    Quality targets are computed as::

        q_target = exp(-mae_per_segment / quality_temp)

    where ``quality_temp`` (default 5.0 mmHg) controls the sensitivity of the
    error-to-quality mapping.  The combined training loss is::

        loss = bp_loss + quality_weight * MSE(sigmoid(q_logit), q_target)

    ``compute_loss`` is called automatically by the Trainer when present.

    Default configuration (L=1000, seg_len=125, stride=62):
      - 15 overlapping segments, each 125 samples (1 s @ 125 Hz)
      - ~50 % overlap between adjacent segments

    Interface:
      forward(x)               → (B, 2)  [SBP, DBP] in mmHg
      forward_with_quality(x)  → (B, 3)  [SBP, DBP, quality ∈ (0, 1)]

    Input:  (B, L) or (B, 1, L)   PPG waveform, default L=1000 (8 s @ 125 Hz)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 2,
        base_channels: int = 16,
        seg_len: int = 125,
        stride: int = 62,
        dropout: float = 0.1,
        quality_temp: float = 5.0,
        quality_weight: float = 0.5,
    ):
        super().__init__()
        self.seg_len = seg_len
        self.stride = stride
        self.out_features = out_features
        self.quality_temp = quality_temp
        self.quality_weight = quality_weight
        # backbone outputs [SBP, DBP, quality_logit] per segment
        self.backbone = PulseBackbone(
            in_channels, out_features + 1, base_channels, dropout
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _segment_forward(self, x: torch.Tensor):
        """Extract overlapping segments and run the backbone.

        Returns:
            bp (B, S, F):  per-segment BP predictions
            q  (B, S):     per-segment quality logits (un-bounded)
        """
        x = ensure_3d(x)                                           # (B, C, L)
        B, C, _ = x.shape
        x = x.unfold(2, self.seg_len, self.stride)                 # (B, C, S, seg_len)
        S = x.size(2)
        x = x.permute(0, 2, 1, 3).contiguous()                    # (B, S, C, seg_len)
        x = x.view(B * S, C, self.seg_len)                         # (B*S, C, seg_len)
        out = self.backbone(x)                                     # (B*S, F+1)
        out = out.view(B, S, self.out_features + 1)                # (B, S, F+1)
        bp  = out[:, :, : self.out_features]                       # (B, S, F)
        q   = out[:, :,   self.out_features]                       # (B, S)
        return bp, q

    def _weighted_bp(self, bp: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Softmax quality-weighted average of per-segment BP predictions."""
        w = F.softmax(q, dim=1).unsqueeze(-1)                      # (B, S, 1)
        return (w * bp).sum(dim=1)                                  # (B, F)

    # ── Public interface ──────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return quality-weighted [SBP, DBP] prediction."""
        bp, q = self._segment_forward(x)
        return self._weighted_bp(bp, q)                            # (B, F)

    def forward_with_quality(self, x: torch.Tensor) -> torch.Tensor:
        """Return [SBP, DBP, quality] with quality ∈ (0, 1).

        Quality is the softmax-weighted mean of per-segment sigmoid quality
        scores, reflecting the overall confidence of the BP prediction.
        Higher quality means the model's segments agreed and each had low
        individual BP error during training.
        """
        bp, q = self._segment_forward(x)
        pred  = self._weighted_bp(bp, q)                           # (B, F)
        w     = F.softmax(q, dim=1)                                # (B, S)
        # expected quality: weighted mean of per-segment sigmoid scores
        qual  = (w * torch.sigmoid(q)).sum(dim=1, keepdim=True)   # (B, 1)
        return torch.cat([pred, qual], dim=1)                      # (B, F+1)

    def compute_loss(
        self, x: torch.Tensor, y: torch.Tensor, criterion
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combined BP + quality loss called by the Trainer.

        Quality targets are computed from per-segment BP prediction errors
        (no gradient flows through the targets)::

            q_target = exp(-mean_abs_error / quality_temp)

        Args:
            x:         PPG input (B, L) or (B, 1, L)
            y:         Ground-truth [SBP, DBP] (B, 2)
            criterion: BP loss function (e.g. HuberLoss)

        Returns:
            (loss, pred) where pred is the quality-weighted BP (B, F).
        """
        bp, q = self._segment_forward(x)                          # (B, S, F), (B, S)

        # derive quality targets from per-segment BP errors — no gradient
        with torch.no_grad():
            bp_err   = (bp - y.unsqueeze(1)).abs().mean(dim=-1)   # (B, S), mmHg MAE
            q_target = torch.exp(-bp_err / self.quality_temp)     # (B, S), ∈ (0, 1]

        pred    = self._weighted_bp(bp, q)                        # (B, F)
        bp_loss = criterion(pred, y)
        q_loss  = F.mse_loss(torch.sigmoid(q), q_target)
        loss    = bp_loss + self.quality_weight * q_loss

        return loss, pred
