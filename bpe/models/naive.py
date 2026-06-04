"""Naive constant-prediction baseline for SBP/DBP estimation.

The model ignores its input entirely and returns a learned pair of constants
[sbp, dbp] that are optimised by the standard training loop.

With HuberLoss the optimum is the Huber M-estimator of the target
distribution — for distributions close to Gaussian this is indistinguishable
from the arithmetic mean of the training labels.

Trainable parameters: 2 (one per output).
"""

import torch
from torch import nn

from bpe.models.registry import register_model


@register_model("naive")
class NaiveConstant(nn.Module):
    """Predict a fixed [SBP, DBP] pair regardless of the input waveform.

    The two constants are ``nn.Parameter`` values so the standard training
    loop optimises them via back-propagation.  They are initialised to
    typical resting arterial blood-pressure values so the first epoch does
    not start from an absurd prediction.

    Args:
        sbp_init:  Initial SBP prediction in mmHg.  (default: 120.0)
        dbp_init:  Initial DBP prediction in mmHg.  (default: 75.0)
    """

    def __init__(self, sbp_init: float = 120.0, dbp_init: float = 75.0):
        super().__init__()
        # Shape (2,): [sbp, dbp]
        self.bias = nn.Parameter(torch.tensor([sbp_init, dbp_init]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the learned constant for every sample in the batch.

        Args:
            x: PPG waveform tensor of shape ``(batch, samples)`` or
               ``(batch, 1, samples)``.  Values are ignored.

        Returns:
            Tensor of shape ``(batch, 2)`` with ``[:, 0]`` = SBP and
            ``[:, 1]`` = DBP.
        """
        batch = x.size(0)
        return self.bias.unsqueeze(0).expand(batch, -1)
