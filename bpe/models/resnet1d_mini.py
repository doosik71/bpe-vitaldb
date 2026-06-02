"""ResNet1D-Mini: 50 % of the original ResNet1D layer count.

Original ResNet1D: 4 stages × 2 BasicBlock1D = 8 blocks.
Mini             : 4 stages × 1 BasicBlock1D = 4 blocks.
Channel widths and the stem are unchanged.
"""

from bpe.models.registry import register_model
from bpe.models.resnet1d import BasicBlock1D, ResNet1D


@register_model("resnet1d_mini")
class ResNet1DMini(ResNet1D):
    """Halved-depth 1D ResNet — 4 residual blocks across 4 stages."""

    def __init__(
        self,
        in_channels: int = 1,
        out_features: int = 2,
        base_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__(
            in_channels=in_channels,
            out_features=out_features,
            base_channels=base_channels,
            layers=(1, 1, 1, 1),
            block=BasicBlock1D,
            dropout=dropout,
        )
