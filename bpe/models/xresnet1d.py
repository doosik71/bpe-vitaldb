"""Deep XResNet-style 1D SBP/DBP regression model."""

from bpe.models.registry import register_model
from bpe.models.resnet1d import BottleneckBlock1D, ResNet1D


@register_model("xresnet1d")
class XResNet1D101(ResNet1D):
    """Deeper ResNet-101-like 1D model for stronger baselines."""

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
            layers=(3, 4, 23, 3),
            block=BottleneckBlock1D,
            dropout=dropout,
        )
