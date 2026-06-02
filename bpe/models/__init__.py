"""Model registry and direct SBP/DBP regression architectures."""

from bpe.models.minception import MInception
from bpe.models.minception_demographic import MInceptionDemographic
from bpe.models.registry import (
    create_model,
    get_model_class,
    list_models,
    register_model,
)
from bpe.models.resnet1d import ResNet1D
from bpe.models.resnet1d_micro import ResNet1DMicro
from bpe.models.resnet1d_mini import ResNet1DMini
from bpe.models.resnet1d_tiny import ResNet1DTiny
from bpe.models.st_resnet import SpectroTemporalResNet
from bpe.models.xresnet1d import XResNet1D101

__all__ = [
    "MInception",
    "MInceptionDemographic",
    "ResNet1D",
    "ResNet1DMicro",
    "ResNet1DMini",
    "ResNet1DTiny",
    "SpectroTemporalResNet",
    "XResNet1D101",
    "create_model",
    "get_model_class",
    "list_models",
    "register_model",
]
