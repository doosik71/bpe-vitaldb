"""Model registry and direct SBP/DBP regression architectures."""

from bpe.models.acfa import ACFA
from bpe.models.minception import MInception
from bpe.models.mtae import MTAE
from bpe.models.mtae_tr import MTAE_TR
from bpe.models.naive import NaiveConstant
from bpe.models.pulse_resnet1d import PulseResNet1D
from bpe.models.pulsew_resnet1d import PulseWResNet1D
from bpe.models.pulsewo_resnet1d import PulseWOResNet1D
from bpe.models.pulsewoq_resnet1d import PulseWOQResNet1D
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
    "ACFA",
    "MInception",
    "MTAE",
    "MTAE_TR",
    "NaiveConstant",
    "PulseResNet1D",
    "PulseWResNet1D",
    "PulseWOResNet1D",
    "PulseWOQResNet1D",
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
