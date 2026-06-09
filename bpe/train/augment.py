"""PPG signal augmentation transforms for training.

Each transform operates on a 1-D float32 torch.Tensor (a single PPG
segment, already z-score normalised).  They are composable via PPGAugment.

All transforms are deterministic in the sense that they draw random state
fresh on every call, making them suitable for per-sample online augmentation
inside DataLoader workers.
"""

import torch


class GaussianNoise:
    """Add zero-mean Gaussian noise to the signal.

    Args:
        std: Noise standard deviation in normalised signal units (default 0.01).
    """

    def __init__(self, std: float = 0.01) -> None:
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std


class AmplitudeScaling:
    """Multiply the signal by a random scalar drawn uniformly from [lo, hi].

    Applied after z-score normalisation, so the output mean remains ~0 while
    the variance is perturbed.  Teaches the model to be invariant to moderate
    gain variations (e.g. PPG contact quality differences).

    Args:
        lo: Lower bound of the scale factor (default 0.8).
        hi: Upper bound of the scale factor (default 1.2).
    """

    def __init__(self, lo: float = 0.8, hi: float = 1.2) -> None:
        self.lo = lo
        self.hi = hi

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.empty(1).uniform_(self.lo, self.hi).item()
        return x * scale


class TimeShift:
    """Circularly shift the signal by a random number of samples.

    torch.roll is used so no zero-padding artefacts are introduced at the
    boundary.  A uniform random offset is drawn from [-max_shift, +max_shift]
    at every call.

    Args:
        max_shift: Maximum absolute shift in samples (default 50, i.e. 0.4 s
                   at 125 Hz).
    """

    def __init__(self, max_shift: int = 50) -> None:
        self.max_shift = max_shift

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)))
        return torch.roll(x, shift)


class RandomMasking:
    """Zero-out one random contiguous span of samples.

    A masking fraction is drawn uniformly from [lo_frac, hi_frac] to decide
    the target span length, then clipped to at most 125 samples (1.0 s at
    125 Hz).  The masked values are set to 0.0, which equals the mean of a
    z-score normalised signal.

    Args:
        lo_frac: Minimum fraction of samples to mask (default 0.05).
        hi_frac: Maximum fraction of samples to mask (default 0.10).
    """

    def __init__(self, lo_frac: float = 0.05, hi_frac: float = 0.10) -> None:
        self.lo_frac = lo_frac
        self.hi_frac = hi_frac

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        frac = torch.empty(1).uniform_(self.lo_frac, self.hi_frac).item()
        n_mask = min(max(1, int(n * frac)), min(n, 125))
        start = int(torch.randint(0, n - n_mask + 1, (1,)))
        x = x.clone()
        x[start:start + n_mask] = 0.0
        return x


class PPGAugment:
    """Sequentially apply a list of augmentation transforms.

    Args:
        transforms: Ordered list of callables (each x -> x).
                    An empty list is a no-op.
    """

    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x

    def __bool__(self) -> bool:
        return bool(self.transforms)

    def __repr__(self) -> str:
        names = [type(t).__name__ for t in self.transforms]
        return f"PPGAugment([{', '.join(names)}])"
