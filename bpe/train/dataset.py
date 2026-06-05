"""PyTorch Dataset for PPG/BP segment NPZ files produced by construct-dataset.py.

Each NPZ file contains all segments from one case:
    x  float32  (N, segment_samples)   PPG waveforms
    y  float32  (N, 2)                 [SBP_mean, DBP_mean] in mmHg

PPGDataset builds a flat segment index over all NPZ files in a split
directory.  Files are loaded and cached on first access so the full
dataset does not need to fit in RAM before the first epoch.

When used with DataLoader(num_workers > 0), each worker process receives
its own (initially empty) cache and populates it independently as it
processes batches.  After the first epoch every worker has cached the
files it is responsible for.
"""

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class PPGDataset(Dataset):
    """Flat segment index over per-case NPZ files.

    Args:
        split_dir:  Directory containing ``<caseid>.npz`` files.
        normalize:  If True, apply per-segment z-score normalization to the
                    PPG signal (zero mean, unit variance).  The models that
                    use DerivativeChannels already perform their own
                    normalization internally; for those, this flag can be
                    set to False to avoid double-normalization, though in
                    practice the difference is negligible.
        preload:    Load *all* x/y arrays into contiguous NumPy arrays at
                    construction time.  Faster training at the cost of RAM
                    (roughly ``N_segments × segment_len × 4`` bytes).
        augment:    Optional callable ``(x: Tensor) -> Tensor`` applied to
                    each PPG segment after normalization.  Pass ``None``
                    (default) for validation / test sets.
    """

    def __init__(
        self,
        split_dir: Path,
        *,
        normalize: bool = True,
        preload: bool = False,
        augment: Callable | None = None,
    ):
        self._normalize = normalize
        self._preload   = preload
        self._augment   = augment

        # Build flat segment index: list of (file_idx, local_seg_idx)
        self._files: list[Path] = []
        self._segs:  list[tuple[int, int]] = []   # (file_idx, local_idx)

        npz_files = sorted(
            split_dir.glob("*.npz"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
        )
        if not npz_files:
            raise FileNotFoundError(f"No .npz files found in {split_dir}")

        for path in tqdm(npz_files, desc=f"Indexing {split_dir.name}", leave=False):
            try:
                with np.load(path) as data:
                    n = int(data["x"].shape[0])
            except Exception:
                continue
            file_idx = len(self._files)
            self._files.append(path)
            for local_idx in range(n):
                self._segs.append((file_idx, local_idx))

        if not self._segs:
            raise RuntimeError(f"Dataset at {split_dir} is empty after scanning.")

        # Lazy cache:  {file_idx: (x_array, y_array)}
        # Populated on first access; kept in memory thereafter.
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        # Preload: concatenate everything into two large arrays
        self._px: np.ndarray | None = None
        self._py: np.ndarray | None = None
        if preload:
            self._do_preload()

    # ── Preload ───────────────────────────────────────────────────────────────

    def _do_preload(self) -> None:
        xs, ys = [], []
        for path in tqdm(self._files, desc="Preloading", unit="file", leave=False):
            with np.load(path) as data:
                xs.append(data["x"].copy())
                ys.append(data["y"].copy())
        self._px = np.concatenate(xs, axis=0)
        self._py = np.concatenate(ys, axis=0)

    # ── Lazy file cache ───────────────────────────────────────────────────────

    def _get_arrays(self, file_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if file_idx not in self._cache:
            with np.load(self._files[file_idx]) as data:
                self._cache[file_idx] = (data["x"].copy(), data["y"].copy())
        return self._cache[file_idx]

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._segs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_idx, local_idx = self._segs[idx]

        if self._preload and self._px is not None:
            x_np = self._px[idx]
            y_np = self._py[idx]
        else:
            x_arr, y_arr = self._get_arrays(file_idx)
            x_np = x_arr[local_idx]
            y_np = y_arr[local_idx]

        x = torch.from_numpy(x_np.copy())
        y = torch.from_numpy(y_np.copy())

        if self._normalize:
            std = x.std()
            x = (x - x.mean()) / std.clamp_min(1e-6)

        if self._augment is not None:
            x = self._augment(x)

        return x, y

    # ── Metadata helpers ──────────────────────────────────────────────────────

    @property
    def n_files(self) -> int:
        """Number of case NPZ files in this split."""
        return len(self._files)

    def segment_length(self) -> int:
        """Length (number of samples) of a single PPG segment."""
        return int(self._segs and self._get_arrays(0)[0].shape[1] or 0)

    def sample_weights(self) -> torch.Tensor:
        """Per-segment weights for WeightedRandomSampler.

        Each segment is assigned weight ``1 / n`` where ``n`` is the total
        number of segments belonging to the same source file (patient case).
        This gives every patient equal expected representation per epoch,
        regardless of how many segments their recording contributes.

        Returns:
            Float tensor of shape ``(len(self),)`` suitable for passing
            directly to ``torch.utils.data.WeightedRandomSampler``.
        """
        file_counts: list[int] = [0] * len(self._files)
        for file_idx, _ in self._segs:
            file_counts[file_idx] += 1

        weights = torch.zeros(len(self._segs))
        for i, (file_idx, _) in enumerate(self._segs):
            weights[i] = 1.0 / file_counts[file_idx]
        return weights
