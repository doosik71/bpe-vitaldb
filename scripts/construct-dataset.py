"""
Construct train / val / test datasets from VitalDB .vital files.

Each case's PPG waveform (SNUADC/PLETH, 500 Hz) is decimated to
--target-hz and split into overlapping windows of --segment-sec seconds
with 50 % overlap (stride = segment_sec / 2).

Labels (SBP and DBP in mmHg) are derived from the numeric tracks
Solar8000/ART_SBP and Solar8000/ART_DBP (~1 Hz) by averaging the valid
samples that fall inside each window.

A window is discarded when:
  * the PPG segment contains any NaN or Inf value
  * fewer than 50 % of the 1-Hz BP samples are finite and in-range
      SBP: [50, 250] mmHg   DBP: [20, 150] mmHg
  * the computed mean SBP <= mean DBP (physiologically inconsistent)

Cases (not segments) are shuffled and split into train / val / test to
prevent data leakage across splits.

Output layout:
    <output-dir>/
        train/  <caseid>.npz
        val/    <caseid>.npz
        test/   <caseid>.npz

Each .npz contains:
    x  float32  (N, segment_samples)   PPG segments
    y  float32  (N, 2)                 [SBP_mean, DBP_mean] per segment

Usage:
    uv run python scripts/construct-dataset.py [OPTIONS]

Options:
    --data-dir      Directory with .vital files   (default: data/vitaldb)
    --output-dir    Root output directory         (default: data/dataset)
    --split         Train val test ratios         (default: 0.6 0.2 0.2)
    --target-hz     Output PPG sample rate (Hz)   (default: 125)
    --segment-sec   Window length in seconds      (default: 8)
    --seed          Shuffle seed                  (default: 42)
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm
from vitaldb.utils import VitalFile

SOURCE_HZ = 500          # native VitalDB PPG sample rate
SBP_RANGE = (50, 250)    # physiological bounds (mmHg)
DBP_RANGE = (20, 150)
BP_VALID_MIN_FRAC = 0.5  # minimum fraction of valid 1-Hz BP samples per window

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build train/val/test NPZ datasets from VitalDB .vital files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--data-dir",    type=Path,  default=Path("data/vitaldb"),
                   help="Directory containing .vital files (default: data/vitaldb)")
    p.add_argument("--output-dir",  type=Path,  default=Path("data/dataset"),
                   help="Root output directory (default: data/dataset)")
    p.add_argument("--split",       type=float, nargs=3,
                   metavar=("TRAIN", "VAL", "TEST"), default=[0.6, 0.2, 0.2],
                   help="Train/val/test ratios that sum to 1.0 (default: 0.6 0.2 0.2)")
    p.add_argument("--target-hz",   type=int,   default=125,
                   help="Target PPG sample rate in Hz (default: 125)")
    p.add_argument("--segment-sec", type=int,   default=8,
                   help="Segment / window duration in seconds (default: 8)")
    p.add_argument("--seed",        type=int,   default=42,
                   help="Random seed for case shuffling (default: 42)")
    return p.parse_args()


def _bp_label(samples: np.ndarray, bounds: tuple[float, float]) -> float | None:
    """Return the mean of valid BP samples or None if below the validity threshold."""
    mask = np.isfinite(samples) & (samples >= bounds[0]) & (samples <= bounds[1])
    if mask.mean() < BP_VALID_MIN_FRAC:
        return None
    return float(np.mean(samples[mask]))


def process_case(
    path: Path,
    *,
    target_hz: int,
    segment_sec: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Extract all valid (x, y) pairs from a single .vital file.

    Returns (x, y) arrays or None when the case must be skipped entirely.
    """
    if SOURCE_HZ % target_hz != 0:
        raise ValueError(
            f"SOURCE_HZ ({SOURCE_HZ}) must be divisible by --target-hz ({target_hz})"
        )

    factor          = SOURCE_HZ // target_hz        # decimation factor
    segment_samples = segment_sec * target_hz       # samples per window at target_hz
    stride_samples  = (segment_sec // 2) * target_hz  # 50 % overlap
    stride_sec      = segment_sec // 2              # stride in seconds (for 1-Hz arrays)

    # ── open file ────────────────────────────────────────────────────────────
    try:
        vf        = VitalFile(str(path))
        available = set(vf.get_track_names())
    except Exception as exc:
        log.warning("%s: cannot open — %s", path.stem, exc)
        return None

    required = {"SNUADC/PLETH", "Solar8000/ART_SBP", "Solar8000/ART_DBP"}
    if not required.issubset(available):
        return None

    # ── load tracks ──────────────────────────────────────────────────────────
    try:
        ppg_raw = vf.to_numpy(["SNUADC/PLETH"], interval=1 / SOURCE_HZ)[:, 0]
        bp_raw  = vf.to_numpy(
            ["Solar8000/ART_SBP", "Solar8000/ART_DBP"], interval=1.0
        )
        sbp_1hz = bp_raw[:, 0]
        dbp_1hz = bp_raw[:, 1]
    except Exception as exc:
        log.warning("%s: track read error — %s", path.stem, exc)
        return None

    # ── decimate PPG ─────────────────────────────────────────────────────────
    ppg = ppg_raw[::factor]   # shape: (T * target_hz / SOURCE_HZ,)

    # ── compute window count ─────────────────────────────────────────────────
    total_sec = min(len(ppg) / target_hz, len(sbp_1hz), len(dbp_1hz))
    if total_sec < segment_sec:
        return None

    n_windows = int((total_sec - segment_sec) / stride_sec) + 1

    xs: list[np.ndarray] = []
    ys: list[list[float]] = []

    for w in range(n_windows):
        ps = w * stride_samples
        pe = ps + segment_samples
        bs = w * stride_sec
        be = bs + segment_sec

        if pe > len(ppg) or be > len(sbp_1hz) or be > len(dbp_1hz):
            break

        ppg_seg = ppg[ps:pe]
        if not np.all(np.isfinite(ppg_seg)):
            continue

        sbp_val = _bp_label(sbp_1hz[bs:be], SBP_RANGE)
        dbp_val = _bp_label(dbp_1hz[bs:be], DBP_RANGE)
        if sbp_val is None or dbp_val is None:
            continue

        if sbp_val <= dbp_val:   # physiologically inconsistent
            continue

        xs.append(ppg_seg)
        ys.append([sbp_val, dbp_val])

    if not xs:
        return None

    return (
        np.array(xs, dtype=np.float32),   # (N, segment_samples)
        np.array(ys, dtype=np.float32),   # (N, 2)
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()

    # ── validate split ratios ────────────────────────────────────────────────
    ratio_sum = sum(args.split)
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"--split ratios must sum to 1.0, got {ratio_sum:.4f}")

    # ── discover .vital files ────────────────────────────────────────────────
    vital_files: list[Path] = sorted(
        args.data_dir.glob("*.vital"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )
    if not vital_files:
        log.error("No .vital files found in %s", args.data_dir)
        return

    log.info("Found %d .vital files in %s", len(vital_files), args.data_dir)
    log.info(
        "Settings: target_hz=%d  segment_sec=%ds  overlap=%ds  split=%.0f/%.0f/%.0f",
        args.target_hz, args.segment_sec, args.segment_sec // 2,
        args.split[0] * 100, args.split[1] * 100, args.split[2] * 100,
    )

    # ── shuffle and split at the case level ──────────────────────────────────
    rng      = random.Random(args.seed)
    shuffled = vital_files[:]
    rng.shuffle(shuffled)

    n       = len(shuffled)
    n_train = int(n * args.split[0])
    n_val   = int(n * args.split[1])
    splits: dict[str, list[Path]] = {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train : n_train + n_val],
        "test":  shuffled[n_train + n_val :],
    }

    for name, files in splits.items():
        log.info("  %-5s : %d cases", name, len(files))

    # ── process each split ───────────────────────────────────────────────────
    seg_counts: dict[str, int] = {}

    for split_name, files in splits.items():
        out_dir = args.output_dir / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        skipped   = 0
        n_segs    = 0

        for path in tqdm(files, desc=f"{split_name:5s}", unit="case"):
            result = process_case(
                path,
                target_hz=args.target_hz,
                segment_sec=args.segment_sec,
            )
            if result is None:
                skipped += 1
                continue

            x, y = result
            np.savez_compressed(out_dir / f"{path.stem}.npz", x=x, y=y)
            n_segs += len(x)

        seg_counts[split_name] = n_segs
        log.info(
            "  %-5s done — %d segments from %d cases (%d skipped)",
            split_name, n_segs, len(files) - skipped, skipped,
        )

    # ── summary ──────────────────────────────────────────────────────────────
    log.info("=" * 60)
    total = sum(seg_counts.values())
    for name, cnt in seg_counts.items():
        log.info("  %-5s : %7d segments", name, cnt)
    log.info("  total : %7d segments", total)
    log.info("Output written to %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
