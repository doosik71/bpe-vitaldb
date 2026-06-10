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
    --split         Train val test ratios         (default: 0.7 0.1 0.2)
    --target-hz     Output PPG sample rate (Hz)   (default: 125)
    --segment-sec   Window length in seconds      (default: 8)
    --guard-sec     Guard-band duration in sec    (default: 1)
    --no-guard      Disable guard-band filtering
    --nproc         Worker process count          (default: os.cpu_count())
    --no-resume     Reprocess cases even if NPZ already exists
    --seed          Shuffle seed                  (default: 42)
"""

import argparse
import csv
import logging
import multiprocessing as mp
import os
import random
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm
from vitaldb.utils import VitalFile

SOURCE_HZ = 500          # native VitalDB PPG sample rate
SBP_RANGE = (50, 250)    # physiological bounds (mmHg)
DBP_RANGE = (20, 150)
BP_VALID_MIN_FRAC = 0.5  # minimum fraction of valid 1-Hz BP samples per window

BANDPASS_LO    = 0.5     # Hz — high-pass cut: removes slow baseline drift
BANDPASS_HI    = 10.0    # Hz — low-pass cut: removes high-frequency noise
BANDPASS_ORDER = 4       # filter order (4th-order Butterworth)

INDEX_FILE = "index.csv"  # per-split resume index: case_id, n_segments

log = logging.getLogger(__name__)


def _bandpass_filter(signal: np.ndarray, fs: int) -> np.ndarray:
    """Apply a 4th-order zero-phase Butterworth bandpass filter (0.5–10 Hz).

    sosfiltfilt performs forward-backward filtering, yielding zero phase
    distortion and effectively doubling the filter order.
    """
    sos = butter(BANDPASS_ORDER, [BANDPASS_LO, BANDPASS_HI],
                 btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


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
                   metavar=("TRAIN", "VAL", "TEST"), default=[0.7, 0.1, 0.2],
                   help="Train/val/test ratios that sum to 1.0 (default: 0.7 0.1 0.2)")
    p.add_argument("--target-hz",   type=int,   default=125,
                   help="Target PPG sample rate in Hz (default: 125)")
    p.add_argument("--segment-sec", type=int,   default=8,
                   help="Segment / window duration in seconds (default: 8)")
    p.add_argument("--guard-sec",   type=int,   default=1,
                   help="Guard-band duration added on each side before filtering (default: 1)")
    p.add_argument("--no-guard",    action="store_true",
                   help="Disable guard-band: filter each window segment directly")
    p.add_argument("--nproc",       type=int,   default=None,
                   help="Worker process count (default: os.cpu_count())")
    p.add_argument("--no-resume",   action="store_true",
                   help="Reprocess cases even if the output NPZ already exists")
    p.add_argument("--seed",        type=int,   default=42,
                   help="Random seed for case shuffling (default: 42)")
    return p.parse_args()


def _bp_label(samples: np.ndarray, bounds: tuple[float, float]) -> float | None:
    """Return the mean of valid BP samples or None if below the validity threshold."""
    mask = np.isfinite(samples) & (samples >= bounds[0]) & (samples <= bounds[1])
    if mask.mean() < BP_VALID_MIN_FRAC:
        return None
    return float(np.mean(samples[mask]))


def _npz_segment_count(npz_path: Path) -> int:
    """Return the number of segments stored in an NPZ file."""
    return int(np.load(npz_path)["x"].shape[0])


def _read_index(csv_path: Path) -> dict[str, int]:
    """Load index.csv and return {case_id: n_segments}.

    When duplicate case_id rows exist (e.g. after a re-process) the last one wins.
    """
    index: dict[str, int] = {}
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2 and row[0] != "case_id":
                try:
                    index[row[0]] = int(row[1])
                except ValueError:
                    pass
    return index


def _write_index(csv_path: Path, index: dict[str, int]) -> None:
    """Write (or overwrite) index.csv from a {case_id: n_segments} dict."""
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "n_segments"])
        for case_id, n_segs in sorted(index.items()):
            w.writerow([case_id, n_segs])


def _append_index_row(csv_path: Path, case_id: str, n_segs: int, lock) -> None:
    """Append one row to index.csv under a multiprocessing lock."""
    with lock:
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([case_id, n_segs])


def process_case(
    path: Path,
    *,
    target_hz: int,
    segment_sec: int,
    guard_sec: int,
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
    guard_samples   = guard_sec * target_hz         # guard-band length (0 = disabled)

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
    # Bandpass filter is applied per-window (after NaN check) because
    # sosfiltfilt propagates NaN from any single bad sample to the entire signal.
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

        # Guard-band: extend the filter region by guard_samples on each side.
        # When guard_sec=0 (--no-guard), fs=ps and fe=pe — identical to no-guard.
        fs = ps - guard_samples
        fe = pe + guard_samples
        if fs < 0 or fe > len(ppg):
            continue  # guard region out of bounds; skip (first/last few windows)

        filter_region = ppg[fs:fe]
        if not np.all(np.isfinite(filter_region)):
            continue

        filtered = _bandpass_filter(filter_region, target_hz)
        ppg_seg = filtered[guard_samples : guard_samples + segment_samples]

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


def _process_chunk(args: tuple) -> tuple[dict[str, int], dict[str, int]]:
    """Worker: process a list of (path, out_dir, split_name) tasks and save NPZ files."""
    worker_id, tasks, target_hz, segment_sec, guard_sec, csv_paths, csv_lock = args

    pbar = tqdm(
        total=len(tasks),
        position=worker_id,
        desc=f"  proc {worker_id:2d}",
        unit="case",
        ascii=" -+=",
        leave=True,
        dynamic_ncols=True,
    )

    seg_counts:  dict[str, int] = {}
    skip_counts: dict[str, int] = {}

    for path, out_dir, split_name in tasks:
        result = process_case(
            path,
            target_hz=target_hz,
            segment_sec=segment_sec,
            guard_sec=guard_sec,
        )
        if result is None:
            skip_counts[split_name] = skip_counts.get(split_name, 0) + 1
            n_segs = 0
        else:
            x, y = result
            out_path = out_dir / f"{path.stem}.npz"
            tmp_path = out_dir / f".{path.stem}.tmp.npz"
            np.savez_compressed(tmp_path, x=x, y=y)
            tmp_path.rename(out_path)  # atomic on POSIX — no partial file on crash
            seg_counts[split_name] = seg_counts.get(split_name, 0) + len(x)
            n_segs = len(x)

        _append_index_row(csv_paths[split_name], path.stem, n_segs, csv_lock)
        pbar.update(1)

    pbar.close()
    return seg_counts, skip_counts


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

    guard_sec = 0 if args.no_guard else args.guard_sec
    nproc     = args.nproc if args.nproc is not None else (os.cpu_count() or 1)
    resume    = not args.no_resume

    log.info("Found %d .vital files in %s", len(vital_files), args.data_dir)
    log.info(
        "Settings: target_hz=%d  segment_sec=%ds  overlap=%ds  guard=%s  nproc=%d  resume=%s  split=%.0f/%.0f/%.0f",
        args.target_hz, args.segment_sec, args.segment_sec // 2,
        "disabled" if guard_sec == 0 else f"{guard_sec}s",
        nproc,
        "on" if resume else "off",
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

    # ── create output directories ─────────────────────────────────────────────
    for split_name in splits:
        (args.output_dir / split_name).mkdir(parents=True, exist_ok=True)

    # ── warn about NPZ files not assigned to the current split ───────────────
    for split_name, files in splits.items():
        expected = {p.stem for p in files}
        orphans  = [
            npz.stem for npz in (args.output_dir / split_name).glob("*.npz")
            if npz.stem not in expected
        ]
        if orphans:
            sample = ", ".join(sorted(orphans)[:5]) + ("…" if len(orphans) > 5 else "")
            log.warning(
                "  %-5s : %d NPZ file(s) not in current split assignment "
                "(seed/ratio/file-list change?) — %s",
                split_name, len(orphans), sample,
            )

    # ── build flat task list (all splits combined) ────────────────────────────
    all_tasks: list[tuple[Path, Path, str]] = [
        (path, args.output_dir / split_name, split_name)
        for split_name, files in splits.items()
        for path in files
    ]

    csv_paths: dict[str, Path] = {
        split_name: args.output_dir / split_name / INDEX_FILE
        for split_name in splits
    }

    # ── clear index files when a full re-run is requested ────────────────────
    if not resume:
        for csv_path in csv_paths.values():
            csv_path.unlink(missing_ok=True)

    # ── load or initialise per-split indexes; pre-filter tasks ───────────────
    # index.csv tracks every attempted case: case_id, n_segments
    # n_segments == 0 means the case failed; skipped on subsequent runs.
    resume_counts:     dict[str, int] = {name: 0 for name in splits}
    resumed_seg_counts: dict[str, int] = {name: 0 for name in splits}
    pending: list[tuple[Path, Path, str]] = []

    if resume:
        indexes: dict[str, dict[str, int]] = {}
        for split_name, csv_path in csv_paths.items():
            if not csv_path.exists():
                # First run for this split: build initial index from existing NPZs.
                index: dict[str, int] = {}
                existing_npzs = sorted((args.output_dir / split_name).glob("*.npz"))
                if existing_npzs:
                    log.info(
                        "  %-5s : building index.csv from %d existing NPZ files ...",
                        split_name, len(existing_npzs),
                    )
                    for npz in tqdm(
                        existing_npzs,
                        desc=f"  {split_name:<5} index",
                        unit="case",
                        ascii=" -+=",
                        dynamic_ncols=True,
                    ):
                        if npz.stat().st_size > 0:
                            try:
                                index[npz.stem] = _npz_segment_count(npz)
                            except Exception:
                                index[npz.stem] = 0
                _write_index(csv_path, index)
                indexes[split_name] = index
            else:
                indexes[split_name] = _read_index(csv_path)

        for path, out_dir, split_name in all_tasks:
            case_id  = path.stem
            out_file = out_dir / f"{case_id}.npz"
            index    = indexes[split_name]

            if case_id in index:
                n_segs = index[case_id]
                if n_segs == 0:
                    resume_counts[split_name] += 1          # failed before — skip
                elif out_file.exists():
                    resume_counts[split_name]      += 1     # success + NPZ present — skip
                    resumed_seg_counts[split_name] += n_segs
                else:
                    pending.append((path, out_dir, split_name))  # NPZ lost — re-process
            else:
                if out_file.exists() and out_file.stat().st_size > 0:
                    # Legacy NPZ not yet in index — record it and skip
                    try:
                        n_segs = _npz_segment_count(out_file)
                    except Exception:
                        n_segs = 0
                    with open(csv_paths[split_name], "a", newline="") as f:
                        csv.writer(f).writerow([case_id, n_segs])
                    resume_counts[split_name]      += 1
                    resumed_seg_counts[split_name] += n_segs
                else:
                    pending.append((path, out_dir, split_name))
    else:
        pending = list(all_tasks)

    all_tasks = pending

    # ── divide tasks into per-worker chunks ───────────────────────────────────
    seg_counts:  dict[str, int] = {name: 0 for name in splits}
    skip_counts: dict[str, int] = {name: 0 for name in splits}

    if all_tasks:
        n_workers = min(nproc, len(all_tasks))
        chunks    = [all_tasks[i::n_workers] for i in range(n_workers)]

        # ── run workers with per-worker progress bars ─────────────────────────
        tqdm.set_lock(mp.RLock())
        with mp.Manager() as manager:
            csv_lock    = manager.Lock()
            worker_args = [
                (i, chunk, args.target_hz, args.segment_sec, guard_sec, csv_paths, csv_lock)
                for i, chunk in enumerate(chunks)
            ]
            with mp.Pool(
                processes=n_workers,
                initializer=tqdm.set_lock,
                initargs=(tqdm.get_lock(),),
            ) as pool:
                results = pool.map(_process_chunk, worker_args)

        print()  # move cursor below progress bars

        for worker_seg, worker_skip in results:
            for split, n in worker_seg.items():
                seg_counts[split] += n
            for split, n in worker_skip.items():
                skip_counts[split] += n

    for split_name, files in splits.items():
        n_segs    = seg_counts[split_name]
        n_skipped = skip_counts[split_name]
        n_resumed = resume_counts[split_name]
        log.info(
            "  %-5s done — %s segments from %d new cases (%d resumed, %d skipped)",
            split_name, f"{n_segs:,}", len(files) - n_skipped - n_resumed, n_resumed, n_skipped,
        )

    # ── summary ──────────────────────────────────────────────────────────────
    log.info("=" * 60)
    total_new = sum(seg_counts.values())
    total_all = total_new + sum(resumed_seg_counts.values())
    log.info("  %-5s   %12s   %12s   %s", "", "new segs", "total segs", "  %")
    log.info("  " + "-" * 42)
    for name in splits:
        new_segs   = seg_counts[name]
        total_segs = new_segs + resumed_seg_counts[name]
        pct        = total_segs / total_all * 100 if total_all else 0.0
        log.info("  %-5s   %12s   %12s   %5.1f%%", name, f"{new_segs:,}", f"{total_segs:,}", pct)
    log.info("  " + "-" * 42)
    log.info("  %-5s   %12s   %12s   %5.1f%%", "total", f"{total_new:,}", f"{total_all:,}", 100.0)
    log.info("Output written to %s", args.output_dir.resolve())


if __name__ == "__main__":
    main()
