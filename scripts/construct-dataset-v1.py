"""
Construct a filtered train / val / test dataset from existing NPZ dataset files.

This script reads `data/dataset/{train,val,test}/*.npz`, computes

    power_ratio = Power(0.67-3.0 Hz) / Power(0.5-10.0 Hz)

for every PPG segment, and keeps only segments whose power_ratio is at least
`--power-ratio-min`.

Unlike construct-dataset.py, this script does not read raw `.vital` files.
It reuses the already-built NPZ dataset to avoid repeating VitalDB parsing,
resampling, segmentation, and label generation.

Output layout:
    <dataset-dir>/
        train/  <caseid>.npz
        val/    <caseid>.npz
        test/   <caseid>.npz

Each output `.npz` contains:
    x  float32  (N, segment_samples)   filtered PPG segments
    y  float32  (N, 2)                 [SBP_mean, DBP_mean] per segment

Usage:
    uv run python scripts/construct-dataset-v1.py [OPTIONS]

Options:
    --input-dir         Root input dataset directory     (default: data/dataset)
    --dataset-dir       Root output dataset directory    (default: data/dataset-v1)
    --target-hz         PPG sample rate in Hz            (default: 125)
    --nperseg           Welch segment length             (default: 256)
    --power-ratio-min   Minimum allowed power_ratio      (default: 0.6)
    --nproc             Worker process count             (default: os.cpu_count())
    --no-resume         Reprocess cases even if output NPZ already exists
"""

import argparse
import csv
import logging
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from scipy.signal import welch
from tqdm import tqdm

SPLITS = ("train", "val", "test")
INDEX_FILE = "index.csv"
PASSBAND = (0.5, 10.0)
HEART_BAND = (0.67, 3.0)

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build filtered NPZ dataset from an existing dataset directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/dataset"),
        help="Root input dataset directory (default: data/dataset)",
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/dataset-v1"),
        help="Root output dataset directory (default: data/dataset-v1)",
    )
    p.add_argument(
        "--target-hz",
        type=int,
        default=125,
        help="PPG sample rate in Hz (default: 125)",
    )
    p.add_argument(
        "--nperseg",
        type=int,
        default=256,
        help="Welch segment length (default: 256)",
    )
    p.add_argument(
        "--power-ratio-min",
        type=float,
        default=0.6,
        help="Minimum allowed power_ratio (default: 0.6)",
    )
    p.add_argument(
        "--nproc",
        type=int,
        default=None,
        help="Worker process count (default: os.cpu_count())",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Reprocess cases even if the output NPZ already exists",
    )
    return p.parse_args()


def compute_psd(signal: np.ndarray, fs: int, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    freqs, psd = welch(
        signal,
        fs=fs,
        window="hann",
        nperseg=min(len(signal), nperseg),
        noverlap=None,
        detrend="constant",
        scaling="density",
    )
    return freqs, psd


def band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return float("nan")
    return float(np.trapezoid(psd[mask], freqs[mask]))


def power_ratio(signal: np.ndarray, fs: int, nperseg: int) -> float:
    freqs, psd = compute_psd(signal, fs, nperseg)
    heart_power = band_power(freqs, psd, HEART_BAND)
    passband_power = band_power(freqs, psd, PASSBAND)
    if not np.isfinite(heart_power) or not np.isfinite(passband_power) or passband_power <= 0:
        return float("nan")
    return heart_power / passband_power


def _npz_segment_count(npz_path: Path) -> int:
    return int(np.load(npz_path)["x"].shape[0])


def _read_index(csv_path: Path) -> dict[str, int]:
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
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "n_segments"])
        for case_id, n_segs in sorted(index.items()):
            w.writerow([case_id, n_segs])


def _append_index_row(csv_path: Path, case_id: str, n_segs: int, lock) -> None:
    with lock:
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([case_id, n_segs])


def process_case(
    path: Path,
    *,
    target_hz: int,
    nperseg: int,
    power_ratio_min: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        with np.load(path) as data:
            x = np.asarray(data["x"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.float32)
    except Exception as exc:
        log.warning("%s: cannot load NPZ - %s", path.stem, exc)
        return None

    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        log.warning("%s: unexpected NPZ shapes x=%s y=%s", path.stem, x.shape, y.shape)
        return None

    keep_indices: list[int] = []
    for idx, signal in enumerate(x):
        if not np.all(np.isfinite(signal)):
            continue
        ratio = power_ratio(signal, target_hz, nperseg)
        if np.isfinite(ratio) and ratio >= power_ratio_min:
            keep_indices.append(idx)

    if not keep_indices:
        return None

    keep = np.asarray(keep_indices, dtype=np.int64)
    return x[keep], y[keep]


def _process_chunk(args: tuple) -> tuple[dict[str, int], dict[str, int]]:
    worker_id, tasks, target_hz, nperseg, power_ratio_min, csv_paths, csv_lock = args

    pbar = tqdm(
        total=len(tasks),
        position=worker_id,
        desc=f"  proc {worker_id:2d}",
        unit="case",
        ascii=" -+=",
        leave=True,
        dynamic_ncols=True,
    )

    seg_counts: dict[str, int] = {}
    skip_counts: dict[str, int] = {}

    for path, out_dir, split_name in tasks:
        result = process_case(
            path,
            target_hz=target_hz,
            nperseg=nperseg,
            power_ratio_min=power_ratio_min,
        )
        out_path = out_dir / f"{path.stem}.npz"
        if result is None:
            skip_counts[split_name] = skip_counts.get(split_name, 0) + 1
            n_segs = 0
            out_path.unlink(missing_ok=True)
        else:
            x, y = result
            tmp_path = out_dir / f".{path.stem}.tmp.npz"
            np.savez_compressed(tmp_path, x=x, y=y)
            tmp_path.rename(out_path)
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
    resume = not args.no_resume
    nproc = args.nproc if args.nproc is not None else (os.cpu_count() or 1)

    splits: dict[str, list[Path]] = {}
    for split_name in SPLITS:
        split_dir = args.input_dir / split_name
        files = sorted(
            split_dir.glob("*.npz"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
        ) if split_dir.exists() else []
        splits[split_name] = files

    total_cases = sum(len(files) for files in splits.values())
    if total_cases == 0:
        log.error("No NPZ files found under %s/{train,val,test}", args.input_dir)
        return

    log.info("Found %d NPZ case files in %s", total_cases, args.input_dir)
    log.info(
        "Settings: target_hz=%d  nperseg=%d  power_ratio_min=%.3f  nproc=%d  resume=%s",
        args.target_hz,
        args.nperseg,
        args.power_ratio_min,
        nproc,
        "on" if resume else "off",
    )

    for split_name, files in splits.items():
        log.info("  %-5s : %d cases", split_name, len(files))
        (args.dataset_dir / split_name).mkdir(parents=True, exist_ok=True)

    all_tasks: list[tuple[Path, Path, str]] = [
        (path, args.dataset_dir / split_name, split_name)
        for split_name, files in splits.items()
        for path in files
    ]

    csv_paths: dict[str, Path] = {
        split_name: args.dataset_dir / split_name / INDEX_FILE
        for split_name in SPLITS
    }

    if not resume:
        for csv_path in csv_paths.values():
            csv_path.unlink(missing_ok=True)

    resume_counts: dict[str, int] = {name: 0 for name in SPLITS}
    resumed_seg_counts: dict[str, int] = {name: 0 for name in SPLITS}
    pending: list[tuple[Path, Path, str]] = []

    if resume:
        indexes: dict[str, dict[str, int]] = {}
        for split_name, csv_path in csv_paths.items():
            if not csv_path.exists():
                index: dict[str, int] = {}
                existing_npzs = sorted((args.dataset_dir / split_name).glob("*.npz"))
                if existing_npzs:
                    log.info(
                        "  %-5s : building index.csv from %d existing NPZ files ...",
                        split_name,
                        len(existing_npzs),
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
            case_id = path.stem
            out_file = out_dir / f"{case_id}.npz"
            index = indexes[split_name]

            if case_id in index:
                n_segs = index[case_id]
                if n_segs == 0:
                    resume_counts[split_name] += 1
                elif out_file.exists():
                    resume_counts[split_name] += 1
                    resumed_seg_counts[split_name] += n_segs
                else:
                    pending.append((path, out_dir, split_name))
            else:
                if out_file.exists() and out_file.stat().st_size > 0:
                    try:
                        n_segs = _npz_segment_count(out_file)
                    except Exception:
                        n_segs = 0
                    with open(csv_paths[split_name], "a", newline="") as f:
                        csv.writer(f).writerow([case_id, n_segs])
                    resume_counts[split_name] += 1
                    resumed_seg_counts[split_name] += n_segs
                else:
                    pending.append((path, out_dir, split_name))
    else:
        pending = list(all_tasks)

    all_tasks = pending
    seg_counts: dict[str, int] = {name: 0 for name in SPLITS}
    skip_counts: dict[str, int] = {name: 0 for name in SPLITS}

    if all_tasks:
        n_workers = min(nproc, len(all_tasks))
        chunks = [all_tasks[i::n_workers] for i in range(n_workers)]

        tqdm.set_lock(mp.RLock())
        with mp.Manager() as manager:
            csv_lock = manager.Lock()
            worker_args = [
                (i, chunk, args.target_hz, args.nperseg, args.power_ratio_min, csv_paths, csv_lock)
                for i, chunk in enumerate(chunks)
            ]
            with mp.Pool(
                processes=n_workers,
                initializer=tqdm.set_lock,
                initargs=(tqdm.get_lock(),),
            ) as pool:
                results = pool.map(_process_chunk, worker_args)

        print()

        for worker_seg, worker_skip in results:
            for split, n in worker_seg.items():
                seg_counts[split] += n
            for split, n in worker_skip.items():
                skip_counts[split] += n

    for split_name, files in splits.items():
        n_segs = seg_counts[split_name]
        n_skipped = skip_counts[split_name]
        n_resumed = resume_counts[split_name]
        log.info(
            "  %-5s done - %s segments from %d new cases (%d resumed, %d filtered-out)",
            split_name,
            f"{n_segs:,}",
            len(files) - n_skipped - n_resumed,
            n_resumed,
            n_skipped,
        )

    log.info("=" * 60)
    total_new = sum(seg_counts.values())
    total_all = total_new + sum(resumed_seg_counts.values())
    log.info("  %-5s   %12s   %12s   %s", "", "new segs", "total segs", "  %")
    log.info("  " + "-" * 42)
    for name in SPLITS:
        new_segs = seg_counts[name]
        total_segs = new_segs + resumed_seg_counts[name]
        pct = total_segs / total_all * 100 if total_all else 0.0
        log.info("  %-5s   %12s   %12s   %5.1f%%", name, f"{new_segs:,}", f"{total_segs:,}", pct)
    log.info("  " + "-" * 42)
    log.info("  %-5s   %12s   %12s   %5.1f%%", "total", f"{total_new:,}", f"{total_all:,}", 100.0)
    log.info("Output written to %s", args.dataset_dir.resolve())


if __name__ == "__main__":
    main()
