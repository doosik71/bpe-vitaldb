"""
Construct a filtered train / val / test dataset directly from VitalDB .vital files.

This script reads both waveform tracks

    SNUADC/PLETH   (PPG, 500 Hz)
    SNUADC/ART     (ABP, 500 Hz)

decimates them to --target-hz, applies per-window bandpass filtering with an
optional guard band, and keeps only segments that pass the v2 nine-step
cleaning rules documented in docs/construct-dataset-v2.md.

Labels are derived directly from ABP waveform peaks and foot points:

    SBP = mean(ABP[peak_indices])
    DBP = mean(ABP[foot_indices])

Output layout:
    <dataset-dir>/
        train/  <caseid>.npz
        val/    <caseid>.npz
        test/   <caseid>.npz

Each .npz contains:
    x  float32  (N, segment_samples)   filtered PPG segments
    y  float32  (N, 2)                 [SBP_mean, DBP_mean] per segment
"""

import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm
from vitaldb.utils import VitalFile

from bpe.utils.qc_v2 import QCParams, check_segment_quality

SOURCE_HZ = 500

BANDPASS_LO = 0.5
BANDPASS_HI = 10.0
BANDPASS_ORDER = 4

INDEX_FILE = "index.csv"

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build filtered NPZ dataset-v2 directly from VitalDB .vital files",
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/vitaldb"))
    p.add_argument("--dataset-dir", type=Path, default=Path("data/dataset-v2"))
    p.add_argument(
        "--split",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=[0.7, 0.1, 0.2],
    )
    p.add_argument("--target-hz", type=int, default=125)
    p.add_argument("--segment-sec", type=int, default=8)
    p.add_argument("--guard-sec", type=int, default=1)
    p.add_argument("--no-guard", action="store_true")
    p.add_argument("--nproc", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--contlen", type=int, default=10)
    p.add_argument("--sbp-min", type=float, default=60.0)
    p.add_argument("--sbp-max", type=float, default=180.0)
    p.add_argument("--dbp-min", type=float, default=40.0)
    p.add_argument("--dbp-max", type=float, default=120.0)
    p.add_argument("--hr-min", type=float, default=30.0)
    p.add_argument("--hr-max", type=float, default=150.0)
    p.add_argument("--hr-diff-max", type=float, default=10.0)
    p.add_argument("--peak-foot-diff-max", type=int, default=2)
    p.add_argument("--min-peaks", type=int, default=4)
    p.add_argument("--sbp-range-max", type=float, default=40.0)
    p.add_argument("--dbp-range-max", type=float, default=20.0)
    p.add_argument("--fasqa-psd-low-max", type=float, default=0.15)
    p.add_argument("--fasqa-psd-tgt-min", type=float, default=0.10)
    p.add_argument("--fasqa-psd-high-max", type=float, default=0.05)
    return p.parse_args()


def _bandpass_filter(
    signal: np.ndarray,
    fs: int,
    *,
    lo: float = BANDPASS_LO,
    hi: float = BANDPASS_HI,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


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
    args: argparse.Namespace,
    guard_sec: int,
) -> tuple[np.ndarray | None, np.ndarray | None, int, dict[int, int]]:
    """
    Returns (x, y, n_windows_evaluated, rule_rejects).
    x / y are None when no segments passed QC.
    n_windows_evaluated counts windows that entered quality evaluation
    (guard-band-edge windows are excluded but non-finite windows are counted under rule 1).
    """
    if SOURCE_HZ % args.target_hz != 0:
        raise ValueError(
            f"SOURCE_HZ ({SOURCE_HZ}) must be divisible by --target-hz ({args.target_hz})"
        )

    factor = SOURCE_HZ // args.target_hz
    segment_samples = args.segment_sec * args.target_hz
    stride_sec = args.segment_sec // 2
    stride_samples = stride_sec * args.target_hz
    guard_samples = guard_sec * args.target_hz

    qc_params = QCParams(
        contlen=args.contlen,
        sbp_min=args.sbp_min,
        sbp_max=args.sbp_max,
        dbp_min=args.dbp_min,
        dbp_max=args.dbp_max,
        hr_min=args.hr_min,
        hr_max=args.hr_max,
        hr_diff_max=args.hr_diff_max,
        peak_foot_diff_max=args.peak_foot_diff_max,
        min_peaks=args.min_peaks,
        sbp_range_max=args.sbp_range_max,
        dbp_range_max=args.dbp_range_max,
        fasqa_psd_low_max=args.fasqa_psd_low_max,
        fasqa_psd_tgt_min=args.fasqa_psd_tgt_min,
        fasqa_psd_high_max=args.fasqa_psd_high_max,
    )

    try:
        vf = VitalFile(str(path))
        available = set(vf.get_track_names())
    except Exception as exc:
        log.warning("%s: cannot open - %s", path.stem, exc)
        return None, None, 0, {}

    required = {"SNUADC/PLETH", "SNUADC/ART"}
    if not required.issubset(available):
        return None, None, 0, {}

    try:
        ppg_raw = vf.to_numpy(["SNUADC/PLETH"], interval=1 / SOURCE_HZ)[:, 0]
        abp_raw = vf.to_numpy(["SNUADC/ART"], interval=1 / SOURCE_HZ)[:, 0]
    except Exception as exc:
        log.warning("%s: track read error - %s", path.stem, exc)
        return None, None, 0, {}

    ppg = ppg_raw[::factor]
    abp = abp_raw[::factor]
    total_samples = min(len(ppg), len(abp))
    if total_samples < segment_samples:
        return None, None, 0, {}
    ppg = ppg[:total_samples]
    abp = abp[:total_samples]

    n_windows = int((total_samples - segment_samples) / stride_samples) + 1
    xs: list[np.ndarray] = []
    ys: list[list[float]] = []
    n_windows_evaluated = 0
    rule_rejects: dict[int, int] = {}

    for w in range(n_windows):
        ps = w * stride_samples
        pe = ps + segment_samples
        if pe > total_samples:
            break

        filter_start = ps - guard_samples
        filter_end   = pe + guard_samples
        if filter_start < 0 or filter_end > total_samples:
            continue

        n_windows_evaluated += 1

        ppg_region = ppg[filter_start:filter_end]
        abp_region = abp[filter_start:filter_end]
        if not np.all(np.isfinite(ppg_region)) or not np.all(np.isfinite(abp_region)):
            rule_rejects[1] = rule_rejects.get(1, 0) + 1
            continue

        try:
            ppg_filtered = _bandpass_filter(ppg_region, args.target_hz)
            abp_filtered = _bandpass_filter(abp_region, args.target_hz)
        except Exception:
            rule_rejects[1] = rule_rejects.get(1, 0) + 1
            continue

        ppg_seg     = ppg_filtered[guard_samples : guard_samples + segment_samples]
        abp_seg     = abp_filtered[guard_samples : guard_samples + segment_samples]
        abp_raw_seg = abp[ps:pe]  # BPF 미적용 decimated 원신호 — 레이블 mmHg 값 산출용

        qc = check_segment_quality(ppg_seg, abp_seg, abp_raw_seg, args.target_hz, t_start=0.0, params=qc_params)
        if not qc.passed:
            rule = qc.failed_rule or 1
            rule_rejects[rule] = rule_rejects.get(rule, 0) + 1
            continue

        xs.append(ppg_seg.astype(np.float32))
        ys.append([qc.avg_sbp, qc.avg_dbp])

    if not xs:
        return None, None, n_windows_evaluated, rule_rejects

    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        n_windows_evaluated,
        rule_rejects,
    )


def _process_chunk(
    args_tuple: tuple,
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, dict[int, int]]]:
    worker_id, tasks, args, guard_sec, csv_paths, csv_lock = args_tuple

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
    window_counts: dict[str, int] = {}
    rule_rejects: dict[str, dict[int, int]] = {}

    for path, out_dir, split_name in tasks:
        x, y, n_windows_eval, case_rejects = process_case(path, args=args, guard_sec=guard_sec)
        out_path = out_dir / f"{path.stem}.npz"

        window_counts[split_name] = window_counts.get(split_name, 0) + n_windows_eval
        split_rejects = rule_rejects.setdefault(split_name, {})
        for rule, count in case_rejects.items():
            split_rejects[rule] = split_rejects.get(rule, 0) + count

        if x is None:
            skip_counts[split_name] = skip_counts.get(split_name, 0) + 1
            out_path.unlink(missing_ok=True)
            n_segs = 0
        else:
            tmp_path = out_dir / f".{path.stem}.tmp.npz"
            np.savez_compressed(tmp_path, x=x, y=y)
            tmp_path.rename(out_path)
            seg_counts[split_name] = seg_counts.get(split_name, 0) + len(x)
            n_segs = len(x)

        _append_index_row(csv_paths[split_name], path.stem, n_segs, csv_lock)
        pbar.update(1)

    pbar.close()
    return seg_counts, skip_counts, window_counts, rule_rejects


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    ratio_sum = sum(args.split)
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"--split ratios must sum to 1.0, got {ratio_sum:.4f}")

    vital_files: list[Path] = sorted(
        args.data_dir.glob("*.vital"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else 0,
    )
    if not vital_files:
        log.error("No .vital files found in %s", args.data_dir)
        return

    guard_sec = 0 if args.no_guard else args.guard_sec
    nproc = args.nproc if args.nproc is not None else (os.cpu_count() or 1)
    resume = not args.no_resume

    log.info("Found %d .vital files in %s", len(vital_files), args.data_dir)
    log.info(
        "Settings: target_hz=%d  segment_sec=%ds  overlap=%ds  guard=%s  nproc=%d  resume=%s",
        args.target_hz,
        args.segment_sec,
        args.segment_sec // 2,
        "disabled" if guard_sec == 0 else f"{guard_sec}s",
        nproc,
        "on" if resume else "off",
    )

    rng = random.Random(args.seed)
    shuffled = vital_files[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * args.split[0])
    n_val = int(n * args.split[1])
    splits: dict[str, list[Path]] = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }

    for split_name, files in splits.items():
        log.info("  %-5s : %d cases", split_name, len(files))
        (args.dataset_dir / split_name).mkdir(parents=True, exist_ok=True)

    for split_name, files in splits.items():
        expected = {p.stem for p in files}
        orphans = [
            npz.stem
            for npz in (args.dataset_dir / split_name).glob("*.npz")
            if npz.stem not in expected
        ]
        if orphans:
            sample = ", ".join(sorted(orphans)[:5]) + ("..." if len(orphans) > 5 else "")
            log.warning(
                "  %-5s : %d NPZ file(s) not in current split assignment - %s",
                split_name,
                len(orphans),
                sample,
            )

    all_tasks: list[tuple[Path, Path, str]] = [
        (path, args.dataset_dir / split_name, split_name)
        for split_name, files in splits.items()
        for path in files
    ]

    csv_paths: dict[str, Path] = {
        split_name: args.dataset_dir / split_name / INDEX_FILE
        for split_name in splits
    }

    if not resume:
        for csv_path in csv_paths.values():
            csv_path.unlink(missing_ok=True)

    resume_counts: dict[str, int] = {name: 0 for name in splits}
    resumed_seg_counts: dict[str, int] = {name: 0 for name in splits}
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

    seg_counts: dict[str, int] = {name: 0 for name in splits}
    skip_counts: dict[str, int] = {name: 0 for name in splits}
    window_counts: dict[str, int] = {name: 0 for name in splits}
    rule_rejects: dict[str, dict[int, int]] = {name: {} for name in splits}

    if pending:
        n_workers = min(nproc, len(pending))
        chunks = [pending[i::n_workers] for i in range(n_workers)]

        tqdm.set_lock(mp.RLock())
        with mp.Manager() as manager:
            csv_lock = manager.Lock()
            worker_args = [
                (i, chunk, args, guard_sec, csv_paths, csv_lock)
                for i, chunk in enumerate(chunks)
            ]
            with mp.Pool(
                processes=n_workers,
                initializer=tqdm.set_lock,
                initargs=(tqdm.get_lock(),),
            ) as pool:
                results = pool.map(_process_chunk, worker_args)

        print()

        for worker_seg, worker_skip, worker_windows, worker_rejects in results:
            for split, count in worker_seg.items():
                seg_counts[split] += count
            for split, count in worker_skip.items():
                skip_counts[split] += count
            for split, count in worker_windows.items():
                window_counts[split] += count
            for split, rejects in worker_rejects.items():
                for rule, count in rejects.items():
                    rule_rejects[split][rule] = rule_rejects[split].get(rule, 0) + count

    for split_name, files in splits.items():
        log.info(
            "  %-5s done - %s segments from %d new cases (%d resumed, %d filtered-out)",
            split_name,
            f"{seg_counts[split_name]:,}",
            len(files) - skip_counts[split_name] - resume_counts[split_name],
            resume_counts[split_name],
            skip_counts[split_name],
        )

    log.info("=" * 60)
    total_new = sum(seg_counts.values())
    total_all = total_new + sum(resumed_seg_counts.values())
    log.info("  %-5s   %12s   %12s   %s", "", "new segs", "total segs", "  %")
    log.info("  " + "-" * 42)
    for split_name in splits:
        new_segs = seg_counts[split_name]
        total_segs = new_segs + resumed_seg_counts[split_name]
        pct = total_segs / total_all * 100 if total_all else 0.0
        log.info(
            "  %-5s   %12s   %12s   %5.1f%%",
            split_name,
            f"{new_segs:,}",
            f"{total_segs:,}",
            pct,
        )
    log.info("  " + "-" * 42)
    log.info("  %-5s   %12s   %12s   %5.1f%%", "total", f"{total_new:,}", f"{total_all:,}", 100.0)
    log.info("Output written to %s", args.dataset_dir.resolve())

    # ------------------------------------------------------------------
    # Write construction-results.json
    # ------------------------------------------------------------------
    total_windows_all = sum(window_counts.values())
    total_segs_new = sum(seg_counts.values())
    total_segs_resumed = sum(resumed_seg_counts.values())
    total_segs_all = total_segs_new + total_segs_resumed
    total_cases_resumed = sum(resume_counts.values())

    json_splits: dict[str, dict] = {}
    total_rule_rejects: dict[str, int] = {}

    for split_name in splits:
        n_cases_assigned = len(splits[split_name])
        n_cases_resumed = resume_counts[split_name]
        n_cases_no_output = skip_counts[split_name]
        n_win = window_counts[split_name]
        n_segs_new = seg_counts[split_name]
        n_segs_total = n_segs_new + resumed_seg_counts[split_name]
        rejects_by_rule = {
            f"rule_{r}": rule_rejects[split_name].get(r, 0) for r in range(1, 10)
        }
        yield_pct = round(n_segs_new / n_win * 100, 2) if n_win > 0 else 0.0

        json_splits[split_name] = {
            "n_cases": n_cases_assigned,
            "n_cases_resumed": n_cases_resumed,
            "n_cases_processed": n_cases_assigned - n_cases_resumed,
            "n_cases_no_output": n_cases_no_output,
            "n_windows_evaluated": n_win,
            "n_segments_generated": n_segs_new,
            "n_segments_total": n_segs_total,
            "yield_pct": yield_pct,
            "rejections_by_rule": rejects_by_rule,
        }

        for key, count in rejects_by_rule.items():
            total_rule_rejects[key] = total_rule_rejects.get(key, 0) + count

    total_yield_pct = (
        round(total_segs_new / total_windows_all * 100, 2)
        if total_windows_all > 0 else 0.0
    )

    results_doc: dict = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "args": {
            "target_hz": args.target_hz,
            "segment_sec": args.segment_sec,
            "guard_sec": guard_sec,
            "split": args.split,
            "seed": args.seed,
            "contlen": args.contlen,
            "sbp_min": args.sbp_min,
            "sbp_max": args.sbp_max,
            "dbp_min": args.dbp_min,
            "dbp_max": args.dbp_max,
            "hr_min": args.hr_min,
            "hr_max": args.hr_max,
            "hr_diff_max": args.hr_diff_max,
            "peak_foot_diff_max": args.peak_foot_diff_max,
            "min_peaks": args.min_peaks,
            "sbp_range_max": args.sbp_range_max,
            "dbp_range_max": args.dbp_range_max,
            "fasqa_psd_low_max": args.fasqa_psd_low_max,
            "fasqa_psd_tgt_min": args.fasqa_psd_tgt_min,
            "fasqa_psd_high_max": args.fasqa_psd_high_max,
        },
        "total": {
            "n_cases": len(vital_files),
            "n_cases_resumed": total_cases_resumed,
            "n_cases_processed": len(vital_files) - total_cases_resumed,
            "n_cases_no_output": sum(skip_counts.values()),
            "n_windows_evaluated": total_windows_all,
            "n_segments_generated": total_segs_new,
            "n_segments_total": total_segs_all,
            "yield_pct": total_yield_pct,
            "rejections_by_rule": total_rule_rejects,
        },
        "splits": json_splits,
    }

    if total_cases_resumed > 0:
        results_doc["note"] = (
            "rejections_by_rule and n_windows_evaluated cover only newly processed cases; "
            f"{total_cases_resumed} resumed case(s) are not included in those counts."
        )

    json_path = args.dataset_dir / "construction-results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_doc, f, indent=2, ensure_ascii=False)
    log.info("Construction results written to %s", json_path)


if __name__ == "__main__":
    main()
