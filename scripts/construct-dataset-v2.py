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
    <output-dir>/
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
from scipy.fft import fft
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.stats import skew
from tqdm import tqdm
from vitaldb.utils import VitalFile

SOURCE_HZ = 500

BANDPASS_LO = 0.5
BANDPASS_HI = 10.0
BANDPASS_ORDER = 4

ABP_PEAK_LO = 0.5
ABP_PEAK_HI = 8.0
ABP_PEAK_ORDER = 3

INDEX_FILE = "index.csv"
SENTINEL_NAN = -9999.0

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build filtered NPZ dataset-v2 directly from VitalDB .vital files",
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/vitaldb"))
    p.add_argument("--output-dir", type=Path, default=Path("data/dataset-v2"))
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


def _nan_linear_interp(signal: np.ndarray) -> np.ndarray:
    out = signal.astype(np.float32, copy=True)
    nan_mask = np.isnan(out)
    if not np.any(nan_mask):
        return out
    valid = np.flatnonzero(~nan_mask)
    if valid.size == 0:
        return out
    out[nan_mask] = np.interp(np.flatnonzero(nan_mask), valid, out[valid])
    return out


def _repetition_fail(signal: np.ndarray, contlen: int) -> bool:
    replaced = np.where(np.isnan(signal), SENTINEL_NAN, signal)
    patience = 0
    prev = None
    for value in replaced:
        patience = patience + 1 if prev is not None and value == prev else 1
        if patience > contlen:
            return True
        prev = value
    return False


def _rule1_repetition_and_interpolation(
    abp: np.ndarray,
    ppg: np.ndarray,
    contlen: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if _repetition_fail(abp, contlen):
        return None
    if _repetition_fail(ppg, contlen):
        return None
    return _nan_linear_interp(abp), _nan_linear_interp(ppg)


def _safe_peak_distance(value: float) -> int:
    return max(1, int(round(value)))


def _threshold_from_candidates(signal: np.ndarray, candidates: np.ndarray) -> float | None:
    if candidates.size == 0:
        return None
    vals = signal[candidates]
    return float((np.mean(vals) - np.min(signal)) * 0.6 + np.min(signal))


def _find_abp_peak_foots(signal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        filtered = _bandpass_filter(
            signal,
            fs,
            lo=ABP_PEAK_LO,
            hi=ABP_PEAK_HI,
            order=ABP_PEAK_ORDER,
        )
        src = signal if np.isnan(filtered).any() else filtered
    except Exception:
        src = signal

    peak_candidates, _ = find_peaks(
        src,
        distance=_safe_peak_distance(fs * 0.35),
        width=max(1, fs * 0.05),
    )
    peak_thres = _threshold_from_candidates(src, peak_candidates)
    if peak_thres is None:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    peaks, _ = find_peaks(
        src,
        distance=_safe_peak_distance(fs * 0.35),
        height=peak_thres,
        width=max(1, fs * 0.05),
    )
    if peaks.size < 2:
        return peaks.astype(np.int64), np.array([], dtype=np.int64)

    inv = np.max(src) - src
    ppi = float(np.mean(np.diff(peaks)))
    foot_candidates, _ = find_peaks(
        inv,
        distance=_safe_peak_distance(fs * 0.5),
        width=max(1, fs * 0.06),
    )
    foot_thres = _threshold_from_candidates(inv, foot_candidates)
    if foot_thres is None:
        return peaks.astype(np.int64), np.array([], dtype=np.int64)
    foots, _ = find_peaks(
        inv,
        distance=_safe_peak_distance(ppi * 0.5),
        height=foot_thres,
        width=max(1, fs * 0.06),
    )
    return peaks.astype(np.int64), foots.astype(np.int64)


def _find_ppg_peak_foots(signal: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    peak_candidates, _ = find_peaks(
        signal,
        distance=_safe_peak_distance(fs * 0.35),
        width=max(1, fs * 0.1),
    )
    peak_thres = _threshold_from_candidates(signal, peak_candidates)
    if peak_thres is None:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    peaks, _ = find_peaks(
        signal,
        distance=_safe_peak_distance(fs * 0.35),
        height=peak_thres,
        width=max(1, fs * 0.1),
    )
    if peaks.size < 2:
        return peaks.astype(np.int64), np.array([], dtype=np.int64)

    inv = np.max(signal) - signal
    ppi = float(np.mean(np.diff(peaks)))
    foot_candidates, _ = find_peaks(
        inv,
        distance=_safe_peak_distance(ppi * 0.5),
        width=max(1, fs * 0.1),
    )
    foot_thres = _threshold_from_candidates(inv, foot_candidates)
    if foot_thres is None:
        return peaks.astype(np.int64), np.array([], dtype=np.int64)
    foots, _ = find_peaks(
        inv,
        distance=_safe_peak_distance(ppi * 0.5),
        height=foot_thres,
        width=max(1, fs * 0.1),
    )
    return peaks.astype(np.int64), foots.astype(np.int64)


def _heart_rate_from_points(points: np.ndarray, fs: int) -> float | None:
    if points.size < 2:
        return None
    interval = float(np.mean(np.diff(points))) / fs
    if interval <= 0:
        return None
    return float(60.0 / interval)


def _avg_heart_rate(peaks: np.ndarray, foots: np.ndarray, fs: int) -> float | None:
    hr_peak = _heart_rate_from_points(peaks, fs)
    hr_foot = _heart_rate_from_points(foots, fs)
    if hr_peak is None or hr_foot is None:
        return None
    return float((hr_peak + hr_foot) / 2.0)


def _minmax_scale(signal: np.ndarray) -> np.ndarray:
    lo = float(np.min(signal))
    hi = float(np.max(signal))
    if hi <= lo:
        return np.zeros_like(signal, dtype=np.float32)
    return ((signal - lo) / (hi - lo)).astype(np.float32)


def _fasqa_adaptive(
    signal: np.ndarray,
    fs: int,
    peaks: np.ndarray,
    foots: np.ndarray,
    *,
    psd_low_max: float,
    psd_tgt_min: float,
    psd_high_max: float,
) -> tuple[bool, tuple[float, float, float, float]]:
    n = len(signal)
    if n < 4:
        return False, (0.0, 0.0, 0.0, 0.0)

    hr_peak = _heart_rate_from_points(peaks, fs)
    hr_foot = _heart_rate_from_points(foots, fs)
    if foots.size == 0 or hr_peak is None or hr_foot is None:
        return False, (0.0, 0.0, 0.0, 0.0)
    if hr_foot < 40.0 or abs(hr_peak - hr_foot) > 5.0:
        return False, (0.0, 0.0, 0.0, 0.0)

    yf = fft(signal)[: n // 2]
    psd = 2.0 * np.abs(yf) / n
    if psd.size <= 1:
        return False, (0.0, 0.0, 0.0, 0.0)

    total = float(np.sum(psd[1:]))
    if total == 0.0:
        return False, (0.0, 0.0, 0.0, 0.0)

    delta_freq = fs / n
    low_idx  = max(1, int(round(((hr_foot / 60.0) - 0.25) / delta_freq)))
    tgt_idx  = max(low_idx + 1, int(round(((hr_foot / 60.0) + 0.25) / delta_freq)))
    high_idx = int(round(7.0 / delta_freq))

    psd_low    = float(np.sum(psd[1:low_idx]) / total)
    psd_target = float(np.sum(psd[low_idx:tgt_idx]) / total)
    psd_high   = float(np.sum(psd[high_idx:]) / total) if high_idx < len(psd) else 0.0
    psdr       = float(np.sum(psd[low_idx:]) / total)  # PSDR: HR 대역 이상 전체 비율

    passed = (
        psd_low    < psd_low_max
        and psd_target > psd_tgt_min
        and psd_high   < psd_high_max
    )
    return passed, (psd_low, psd_target, psd_high, psdr)


def _average_cycle_skewness(foots: np.ndarray, signal: np.ndarray) -> float:
    if foots.size < 2:
        return -1.0
    values: list[float] = []
    for start, end in zip(foots[:-1], foots[1:]):
        if end - start < 3:
            continue
        skew_val = float(skew(signal[start:end]))
        if np.isfinite(skew_val):
            values.append(skew_val)
    if not values:
        return -1.0
    return float(np.mean(values))


def _segment_label_and_quality(
    ppg_seg: np.ndarray,
    abp_seg: np.ndarray,
    abp_raw_seg: np.ndarray,
    fs: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[float]] | int:
    """
    ppg_seg     — BPF 적용 PPG 세그먼트 (peak 검출 및 FASQA에 사용)
    abp_seg     — BPF 적용 ABP 세그먼트 (peak 검출 및 FASQA에 사용)
    abp_raw_seg — BPF 미적용 decimated ABP 원본 (레이블·혈압 범위·변동성 검사에 사용)

    Returns (ppg_seg, [avg_sbp, avg_dbp]) on success, or int failed_rule (1-9) on failure.
    """
    if not np.all(np.isfinite(ppg_seg)) or not np.all(np.isfinite(abp_seg)):
        return 1

    fixed = _rule1_repetition_and_interpolation(abp_seg, ppg_seg, args.contlen)
    if fixed is None:
        return 1
    abp_seg, ppg_seg = fixed

    abp_peaks, abp_foots = _find_abp_peak_foots(abp_seg, fs)
    ppg_peaks, ppg_foots = _find_ppg_peak_foots(ppg_seg, fs)

    # Rule 2: FASQA — ABP
    abp_ok, _ = _fasqa_adaptive(
        abp_seg,
        fs,
        abp_peaks,
        abp_foots,
        psd_low_max=args.fasqa_psd_low_max,
        psd_tgt_min=args.fasqa_psd_tgt_min,
        psd_high_max=args.fasqa_psd_high_max,
    )
    if not abp_ok:
        return 2

    # Rule 2: FASQA — PPG (Min-Max 정규화 후)
    ppg_ok, _ = _fasqa_adaptive(
        _minmax_scale(ppg_seg),
        fs,
        ppg_peaks,
        ppg_foots,
        psd_low_max=args.fasqa_psd_low_max,
        psd_tgt_min=args.fasqa_psd_tgt_min,
        psd_high_max=args.fasqa_psd_high_max,
    )
    if not ppg_ok:
        return 2

    # 레이블 사전 산출: BPF 미적용 원신호에서 절대 mmHg 값 추출 (Rule 3, 8에서 필요)
    if abp_peaks.size == 0 or abp_foots.size == 0:
        return 3
    avg_sbp = float(np.mean(abp_raw_seg[abp_peaks]))
    avg_dbp = float(np.mean(abp_raw_seg[abp_foots]))

    # Rule 3: 혈압 범위 검사
    if not (args.sbp_min <= avg_sbp <= args.sbp_max and args.dbp_min <= avg_dbp <= args.dbp_max):
        return 3
    # 기본: SBP > DBP 생리적 일관성 검사 — Rule 3으로 분류
    if avg_sbp <= avg_dbp:
        return 3

    # Rule 4: 심박수 범위 검사
    hr_abp = _avg_heart_rate(abp_peaks, abp_foots, fs)
    hr_ppg = _avg_heart_rate(ppg_peaks, ppg_foots, fs)
    if hr_abp is None or hr_ppg is None:
        return 4
    if not (args.hr_min <= hr_abp <= args.hr_max and args.hr_min <= hr_ppg <= args.hr_max):
        return 4

    # Rule 5: ABP–PPG 심박수 일치 검사
    if abs(hr_abp - hr_ppg) > args.hr_diff_max:
        return 5

    # Rule 6: Peak/Foot 개수 차이 검사
    if (
        abs(len(abp_peaks) - len(abp_foots)) > args.peak_foot_diff_max
        or abs(len(ppg_peaks) - len(ppg_foots)) > args.peak_foot_diff_max
    ):
        return 6

    # Rule 7: 최소 Peak/Foot 개수 검사
    if len(abp_peaks) < args.min_peaks or len(abp_foots) < args.min_peaks:
        return 7

    # Rule 8: 세그먼트 내 혈압 변동 범위 검사 (원신호 기준)
    if np.ptp(abp_raw_seg[abp_peaks]) > args.sbp_range_max:
        return 8
    if np.ptp(abp_raw_seg[abp_foots]) > args.dbp_range_max:
        return 8

    # Rule 9: PPG Skewness 검사
    if _average_cycle_skewness(ppg_foots, ppg_seg) <= 0.0:
        return 9

    return ppg_seg.astype(np.float32), [avg_sbp, avg_dbp]


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
        result = _segment_label_and_quality(ppg_seg, abp_seg, abp_raw_seg, args.target_hz, args)
        if isinstance(result, int):
            rule_rejects[result] = rule_rejects.get(result, 0) + 1
            continue

        x_seg, label = result
        xs.append(x_seg)
        ys.append(label)

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
        (args.output_dir / split_name).mkdir(parents=True, exist_ok=True)

    for split_name, files in splits.items():
        expected = {p.stem for p in files}
        orphans = [
            npz.stem
            for npz in (args.output_dir / split_name).glob("*.npz")
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
        (path, args.output_dir / split_name, split_name)
        for split_name, files in splits.items()
        for path in files
    ]

    csv_paths: dict[str, Path] = {
        split_name: args.output_dir / split_name / INDEX_FILE
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
                existing_npzs = sorted((args.output_dir / split_name).glob("*.npz"))
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
    log.info("Output written to %s", args.output_dir.resolve())

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

    json_path = args.output_dir / "construction-results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_doc, f, indent=2, ensure_ascii=False)
    log.info("Construction results written to %s", json_path)


if __name__ == "__main__":
    main()
