"""
Dataset-v2 nine-step signal quality control.

Extracted from scripts/construct-dataset-v2.py so that other tools
(e.g. scripts/vitaldb-browser.py) can import and reuse the logic.

Public API
----------
QCParams          - threshold parameters (mirrors construct-dataset-v2 CLI defaults)
QCResult          - per-segment quality result with rule-level detail
check_segment_quality(ppg_seg, abp_seg, abp_raw_seg, fs, t_start, params)
    → QCResult    - evaluate one 8-second segment
compute_case_qc(ppg_raw, abp_raw, ...)
    → list[QCResult]  - evaluate all sliding windows in a full case
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
from scipy.fft import fft
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.stats import skew

# ---------------------------------------------------------------------------
# Constants - must match construct-dataset-v2.py
# ---------------------------------------------------------------------------

SOURCE_HZ = 500
TARGET_HZ = 125
SEGMENT_SEC = 8
GUARD_SEC = 1
STRIDE_SEC = 4

BANDPASS_LO = 0.5
BANDPASS_HI = 10.0
BANDPASS_ORDER = 4

ABP_PEAK_LO = 0.5
ABP_PEAK_HI = 8.0
ABP_PEAK_ORDER = 3

SENTINEL_NAN = -9999.0


# ---------------------------------------------------------------------------
# Parameter / result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QCParams:
    """Threshold parameters - defaults match construct-dataset-v2.py CLI."""
    contlen: int = 10
    sbp_min: float = 60.0
    sbp_max: float = 180.0
    dbp_min: float = 40.0
    dbp_max: float = 120.0
    hr_min: float = 30.0
    hr_max: float = 150.0
    hr_diff_max: float = 10.0
    peak_foot_diff_max: int = 2
    min_peaks: int = 4
    sbp_range_max: float = 40.0
    dbp_range_max: float = 20.0
    fasqa_psd_low_max: float = 0.15
    fasqa_psd_tgt_min: float = 0.10
    fasqa_psd_high_max: float = 0.05
    fasqa_ppg_psd_low_max: float = 0.30
    fasqa_ppg_psd_tgt_min: float = 0.20
    fasqa_ppg_psd_high_max: float = 0.05
    fasqa_abp_psd_low_max: float = 0.25
    fasqa_abp_psd_tgt_min: float = 0.20
    fasqa_abp_psd_high_max: float = 0.08


@dataclass
class QCResult:
    """Quality-control result for a single 8-second segment."""
    t_start: float                        # segment start (seconds within case)
    passed: bool
    failed_rule: int | None               # first failing rule 1-9, None = passed
    # per-rule result (None = not reached)
    rules: dict[int, bool | None]
    metrics: dict[str, float] = field(default_factory=dict)
    avg_sbp: float = 0.0
    avg_dbp: float = 0.0


# ---------------------------------------------------------------------------
# Signal processing helpers - verbatim from construct-dataset-v2.py
# ---------------------------------------------------------------------------

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


def _safe_peak_distance(value: float) -> int:
    return max(1, int(round(value)))


def _threshold_from_candidates(
    signal: np.ndarray, candidates: np.ndarray
) -> float | None:
    if candidates.size == 0:
        return None
    vals = signal[candidates]
    return float((np.mean(vals) - np.min(signal)) * 0.6 + np.min(signal))


def _find_abp_peak_foots(
    signal: np.ndarray, fs: int
) -> tuple[np.ndarray, np.ndarray]:
    try:
        filtered = _bandpass_filter(
            signal, fs, lo=ABP_PEAK_LO, hi=ABP_PEAK_HI, order=ABP_PEAK_ORDER
        )
        src = signal if np.isnan(filtered).any() else filtered
    except Exception:
        src = signal

    peak_cands, _ = find_peaks(
        src,
        distance=_safe_peak_distance(fs * 0.35),
        width=max(1, fs * 0.05),
    )
    peak_thres = _threshold_from_candidates(src, peak_cands)
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
    foot_cands, _ = find_peaks(
        inv,
        distance=_safe_peak_distance(fs * 0.5),
        width=max(1, fs * 0.06),
    )
    foot_thres = _threshold_from_candidates(inv, foot_cands)
    if foot_thres is None:
        return peaks.astype(np.int64), np.array([], dtype=np.int64)

    foots, _ = find_peaks(
        inv,
        distance=_safe_peak_distance(ppi * 0.5),
        height=foot_thres,
        width=max(1, fs * 0.06),
    )
    return peaks.astype(np.int64), foots.astype(np.int64)


def _find_ppg_peak_foots(
    signal: np.ndarray, fs: int
) -> tuple[np.ndarray, np.ndarray]:
    peak_cands, _ = find_peaks(
        signal,
        distance=_safe_peak_distance(fs * 0.35),
        width=max(1, fs * 0.1),
    )
    peak_thres = _threshold_from_candidates(signal, peak_cands)
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
    foot_cands, _ = find_peaks(
        inv,
        distance=_safe_peak_distance(ppi * 0.5),
        width=max(1, fs * 0.1),
    )
    foot_thres = _threshold_from_candidates(inv, foot_cands)
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


def _avg_heart_rate(
    peaks: np.ndarray, foots: np.ndarray, fs: int
) -> float | None:
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
    low_idx = max(1, int(round(((hr_foot / 60.0) - 0.25) / delta_freq)))
    tgt_idx = max(
        low_idx + 1, int(round(((hr_foot / 60.0) + 0.25) / delta_freq)))
    high_idx = int(round(7.0 / delta_freq))

    psd_low = float(np.sum(psd[1:low_idx]) / total)
    psd_target = float(np.sum(psd[low_idx:tgt_idx]) / total)
    psd_high = float(np.sum(psd[high_idx:]) /
                     total) if high_idx < len(psd) else 0.0

    passed = (
        psd_low < psd_low_max
        and psd_target > psd_tgt_min
        and psd_high < psd_high_max
    )
    return passed, (psd_low, psd_target, psd_high, 0.0)


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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_segment_quality(
    ppg_seg: np.ndarray,
    abp_seg: np.ndarray,
    abp_raw_seg: np.ndarray,
    fs: int,
    t_start: float,
    params: QCParams | None = None,
) -> QCResult:
    """
    Apply all nine dataset-v2 quality rules to a single segment.

    Parameters
    ----------
    ppg_seg     : BPF-filtered PPG (target_hz samples)
    abp_seg     : BPF-filtered ABP (target_hz samples)
    abp_raw_seg : decimated raw ABP without BPF - for mmHg label extraction
    fs          : sample rate of all three arrays (target_hz)
    t_start     : segment start time within the case (seconds)
    params      : QCParams; None → use defaults

    Returns
    -------
    QCResult with rules dict and metrics populated up to the failing rule.

    Rules
    -----
    Rule 1 - 반복값 / NaN 검사
        동일 값이 contlen(기본 10 ticks, 80ms) 샘플 이상 연속되면 신호 동결(flatline)로 판정.
        Inf / NaN을 포함하는 세그먼트도 이 단계에서 기각.
        통과 시 잔여 NaN은 선형 보간으로 대체.

    Rule 2 (NEW) - FASQA 스펙트럴 품질 검사 (ABP + PPG 각각)
        Frequency-domain Adaptive Signal Quality Assessment.
        FFT로 PSD를 계산하고 세 구간 비율로 신호 품질을 평가한다.
        단, ABP와 PPG의 파형 특성이 다르므로 동일 임계값을 적용하지 않고
        신호별 임계값을 분리하여 적용한다.

        psd_low  : HR 기본 주파수 이하 저주파 비율
            → PPG는 fasqa_ppg_psd_low_max(0.30) 미만이어야 함
            → ABP는 fasqa_abp_psd_low_max(0.25) 미만이어야 함

        psd_tgt  : HR ±0.25 Hz 대역의 집중도
            → PPG는 fasqa_ppg_psd_tgt_min(0.20) 초과해야 함
            → ABP는 fasqa_abp_psd_tgt_min(0.20) 초과해야 함

        psd_high : 7 Hz 이상 고주파 비율
            → PPG는 fasqa_ppg_psd_high_max(0.05) 미만이어야 함
            → ABP는 fasqa_abp_psd_high_max(0.08) 미만이어야 함

        ABP와 PPG가 각각의 기준을 모두 통과해야 한다.

    Rule 2 (비활성화) - FASQA 스펙트럴 품질 검사 (ABP + PPG 각각)
        Frequency-domain Adaptive Signal Quality Assessment.
        FFT로 PSD를 계산하고 세 구간 비율로 신호 품질을 평가한다.
          psd_low  : HR 기본 주파수 이하 저주파 비율  → fasqa_psd_low_max(0.15) 미만이어야 함
          psd_tgt  : HR ±0.25 Hz 대역의 집중도        → fasqa_psd_tgt_min(0.10) 초과해야 함
          psd_high : 7 Hz 이상 고주파 비율            → fasqa_psd_high_max(0.05) 미만이어야 함
        ABP와 PPG 모두 통과해야 한다. PPG는 min-max 정규화 후 적용.

    Rule 3 - 혈압 범위 검사
        abp_raw_seg(BPF 미적용 원신호)에서 추출한 평균 SBP / DBP가
        정의된 생리적 범위 안에 있어야 한다.
          SBP: sbp_min(60) ~ sbp_max(180) mmHg
          DBP: dbp_min(40) ~ dbp_max(120) mmHg
        SBP ≤ DBP인 생리적으로 불가능한 경우도 이 룰로 기각.

    Rule 4 - 심박수 범위 검사
        ABP와 PPG 각각에서 Peak 간격, Foot 간격으로 HR을 추정하고
        (peak HR + foot HR) / 2 의 평균을 사용한다.
          hr_min(30) ~ hr_max(150) bpm 범위를 벗어나면 기각.

    Rule 5 - ABP-PPG 심박수 일치 검사
        Rule 4에서 구한 HR_ABP와 HR_PPG의 차이가
        hr_diff_max(10) bpm 이내이어야 한다.
        두 신호의 심박수가 크게 다르면 한쪽 신호가 노이즈이거나
        신호 간 시간 동기화 오류일 가능성이 높다.

    Rule 6 - Peak / Foot 개수 차이 검사
        ABP 또는 PPG 신호 내에서 검출된 Peak 수와 Foot 수의 차이가
        peak_foot_diff_max(2) 이하이어야 한다.
        차이가 크면 피크/풋 검출 알고리즘이 노이즈를 오검출한 것으로 판단.

    Rule 7 - 최소 Peak / Foot 개수 검사
        ABP에서 검출된 Peak와 Foot 각각의 수가 min_peaks(4) 이상이어야 한다.
        8초 세그먼트에서 심박수 30 bpm 이상이면 최소 4박자가 포함되어야 한다.

    Rule 8 - 세그먼트 내 혈압 변동 범위 검사
        abp_raw_seg 기준으로 Peak값(SBP)과 Foot값(DBP)의 최대-최소 범위가
        각각 sbp_range_max(40) mmHg, dbp_range_max(20) mmHg 이하이어야 한다.
        범위가 너무 넓으면 혈압 변동이 심하거나 이상치 피크가 포함된 것.

    Rule 9 - PPG 파형 스큐니스 검사
        Foot-to-Foot으로 분할한 각 PPG 박동의 왜도(skewness) 평균이
        양수(> 0)이어야 한다.
        생리적으로 정상 PPG는 급격히 상승 후 완만히 하강하므로 양의 왜도를 가진다.
        왜도 ≤ 0이면 파형이 뒤집히거나 노이즈가 심한 것으로 판단.
    """
    if params is None:
        params = QCParams()

    rules: dict[int, bool | None] = {i: None for i in range(1, 10)}
    metrics: dict[str, float] = {}

    def _fail(rule: int) -> QCResult:
        rules[rule] = False
        return QCResult(
            t_start=t_start, passed=False, failed_rule=rule,
            rules=rules, metrics=metrics,
        )

    # Pre-check: finite values
    if not np.all(np.isfinite(ppg_seg)) or not np.all(np.isfinite(abp_seg)):
        return _fail(1)

    # Rule 1: repetition check + NaN interpolation
    if _repetition_fail(abp_seg, params.contlen) or _repetition_fail(ppg_seg, params.contlen):
        return _fail(1)
    rules[1] = True
    abp_seg = _nan_linear_interp(abp_seg)
    ppg_seg = _nan_linear_interp(ppg_seg)

    # Peak / foot detection (reused by rules 2-9)
    abp_peaks, abp_foots = _find_abp_peak_foots(abp_seg, fs)
    ppg_peaks, ppg_foots = _find_ppg_peak_foots(ppg_seg, fs)

    # # Rule 2: FASQA - ABP
    # abp_ok, (psd_lo_a, psd_tg_a, psd_hi_a, _) = _fasqa_adaptive(
    #     abp_seg, fs, abp_peaks, abp_foots,
    #     psd_low_max=params.fasqa_abp_psd_low_max,
    #     psd_tgt_min=params.fasqa_abp_psd_tgt_min,
    #     psd_high_max=params.fasqa_abp_psd_high_max,
    # )
    # metrics.update(abp_psd_low=psd_lo_a, abp_psd_tgt=psd_tg_a,
    #                abp_psd_high=psd_hi_a)

    # # Rule 2: FASQA - PPG (min-max normalised)
    # ppg_ok, (psd_lo_p, psd_tg_p, psd_hi_p, _) = _fasqa_adaptive(
    #     _minmax_scale(ppg_seg), fs, ppg_peaks, ppg_foots,
    #     psd_low_max=params.fasqa_ppg_psd_low_max,
    #     psd_tgt_min=params.fasqa_ppg_psd_tgt_min,
    #     psd_high_max=params.fasqa_ppg_psd_high_max,
    # )
    # metrics.update(ppg_psd_low=psd_lo_p, ppg_psd_tgt=psd_tg_p,
    #                ppg_psd_high=psd_hi_p)

    # if not abp_ok or not ppg_ok:
    #     return _fail(2)
    # rules[2] = True

    # Labels - needed by Rules 3 and 8 (raw signal → absolute mmHg)
    if abp_peaks.size == 0 or abp_foots.size == 0:
        return _fail(3)
    avg_sbp = float(np.mean(abp_raw_seg[abp_peaks]))
    avg_dbp = float(np.mean(abp_raw_seg[abp_foots]))
    metrics.update(avg_sbp=avg_sbp, avg_dbp=avg_dbp)

    # Rule 3: blood pressure range
    if not (params.sbp_min <= avg_sbp <= params.sbp_max
            and params.dbp_min <= avg_dbp <= params.dbp_max):
        return _fail(3)
    # Basic physiological sanity (SBP > DBP) - classify under Rule 3
    if avg_sbp <= avg_dbp:
        return _fail(3)
    rules[3] = True

    # Rule 4: heart rate range
    hr_abp = _avg_heart_rate(abp_peaks, abp_foots, fs)
    hr_ppg = _avg_heart_rate(ppg_peaks, ppg_foots, fs)
    if hr_abp is None or hr_ppg is None:
        return _fail(4)
    metrics.update(hr_abp=hr_abp, hr_ppg=hr_ppg)
    if not (params.hr_min <= hr_abp <= params.hr_max
            and params.hr_min <= hr_ppg <= params.hr_max):
        return _fail(4)
    rules[4] = True

    # Rule 5: ABP-PPG heart rate consistency
    hr_diff = abs(hr_abp - hr_ppg)
    metrics["hr_diff"] = hr_diff
    if hr_diff > params.hr_diff_max:
        return _fail(5)
    rules[5] = True

    # Rule 6: peak / foot count difference
    if (abs(len(abp_peaks) - len(abp_foots)) > params.peak_foot_diff_max
            or abs(len(ppg_peaks) - len(ppg_foots)) > params.peak_foot_diff_max):
        return _fail(6)
    rules[6] = True

    # Rule 7: minimum peak / foot count
    if len(abp_peaks) < params.min_peaks or len(abp_foots) < params.min_peaks:
        return _fail(7)
    rules[7] = True

    # Rule 8: intra-segment BP variability
    sbp_range = float(np.ptp(abp_raw_seg[abp_peaks]))
    dbp_range = float(np.ptp(abp_raw_seg[abp_foots]))
    metrics.update(sbp_range=sbp_range, dbp_range=dbp_range)
    if sbp_range > params.sbp_range_max or dbp_range > params.dbp_range_max:
        return _fail(8)
    rules[8] = True

    # Rule 9: PPG skewness (positive = physiological forward-slope)
    skewness = _average_cycle_skewness(ppg_foots, ppg_seg)
    metrics["ppg_skewness"] = skewness
    if skewness <= 0.0:
        return _fail(9)
    rules[9] = True

    return QCResult(
        t_start=t_start,
        passed=True,
        failed_rule=None,
        rules=rules,
        metrics=metrics,
        avg_sbp=avg_sbp,
        avg_dbp=avg_dbp,
    )


def compute_case_qc(
    ppg_raw: np.ndarray,
    abp_raw: np.ndarray,
    *,
    source_hz: int = SOURCE_HZ,
    target_hz: int = TARGET_HZ,
    segment_sec: int = SEGMENT_SEC,
    stride_sec: int = STRIDE_SEC,
    guard_sec: int = GUARD_SEC,
    params: QCParams | None = None,
    cancel_event: threading.Event | None = None,
) -> list[QCResult]:
    """
    Compute QC results for every sliding window in a full case.

    Parameters
    ----------
    ppg_raw / abp_raw : raw waveforms at source_hz (500 Hz from VitalDB)
    cancel_event      : set this Event to abort early (e.g. when a new case loads)

    Returns
    -------
    list[QCResult], one entry per 8-second window that had enough guard samples.
    """
    if source_hz % target_hz != 0:
        raise ValueError(
            f"source_hz ({source_hz}) must be divisible by target_hz ({target_hz})"
        )

    factor = source_hz // target_hz
    segment_samples = segment_sec * target_hz
    stride_samples = stride_sec * target_hz
    guard_samples = guard_sec * target_hz

    ppg = ppg_raw[::factor].astype(np.float32)
    abp = abp_raw[::factor].astype(np.float32)
    total = min(len(ppg), len(abp))
    ppg = ppg[:total]
    abp = abp[:total]

    n_windows = (total - segment_samples) // stride_samples + 1
    results: list[QCResult] = []

    for w in range(n_windows):
        if cancel_event is not None and cancel_event.is_set():
            break

        ps = w * stride_samples
        pe = ps + segment_samples
        if pe > total:
            break

        fs_start = ps - guard_samples
        fe_end = pe + guard_samples
        t_start = ps / target_hz

        # Guard band falls outside the recording
        if fs_start < 0 or fe_end > total:
            continue

        ppg_region = ppg[fs_start:fe_end]
        abp_region = abp[fs_start:fe_end]

        if not np.all(np.isfinite(ppg_region)) or not np.all(np.isfinite(abp_region)):
            results.append(QCResult(
                t_start=t_start, passed=False, failed_rule=1,
                rules={i: None for i in range(1, 10)}, metrics={},
            ))
            continue

        try:
            ppg_filtered = _bandpass_filter(ppg_region, target_hz)
            abp_filtered = _bandpass_filter(abp_region, target_hz)
        except Exception:
            results.append(QCResult(
                t_start=t_start, passed=False, failed_rule=1,
                rules={i: None for i in range(1, 10)}, metrics={},
            ))
            continue

        ppg_seg = ppg_filtered[guard_samples: guard_samples + segment_samples]
        abp_seg = abp_filtered[guard_samples: guard_samples + segment_samples]
        abp_raw_seg = abp[ps:pe]

        results.append(
            check_segment_quality(
                ppg_seg, abp_seg, abp_raw_seg,
                fs=target_hz,
                t_start=t_start,
                params=params,
            )
        )

    return results


def compute_window_qc(
    ppg_dec: np.ndarray,
    abp_dec: np.ndarray,
    t0: float,
    t1: float,
    *,
    target_hz: int = TARGET_HZ,
    segment_sec: int = SEGMENT_SEC,
    stride_sec: int = STRIDE_SEC,
    guard_sec: int = GUARD_SEC,
    params: QCParams | None = None,
    cache: "dict[float, QCResult] | None" = None,
) -> "list[QCResult]":
    """
    Compute QC only for segments overlapping [t0, t1) seconds.

    Parameters
    ----------
    ppg_dec / abp_dec : pre-decimated arrays at target_hz (float32 recommended)
    t0, t1            : visible window bounds in seconds
    cache             : if provided, already-computed segments are re-used and
                        new results are stored back for future calls.

    Returns
    -------
    list[QCResult] for all segments whose time span overlaps [t0, t1).
    """
    segment_samples = segment_sec * target_hz
    stride_samples = stride_sec * target_hz
    guard_samples = guard_sec * target_hz

    total = min(len(ppg_dec), len(abp_dec))
    n_windows = (total - segment_samples) // stride_samples + 1
    results: list[QCResult] = []

    for w in range(n_windows):
        ps = w * stride_samples
        pe = ps + segment_samples
        if pe > total:
            break

        t_start = ps / target_hz
        t_end_seg = t_start + segment_sec

        # Skip segments that do not overlap the requested window
        if t_end_seg <= t0 or t_start >= t1:
            continue

        # Guard band falls outside the recording
        fs_start = ps - guard_samples
        fe_end = pe + guard_samples
        if fs_start < 0 or fe_end > total:
            continue

        if cache is not None and t_start in cache:
            results.append(cache[t_start])
            continue

        ppg_region = ppg_dec[fs_start:fe_end]
        abp_region = abp_dec[fs_start:fe_end]

        if not np.all(np.isfinite(ppg_region)) or not np.all(np.isfinite(abp_region)):
            qr = QCResult(
                t_start=t_start, passed=False, failed_rule=1,
                rules={i: None for i in range(1, 10)}, metrics={},
            )
        else:
            try:
                ppg_filtered = _bandpass_filter(ppg_region, target_hz)
                abp_filtered = _bandpass_filter(abp_region, target_hz)
                ppg_seg = ppg_filtered[guard_samples: guard_samples +
                                       segment_samples]
                abp_seg = abp_filtered[guard_samples: guard_samples +
                                       segment_samples]
                abp_raw_seg = abp_dec[ps:pe]
                qr = check_segment_quality(
                    ppg_seg, abp_seg, abp_raw_seg,
                    fs=target_hz, t_start=t_start, params=params,
                )
            except Exception:
                qr = QCResult(
                    t_start=t_start, passed=False, failed_rule=1,
                    rules={i: None for i in range(1, 10)}, metrics={},
                )

        if cache is not None:
            cache[t_start] = qr
        results.append(qr)

    return results
