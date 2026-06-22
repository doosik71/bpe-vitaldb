# `construct-dataset-v2.py` 사용 및 상세 설계

작성일: 2026-06-16  
관련 문서: [docs/construct-dataset.md](construct-dataset.md), [docs/construct-dataset-v1.md](construct-dataset-v1.md)

## 1. 목적

`scripts/construct-dataset-v2.py`는 원본 `.vital` 파일을 직접 읽어
9단계 정제 룰을 모두 적용한 고품질 PPG-혈압 데이터셋 `data/dataset-v2`를 구축하는 스크립트다.

기본 파이프라인(decimation, BPF, guard-band, 슬라이딩 윈도우, 병렬처리, resume)은
`construct-dataset.py`와 동일하게 따른다.
추가된 9개 정제 룰과 ABP 파형 기반 레이블 산출이 이 스크립트의 핵심 차별점이다.

### v1 대비 주요 개선점

| 항목               | dataset-v1                      | dataset-v2                                          |
| ------------------ | ------------------------------- | --------------------------------------------------- |
| 입력               | 기존 `data/dataset` NPZ (PPG만) | 원본 `.vital` 파일 (PPG + ABP 파형)                 |
| 품질 평가 기준     | PPG power_ratio 1개             | 9개 정제 룰 순차 적용                               |
| ABP 활용           | 레이블(SBP/DBP 숫자값)만        | ABP 파형으로 Peak/Foot 검출 후 레이블 직접 산출     |
| 신호 품질 평가     | PPG 주파수 비율만               | ABP·PPG 각각 Adaptive FASQA                         |
| 생리적 타당성 검증 | 없음                            | 혈압 범위, HR 범위, ABP-PPG HR 일치도               |
| 파형 형태 검증     | 없음                            | Peak/Foot 개수 신뢰도, 혈압 변동 범위, PPG skewness |

## 2. 처리 흐름

```
.vital 파일 로드 (PPG 500 Hz, ABP 500 Hz)
    │
    ▼
500 Hz → 125 Hz Decimation (factor = 4, 단순 슬라이싱 [::4])
    │
    ├─── abp_raw_seg (BPF 미적용 원신호) ─────────────────────────┐
    │                                                             │
    ▼                                                             │
[기본] NaN / Inf 포함 윈도우(guard-band 포함 구간) 폐기           │
    │                                                             │
    ▼                                                             │
4차 버터워스 BPF (0.5–10 Hz, zero-phase, sosfiltfilt, guard-band) │
    │                                                             │
    ▼                                                             │
guard-band 제거 → ppg_seg, abp_seg (BPF 적용 세그먼트)            │
    │                                                             │
    ▼                                                             │
[룰 1] 연속값 검사 (flat line / 센서 탈락 → 폐기)                 │
    │                                                             │
    ▼                                                             │
ABP Peak / Foot 검출 (abp_seg에 협대역 BPF 후 2단계 find_peaks)   │
PPG Peak / Foot 검출 (ppg_seg 2단계 find_peaks)                   │
    │                                                             │
    ▼                                                             │
[룰 2] FASQA — ABP 주파수 품질 평가 (adaptive)                    │
    │ 실패 → 폐기                                                 │
    ▼                                                             │
[룰 2] FASQA — PPG 주파수 품질 평가 (adaptive, Min-Max 정규화 후) │
    │ 실패 → 폐기                                                 │
    ▼                                                             │
레이블 사전 산출 ◄── abp_raw_seg[peak/foot 인덱스] ───────────────┘
(SBP = mean(abp_raw_seg[peaks]), DBP = mean(abp_raw_seg[foots]))
    │
    ▼
[룰 3] 혈압 범위 검사 (SBP 60–180 mmHg, DBP 40–120 mmHg)
    │ 실패 → 폐기
    ▼
[룰 4] 심박수 범위 검사 (ABP·PPG 모두 30–150 bpm)
    │ 실패 → 폐기
    ▼
[룰 5] ABP–PPG 심박수 일치 검사 (차이 ≤ 10 bpm)
    │ 실패 → 폐기
    ▼
[룰 6] Peak/Foot 개수 차이 검사 (ABP·PPG 모두 ≤ 2)
    │ 실패 → 폐기
    ▼
[룰 7] 최소 Peak/Foot 개수 검사 (ABP peak ≥ 4, foot ≥ 4)
    │ 실패 → 폐기
    ▼
[룰 8] 세그먼트 내 혈압 변동 범위 검사 (abp_raw_seg 기준, SBP ≤ 40, DBP ≤ 20 mmHg)
    │ 실패 → 폐기
    ▼
[룰 9] PPG Skewness 검사 (평균 skewness > 0)
    │ 실패 → 폐기
    ▼
[기본] SBP > DBP 검사 (레이블 생리적 일관성)
    │ 실패 → 폐기
    ▼
케이스 단위 NPZ 저장 → data/dataset-v2/{split}/{caseid}.npz
```

## 3. 입력과 출력

### 입력

| 항목          | 기본값           | 설명                            |
| ------------- | ---------------- | ------------------------------- |
| 루트 디렉터리 | `data/vitaldb`   | `.vital` 파일이 저장된 디렉터리 |
| 파일 형식     | `{caseid}.vital` | VitalDB 원본 파일               |
| PPG 트랙      | `SNUADC/PLETH`   | 500 Hz                          |
| ABP 트랙      | `SNUADC/ART`     | 500 Hz                          |

PPG 또는 ABP 트랙이 없는 케이스는 자동으로 건너뛴다.

### 출력

| 항목          | 기본값            | 설명                                         |
| ------------- | ----------------- | -------------------------------------------- |
| 루트 디렉터리 | `data/dataset-v2` | `train/`, `val/`, `test/` 하위 디렉터리 생성 |
| 파일 형식     | `{caseid}.npz`    | 정제된 세그먼트만 포함                       |
| 케이스 분할   | 70 / 10 / 20      | 케이스 단위 분할 (세그먼트 단위 아님)        |

출력 NPZ 배열 구조:

```text
x  float32  (N, 1000)   PPG 세그먼트 (125 Hz, 8초)
y  float32  (N, 2)      [SBP_mean, DBP_mean] mmHg — ABP peak/foot 평균
```

## 4. CLI 옵션

### 기본 파이프라인 (construct-dataset.md 준용)

| 옵션            | 기본값            | 설명                                              |
| --------------- | ----------------- | ------------------------------------------------- |
| `--data-dir`    | `data/vitaldb`    | 원본 `.vital` 파일 디렉터리                       |
| `--dataset-dir`  | `data/dataset-v2` | 출력 루트 디렉터리                                |
| `--split`       | `0.7 0.1 0.2`     | train / val / test 케이스 비율 (합계 = 1.0)       |
| `--target-hz`   | `125`             | 출력 PPG 샘플링 주파수 (Hz); 500의 약수여야 함    |
| `--segment-sec` | `8`               | 세그먼트 길이 (초); stride는 자동으로 절반(`//2`) |
| `--guard-sec`   | `1`               | BPF guard-band 길이 (초); edge artifact 완화      |
| `--no-guard`    | off               | guard-band 없이 윈도우 구간만 필터링              |
| `--nproc`       | CPU 코어 수       | 병렬 워커 프로세스 수                             |
| `--no-resume`   | off               | 기존 처리 결과 무시하고 처음부터 재처리           |
| `--seed`        | `42`              | 케이스 셔플 랜덤 시드                             |

### 정제 룰 파라미터 (v2 추가)

| 옵션                   | 기본값 | 룰   | 설명                                                  |
| ---------------------- | ------ | ---- | ----------------------------------------------------- |
| `--contlen`            | `10`   | 룰 1 | 연속 동일값 허용 횟수 상한                            |
| `--sbp-min`            | `60`   | 룰 3 | SBP 하한 (mmHg)                                       |
| `--sbp-max`            | `180`  | 룰 3 | SBP 상한 (mmHg)                                       |
| `--dbp-min`            | `40`   | 룰 3 | DBP 하한 (mmHg)                                       |
| `--dbp-max`            | `120`  | 룰 3 | DBP 상한 (mmHg)                                       |
| `--hr-min`             | `30`   | 룰 4 | HR 하한 (bpm)                                         |
| `--hr-max`             | `150`  | 룰 4 | HR 상한 (bpm)                                         |
| `--hr-diff-max`        | `10`   | 룰 5 | ABP-PPG HR 최대 차이 (bpm)                            |
| `--peak-foot-diff-max` | `2`    | 룰 6 | Peak/Foot 개수 최대 차이                              |
| `--min-peaks`          | `4`    | 룰 7 | 최소 ABP peak 개수                                    |
| `--sbp-range-max`      | `40`   | 룰 8 | 세그먼트 내 SBP 변동 상한 (mmHg)                      |
| `--dbp-range-max`      | `20`   | 룰 8 | 세그먼트 내 DBP 변동 상한 (mmHg)                      |
| `--fasqa-psd-low-max`  | `0.15` | 룰 2 | FASQA 저주파 성분 최대 허용값 (초과 시 폐기)          |
| `--fasqa-psd-tgt-min`  | `0.10` | 룰 2 | FASQA 목표 주파수(HR 대역) 최소 에너지 (미달 시 폐기) |
| `--fasqa-psd-high-max` | `0.05` | 룰 2 | FASQA 고주파 노이즈 최대 허용값 (초과 시 폐기)        |

### 자주 쓰는 예시

```bash
# 기본 실행
uv run python scripts/construct-dataset-v2.py

# 입출력 디렉터리 명시
uv run python scripts/construct-dataset-v2.py \
  --data-dir data/vitaldb \
  --dataset-dir data/dataset-v2

# 더 엄격한 HR 범위 (40–130 bpm)
uv run python scripts/construct-dataset-v2.py \
  --hr-min 40 --hr-max 130

# guard-band 비활성화
uv run python scripts/construct-dataset-v2.py --no-guard

# 파라미터 변경 후 전체 재구축
uv run python scripts/construct-dataset-v2.py --no-resume
```

## 5. 기본 파이프라인 상세

기본 파이프라인은 `construct-dataset.py`와 동일하게 따른다. 상세 설명은
[docs/construct-dataset.md](construct-dataset.md)를 참조하고, 여기서는 차이점만 기술한다.

### 5.1 PPG Decimation

원본 PPG 500 Hz를 `[::factor]` 단순 슬라이싱으로 목표 주파수로 낮춘다.
`target_hz=125`이면 `factor=4`. `500 % target_hz ≠ 0`이면 `ValueError`.

### 5.2 4차 버터워스 BPF (0.5–10 Hz, zero-phase)

`scipy.signal.sosfiltfilt`로 윈도우 단위 필터링을 수행한다.

```text
통과 대역: 0.5 Hz – 10 Hz
필터 차수: 4차 버터워스 (sosfiltfilt 적용 시 등가 8차 zero-phase)
```

- 0.5 Hz 고역통과: 호흡·체동에 의한 baseline wander 제거
- 10 Hz 저역통과: 전기적 잡음 및 고주파 간섭 제거

**윈도우 단위 적용 이유**: `sosfiltfilt`는 NaN이 하나라도 있으면 출력 전체가 NaN이 된다.
NaN 체크를 통과한 윈도우에만 필터를 적용한다.

**ABP 신호 분리 사용**: ABP는 두 가지 복사본을 유지한다.

| 복사본                     | BPF                    | 용도                                          |
| -------------------------- | ---------------------- | --------------------------------------------- |
| `abp_seg` (BPF 적용)       | 0.5–10 Hz 4차 버터워스 | Peak/Foot 인덱스 검출, FASQA 품질 평가        |
| `abp_raw_seg` (BPF 미적용) | 없음 (decimation만)    | SBP/DBP 레이블 mmHg 값, 혈압 범위/변동성 검사 |

BPF는 DC 성분(기준 혈압)을 제거하므로 `abp_seg`에서 직접 peak/foot 진폭을 읽으면 절대 mmHg 값이 아닌 편차값이 된다. 레이블은 반드시 `abp_raw_seg`에서 산출해야 한다.

Peak/Foot **검출**(인덱스 산출)에는 `abp_seg`에 추가로 협대역 BPF(0.5–8 Hz, 3차)를 적용한다.

### 5.3 Guard-band 기법 (기본 활성)

0.5 Hz 고역통과 필터의 settling time은 약 1–2초다.
Edge artifact를 완화하기 위해 필터링 구간을 양쪽으로 `guard_sec`초 확장한 뒤
guard 구간을 제거하여 정상 상태(steady-state) 응답만 취한다.

```text
[─ guard_sec ─][──── segment_sec ────][─ guard_sec ─]
←         이 전체 구간으로 sosfiltfilt 적용         →
               └── 이 구간만 저장 ───┘
```

guard 구간에 NaN이 포함되거나 배열 범위를 벗어나면 해당 윈도우는 폐기한다.

### 5.4 슬라이딩 윈도우

```
stride = segment_sec // 2   # 기본 50 % overlap
```

기본값 `segment_sec=8`에서는 stride=4초, 50% overlap.

### 5.5 기본 세그먼트 필터링

BPF 적용 **전에** guard-band 포함 구간(`ppg_region`, `abp_region`) 전체를 검사한다.

- `NaN` 또는 `Inf`가 하나라도 포함되면 해당 윈도우를 **폐기**한다.

`sosfiltfilt`는 NaN이 하나라도 있으면 출력 전체가 NaN이 되므로, NaN 포함 구간은 BPF 전에 반드시 제거해야 한다. 따라서 룰 1 이후 단계에서 NaN을 마주치는 일은 없다. 룰 1의 연속값 검사는 flat line 및 센서 탈락(동일값 연속) 감지에 집중한다.

정제 룰을 모두 통과한 뒤에는 레이블 일관성을 최종 확인한다.

- `SBP_mean > DBP_mean` 조건 불만족 시 폐기

### 5.6 병렬처리 및 Resume

병렬처리와 resume 방식은 `construct-dataset.py`와 동일하다.

- 전체 케이스를 `i::nproc` 라운드로빈으로 워커에 균등 분배한다.
- 각 워커는 독립적인 `tqdm` 진행 막대를 출력한다.
- Resume(기본 활성): 출력 NPZ 파일이 이미 존재하면 해당 케이스를 건너뛴다.
- `--no-resume`: 기존 파일 여부와 무관하게 모든 케이스를 재처리한다.

## 6. 정제 룰 상세

### 룰 1 — 연속값 검사

**목적**: Flat line, Flat peak, 센서 탈락으로 인한 동일값 연속 구간을 검출한다.

**파라미터**: `--contlen` (기본 10회; 125 Hz 기준 80 ms에 해당)

**처리 순서**:

1. **ABP를 먼저** 순회하며 직전 값과 동일한 샘플이 연속 발생하면 카운터를 증가시킨다.
   - `patience > contlen`이 되는 순간 세그먼트를 **폐기**한다.
2. ABP 통과 후 **PPG도 동일하게** 검사한다.

```python
# 핵심 로직
patience = 0
prev = None
for val in signal:
    patience = patience + 1 if prev is not None and val == prev else 1
    if patience > contlen:
        discard(); break
    prev = val
```

> **NaN 보간 없음**: 5.5절에서 설명한 것처럼 NaN은 BPF 전에 미리 제거되므로,
> 룰 1 도달 시점에 NaN은 존재하지 않는다. 연속값 검사만 수행한다.

**판단 근거**: 동일값이 80 ms 이상 지속되면 맥박 주기(최소 375 ms @ 160 bpm)의
일부가 손실된 것이므로 해당 세그먼트는 신뢰할 수 없다.

### 룰 2 — FASQA: 주파수 기반 신호 품질 평가 (Adaptive)

**목적**: FFT 전력 스펙트럼을 분석하여 심박수 대역 에너지가 충분하고,
저주파 drift와 고주파 노이즈가 억제되어 있는지 확인한다.

**적용 대상**: ABP와 PPG 각각에 독립적으로 적용한다.
PPG에는 Min-Max 정규화를 먼저 적용한 후 FASQA를 수행한다.

**PSDR 계산**:

```python
yf = scipy.fft.fft(signal)[0:N//2]
psd = 2.0 * np.abs(yf) / N
PSDR = sum(psd[f_start:f_end]) / sum(psd[1:])
```

**Adaptive 알고리즘 절차**:

1. ABP/PPG peak·foot detection으로 심박수(HR)를 추정한다.
2. 다음 조건이 하나라도 해당하면 즉시 **실패**:
   - foot이 하나도 검출되지 않음
   - HR < 40 bpm (비생리적으로 느린 맥박)
   - `|HR_peak - HR_foot| > 5 bpm` (peak/foot 기반 HR 추정치 불일치)
3. 추정된 HR로부터 주파수 범위를 동적으로 산출한다:
   - 저주파 영역 상한: `HR/60 - 0.25` Hz
   - 목표 주파수 상한: `HR/60 + 0.25` Hz
   - 고주파 노이즈 영역: 7 Hz 이상
4. 세 조건을 **모두** 만족해야 통과:

| 조건                     | CLI 옵션               | 기본값 | 의미                                    |
| ------------------------ | ---------------------- | ------ | --------------------------------------- |
| `psd_low < threshold`    | `--fasqa-psd-low-max`  | 0.15   | DC 성분 및 호흡 노이즈 기여가 낮아야 함 |
| `psd_target > threshold` | `--fasqa-psd-tgt-min`  | 0.10   | 맥박 주기적 성분이 충분해야 함          |
| `psd_high < threshold`   | `--fasqa-psd-high-max` | 0.05   | 7 Hz 이상 노이즈가 낮아야 함            |

**v1 대비 개선**: v1은 Welch PSD 기반 단일 임계값(PPG만)을 사용했다.
v2의 Adaptive FASQA는 HR에 따라 평가 주파수 범위를 동적으로 조정하고
ABP·PPG 양쪽에 독립적으로 적용하므로 더 정밀하다.

### 룰 3 — 혈압 범위 제한

**목적**: 생리적으로 불가능한 혈압값을 가진 세그먼트를 제거한다.

ABP peak 평균값(SBP)과 foot 평균값(DBP)이 아래 범위를 벗어나면 폐기한다.

| 신호                | 하한 (`--sbp/dbp-min`) | 상한 (`--sbp/dbp-max`) |
| ------------------- | ---------------------- | ---------------------- |
| SBP (ABP peak 평균) | 60 mmHg                | 180 mmHg               |
| DBP (ABP foot 평균) | 40 mmHg                | 120 mmHg               |

### 룰 4 — 심박수 범위 제한

**목적**: 검출 오류이거나 임상적으로 극단적인 심박수 세그먼트를 제거한다.

ABP 기반 HR과 PPG 기반 HR **모두** 허용 범위 안에 있어야 한다.

- 허용 HR 범위: 30–150 bpm (`--hr-min`, `--hr-max`)

HR은 peak-to-peak interval(PPI)과 foot-to-foot interval(FFI)의 평균으로 추정한다:

```
avg_HR = (HR_peak + HR_foot) / 2,  where HR = 60 / mean_interval × fs
```

### 룰 5 — ABP–PPG 심박수 일치 검사

**목적**: 동시간에 기록된 ABP와 PPG는 동일한 심박 주기를 반영해야 한다.
HR 추정치 차이가 크면 신호 정합 불량(타임스탬프 오류, 채널 혼용 등)으로 판단한다.

```
|avg_HR_ABP - avg_HR_PPG| ≤ 10 bpm   (--hr-diff-max)
```

### 룰 6 — Peak/Foot 개수 차이 제한

**목적**: 검출 알고리즘 실패를 간접적으로 감지한다.
정상 신호에서는 한 세그먼트 내 peak 수와 foot 수가 거의 같아야 한다.

- ABP: `|N_peaks - N_foots| ≤ 2` (`--peak-foot-diff-max`)
- PPG: `|N_peaks - N_foots| ≤ 2`

### 룰 7 — 최소 Peak/Foot 개수 (레이블 신뢰도)

**목적**: SBP·DBP 레이블은 여러 심박 주기의 평균으로 산출되므로,
최소한의 주기 수가 확보되어야 신뢰도가 보장된다.

- ABP peak 개수 ≥ **4** (`--min-peaks`)
- ABP foot 개수 ≥ **4**

8초 세그먼트에서 60 bpm 기준 약 8개의 주기가 예상되며, 4개는 보수적 하한이다.

### 룰 8 — 세그먼트 내 혈압 변동 범위 제한

**목적**: 단일 8초 세그먼트 안에서 혈압이 과도하게 변동하면
혈역학적으로 불안정한 과도기 구간으로 판단한다.
이런 구간은 레이블의 대표값(평균)이 신호의 실제 특성을 반영하지 못한다.

```
max(SBP_peaks) - min(SBP_peaks) ≤ 40 mmHg   (--sbp-range-max)
max(DBP_foots) - min(DBP_foots) ≤ 20 mmHg   (--dbp-range-max)
```

### 룰 9 — PPG Skewness 검사

**목적**: 정상 PPG 파형은 급격한 수축기 상승부와 완만한 이완기 하강부로 구성되어
양의 비대칭(right-skewed) 분포를 가진다. Skewness가 음수이면 파형이 역전되었거나
심하게 왜곡된 것으로 판단한다.

**계산 방법**: foot-to-foot 구간마다 scipy `skew()`를 계산하여 평균한다.

```
avg_skew_PPG = mean( skew(PPG[foot_i : foot_{i+1}]) for each cycle )
avg_skew_PPG > 0   이어야 통과
```

## 7. Peak / Foot 검출 방법

ABP와 PPG의 peak·foot 검출에는 `scipy.signal.find_peaks`를 사용한다.
후보 검출 → 진폭 임계값 산출 → 최종 검출의 2단계 방식을 취한다.

### ABP Peak 검출

1. Butterworth bandpass filter (0.5–8 Hz, 3차)를 적용한다.
   필터 후 NaN이 포함되면 원신호를 대신 사용한다.
2. 1차 후보 peak 검출: `distance=0.35×fs`, `width=0.05×fs`
3. 진폭 임계값 산출: `thres = (mean(candidates) - min(signal)) × 0.6 + min(signal)`
4. 최종 peak 검출: `height=thres` 조건 추가 재검출

### ABP Foot 검출

1. 신호를 반전: `inv = max(signal) - signal`
2. 검출된 peak 간 평균 PPI 계산
3. 반전 신호에서 foot 후보 검출: `distance=0.5×fs`, `width=0.06×fs`
4. 동일한 진폭 임계값 방식 + PPI 기반 `distance`로 최종 foot 검출

### PPG Peak / Foot 검출

0.5–10 Hz BPF가 적용된 `ppg_seg`에 동일한 2단계 방식을 적용한다.
BPF로 노이즈를 제거한 신호에서 검출하면 오검출이 줄어든다.

- Peak: `distance=0.35×fs`, `width=0.1×fs`
- Foot: 신호 반전 후 동일하게 적용

## 8. 레이블 산출

Ground truth SBP·DBP는 ABP 파형에서 직접 추출한다.
`Solar8000/ART_SBP`, `ART_DBP` 수치 트랙에 의존하지 않으므로
각 세그먼트의 실제 파형과 레이블의 시간적 정합이 보장된다.

peak·foot **인덱스**는 BPF 적용 신호(`abp_seg`)에서 검출하지만,
레이블 **값**은 BPF 미적용 decimated 원신호(`abp_raw_seg`)에서 읽는다.
BPF(0.5 Hz 고역통과)는 DC 성분(기준 혈압)을 제거하므로
필터된 신호의 peak/foot 진폭은 절대 mmHg 값이 아닌 편차값이 된다.

```
peak_indices, foot_indices ← abp_seg (BPF 적용) 에서 검출

SBP = mean(abp_raw_seg[peak_indices])   # 원신호 peak 평균 (수축기 혈압, mmHg)
DBP = mean(abp_raw_seg[foot_indices])   # 원신호 foot 평균 (이완기 혈압, mmHg)
```

## 9. v1과의 정제 기준 비교

| 정제 기준                                |      dataset-v1       |          dataset-v2          |
| ---------------------------------------- | :-------------------: | :--------------------------: |
| PPG Welch power_ratio ≥ 0.6              |           ✓           |       — (FASQA로 대체)       |
| 기본: NaN / Inf 포함 세그먼트 폐기       |           ✓           |              ✓               |
| 기본: SBP > DBP 검사                     |           ✓           |              ✓               |
| 룰 1: 연속값 검사 + 선형 보간            |           —           |              ✓               |
| 룰 2: FASQA Adaptive (ABP)               |           —           |              ✓               |
| 룰 2: FASQA Adaptive (PPG)               |           —           |              ✓               |
| 룰 3: 혈압 범위 (SBP 60–180, DBP 40–120) |           —           |              ✓               |
| 룰 4: HR 범위 (30–150 bpm)               |           —           |              ✓               |
| 룰 5: ABP–PPG HR 차이 ≤ 10 bpm           |           —           |              ✓               |
| 룰 6: Peak/Foot 개수 차이 ≤ 2            |           —           |              ✓               |
| 룰 7: 최소 Peak/Foot 개수 ≥ 4            |           —           |              ✓               |
| 룰 8: SBP 변동 ≤ 40, DBP 변동 ≤ 20 mmHg  |           —           |              ✓               |
| 룰 9: PPG skewness > 0                   |           —           |              ✓               |
| 레이블 출처                              | `Solar8000` 수치 트랙 | ABP 파형 peak/foot 직접 추출 |

## 10. 주의할 점

### 10.1 전제 조건

PPG와 ABP 트랙을 **모두** 포함하는 `.vital` 파일이 필요하다.

```bash
bin/download-vitaldb --filter-tracks
```

### 10.2 `--target-hz`는 500의 약수여야 한다

Decimation을 단순 슬라이싱으로 구현하므로 `target_hz=128` 같은 값은 허용되지 않는다.
허용 대표값: `250`, `125`, `100`, `50`, `25`.

### 10.3 파라미터 변경 시 재처리

resume 모드는 출력 NPZ 존재 여부만 확인하므로,
정제 파라미터를 변경한 경우 반드시 `--no-resume`으로 전체 재처리해야 한다.

### 10.4 세그먼트 수 감소

9개 정제 룰을 순차 적용하면 데이터셋보다 세그먼트 수가 크게 줄어든다.
데이터 부족이 우려되면 `--segment-sec`을 줄이거나 (단, 레이블 신뢰도 하락),
룰 7 `--min-peaks`를 3으로 낮추는 방법을 검토할 수 있다.

### 10.5 레이블 방식 변경에 따른 수치 차이

v2의 레이블은 ABP 파형 peak·foot 평균이다.
기존 `Solar8000` 수치 트랙 기반 레이블과는 동일 케이스에서도 수 mmHg 차이가 날 수 있으므로,
v2로 학습한 모델의 성능을 기존 데이터셋 기반 모델과 직접 비교할 때 이 점을 감안해야 한다.

### 10.6 split 재현성

같은 파일 집합과 같은 `--seed`면 split은 재현 가능하다.
`.vital` 파일 수가 바뀌면 shuffle 결과와 경계 위치도 함께 달라진다.
