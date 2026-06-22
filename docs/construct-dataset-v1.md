# `construct-dataset-v1.py` 사용 및 상세 설계

작성일: 2026-06-15  
관련 코드: [scripts/construct-dataset-v1.py](../scripts/construct-dataset-v1.py)  
관련 문서: [docs/construct-dataset.md](construct-dataset.md), [README.md](../README.md)

## 1. 목적

`scripts/construct-dataset-v1.py`는 `data/dataset`에 이미 구축된 NPZ 데이터셋을 입력으로 받아,
PPG 신호 품질이 낮은 세그먼트를 제거하고 고품질 세그먼트만 남긴 `data/dataset-v1`을 생성하는
필터링 스크립트다.

이 스크립트의 역할은 다음과 같다.

- 기존 NPZ 파일에서 PPG 세그먼트를 하나씩 읽는다.
- 각 세그먼트에 대해 Welch 방법으로 전력 스펙트럼 밀도(PSD)를 계산한다.
- 심박수 대역(0.67–3.0 Hz)의 전력이 전체 패스밴드(0.5–10.0 Hz) 전력에서
  차지하는 비율(`power_ratio`)을 계산한다.
- `power_ratio`가 임계값 이상인 세그먼트만 선별하여 새 NPZ 파일로 저장한다.
- `index.csv`를 사용해 재실행 시 이미 처리된 케이스를 자동으로 건너뛴다(resume).
- `multiprocessing`으로 CPU 코어 수만큼 병렬 처리한다.

`construct-dataset.py`와 달리 원본 `.vital` 파일이 필요 없다.
이미 파싱·리샘플링·세그멘테이션이 완료된 NPZ 파일을 재활용한다.

## 2. 입력과 출력

### 입력

| 항목          | 기본값         | 설명                                         |
| ------------- | -------------- | -------------------------------------------- |
| 루트 디렉터리 | `data/dataset` | `train/`, `val/`, `test/` 하위 디렉터리 포함 |
| 파일 형식     | `{caseid}.npz` | `x (N, L)`, `y (N, 2)` 배열 포함             |

각 NPZ 파일 내 배열:

```text
x  float32  (N, segment_samples)   PPG 세그먼트 (케이스별 N개)
y  float32  (N, 2)                 [SBP_mean, DBP_mean] per segment
```

### 출력

| 항목          | 기본값              | 설명                                         |
| ------------- | ------------------- | -------------------------------------------- |
| 루트 디렉터리 | `data/dataset-v1`   | `train/`, `val/`, `test/` 하위 디렉터리 생성 |
| 파일 형식     | `{caseid}.npz`      | 선별된 세그먼트만 포함                       |
| 인덱스 파일   | `{split}/index.csv` | 케이스별 선별 세그먼트 수 기록               |

출력 NPZ 배열 구조는 입력과 동일하다.
세그먼트 수 `N`이 줄어들거나, 품질 조건을 만족하는 세그먼트가 하나도 없으면
해당 케이스의 NPZ 파일이 생성되지 않는다.

## 3. 사용 방법

### 기본 실행

```bash
uv run python scripts/construct-dataset-v1.py
```

### 자주 쓰는 예시

```bash
# 입력/출력 디렉터리 명시
uv run python scripts/construct-dataset-v1.py \
  --input-dir data/dataset \
  --dataset-dir data/dataset-v1

# 임계값 높이기 (더 엄격한 선별)
uv run python scripts/construct-dataset-v1.py \
  --power-ratio-min 0.7

# 워커 수 제한 (메모리 절약)
uv run python scripts/construct-dataset-v1.py \
  --nproc 4

# 이미 처리된 케이스 무시하고 전체 재처리
uv run python scripts/construct-dataset-v1.py \
  --no-resume

# 전체 옵션 조합 예시
uv run python scripts/construct-dataset-v1.py \
  --input-dir data/dataset \
  --dataset-dir data/dataset-v1 \
  --target-hz 125 \
  --nperseg 256 \
  --power-ratio-min 0.6 \
  --nproc 8
```

## 4. CLI 옵션

| 옵션                | 기본값            | 설명                                                |
| ------------------- | ----------------- | --------------------------------------------------- |
| `--input-dir`       | `data/dataset`    | 원본 NPZ 데이터셋 루트 디렉터리                     |
| `--dataset-dir`      | `data/dataset-v1` | 선별된 NPZ 출력 루트 디렉터리                       |
| `--target-hz`       | `125`             | PPG 샘플링 주파수 (Hz); Welch PSD 계산에 사용       |
| `--nperseg`         | `256`             | Welch 세그먼트 길이 (샘플 수); 주파수 해상도 결정   |
| `--power-ratio-min` | `0.6`             | 허용 최소 power_ratio; 이 값 미만인 세그먼트는 제거 |
| `--nproc`           | CPU 코어 수       | 병렬 워커 프로세스 수                               |
| `--no-resume`       | off               | 기존 출력 NPZ가 있어도 무시하고 처음부터 재처리     |

### `--nperseg` 설정 가이드

Welch 방법의 주파수 해상도는 `fs / nperseg`다.

| `nperseg` | 해상도 (125 Hz 기준) | 비고                                 |
| --------- | -------------------- | ------------------------------------ |
| 128       | ~0.98 Hz             | 거친 추정, 빠름                      |
| 256       | ~0.49 Hz (기본값)    | 심박수 대역 구분에 적합              |
| 512       | ~0.24 Hz             | 세밀한 추정, 신호 길이가 충분해야 함 |

신호 길이보다 `nperseg`가 크면 `min(len(signal), nperseg)`로 자동 조정된다.

### `--power-ratio-min` 설정 가이드

| 값  | 선별 강도 | 특성                                         |
| --- | --------- | -------------------------------------------- |
| 0.5 | 느슨함    | 노이즈가 많은 세그먼트 일부 포함될 수 있음   |
| 0.6 | 기본값    | 실험적으로 적합한 균형점                     |
| 0.7 | 엄격함    | 더 깨끗한 데이터, 케이스 당 세그먼트 수 감소 |
| 0.8 | 매우 엄격 | 데이터셋 크기가 크게 줄어들 수 있음          |

## 5. 동작 방식

### 5.1 전체 흐름

```
parse_args()
    │
    ├─ input_dir/{train,val,test}/*.npz 목록 수집
    ├─ output_dir/{train,val,test}/ 디렉터리 생성
    │
    ├─ resume 모드
    │   ├─ index.csv 읽기 (없으면 기존 NPZ에서 재구성)
    │   └─ 미처리 케이스만 pending 목록에 추가
    │
    ├─ multiprocessing.Pool 생성 (nproc 워커)
    │   └─ _process_chunk(worker_id, tasks, ...) × nproc
    │           └─ process_case(path, ...) × 케이스 수
    │                   ├─ NPZ 로드 → x, y 배열
    │                   ├─ 세그먼트별 power_ratio 계산
    │                   ├─ 조건 미달 세그먼트 제거
    │                   └─ (x_filtered, y_filtered) 반환
    │
    └─ 결과 요약 출력 (split별 신규 세그먼트 수, resume 수, 필터 제거 수)
```

### 5.2 power_ratio 계산

각 PPG 세그먼트에 대해 다음 순서로 신호 품질을 정량화한다.

**Step 1 — Welch PSD 계산**

```python
freqs, psd = welch(
    signal,
    fs=target_hz,          # 125 Hz
    window="hann",
    nperseg=min(len(signal), nperseg),
    noverlap=None,         # 기본값: nperseg // 2
    detrend="constant",    # DC 성분 제거
    scaling="density",     # 단위: V²/Hz
)
```

**Step 2 — 대역 전력 계산**

사다리꼴 적분(`np.trapezoid`)으로 두 주파수 대역의 전력을 계산한다.

```
heart_power    = ∫[0.67, 3.0] PSD(f) df   # 심박수 대역 (40–180 BPM)
passband_power = ∫[0.5, 10.0] PSD(f) df   # 전체 PPG 유효 대역
```

**Step 3 — 비율 계산**

```
power_ratio = heart_power / passband_power
```

`power_ratio`가 높을수록 PPG 신호의 에너지가 심박수 대역에 집중되어 있다.
즉, 주기적인 맥박 파형이 지배적이고 노이즈가 적다는 의미다.

**예외 처리**: 다음 경우 해당 세그먼트를 제거한다.

- `signal`에 `nan` 또는 `inf` 포함
- `heart_power` 또는 `passband_power`가 유한하지 않음
- `passband_power <= 0`
- `power_ratio < power_ratio_min`

### 5.3 파일 단위 처리 (`process_case`)

```python
def process_case(path, *, target_hz, nperseg, power_ratio_min):
```

1. NPZ 파일에서 `x`(세그먼트), `y`(레이블)를 로드한다.
2. 배열 형상을 검증한다 (`x.ndim == 2`, `y.ndim == 2`, `len(x) == len(y)`).
3. 세그먼트를 순회하며 `power_ratio`를 계산하고 통과한 인덱스를 수집한다.
4. 통과한 인덱스가 없으면 `None`을 반환한다 (파일 미생성).
5. 통과한 세그먼트만 슬라이싱하여 `(x_filtered, y_filtered)`를 반환한다.

### 5.4 병렬 처리 (`_process_chunk`)

전체 태스크를 워커 수로 나누어 각 워커에게 균등 분배한다.

```python
chunks = [all_tasks[i::n_workers] for i in range(n_workers)]
```

`i::n_workers` 방식의 라운드로빈 분배로 각 워커가 비슷한 수의 케이스를 담당하면서
처리 시간도 고르게 분산된다.

각 워커는 독립적인 `tqdm` 진행 막대를 유지한다.
`mp.RLock`으로 진행 막대 출력이 섞이지 않도록 보호한다.

### 5.5 원자적 파일 쓰기

출력 NPZ 파일은 임시 파일에 먼저 저장한 뒤 최종 이름으로 교체한다.

```python
tmp_path = out_dir / f".{path.stem}.tmp.npz"
np.savez_compressed(tmp_path, x=x, y=y)
tmp_path.rename(out_path)
```

이 방식으로 중간에 프로세스가 죽어도 불완전한 NPZ 파일이 남지 않는다.

### 5.6 index.csv

각 split 디렉터리에 `index.csv`가 생성된다.

```
case_id,n_segments
10001,842
10002,0
10007,1205
```

`n_segments == 0`은 해당 케이스의 세그먼트가 전부 필터링되어 NPZ 파일이 없음을 의미한다.
`index.csv`는 여러 워커가 동시에 쓰므로 `multiprocessing.Manager().Lock()`으로 보호된다.

### 5.7 resume 로직

`--no-resume`을 주지 않으면 이미 처리된 케이스는 건너뛴다.

판단 기준은 다음 순서로 확인한다.

1. `index.csv`가 있으면 해당 케이스가 기록되어 있는지 확인한다.
   - 기록이 있고 `n_segments == 0`이면 → resume (필터 탈락 케이스)
   - 기록이 있고 `n_segments > 0`이며 NPZ 파일도 있으면 → resume (정상 처리)
   - 기록이 있지만 NPZ 파일이 없으면 → 재처리 (파일 삭제 후 재실행된 경우)
   - 기록이 없으면 → 재처리

2. `index.csv`가 없으면 기존 NPZ 파일에서 세그먼트 수를 읽어 `index.csv`를 새로 구성한다.

이 덕분에 중간에 중단된 작업을 재시작해도 처음부터 다시 처리하지 않는다.

## 6. 출력 요약

실행 완료 후 다음과 같은 요약이 터미널에 출력된다.

```
HH:MM:SS [INFO]   train done - 1,234,567 segments from 512 new cases (88 resumed, 12 filtered-out)
HH:MM:SS [INFO]   val   done -   123,456 segments from  51 new cases ( 9 resumed,  1 filtered-out)
HH:MM:SS [INFO]   test  done -   123,789 segments from  51 new cases ( 9 resumed,  1 filtered-out)
HH:MM:SS [INFO] ============================================================
HH:MM:SS [INFO]             new segs   total segs        %
HH:MM:SS [INFO]   ------------------------------------------
HH:MM:SS [INFO]   train    1,234,567    1,456,789    82.1%
HH:MM:SS [INFO]   val        123,456      145,678     8.2%
HH:MM:SS [INFO]   test       123,789      146,001     8.2%
HH:MM:SS [INFO]   ------------------------------------------
HH:MM:SS [INFO]   total    1,481,812    1,748,468   100.0%
HH:MM:SS [INFO] Output written to /path/to/data/dataset-v1
```

각 항목의 의미:

| 항목           | 설명                                                |
| -------------- | --------------------------------------------------- |
| `new segs`     | 이번 실행에서 새로 처리된 케이스의 선별 세그먼트 수 |
| `total segs`   | resume된 케이스 포함 전체 선별 세그먼트 수          |
| `new cases`    | 이번에 실제로 처리한 케이스 수                      |
| `resumed`      | index.csv 기준 건너뛴 케이스 수                     |
| `filtered-out` | 품질 조건 미달로 NPZ가 생성되지 않은 케이스 수      |

## 7. 내부 모듈 설계

### 주요 상수

| 상수         | 값                         | 의미                          |
| ------------ | -------------------------- | ----------------------------- |
| `PASSBAND`   | `(0.5, 10.0)`              | 분모 대역: 전체 PPG 유효 범위 |
| `HEART_BAND` | `(0.67, 3.0)`              | 분자 대역: 40–180 BPM 심박수  |
| `INDEX_FILE` | `"index.csv"`              | split별 처리 결과 인덱스 파일 |
| `SPLITS`     | `("train", "val", "test")` | 처리 대상 split 목록          |

### 함수 요약

| 함수                 | 입력                          | 반환                      | 역할                           |
| -------------------- | ----------------------------- | ------------------------- | ------------------------------ |
| `compute_psd`        | signal, fs, nperseg           | (freqs, psd)              | Welch PSD 계산                 |
| `band_power`         | freqs, psd, band              | float                     | 특정 대역 전력 (사다리꼴 적분) |
| `power_ratio`        | signal, fs, nperseg           | float                     | 심박수 대역 비율 계산          |
| `process_case`       | path, target_hz, nperseg, min | (x, y) \| None            | 케이스 단위 필터링             |
| `_process_chunk`     | (worker_id, tasks, ...)       | (seg_counts, skip_counts) | 워커 함수 (여러 케이스 처리)   |
| `_read_index`        | csv_path                      | dict[str, int]            | index.csv 읽기                 |
| `_write_index`       | csv_path, index               | None                      | index.csv 쓰기 (전체 재기록)   |
| `_append_index_row`  | csv_path, case_id, n, lock    | None                      | index.csv에 행 추가 (락 사용)  |
| `_npz_segment_count` | npz_path                      | int                       | NPZ 파일에서 세그먼트 수 조회  |

## 8. 전제 조건

이 스크립트를 실행하기 전에 `data/dataset`이 구축되어 있어야 한다.

```bash
bin/construct-dataset      # Linux / macOS
bin\construct-dataset.bat  # Windows
```

`data/dataset/{train,val,test}/*.npz`가 존재해야 한다.
파일이 없으면 스크립트가 오류를 출력하고 종료한다.

## 9. 주의할 점

### 9.1 `--target-hz`는 원본과 일치해야 함

`--target-hz`는 NPZ 파일 내 PPG 세그먼트의 실제 샘플링 주파수여야 한다.
`construct-dataset.py`의 기본값과 동일한 `125`를 사용하면 된다.
이 값이 틀리면 PSD의 주파수 축이 왜곡되어 잘못된 세그먼트가 선별된다.

### 9.2 `power_ratio_min` 변경 시 재처리 필요

이미 처리된 케이스는 resume으로 건너뛰기 때문에,
`--power-ratio-min` 값을 바꿔 재실행하려면 `--no-resume`을 함께 사용해야 한다.

```bash
uv run python scripts/construct-dataset-v1.py \
  --power-ratio-min 0.7 \
  --no-resume
```

### 9.3 필터 탈락 케이스의 NPZ 파일

`process_case`가 `None`을 반환하면 (`keep_indices`가 비어 있으면)
이전에 존재하던 출력 NPZ 파일을 `unlink`로 삭제한다.
재실행 후 해당 케이스의 NPZ가 없는 것은 정상이다.

### 9.4 병렬 처리와 메모리

각 워커가 NPZ 파일 전체를 메모리에 올린다.
`--nproc`를 줄이면 메모리 사용량을 줄일 수 있다.

## 10. 한계와 향후 확장

현재 구현은 단순함을 우선한다.

현재 한계:

- 필터링 기준이 power_ratio 하나뿐이다 (피크 검출, SNR 등 추가 기준 없음).
- `--target-hz`가 틀려도 오류를 발생시키지 않는다.
