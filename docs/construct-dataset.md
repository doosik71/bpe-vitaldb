# `construct-dataset.py` 사용 및 개발 설명

작성일: 2026-06-08  
관련 코드: [scripts/construct-dataset.py](../scripts/construct-dataset.py)  
관련 문서: [README.md](../README.md)

---

## 1. 목적

`scripts/construct-dataset.py`는 VitalDB의 원본 `.vital` 파일들을 읽어서
학습용 `NPZ` 데이터셋으로 변환하는 전처리 스크립트다.

이 스크립트의 역할은 다음과 같다.

- 각 수술 케이스에서 PPG 파형을 읽는다.
- 수축기 혈압(`SBP`)과 이완기 혈압(`DBP`) 수치 트랙을 읽는다.
- PPG를 목표 샘플링 주파수로 다운샘플링한다.
- 고정 길이 윈도우로 잘라 세그먼트를 만든다.
- 각 세그먼트 구간에 대응하는 평균 `SBP`, `DBP`를 레이블로 계산한다.
- 유효하지 않은 세그먼트를 제거한다.
- 케이스 단위로 `train / val / test`를 분할해 저장한다.

프로젝트 목표 관점에서 보면, 이 스크립트는 원시 수술 파형을
"모델이 바로 학습할 수 있는 `(x, y)` 쌍"으로 바꾸는 데이터셋 생성기다.

---

## 2. 입력과 출력

### 입력

- 디렉터리: 기본값 `data/vitaldb`
- 파일 형식: `*.vital`
- 각 케이스에서 필요한 트랙
  - `SNUADC/PLETH`: 원본 PPG 파형, 500 Hz
  - `Solar8000/ART_SBP`: 1 Hz 수준의 수축기 혈압 수치
  - `Solar8000/ART_DBP`: 1 Hz 수준의 이완기 혈압 수치

중요한 점은 이 스크립트가 레이블 생성을 위해 **침습 ABP 파형 자체(`SNUADC/ART`)를 직접 쓰지 않고**,
모니터 수치 트랙 `ART_SBP`, `ART_DBP`를 사용한다는 것이다.

### 출력

- 루트 디렉터리: 기본값 `data/dataset`
- 하위 디렉터리
  - `data/dataset/train`
  - `data/dataset/val`
  - `data/dataset/test`
- 파일명: 각 케이스별 `{caseid}.npz`

각 `npz` 파일은 다음 배열을 가진다.

```text
x  float32  (N, segment_samples)   PPG 세그먼트
y  float32  (N, 2)                 [SBP_mean, DBP_mean]
```

여기서:

- `N`: 해당 케이스에서 살아남은 유효 세그먼트 개수
- `segment_samples`: `segment_sec * target_hz`

예를 들어 기본값 `segment_sec=8`, `target_hz=125`이면 `x`의 각 행 길이는 `1000`이다.

---

## 3. 실행 방법

### 기본 실행

```bash
uv run python scripts/construct-dataset.py
```

또는 저장소 제공 실행 스크립트를 사용할 수 있다.

```bash
bin/construct-dataset.bat
```

### 주요 옵션

```bash
uv run python scripts/construct-dataset.py \
  --data-dir data/vitaldb \
  --output-dir data/dataset \
  --split 0.7 0.1 0.2 \
  --target-hz 125 \
  --segment-sec 8 \
  --seed 42
```

옵션 설명:

- `--data-dir`
  - 원본 `.vital` 파일 디렉터리
  - 기본값: `data/vitaldb`
- `--output-dir`
  - 생성된 `NPZ` 데이터셋 루트 디렉터리
  - 기본값: `data/dataset`
- `--split TRAIN VAL TEST`
  - 케이스 단위 분할 비율
  - 기본값: `0.7 0.1 0.2`
  - 세 값의 합은 반드시 `1.0`이어야 한다.
- `--target-hz`
  - 출력 PPG 샘플링 주파수
  - 기본값: `125`
  - 반드시 원본 500 Hz를 나눌 수 있어야 한다.
- `--segment-sec`
  - 세그먼트 길이(초)
  - 기본값: `8`
- `--seed`
  - 케이스 shuffle용 난수 시드
  - 기본값: `42`

### 자주 쓰는 예시

125 Hz 대신 100 Hz로 생성:

```bash
uv run python scripts/construct-dataset.py --target-hz 100
```

출력 경로를 별도로 분리:

```bash
uv run python scripts/construct-dataset.py --output-dir data/dataset-125hz
```

분할 비율 변경:

```bash
uv run python scripts/construct-dataset.py --split 0.8 0.1 0.1
```

---

## 4. 처리 파이프라인

스크립트의 처리 순서는 아래와 같다.

### 4.1 `.vital` 파일 목록 수집

`main()`은 `data-dir`에서 `*.vital` 파일을 모아 정렬한다.
파일명 stem이 숫자면 그 숫자를 기준으로 정렬한다.

파일이 하나도 없으면 에러 로그를 남기고 종료한다.

### 4.2 케이스 단위 shuffle 및 split

전체 파일 목록을 `seed` 기반으로 섞은 뒤, 케이스 단위로 `train`, `val`, `test`를 나눈다.

이 설계의 목적은 **세그먼트 누수(data leakage) 방지**다.
같은 환자 케이스에서 나온 여러 세그먼트가 서로 다른 split에 섞이면,
모델이 사실상 같은 수술의 매우 유사한 신호를 학습/평가에 동시에 보게 된다.

현재 구현은 다음 식으로 케이스 수를 계산한다.

```text
n_train = int(n * train_ratio)
n_val   = int(n * val_ratio)
n_test  = 나머지 전부
```

따라서 케이스 수가 작을 때는 정확히 비율대로 떨어지지 않고, 소수점 아래는 버림된다.

### 4.3 케이스별 트랙 로딩

각 파일은 `process_case()`에서 처리된다.

1. `VitalFile`로 파일을 연다.
2. 필요한 트랙이 모두 있는지 확인한다.
3. 아래 데이터를 읽는다.
   - PPG: `vf.to_numpy(["SNUADC/PLETH"], interval=1 / 500)`
   - BP: `vf.to_numpy(["Solar8000/ART_SBP", "Solar8000/ART_DBP"], interval=1.0)`

파일을 열 수 없거나 트랙 읽기에 실패하면 해당 케이스는 skip된다.

### 4.4 PPG 다운샘플링 및 대역통과 필터링

원본 PPG는 500 Hz다. 처리는 두 단계로 이루어진다.

#### 4.4.1 Decimation

단순 슬라이싱 `ppg_raw[::factor]`로 목표 샘플링 주파수로 낮춘다.

- `target_hz=125`이면 `factor=4`
- `target_hz=100`이면 `factor=5`

`500 % target_hz != 0`이면 즉시 `ValueError`를 발생시킨다.
허용되는 대표적인 값은 `250`, `125`, `100`, `50`, `25` 등이다.

#### 4.4.2 4차 버터워스 대역통과 필터

Decimation 직후 `_bandpass_filter(ppg, target_hz)`를 적용한다.

- 통과 대역: **0.5 Hz ~ 10 Hz**
- 필터 차수: **4차 버터워스 (Butterworth)**
- 구현: `scipy.signal.sosfiltfilt` (SOS 방식, 전·후진 필터링)
- 위상 왜곡: **0** (zero-phase)

선택 이유:

| 주파수 대역 | 역할 |
| --- | --- |
| 0.5 Hz 이하 제거 (고역통과) | 호흡, 체동에 의한 느린 기저선 변동(baseline wander) 제거 |
| 10 Hz 이상 제거 (저역통과) | 전기적 잡음, 고주파 간섭 제거; PPG 주요 성분(~5 Hz)은 보존 |

`sosfiltfilt`는 forward-backward 필터링으로 위상 왜곡 없이 반환한다.
필터 계수는 케이스마다 재계산하지 않고 `butter()`로 호출 시 계산된다.
신호 길이가 충분히 짧은 케이스(`total_sec < segment_sec`)는 이미 버려지므로
edge effect 문제는 실질적으로 발생하지 않는다.

### 4.5 시간 길이 정렬

PPG와 BP 수치 트랙의 길이가 다를 수 있으므로, 실제 사용 가능한 전체 길이는 아래 최소값으로 정해진다.

```text
total_sec = min(
    len(ppg) / target_hz,
    len(sbp_1hz),
    len(dbp_1hz),
)
```

즉, 가장 짧은 트랙 길이에 맞춰 나머지를 잘라 사용한다.

### 4.6 슬라이딩 윈도우 생성

기본 설정에서는:

- 윈도우 길이: `8초`
- stride: `4초`
- overlap: `50%`

구현은 다음 값을 사용한다.

- `segment_samples = segment_sec * target_hz`
- `stride_samples = (segment_sec // 2) * target_hz`
- `stride_sec = segment_sec // 2`

기본값 `segment_sec=8`에서는 정확히 50% overlap이 맞다.
다만 홀수 초를 넣으면 `// 2` 때문에 stride가 내림 처리되므로, 문서상 50%와 완전히 같지 않을 수 있다.
실무에서는 짝수 초 사용을 권장한다.

### 4.7 레이블 계산

각 윈도우에 대해:

- PPG는 `target_hz` 기준 연속 샘플 구간을 자른다.
- BP는 1 Hz 수치 배열에서 같은 시간 구간을 자른다.

레이블 계산은 `_bp_label()`이 담당한다.

1. 유한한 값(`finite`)인지 확인한다.
2. 생리적 범위 안에 있는지 확인한다.
   - `SBP`: `50 ~ 250`
   - `DBP`: `20 ~ 150`
3. 유효 샘플 비율이 `50%` 이상인지 확인한다.
4. 조건을 만족한 샘플들의 평균을 계산한다.

조건을 만족하지 못하면 `None`을 반환하고, 해당 세그먼트는 버린다.

### 4.8 세그먼트 필터링

다음 조건 중 하나라도 걸리면 세그먼트는 폐기된다.

- PPG 세그먼트에 `NaN` 또는 `Inf`가 포함됨
- SBP 유효 샘플 비율이 50% 미만
- DBP 유효 샘플 비율이 50% 미만
- 평균 `SBP <= DBP`

마지막 조건은 생리적으로 일관되지 않은 레이블을 제거하기 위한 안전장치다.

### 4.9 저장

한 케이스에서 유효 세그먼트가 하나 이상 남으면:

```python
np.savez_compressed(out_dir / f"{path.stem}.npz", x=x, y=y)
```

형태로 저장한다.

유효 세그먼트가 하나도 없으면 그 케이스는 저장하지 않는다.

---

## 5. 함수별 설명

### `parse_args()`

CLI 인자를 정의한다.
문서 문자열(`__doc__`)을 `argparse` epilog에 그대로 붙여 도움말을 풍부하게 보여준다.

### `_bp_label(samples, bounds)`

BP 수치 배열에서 유효한 값만 골라 평균 레이블을 만든다.

핵심 정책:

- 범위를 벗어난 값 제거
- `NaN`, `Inf` 제거
- 유효 비율이 50% 미만이면 실패 처리

즉, "몇 개 값만 우연히 정상"인 구간은 살리지 않는다.

### `process_case(path, target_hz, segment_sec)`

실제 전처리의 중심 함수다.

역할:

- 파일 열기
- 필수 트랙 확인
- PPG/BP 로드
- PPG decimation
- 슬라이딩 윈도우 생성
- 세그먼트 필터링
- `(x, y)` 배열 반환

반환값:

- 성공 시: `(x, y)`
- 실패 또는 skip 시: `None`

### `main()`

상위 오케스트레이션 함수다.

역할:

- 로깅 설정
- 인자 검증
- 입력 파일 수집
- split 생성
- split별 케이스 처리
- `npz` 저장
- 최종 통계 출력

---

## 6. 로그와 실행 결과 해석

실행 시 대략 아래 정보를 볼 수 있다.

```text
Found 3000 .vital files in data/vitaldb
Settings: target_hz=125  segment_sec=8s  overlap=4s  split=70/10/20
  train : 2100 cases
  val   : 300 cases
  test  : 600 cases
```

split 처리 후에는 다음과 같은 요약이 나온다.

```text
train done — 123456 segments from 1700 cases (100 skipped)
val   done —  40000 segments from  560 cases (40 skipped)
test  done —  41000 segments from  565 cases (35 skipped)
```

여기서 `skipped`는 주로 다음 의미다.

- 필요한 트랙이 없음
- 파일을 열 수 없음
- 트랙 읽기 실패
- 세그먼트가 하나도 남지 않음
- 전체 길이가 `segment_sec`보다 짧음

---

## 7. 개발 시 알아둘 제약과 주의점

### 7.1 `target_hz`는 500의 약수여야 한다

현재 구현은 리샘플링이 아니라 단순 decimation이다.
따라서 `target_hz=128` 같은 값은 허용되지 않는다.

### 7.2 대역통과 필터 설계 근거

PPG 신호는 심박(~1 Hz 기반)과 그 고조파를 포함하며 주요 에너지는 0.5 ~ 5 Hz 대역에 집중되어 있다.
4차 버터워스 0.5–10 Hz 대역통과 필터를 선택한 이유는 다음과 같다.

- **하한 0.5 Hz**: 호흡(0.1–0.4 Hz)이나 체동, 전극 접촉에 의한 느린 baseline wander를 제거한다.
  0.5 Hz 이상을 유지하면 정상 심박(50–200 bpm = 0.83–3.3 Hz)은 모두 통과한다.
- **상한 10 Hz**: PPG 파형의 이차 미분(APG)까지 포함해도 의미 있는 성분은 10 Hz 이내다.
  이 이상은 전기적 잡음, EMG 간섭으로 분류하여 제거한다.
- **4차 버터워스**: 통과 대역이 최대한 평탄(maximally flat)하며 차단 대역 roll-off가 충분히 가파르다.
  `sosfiltfilt`(전·후진 이중 적용)로 등가 8차 zero-phase 응답을 얻어 위상 왜곡을 완전히 제거한다.

### 7.3 레이블은 1 Hz 평균값이다

현재 레이블은 각 세그먼트 내부의 `ART_SBP`, `ART_DBP` 수치 평균이다.
즉, 파형 beat-by-beat 레이블이 아니라 **구간 평균 혈압 회귀** 문제로 정리되어 있다.

이 설계는 다음 장점이 있다.

- 구현이 단순하다.
- 잡음과 순간 이상치에 덜 민감하다.
- 고정 길이 세그먼트와 잘 맞는다.

반면 단점도 있다.

- 세그먼트 내부의 빠른 혈압 변화가 평균으로 눌린다.
- beat-level 정밀도는 잃는다.

### 7.4 기존 출력 정리는 하지 않는다

스크립트는 `output-dir/train|val|test`를 생성만 하고, 기존 파일을 먼저 지우지는 않는다.

즉:

- 같은 케이스 파일명은 새로 덮어써질 수 있다.
- 이번 실행에서 split에서 빠진 오래된 파일은 디렉터리에 남아 있을 수 있다.

새 설정으로 완전히 다시 만들고 싶다면, 실행 전에 출력 디렉터리를 정리하는 것이 안전하다.

### 7.5 split은 재현 가능하지만 데이터 추가 시 달라질 수 있다

같은 파일 집합과 같은 `seed`면 split은 재현 가능하다.
하지만 `.vital` 파일 수가 바뀌면 shuffle 결과와 경계 위치도 함께 달라진다.

---

## 8. 검증 포인트

문서화 기준으로 이 스크립트를 점검할 때는 아래를 우선 보면 된다.

- `uv run python scripts/construct-dataset.py --help`가 옵션 설명과 일치하는가
- `data/vitaldb`에 `.vital` 파일이 있을 때 정상적으로 split 디렉터리가 생성되는가
- 생성된 `npz`에 `x`, `y` 배열이 존재하는가
- `x.shape[1] == segment_sec * target_hz`인가
- `y.shape[1] == 2`인가
- `SBP > DBP`가 유지되는가

간단한 결과 확인 예시:

```bash
uv run python -c "import numpy as np; d=np.load('data/dataset/train/1.npz'); print(d['x'].shape, d['y'].shape)"
```

---

## 9. 요약

`construct-dataset.py`는 이 프로젝트의 핵심 전처리 스크립트이며,
원본 VitalDB 수술 파형을 모델 학습용 세그먼트 데이터셋으로 바꾸는 역할을 담당한다.

현재 구현의 핵심 특징은 다음과 같다.

- 케이스 단위 split(기본값 70 / 10 / 20)으로 데이터 누수를 방지한다.
- PPG를 500 Hz에서 목표 주파수로 단순 decimation 후
  4차 버터워스 대역통과 필터(0.5–10 Hz, zero-phase)를 적용한다.
- 8초 기본 윈도우와 50% overlap으로 세그먼트를 만든다.
- 1 Hz BP 수치의 구간 평균으로 `SBP`, `DBP` 레이블을 만든다.
- 품질이 낮거나 생리적으로 이상한 구간은 적극적으로 제거한다.

학습 성능, 레이블 정의, 샘플 수, split 재현성에 직접 영향을 주는 스크립트이므로,
향후 전처리 정책을 바꾸려면 이 파일을 우선 기준점으로 삼는 것이 좋다.
