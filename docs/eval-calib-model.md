# `eval-calib-model.py` 사용 및 상세 설계

작성일: 2026-06-12  
관련 코드: [scripts/eval-calib-model.py](../scripts/eval-calib-model.py)
관련 문서: [docs/eval-model.md](eval-model.md), [docs/train-model.md](train-model.md)

## 1. 목적

`scripts/eval-calib-model.py`는 이미 학습된 BPE 모델을 단일 테스트 케이스에 대해 calibration한 뒤
그 케이스 전체에 대해 다시 평가하는 스크립트다.

이 스크립트의 역할은 다음과 같다.

- 학습 완료된 모델의 `best.pt`와 `config.json`을 로드한다.
- 테스트 split에서 하나의 케이스를 선택한다.
- 그 케이스의 일부 세그먼트를 calibration set으로 뽑는다.
- 모델의 마지막 BP 출력 선형층만 1 epoch 미세조정한다.
- 보정된 모델을 같은 케이스의 전체 세그먼트에 적용한다.
- MAE, RMSE, ME, SD, BHS, AAMI 지표를 계산한다.
- 산점도, 오차 히스토그램, Bland-Altman 플롯을 저장한다.

즉, 이 도구는 전체 테스트셋 일반화 성능을 보는 `eval-model.py`와 달리,
"특정 환자/케이스에 대해 소량 calibration을 했을 때 얼마나 개선되는가"를 보는 개인화 평가 도구다.

## 2. 사용 방법

### 기본 실행

```bash
uv run python scripts/eval-calib-model.py data/models/resnet1d
```

### 주요 사용 예시

```bash
# 첫 번째 test 케이스를 기본 설정으로 calibration 후 평가
uv run python scripts/eval-calib-model.py data/models/resnet1d

# 특정 케이스를 선택하고 calibration 세그먼트 수를 20개로 지정
uv run python scripts/eval-calib-model.py \
  data/models/resnet1d \
  --case-id 1234 \
  --n-calib 20

# 케이스 세그먼트의 10%만 calibration에 사용
uv run python scripts/eval-calib-model.py \
  data/models/resnet1d \
  --case-id 1234 \
  --n-calib 0.1

# GPU 지정
uv run python scripts/eval-calib-model.py \
  data/models/resnet1d \
  --device cuda:0

# 학습 시 정규화를 쓰지 않았다면 동일하게 비활성화
uv run python scripts/eval-calib-model.py \
  data/models/resnet1d \
  --no-normalize
```

## 3. CLI 옵션

| 옵션             | 기본값         | 설명                                          |
| ---------------- | -------------- | --------------------------------------------- |
| `run_dir`        | *(필수)*       | `best.pt`와 `config.json`이 있는 런 디렉터리  |
| `--case-id`      | 첫 test 케이스 | calibration 및 평가에 사용할 테스트 케이스 ID |
| `--n-calib`      | `10`           | calibration에 사용할 세그먼트 수 또는 비율    |
| `--dataset-dir`  | `data/dataset` | NPZ 데이터셋 루트 디렉터리                    |
| `--device`       | `auto`         | `auto`, `cpu`, `cuda`, `cuda:N`               |
| `--batch-size`   | `512`          | calibration 및 평가 DataLoader 배치 크기      |
| `--lr`           | `1e-4`         | calibration용 학습률                          |
| `--seed`         | `42`           | calibration 세그먼트 샘플링 시드              |
| `--no-normalize` | off            | 세그먼트별 z-score 정규화 비활성화            |

### `--n-calib` 해석 규칙

`--n-calib`는 두 방식으로 해석된다.

- `>= 1`
  - 세그먼트 개수로 해석
  - 예: `--n-calib 20`
- `0 < n < 1`
  - 전체 세그먼트 대비 비율로 해석
  - 예: `--n-calib 0.1` → 전체의 10%

실제 calibration 세그먼트 수는 최소 1개, 최대 전체 세그먼트 수로 clamp된다.

## 4. 입력과 출력

### 입력

필수 입력 파일:

- `<run_dir>/config.json`
- `<run_dir>/best.pt`

데이터셋 입력:

- `<dataset-dir>/test/<case_id>.npz`

각 `npz`는 다음 배열을 포함해야 한다.

```text
x  float32  (N, samples)
y  float32  (N, 2)
```

### 출력

출력 디렉터리:

```text
<run_dir>/<case_id>/
```

저장 파일:

| 파일                | 내용                                                |
| ------------------- | --------------------------------------------------- |
| `calib.pt`          | calibration 후 모델 가중치와 calibration 메타데이터 |
| `eval_results.json` | 정량 지표, calibration 설정, 추론 시간              |
| `eval_plot.png`     | SBP/DBP 예측 vs 실제 산점도                         |
| `error_hist.png`    | SBP/DBP 오차 히스토그램                             |
| `bland_altman.png`  | SBP/DBP Bland-Altman 플롯                           |

## 5. 실행 흐름 개요

```text
parse_args()
    │
    ├─ run_dir/best.pt + config.json 확인
    ├─ config.json에서 model 이름 추출
    ├─ model 생성 후 best.pt 로드
    ├─ dataset_dir/test 에서 case 파일 선택
    ├─ 전체 세그먼트 수 확인
    ├─ n_calib 해석 및 calibration indices 샘플링
    ├─ CaseDataset(calib subset) 생성
    ├─ 마지막 BP 출력 선형층만 1 epoch calibration
    ├─ calib.pt 저장
    ├─ CaseDataset(full case) 생성
    ├─ 전체 세그먼트 추론
    ├─ SBP/DBP 지표 계산
    ├─ eval_results.json 저장
    └─ scatter / hist / Bland-Altman plot 저장
```

## 6. 상세 설계

### 6.1 모델 로드

스크립트는 먼저 `run_dir` 아래의 다음 두 파일이 존재하는지 확인한다.

- `config.json`
- `best.pt`

`config.json`에서 `model` 키를 읽어 모델 이름을 확인한 뒤,
`bpe.models.create_model(model_name)`으로 동일 구조의 모델을 생성한다.
그 다음 `best.pt`의 `model_state_dict`를 로드한다.

즉, calibration은 학습된 모델의 최적 체크포인트 상태에서 시작한다.

### 6.2 테스트 케이스 선택

`dataset_dir/test/*.npz`를 숫자 정렬로 읽는다.

- `--case-id`가 없으면 첫 번째 test 케이스를 사용
- `--case-id`가 있으면 해당 파일명을 직접 찾음

이 단계에서 케이스 전체 세그먼트 수 `n_total`을 확인한다.

### 6.3 calibration 세그먼트 선택

`np.random.default_rng(seed)`를 사용해 calibration용 세그먼트 인덱스를 무작위 비복원 추출한다.

```python
calib_indices = np.sort(rng.choice(n_total, size=n_calib, replace=False))
```

특징:

- 같은 `seed`면 같은 세그먼트 subset이 선택된다.
- `replace=False`이므로 중복 선택이 없다.
- 인덱스를 정렬해 저장하므로 재현과 확인이 쉽다.

### 6.4 `CaseDataset`

`CaseDataset`은 단일 케이스 `NPZ`를 감싸는 최소 Dataset 구현이다.

역할:

- `x`, `y` 배열을 메모리에 로드
- 필요하면 특정 `indices` subset만 선택
- `PPGDataset`과 동일한 per-segment z-score normalization 적용

정규화 방식:

```python
x = (x - x.mean()) / std.clamp_min(1e-6)
```

즉, calibration과 evaluation 모두 일반 학습/평가 파이프라인과 같은 스케일 기준을 유지한다.

### 6.5 마지막 레이어만 미세조정

스크립트는 전체 모델을 다시 학습하지 않는다.
대신 마지막 BP 출력 선형층만 학습 가능하게 두고 나머지는 모두 freeze한다.

#### 출력 레이어 탐색

`get_bp_output_layer(model)`은 다음 우선순위로 레이어를 찾는다.

1. `nn.Linear` 중 `out_features == 2`
2. 그런 레이어가 없으면 마지막 `nn.Linear`

즉, 모델이 SBP/DBP 2차원 출력을 갖는 일반 구조를 우선 가정한다.

#### freeze 정책

```python
for param in model.parameters():
    param.requires_grad_(False)
# 마지막 출력 레이어만 requires_grad=True
```

이 설계의 의도:

- calibration 데이터를 아주 적게 써도 과적합 위험을 줄임
- 케이스 특이적인 bias / scale 보정을 주로 마지막 선형층에 맡김
- calibration 시간을 짧게 유지함

### 6.6 calibration 학습 루프

`calibrate()`는 calibration subset에 대해 정확히 1 epoch만 수행한다.

핵심 설정:

- optimizer: `Adam`
- loss: `MSELoss`
- learning rate: `--lr`
- batch size: `--batch-size`

중요한 점:

- `model.eval()` 상태를 유지한다.
- 즉, Dropout은 비활성화되고 BatchNorm running stats도 바뀌지 않는다.
- 하지만 unfrozen layer에는 gradient가 흐르므로 weight update는 일어난다.

이 설계는 calibration 데이터가 매우 적을 때 BatchNorm 통계가 망가지는 문제를 피하려는 선택이다.

### 6.7 calibration 체크포인트 저장

calibration 후에는 `calib.pt`를 저장한다.

구조 예시:

```python
{
    "model_state_dict": ...,
    "model_name": "resnet1d",
    "base_checkpoint": "data/models/resnet1d/best.pt",
    "case_id": "1234",
    "n_calib": 10,
    "calib_indices": [...],
    "calib_loss": 12.345678,
    "lr": 1e-4,
    "seed": 42,
}
```

즉, 보정된 가중치뿐 아니라
"어떤 케이스를 어떤 세그먼트 subset으로 보정했는가"까지 함께 남긴다.

### 6.8 전체 세그먼트 평가

calibration이 끝나면 동일 케이스의 전체 세그먼트로 다시 추론한다.

`run_inference()`는 다음을 반환한다.

- `preds`
- `targets`
- 전체 추론 시간 `elapsed`

GPU일 때는 `torch.cuda.synchronize()`를 호출해서 실제 wall-clock에 가까운 추론 시간을 측정한다.

## 7. 지표 계산

### 7.1 기본 통계

SBP와 DBP를 각각 독립적으로 계산한다.

| 지표   | 설명             |
| ------ | ---------------- |
| `mae`  | 평균 절대 오차   |
| `me`   | 평균 오차 (bias) |
| `sd`   | 오차 표준편차    |
| `rmse` | 평균 제곱근 오차 |

### 7.2 BHS 등급

누적 오차 비율을 기준으로 `A/B/C/D` 등급을 매긴다.

- `<= 5 mmHg`
- `<= 10 mmHg`
- `<= 15 mmHg`

세 기준을 동시에 만족하는 가장 높은 등급을 채택한다.

### 7.3 AAMI 통과 여부

다음 조건을 동시에 만족하면 `aami_pass = true`다.

- `abs(ME) <= 5.0`
- `SD <= 8.0`

### 7.4 추론 시간

추가로 다음 값을 함께 계산한다.

- `inference_sec`
- `avg_ms_per_sample`

즉, calibration 후 정확도뿐 아니라 해당 케이스 추론 속도도 함께 기록한다.

## 8. 시각화 출력

### 8.1 `eval_plot.png`

SBP와 DBP 각각에 대해:

- x축: 실제값
- y축: 예측값
- `y=x` 기준선 포함

즉, calibration 후 예측이 대각선에 얼마나 가까워졌는지 확인할 수 있다.

### 8.2 `error_hist.png`

SBP와 DBP 각각에 대해:

- 오차 분포 히스토그램
- `0` 기준선
- 평균 오차(`ME`) 수직선

즉, 오차 분포가 좌우 어느 쪽으로 치우치는지와 분산 정도를 볼 수 있다.

### 8.3 `bland_altman.png`

SBP와 DBP 각각에 대해:

- x축: `(actual + predicted) / 2`
- y축: `predicted - actual`
- bias 선
- `±1.96 SD` 한계선

즉, 임상적 agreement 관점에서 calibration 결과를 확인할 수 있다.

## 9. 출력 JSON 구조

`eval_results.json` 예시는 다음과 같다.

```json
{
  "run_dir": "data/models/resnet1d",
  "case_id": "1234",
  "model": "resnet1d",
  "checkpoint": "data/models/resnet1d/best.pt",
  "calib_checkpoint": "data/models/resnet1d/1234/calib.pt",
  "n_total_segments": 2686,
  "n_calib_segments": 10,
  "calib_indices": [12, 104, 233, 401],
  "calib_loss": 8.123456,
  "lr": 0.0001,
  "seed": 42,
  "normalize": true,
  "inference_sec": 0.2314,
  "avg_ms_per_sample": 0.0861,
  "sbp": { ... },
  "dbp": { ... }
}
```

핵심적으로 이 JSON은 다음 세 가지를 함께 담는다.

- 어떤 케이스에 대해 calibration했는지
- 어떤 subset으로 calibration했는지
- calibration 후 전체 케이스 성능이 어땠는지

## 10. 해석 시 주의점

### 10.1 일반화 평가가 아니다

이 스크립트는 같은 케이스의 일부 세그먼트로 calibration하고,
같은 케이스 전체를 평가한다.

따라서 이것은 "완전한 hold-out 일반화 성능"이 아니라
"patient-specific adaptation 성능"에 가깝다.

### 10.2 calibration subset 선택에 따라 결과가 달라질 수 있다

`n_calib`가 작을수록 어떤 세그먼트가 뽑혔는지에 따라 결과 편차가 커질 수 있다.
따라서 공정 비교를 위해서는 `seed`를 고정하거나,
여러 seed로 반복 실험하는 것이 좋다.

### 10.3 마지막 레이어만 보정한다

현재 구현은 마지막 선형층만 미세조정한다.
즉, 표현 학습 전체를 환자별로 다시 맞추는 것이 아니라
출력 매핑만 국소적으로 보정하는 구조다.

이는 안정성과 단순성에는 유리하지만,
더 강한 개인화 성능이 필요한 경우에는 제한이 될 수 있다.

## 11. 추천 사용 흐름

### 단일 케이스 calibration 테스트

```bash
uv run python scripts/eval-calib-model.py data/models/resnet1d --case-id 1234 --n-calib 10
```

### calibration 비율 실험

```bash
uv run python scripts/eval-calib-model.py data/models/resnet1d --case-id 1234 --n-calib 0.05
uv run python scripts/eval-calib-model.py data/models/resnet1d --case-id 1234 --n-calib 0.10
uv run python scripts/eval-calib-model.py data/models/resnet1d --case-id 1234 --n-calib 0.20
```

### CPU 강제 실행

```bash
uv run python scripts/eval-calib-model.py data/models/resnet1d --device cpu
```

## 12. 한계와 향후 확장

현재 구현은 매우 단순한 개인화 baseline이다.

현재 한계:

- 단일 케이스만 지원
- calibration epoch 수가 고정 1 epoch
- 마지막 레이어만 보정
- calibration subset 반복 실험 자동화 없음
- calibration 전/후 성능 비교를 한 번에 저장하지 않음

하지만 다음 목적에는 충분히 유용하다.

- 환자별 소량 calibration 효과 확인
- 모델별 personalization 잠재력 비교
- calibration 세그먼트 수에 따른 민감도 분석
- 임상 지표(BHS/AAMI) 기반 환자 단위 검토
