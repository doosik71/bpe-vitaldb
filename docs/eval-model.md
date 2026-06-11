# `eval-model.py` 사용 및 상세 설계

작성일: 2026-06-11  
관련 코드: [scripts/eval-model.py](../scripts/eval-model.py)  
관련 문서: [README.md](../README.md), [docs/train-model.md](train-model.md)

## 1. 목적

`scripts/eval-model.py`는 학습 완료된 혈압 추정(BPE) 모델을 보류(held-out) 테스트 세트로
평가하는 스크립트다.

이 스크립트의 역할은 다음과 같다.

- 런 디렉터리의 `best.pt` 체크포인트와 `config.json`을 로드한다.
- 테스트 세트(`data/dataset/test/`) 전체에 대해 추론(inference)을 실행한다.
- 임상 표준 지표(BHS, AAMI)를 포함한 정량 지표를 계산한다.
- 결과를 JSON, 산점도, 오차 히스토그램으로 저장한다.
- **Duo 모드**: 두 모델 앙상블 + 예측 불일치(disagreement) 기반 거부(rejection) 평가를 지원한다.

## 2. 사용 방법

### 기본 실행 (단일 모델)

```bash
uv run python scripts/eval-model.py data/models/resnet1d
```

또는 제공 런처 스크립트를 사용한다.

```bash
bin/eval-model     data/models/resnet1d   # Linux / macOS
bin\eval-model.bat data\models\resnet1d   # Windows
```

### Duo 모드 실행

```bash
uv run python scripts/eval-model.py data/models/duo_conv_reg_ds_mtae --duo

# 모델 이름 명시
uv run python scripts/eval-model.py data/models/duo_out --duo \
    --duo-models conv_reg_ds mtae --duo-threshold 5.0

# 거부 임계값 조정
uv run python scripts/eval-model.py data/models/duo_out --duo --duo-threshold 8.0
```

### 주요 사용 예시

```bash
# GPU 지정
uv run python scripts/eval-model.py data/models/resnet1d --device cuda:0

# 정규화 비활성화 (학습 시 --no-normalize를 사용했을 때)
uv run python scripts/eval-model.py data/models/resnet1d --no-normalize

# 데이터셋 경로 변경
uv run python scripts/eval-model.py data/models/resnet1d --dataset-dir /data/my_dataset

# 배치 크기 축소 (GPU 메모리 부족 시)
uv run python scripts/eval-model.py data/models/resnet1d --batch-size 128
```

## 3. CLI 옵션

### 공통 옵션

| 옵션             | 기본값         | 설명                                                                           |
| ---------------- | -------------- | ------------------------------------------------------------------------------ |
| `run_dir`        | *(필수)*       | 런 디렉터리 경로 (단일: `best.pt`+`config.json` 위치; duo: 결과 출력 디렉터리) |
| `--dataset-dir`  | `data/dataset` | NPZ 데이터셋 루트 디렉터리                                                     |
| `--device`       | `auto`         | `auto` \| `cpu` \| `cuda` \| `cuda:N`                                          |
| `--batch-size`   | `512`          | 추론 배치 크기                                                                 |
| `--no-normalize` | off            | PPG z-score 정규화 비활성화                                                    |

### Duo 모드 전용 옵션

| 옵션               | 기본값             | 설명                                                                          |
| ------------------ | ------------------ | ----------------------------------------------------------------------------- |
| `--duo`            | off                | Duo 평가 모드 활성화                                                          |
| `--duo-models A B` | `conv_reg_ds mtae` | 두 모델 ID (공백으로 구분)                                                    |
| `--duo-threshold`  | `5.0`              | 거부 임계값 (mmHg); 두 모델의 예측 차이가 이 값 이상이면 해당 세그먼트를 거부 |
| `--models-dir`     | `data/models`      | 체크포인트를 검색할 루트 모델 디렉터리                                        |

## 4. 출력 파일

### 단일 모드

런 디렉터리(`run_dir`) 아래에 다음 파일이 생성된다.

| 파일                | 내용                                                                                                       |
| ------------------- | ---------------------------------------------------------------------------------------------------------- |
| `eval_results.json` | 정량 지표 전체 (MAE, RMSE, ME, SD; BHS 누적 오차 등급; AAMI 합격 여부; 케이스별 최선/최악 결과; 추론 시간) |
| `eval_plot.png`     | SBP·DBP 예측값 vs 실제값 산점도 (1:1 대각선 포함)                  |
| `error_hist.png`    | SBP·DBP 오차(예측 − 실제) 분포 히스토그램 (평균 오차 수직선 포함)  |
| `bland_altman.png`  | SBP·DBP Bland-Altman 플롯 (bias, ±1.96 SD 한계선 포함)             |

### Duo 모드

출력 디렉터리(`run_dir`, 없으면 자동 생성) 아래에 다음 파일이 생성된다.

| 파일                          | 내용                                                                            |
| ----------------------------- | ------------------------------------------------------------------------------- |
| `eval_results.json`           | 두 모델 정보, 임계값, 거부율; 전체 세그먼트 지표; 수락 세그먼트 지표; 추론 시간 |
| `eval_plot_all.png`           | 전체 세그먼트 앙상블 예측 산점도                                                |
| `error_hist_all.png`          | 전체 세그먼트 오차 히스토그램                                                   |
| `eval_plot.png`               | 수락된 세그먼트만의 산점도 (수락 세그먼트가 있을 때만 생성)                     |
| `error_hist.png`              | 수락된 세그먼트만의 오차 히스토그램 (수락 세그먼트가 있을 때만 생성)            |
| `diff_dist.png`               | 두 모델 간 예측 불일치 분포 (수락/거부 분리, 임계선 포함)                       |
| `bland_altman_all.png`        | 전체 세그먼트 Bland-Altman 플롯                                                  |
| `bland_altman_accepted.png`   | 수락 세그먼트 Bland-Altman 플롯 (수락 세그먼트가 있을 때만 생성)                |

### `eval_results.json` 구조 (단일 모드)

```json
{
  "run_dir":            "data/models/resnet1d",
  "model":              "resnet1d",
  "checkpoint":         "data/models/resnet1d/best.pt",
  "best_epoch":         87,
  "test_dir":           "data/dataset/test",
  "n_segments":         95420,
  "n_cases":            601,
  "inference_sec":      12.3456,
  "avg_ms_per_sample":  0.1294,
  "sbp": {
    "mae": 8.14,  "me": -0.31,  "sd": 10.52,  "rmse": 10.53,
    "bhs_pct_5": 42.1,  "bhs_pct_10": 72.4,  "bhs_pct_15": 88.6,
    "bhs_grade": "C",   "aami_pass": false,   "n_samples": 95420
  },
  "dbp": { ... },
  "best_case_id":      1234,
  "best_case_avg_mae": 2.18,
  "worst_case_id":     5678,
  "worst_case_avg_mae": 24.31
}
```

### `eval_results.json` 구조 (Duo 모드)

```json
{
  "model":                "duo",
  "model_a":              "conv_reg_ds",
  "model_b":              "mtae",
  "threshold_mmhg":       5.0,
  "n_segments_total":     95420,
  "n_segments_accepted":  71560,
  "n_segments_rejected":  23860,
  "acceptance_rate_pct":  75.0,
  "sbp_diff_mean":        3.21,
  "sbp_diff_p95":         11.42,
  "dbp_diff_mean":        2.18,
  "dbp_diff_p95":         8.76,
  "avg_ms_per_sample":    0.2105,
  "sbp_all":     { ... },
  "dbp_all":     { ... },
  "sbp_accepted": { ... },
  "dbp_accepted": { ... }
}
```

## 5. 상세 설계

### 5.1 실행 흐름 개요

```
parse_args()
    │
    ├─ --duo 없음 → _main_single()
    │       │
    │       ├─ config.json + best.pt 로드
    │       ├─ PPGDataset(test/) 구성
    │       ├─ run_inference() → (preds, targets, elapsed_sec)
    │       ├─ compute_metrics() × 2 (SBP, DBP)
    │       ├─ compute_per_case_stats()
    │       └─ JSON 저장 + plot_scatter() + plot_error_hist() + plot_bland_altman()
    │              └─ run_dir/bland_altman.png
    │
    └─ --duo 있음 → _main_duo()
            │
            ├─ _load_model() × 2 (model_a, model_b)
            ├─ PPGDataset(test/) 구성
            ├─ run_duo_inference() → (preds_a, preds_b, targets, elapsed_a, elapsed_b)
            ├─ 앙상블 평균 계산 + 불일치 마스크 생성
            ├─ compute_metrics() × 4 (SBP/DBP × 전체/수락)
            └─ JSON 저장 + 산점도 + 오차 히스토그램 + diff_dist.png + plot_bland_altman() × 2
                   └─ out_dir/bland_altman_all.png
                   └─ out_dir/bland_altman_accepted.png  (수락 세그먼트 있을 때)
```

### 5.2 디바이스 선택 (`resolve_device`)

`--device auto`이면 CUDA가 가능한 경우 자동으로 GPU를 선택하고, 그렇지 않으면 CPU를 사용한다.

```python
def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)
```

### 5.3 임상 지표 계산 (`compute_metrics`)

SBP와 DBP 각 채널별로 독립적으로 호출된다.
입력은 예측값·실제값 NumPy 배열(shape: `(N,)`)이다.

#### 기본 통계 지표

| 지표 | 수식                         | 설명                  |
| ---- | ---------------------------- | --------------------- |
| MAE  | `mean(                       | pred − true           | )` | 평균 절대 오차 |
| ME   | `mean(pred − true)`          | 평균 오차(편향, bias) |
| SD   | `std(pred − true, ddof=1)`   | 오차의 표준편차       |
| RMSE | `sqrt(mean((pred − true)²))` | 평균 제곱근 오차      |

#### BHS (British Hypertension Society) 등급

누적 오차 분포를 기준으로 A~D 등급을 부여한다.

| 등급 | ≤5 mmHg | ≤10 mmHg | ≤15 mmHg |
| ---- | ------- | -------- | -------- |
| A    | ≥ 60 %  | ≥ 85 %   | ≥ 95 %   |
| B    | ≥ 50 %  | ≥ 75 %   | ≥ 90 %   |
| C    | ≥ 40 %  | ≥ 65 %   | ≥ 85 %   |
| D    | C 미만  |          |          |

```python
bhs_5  = sum(|err| <=  5) / N * 100
bhs_10 = sum(|err| <= 10) / N * 100
bhs_15 = sum(|err| <= 15) / N * 100
```

#### AAMI (Association for the Advancement of Medical Instrumentation) 기준

```python
aami_pass = (abs(ME) <= 5.0) and (SD <= 8.0)
```

두 조건을 동시에 만족해야 합격(`True`)이다.

#### 반환 딕셔너리

```python
{
    "mae":        float,   # 평균 절대 오차 (mmHg)
    "me":         float,   # 평균 오차 / 편향 (mmHg)
    "sd":         float,   # 오차 표준편차 (mmHg)
    "rmse":       float,   # 평균 제곱근 오차 (mmHg)
    "bhs_pct_5":  float,   # |오차| ≤ 5 mmHg 비율 (%)
    "bhs_pct_10": float,   # |오차| ≤ 10 mmHg 비율 (%)
    "bhs_pct_15": float,   # |오차| ≤ 15 mmHg 비율 (%)
    "bhs_grade":  str,     # "A" | "B" | "C" | "D"
    "aami_pass":  bool,    # AAMI 합격 여부
    "n_samples":  int,     # 평가에 사용된 세그먼트 수
}
```

### 5.4 단일 모델 추론 (`run_inference`)

```python
def run_inference(model, loader, device) -> (preds, targets, elapsed_sec):
```

- `model.eval()` + `torch.no_grad()` 컨텍스트에서 실행된다.
- **순수 추론 시간**만 측정한다 (데이터 이동 시간 제외).
  - GPU 사용 시 `torch.cuda.synchronize()`를 호출해 CUDA 커널이 완료될 때까지 대기한 후 시간을 측정한다.
- 배치별 결과를 리스트에 누적한 뒤 `np.concatenate`로 합친다.
- 반환값: `preds` `(N, 2)`, `targets` `(N, 2)`, `elapsed_sec` (float)

### 5.5 케이스별 통계 (`compute_per_case_stats`)

전체 세그먼트 중 가장 좋은 케이스와 가장 나쁜 케이스를 SBP·DBP 평균 MAE로 찾아 반환한다.

```python
def compute_per_case_stats(preds, targets, segs, files) -> dict:
```

- `segs`: `PPGDataset._segs` — 각 세그먼트가 어느 파일의 몇 번째 인덱스인지 `(file_idx, seg_idx)` 목록.
- `files`: `PPGDataset._files` — NPZ 파일 경로 목록. 파일 스템(stem)이 숫자이면 정수 케이스 ID로 변환한다.
- 케이스 ID가 정수로 변환 가능하면 `int`, 그렇지 않으면 문자열로 반환한다.

반환 딕셔너리:

```python
{
    "best_case_id":       int | str,
    "best_case_avg_mae":  float,   # (SBP MAE + DBP MAE) / 2 (mmHg)
    "worst_case_id":      int | str,
    "worst_case_avg_mae": float,
}
```

### 5.6 산점도 저장 (`plot_scatter`)

```python
def plot_scatter(pred_sbp, true_sbp, pred_dbp, true_dbp, out_path):
```

- 1행 2열 서브플롯 (12×5 inches, 150 dpi).
- 각 서브플롯: 실제값(x축) vs 예측값(y축) 산점도.
  - 반투명 소점(`alpha=0.15`, `s=4`)으로 밀집 구간 가시화.
  - `rasterized=True`로 파일 크기를 줄인다.
  - `y = x` 점선(이상적 예측선) 표시.
  - 양 축 범위를 동일하게 설정(`set_aspect("equal")`).

### 5.7 오차 히스토그램 저장 (`plot_error_hist`)

```python
def plot_error_hist(err_sbp, err_dbp, out_path):
```

- 1행 2열 서브플롯 (12×5 inches, 150 dpi).
- 각 서브플롯: 오차(예측값 − 실제값) 히스토그램 (80 bins).
  - 수직선 두 개: 오차 0 (검정 점선), 평균 오차 ME (빨간 실선, 범례에 값 표시).

### 5.8 Bland-Altman 플롯 저장 (`plot_bland_altman`)

```python
def plot_bland_altman(pred_sbp, true_sbp, pred_dbp, true_dbp, out_path):
```

- 1행 2열 서브플롯 (12×5 inches, 150 dpi).
- 저장 위치: `eval_results.json`과 동일한 런 디렉터리 (`bland_altman.png` / `bland_altman_all.png` / `bland_altman_accepted.png`).
- 각 서브플롯: x축 = `(예측값 + 실제값) / 2` (평균 측정값), y축 = `예측값 − 실제값` (차이).
  - 수평선 세 개: 편향(bias, ME, 빨간 실선), 상한(`ME + 1.96 × SD`, 주황 점선), 하한(`ME − 1.96 × SD`, 주황 점선).
  - 0 기준선(검정 점선)으로 비편향 기준 표시.
  - 반투명 소점(`alpha=0.15`, `s=4`, `rasterized=True`)으로 밀집 구간 가시화.

Bland-Altman 플롯은 두 측정 방법 간 **일치도(agreement)**를 시각화하는 임상 표준 방법이다. 산점도(`eval_plot.png`)가 선형 상관을 보여주는 반면, Bland-Altman 플롯은 **측정값 범위에 따른 편향 변화**를 확인하는 데 적합하다.

### 5.9 Duo 모드 추론 (`run_duo_inference`)

두 모델을 동일한 DataLoader로 순차 실행하며 각각의 추론 시간을 별도로 측정한다.

```python
def run_duo_inference(model_a, model_b, loader, device)
    -> (preds_a, preds_b, targets, elapsed_a, elapsed_b):
```

- 한 배치를 읽은 뒤 model_a를 실행하고 시간을 측정하고, 이어서 model_b를 실행하고 시간을 측정한다.
- 반환값: 각각 `(N, 2)` float32 배열 + 두 모델의 순수 추론 경과 시간(초).

### 5.10 Duo 모드 거부 로직

```python
avg_preds = (preds_a + preds_b) / 2            # 앙상블 평균 예측
diff      = abs(preds_a - preds_b)             # (N, 2): SBP 불일치, DBP 불일치
accepted  = (diff[:, 0] < threshold) & (diff[:, 1] < threshold)  # (N,) bool
```

- **SBP 불일치와 DBP 불일치 모두** 임계값 미만이어야 수락된다.
- 수락된 세그먼트의 예측값 = 두 모델의 평균.
- 지표는 전체 세그먼트와 수락 세그먼트에 대해 각각 계산하고 비교한다.

### 5.11 Duo 불일치 분포 시각화 (`_plot_duo_diff`)

```python
def _plot_duo_diff(diff, accepted, threshold, out_path):
```

- 1행 2열 서브플롯 (12×5 inches, 150 dpi).
- 각 서브플롯: SBP 또는 DBP 모델 간 절대 불일치 `|A − B|` 히스토그램.
  - 수락(녹색)과 거부(빨간색) 분포를 겹쳐 표시.
  - 임계값 수직선(검정 점선) 표시.

### 5.12 터미널 출력 형식

단일 모드와 Duo 모드 모두 동일한 형식으로 지표 테이블을 출력한다.

```
Metric              SBP         DBP
------------------------------------------
  n_samples              95420       95420
  mae                     8.14        5.22
  me                     -0.31        0.12
  sd                     10.52        6.87
  rmse                   10.53        6.88
  bhs_pct_5              42.1        58.3
  bhs_pct_10             72.4        82.1
  bhs_pct_15             88.6        93.7
  bhs_grade                 C           B
  aami_pass             False        True
  avg_ms/sample         0.129
```

Duo 모드는 전체 세그먼트와 수락 세그먼트 두 테이블을 각각 출력하고, 수락 전후 MAE/SD/RMSE 개선량을 요약한다.

## 6. 전제 조건

평가를 실행하기 전에 다음 두 가지가 완료되어 있어야 한다.

1. **데이터셋 구축**: `data/dataset/test/` 아래에 NPZ 파일이 존재해야 한다.

   ```bash
   bin/construct-dataset   # Linux / macOS
   bin\construct-dataset.bat  # Windows
   ```

2. **모델 학습**: 평가할 런 디렉터리에 `best.pt`와 `config.json`이 존재해야 한다.

   ```bash
   bin/train-model     --model resnet1d   # Linux / macOS
   bin\train-model.bat --model resnet1d   # Windows
   ```

학습 시 `--no-normalize`를 사용했다면 평가 시에도 반드시 동일하게 지정해야 한다.

## 7. 관련 모듈

| 모듈                      | 역할                                                                  |
| ------------------------- | --------------------------------------------------------------------- |
| `bpe/models/__init__.py`  | `create_model(name)` — 모델 레지스트리에서 아키텍처 인스턴스 생성     |
| `bpe/models/duo.py`       | `_load_model(run_dir, device)` — Duo 모드용 체크포인트 로더           |
| `bpe/train/dataset.py`    | `PPGDataset` — NPZ 로더, z-score 정규화, `_segs` / `_files` 속성 제공 |
| `scripts/train-model.py`  | `best.pt`와 `config.json` 생성                                        |
| `scripts/train-status.py` | 학습 곡선(`loss_graph.png`, `mae_graph.png`) 시각화                   |
