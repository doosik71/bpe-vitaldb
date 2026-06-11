# DuoModel 상세 설계서 및 사용 설명서

## 1. 개요

DuoModel은 두 개의 독립적으로 학습된 모델을 병렬 실행하여 혈압을 추정하는
**평가 전용 앙상블 모델**이다. 두 모델이 동일한 세그먼트에 대해 각각 SBP/DBP를
예측하고, 예측값의 차이가 허용 임계값(기본 5 mmHg)을 초과하면 해당 측정을
**거부(rejection)**한다. 이 방식은 신뢰도가 낮은 측정을 사전에 걸러냄으로써
허용된 측정의 정확도를 높이는 것을 목표로 한다.

- **구현 파일**: [`bpe/models/duo.py`](../bpe/models/duo.py)
- **기본 모델 조합**: `conv_reg_ds` (모델 A) + `mtae` (모델 B)
- **거부 임계값**: 5.0 mmHg (기본값)
- **허용 기준**: SBP 불일치 < 임계값 **AND** DBP 불일치 < 임계값
- **허용 예측값**: 두 모델 출력의 단순 평균
- **학습 불가**: 모든 파라미터가 고정된 평가 전용 구조

## 2. 설계 동기

단일 모델은 예측 신뢰도를 자체적으로 추정하기 어렵다. 두 모델이 독립적으로
같은 입력을 처리할 때 **서로 동의**한다면 해당 예측이 더 신뢰할 수 있다는
가정에 기초한다.

```
세그먼트 입력 x
       │
       ├──────────────────────────────┐
       ▼                              ▼
  모델 A(x) → (SBP_A, DBP_A)    모델 B(x) → (SBP_B, DBP_B)
       │                              │
       └──────────────┬───────────────┘
                      ▼
        |SBP_A - SBP_B| < threshold
                 AND
        |DBP_A - DBP_B| < threshold
                      │
           ┌──────────┴──────────┐
           │ True (허용)         │ False (거부)
           ▼                     ▼
   (SBP_A+SBP_B)/2           측정 없음
   (DBP_A+DBP_B)/2
```

## 3. 모듈 구조

### 3.1 공개 인터페이스

```
bpe/models/duo.py
├── _load_model(run_dir, device) → (nn.Module, str)   [내부 함수]
└── DuoModel(nn.Module)
    ├── __init__(model_a_id, model_b_id, models_dir, threshold, device)
    ├── train(mode) → DuoModel                         [오버라이드, 항상 eval 유지]
    ├── forward(x) → Tensor (B, 2)                     [거부 없는 평균 예측]
    └── forward_with_mask(x) → (Tensor (B,2), Tensor (B,)) [예측 + 허용 마스크]
```

### 3.2 `_load_model` 함수

```python
def _load_model(run_dir: Path, device: torch.device) -> tuple[nn.Module, str]
```

학습 완료된 모델 디렉터리에서 모델을 불러온다.

**입력 경로 규칙:**

```
<run_dir>/
├── config.json    ← "model" 키에 모델 등록명 포함
└── best.pt        ← 체크포인트 (model_state_dict 포함)
```

**처리 순서:**

1. `config.json`에서 `"model"` 키 읽기 → 등록명 획득
2. `create_model(model_name)`으로 모델 인스턴스 생성
3. `best.pt`에서 `model_state_dict` 로드 (`weights_only=True`)
4. `model.eval()` + `requires_grad_(False)` — 추론 전용 고정

**예외:**

| 조건               | 예외                |
| ------------------ | ------------------- |
| `config.json` 없음 | `FileNotFoundError` |
| `best.pt` 없음     | `FileNotFoundError` |

### 3.3 `DuoModel` 클래스

#### 생성자 파라미터

| 파라미터     | 타입                   | 기본값                | 설명                                        |
| ------------ | ---------------------- | --------------------- | ------------------------------------------- |
| `model_a_id` | `str`                  | `"conv_reg_ds"`       | 모델 A의 `models_dir` 내 하위 디렉터리명    |
| `model_b_id` | `str`                  | `"mtae"`              | 모델 B의 `models_dir` 내 하위 디렉터리명    |
| `models_dir` | `Path \| str`          | `Path("data/models")` | 학습된 모델들이 위치한 루트 디렉터리        |
| `threshold`  | `float`                | `5.0`                 | 거부 임계값 (mmHg). 이 값 **이상**이면 거부 |
| `device`     | `torch.device \| None` | `None` (→ CPU)        | 두 모델을 올릴 디바이스                     |

#### 인스턴스 속성

| 속성              | 설명                            |
| ----------------- | ------------------------------- |
| `self.model_a`    | 로드된 모델 A 인스턴스 (frozen) |
| `self.model_b`    | 로드된 모델 B 인스턴스 (frozen) |
| `self.model_a_id` | 모델 A 등록명 문자열            |
| `self.model_b_id` | 모델 B 등록명 문자열            |
| `self.threshold`  | 거부 임계값 (mmHg)              |

#### `train(mode)` 오버라이드

```python
def train(self, mode: bool = True) -> "DuoModel":
    return super().train(False)   # mode 인수 무시, 항상 eval
```

`model.train()` 또는 `model.train(True)` 호출 시에도 항상 eval 모드를
유지한다. 생성자 마지막 줄에서도 `super().train(False)`를 호출하여 초기화
시점부터 eval 모드를 보장한다.

#### `forward(x)` — 거부 없는 앙상블 예측

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

| 항목      | 내용                                             |
| --------- | ------------------------------------------------ |
| 입력 `x`  | `(B, 1, 1000)` 또는 각 모델이 허용하는 shape     |
| 출력      | `(B, 2)` — `[:, 0]` = SBP, `[:, 1]` = DBP (mmHg) |
| 거부 처리 | 없음 — 전체 배치에 대해 단순 평균 반환           |

거부 없이 모든 세그먼트를 예측하는 경우, 또는 외부에서 허용 여부를
따로 처리할 때 사용한다.

#### `forward_with_mask(x)` — 거부 포함 앙상블 예측

```python
def forward_with_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
```

| 항목            | 내용                                              |
| --------------- | ------------------------------------------------- |
| 입력 `x`        | `(B, 1, 1000)`                                    |
| 출력 `avg_pred` | `(B, 2)` — 전체 배치의 평균 예측 (허용/거부 무관) |
| 출력 `accepted` | `(B,)` bool 텐서 — `True` = 허용, `False` = 거부  |

**거부 판정 로직:**

```python
pred_a   = self.model_a(x)                         # (B, 2)
pred_b   = self.model_b(x)                         # (B, 2)
diff     = torch.abs(pred_a - pred_b)              # (B, 2)
accepted = (diff[:, 0] < self.threshold)           # SBP 불일치 조건
         & (diff[:, 1] < self.threshold)           # DBP 불일치 조건
```

SBP와 DBP 중 **하나라도** 임계값 이상이면 해당 세그먼트는 거부된다.

## 4. 텐서 흐름 다이어그램

```text
입력 x: (B, 1, 1000)
         │
         ├─────────────────────────────────────────────────┐
         │                                                 │
         ▼                                                 ▼
    model_a(x)                                        model_b(x)
    (B, 1, 1000) → ... → (B, 2)                      (B, 1, 1000) → ... → (B, 2)
    pred_a[:, 0] = SBP_A                              pred_b[:, 0] = SBP_B
    pred_a[:, 1] = DBP_A                              pred_b[:, 1] = DBP_B
         │                                                 │
         └───────────────────┬─────────────────────────────┘
                             │
                    diff = |pred_a - pred_b|          (B, 2)
                             │
               ┌─────────────┴─────────────┐
               │ diff[:, 0] < threshold    │ diff[:, 0] >= threshold
               │ AND                       │ OR
               │ diff[:, 1] < threshold    │ diff[:, 1] >= threshold
               ▼                           ▼
           accepted=True               accepted=False
               │
               ▼
      avg_pred = (pred_a + pred_b) / 2    (B, 2)
```

## 5. 평가 결과 (conv_reg_ds + mtae, threshold=5.0)

테스트셋 기준 실측 결과다 (672 케이스, 1,987,556 세그먼트).

### 5.1 측정 거부 통계

| 항목            | 수치              |
| --------------- | ----------------- |
| 전체 세그먼트   | 1,987,556         |
| 허용 세그먼트   | 1,437,141 (72.3%) |
| 거부 세그먼트   | 550,415 (27.7%)   |
| SBP 불일치 평균 | 3.81 mmHg         |
| DBP 불일치 평균 | 1.96 mmHg         |
| SBP 불일치 p95  | 9.88 mmHg         |
| DBP 불일치 p95  | 5.12 mmHg         |

### 5.2 성능 비교

| 지표              | conv_reg_ds 단독 | Duo 전체 | Duo 허용만 (72.3%) |
| ----------------- | ---------------- | -------- | ------------------ |
| SBP MAE (mmHg)    | 13.04            | 12.91    | **12.47**          |
| SBP SD (mmHg)     | 17.09            | 16.89    | **16.23**          |
| SBP ±10 mmHg 이내 | 48.3%            | 48.3%    | **49.5%**          |
| SBP ±15 mmHg 이내 | 66.4%            | 66.8%    | **68.2%**          |
| SBP BHS 등급      | D                | D        | D                  |
| DBP MAE (mmHg)    | 7.86             | 7.80     | **7.62**           |
| DBP SD (mmHg)     | 10.28            | 10.22    | **9.91**           |
| DBP ±15 mmHg 이내 | 87.0%            | 87.2%    | **88.1%**          |
| DBP BHS 등급      | C                | C        | C                  |

### 5.3 결과 해석

- **앙상블 평균 효과만으로는 개선이 미미하다.** 두 모델의 SBP 불일치
  평균(3.81 mmHg)이 임계값(5.0 mmHg)보다 작아, 이미 유사한 예측을 하는
  세그먼트가 많기 때문이다.
- **허용 세그먼트 기준으로는 SBP MAE 약 4.4%, DBP MAE 약 3.1% 개선**된다.
  두 모델이 크게 불일치하는 세그먼트는 두 모델 모두 예측하기 어려운
  세그먼트일 가능성이 높다는 점에서, 거부 자체는 타당하다.
- **BHS 등급 변화 없음.** 임계값 5.0 mmHg에서 SBP는 D, DBP는 C 등급을
  유지한다. 임계값을 낮추면 허용률이 감소하지만 등급이 올라갈 가능성이 있다.

## 6. 출력 파일 구조

`bin/eval-model-duo` 실행 후 출력 디렉터리에 저장되는 파일들이다.

```
<output_dir>/
├── eval_results.json       ← 전체 통계 + 허용/거부별 지표
├── eval_plot_all.png       ← 예측 vs 실측 산점도 (전체 세그먼트)
├── eval_plot.png           ← 예측 vs 실측 산점도 (허용 세그먼트만)
├── error_hist_all.png      ← 오차 히스토그램 (전체 세그먼트)
├── error_hist.png          ← 오차 히스토그램 (허용 세그먼트만)
└── diff_dist.png           ← 두 모델 간 불일치 분포 (허용/거부 색상 구분)
```

### `eval_results.json` 필드

```json
{
  "model":               "duo",
  "model_a":             "conv_reg_ds",
  "model_b":             "mtae",
  "threshold_mmhg":      5.0,
  "test_dir":            "data/dataset/test",
  "n_cases":             672,
  "n_segments_total":    1987556,
  "n_segments_accepted": 1437141,
  "n_segments_rejected": 550415,
  "acceptance_rate_pct": 72.3069,
  "sbp_diff_mean":       3.8089,
  "dbp_diff_mean":       1.9569,
  "sbp_diff_p95":        9.8831,
  "dbp_diff_p95":        5.1236,
  "inference_sec_a":     5.8403,
  "inference_sec_b":     5.2306,
  "avg_ms_per_sample":   0.0056,
  "sbp_all":             { "mae": ..., "me": ..., "sd": ..., "bhs_grade": "D", ... },
  "dbp_all":             { "mae": ..., "me": ..., "sd": ..., "bhs_grade": "C", ... },
  "sbp_accepted":        { "mae": ..., "me": ..., "sd": ..., "bhs_grade": "D", ... },
  "dbp_accepted":        { "mae": ..., "me": ..., "sd": ..., "bhs_grade": "C", ... }
}
```

## 7. 사용 방법

### 7.1 커맨드라인 평가

```bash
# 기본 설정 (conv_reg_ds + mtae, threshold=5.0)
bin/eval-model-duo data/models/duo

# 임계값 변경
bin/eval-model-duo data/models/duo_3mmhg --duo-threshold 3.0

# 모델 조합 변경
bin/eval-model-duo data/models/duo_resnet --duo-models resnet1d conv_reg_ds

# 모델 디렉터리 명시
bin/eval-model-duo data/models/duo \
    --duo-models conv_reg_ds mtae \
    --models-dir data/models \
    --dataset-dir data/dataset \
    --device cuda
```

Windows:

```bat
bin\eval-model-duo.bat data\models\duo
bin\eval-model-duo.bat data\models\duo_3mmhg --duo-threshold 3.0
```

### 7.2 Python 코드에서 직접 사용

#### 기본 추론 (거부 없음)

```python
import torch
from bpe.models.duo import DuoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

duo = DuoModel(
    model_a_id="conv_reg_ds",
    model_b_id="mtae",
    models_dir="data/models",
    threshold=5.0,
    device=device,
)

x = torch.randn(32, 1, 1000).to(device)   # (B, 1, 1000)
preds = duo(x)                              # (B, 2) — SBP, DBP
```

#### 거부 마스크 포함 추론

```python
avg_pred, accepted = duo.forward_with_mask(x)

# accepted: (B,) bool 텐서
reliable_preds = avg_pred[accepted]         # 허용 세그먼트 예측값만
n_accepted = accepted.sum().item()
acceptance_rate = n_accepted / len(accepted)

print(f"허용률: {acceptance_rate:.1%}  ({n_accepted}/{len(accepted)})")
print(f"허용 SBP 예측 평균: {reliable_preds[:, 0].mean():.1f} mmHg")
print(f"허용 DBP 예측 평균: {reliable_preds[:, 1].mean():.1f} mmHg")
```

#### DataLoader와 함께 배치 추론

```python
import numpy as np
from torch.utils.data import DataLoader
from bpe.train.dataset import PPGDataset

test_ds = PPGDataset("data/dataset/test", normalize=True, preload=False)
loader  = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)

all_preds, all_masks, all_targets = [], [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        pred, mask = duo.forward_with_mask(x)
        all_preds.append(pred.cpu().numpy())
        all_masks.append(mask.cpu().numpy())
        all_targets.append(y.numpy())

preds   = np.concatenate(all_preds)    # (N, 2)
masks   = np.concatenate(all_masks)    # (N,)  bool
targets = np.concatenate(all_targets)  # (N, 2)

# 허용된 세그먼트만 평가
accepted_preds   = preds[masks]
accepted_targets = targets[masks]
sbp_mae = np.mean(np.abs(accepted_preds[:, 0] - accepted_targets[:, 0]))
dbp_mae = np.mean(np.abs(accepted_preds[:, 1] - accepted_targets[:, 1]))
print(f"SBP MAE: {sbp_mae:.2f}  DBP MAE: {dbp_mae:.2f}")
```

#### 임계값 탐색 (후처리)

```python
import numpy as np

# 두 모델 예측을 미리 수집하여 임계값별 허용률/정확도 탐색
preds_a = ...   # (N, 2) — 모델 A 예측
preds_b = ...   # (N, 2) — 모델 B 예측
targets = ...   # (N, 2) — 실제 레이블

diff = np.abs(preds_a - preds_b)

for t in [3.0, 4.0, 5.0, 7.0, 10.0]:
    mask = (diff[:, 0] < t) & (diff[:, 1] < t)
    acc  = preds_a[mask] * 0.5 + preds_b[mask] * 0.5
    sbp_mae = np.mean(np.abs(acc[:, 0] - targets[mask, 0]))
    dbp_mae = np.mean(np.abs(acc[:, 1] - targets[mask, 1]))
    print(f"threshold={t:.1f}  accept={mask.mean():.1%}  "
          f"SBP_MAE={sbp_mae:.2f}  DBP_MAE={dbp_mae:.2f}")
```

## 8. 설계 결정 사항

### 8.1 `nn.Module` 서브클래스로 구현

DuoModel을 `nn.Module`로 구현하면 `DataParallel`, `to(device)`,
`state_dict()` 등 PyTorch 생태계와 자연스럽게 통합된다. 단순 함수로 구현할
수도 있으나 `device` 관리와 `model_a`, `model_b`의 생명주기를 일관되게
관리하기 위해 클래스로 설계했다.

### 8.2 학습 불가 보장 — 세 겹의 방어

| 계층                 | 수단                                   | 효과                              |
| -------------------- | -------------------------------------- | --------------------------------- |
| 파라미터 동결        | `requires_grad_(False)`                | 역전파 시 그래디언트 계산 차단    |
| Eval 모드 초기화     | `super().train(False)` (생성자 마지막) | BatchNorm/Dropout 추론 모드 고정  |
| `train()` 오버라이드 | 항상 `super().train(False)` 반환       | 외부에서 `.train()` 호출해도 무효 |

### 8.3 AND 조건의 거부 판정

SBP **또는** DBP 중 하나라도 불일치하면 거부하는 AND 조건을 사용한다. OR
조건(둘 다 불일치해야 거부)에 비해 더 엄격하여 허용률이 낮아지지만, 임상적으로
SBP와 DBP 중 하나만 정확해서는 의미가 없으므로 AND 조건이 타당하다.

### 8.4 `forward` vs `forward_with_mask` 분리

| 메서드                 | 용도                                               |
| ---------------------- | -------------------------------------------------- |
| `forward(x)`           | `model(x)` 형식의 표준 호출, 파이프라인 통합 시    |
| `forward_with_mask(x)` | 거부 통계 계산, 선택적 측정이 필요한 실시간 시스템 |

두 모델을 두 번 실행하는 오버헤드를 피하기 위해 `forward_with_mask` 내부에서
한 번에 계산한다. `forward`는 `forward_with_mask`를 호출하지 않고 독립적으로
구현되어 mask 계산 비용을 생략한다.

### 8.5 예측값 집계 방식 — 단순 평균

두 모델의 출력을 `(pred_a + pred_b) / 2`로 단순 평균한다. 가중 평균이나
사후 확률 기반 집계도 가능하나, 두 모델의 성능이 유사(SBP MAE 13.04 vs
13.09)하여 동등 가중치가 적절하다.

## 9. 디렉터리 구조 요구 사항

DuoModel 초기화에는 두 모델의 학습 결과 디렉터리가 필요하다.

```
data/models/                     ← models_dir (기본값)
├── conv_reg_ds/                 ← model_a_id (기본값)
│   ├── config.json              ← {"model": "conv_reg_ds", ...}
│   └── best.pt                  ← {"model_state_dict": {...}, "epoch": 5, ...}
└── mtae/                        ← model_b_id (기본값)
    ├── config.json              ← {"model": "mtae", ...}
    └── best.pt
```

`config.json`의 `"model"` 키에는 `bpe.models.registry`에 등록된 모델 이름이
있어야 한다. 학습 스크립트(`scripts/train-model.py`)가 이 형식으로 저장한다.

## 10. 관련 파일

| 파일                                                              | 역할                           |
| ----------------------------------------------------------------- | ------------------------------ |
| [`bpe/models/duo.py`](../bpe/models/duo.py)                       | DuoModel 구현                  |
| [`scripts/eval-model.py`](../scripts/eval-model.py)               | `--duo` 플래그로 duo 평가 실행 |
| [`bin/eval-model-duo`](../bin/eval-model-duo)                     | Linux/macOS 런처               |
| [`bin/eval-model-duo.bat`](../bin/eval-model-duo.bat)             | Windows 런처                   |
| [`bpe/models/conv_reg_ds.py`](../bpe/models/conv_reg_ds.py)       | 기본 모델 A                    |
| [`bpe/models/mtae.py`](../bpe/models/mtae.py)                     | 기본 모델 B                    |
| [`bpe/models/registry.py`](../bpe/models/registry.py)             | `create_model` 팩토리          |
| [`docs/model-design-conv_reg_ds.md`](model-design-conv_reg_ds.md) | 모델 A 설계서                  |
| [`docs/model-design-mtae.md`](model-design-mtae.md)               | 모델 B 설계서                  |
