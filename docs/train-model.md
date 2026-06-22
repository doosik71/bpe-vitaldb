# `train-model.py` 사용 및 상세 설계

작성일: 2026-06-09  
관련 코드: [scripts/train-model.py](../scripts/train-model.py)  
관련 문서: [README.md](../README.md), [docs/data-augmentation.md](data-augmentation.md)

## 1. 목적

`scripts/train-model.py`는 VitalDB NPZ 데이터셋으로 혈압 추정(BPE) 딥러닝 모델을 학습하는
메인 훈련 파이프라인이다.

이 스크립트의 역할은 다음과 같다.

- 모델 레지스트리에서 아키텍처를 생성한다.
- train / val 데이터셋을 로드하고 DataLoader를 구성한다.
- 데이터 증강 파이프라인을 조립한다.
- AdamW + Cosine Annealing 스케줄러로 학습 루프를 실행한다.
- 검증 손실 기준 최적 체크포인트(`best.pt`)와 마지막 체크포인트(`last.pt`)를 저장한다.
- 에폭별 손실/MAE를 `metrics.csv`에 기록한다.
- Early stopping으로 과적합을 방지한다.

## 2. 사용 방법

### 기본 실행

```bash
uv run python scripts/train-model.py --model resnet1d
```

또는 제공 런처 스크립트를 사용한다.

```bash
bin/train-model     --model resnet1d              # Linux / macOS
bin\train-model.bat --model resnet1d              # Windows
```

### 주요 사용 예시

```bash
# 기본 하이퍼파라미터로 학습
uv run python scripts/train-model.py --model resnet1d

# 더 긴 학습, 큰 배치
uv run python scripts/train-model.py --model st_resnet --epochs 150 --batch-size 512

# 학습률, patience 조정
uv run python scripts/train-model.py --model minception --lr 5e-4 --patience 20

# 이전 체크포인트에서 재개
uv run python scripts/train-model.py --model resnet1d --resume data/models/resnet1d/last.pt

# 증강 일부 비활성화
uv run python scripts/train-model.py --model resnet1d --no-aug-noise --no-aug-mask

# GPU 지정
uv run python scripts/train-model.py --model xresnet1d --device cuda:0
```

## 3. CLI 옵션

### 필수

| 옵션      | 설명                                       |
| --------- | ------------------------------------------ |
| `--model` | 모델 이름 (레지스트리에 등록된 이름, 필수) |

### 경로

| 옵션            | 기본값         | 설명                          |
| --------------- | -------------- | ----------------------------- |
| `--dataset-dir` | `data/dataset` | NPZ 데이터셋 루트 디렉터리    |
| `--models-dir`  | `data/models`  | 체크포인트 저장 루트 디렉터리 |
| `--resume`      | —              | 재개할 체크포인트 `.pt` 경로  |

### 학습 하이퍼파라미터

| 옵션             | 기본값 | 설명                           |
| ---------------- | ------ | ------------------------------ |
| `--epochs`       | `100`  | 최대 학습 에폭 수              |
| `--batch-size`   | `256`  | 미니배치 크기                  |
| `--lr`           | `1e-3` | 초기 학습률                    |
| `--weight-decay` | `1e-4` | AdamW weight decay             |
| `--patience`     | `5`    | Early stopping patience (에폭) |
| `--seed`         | `42`   | 난수 시드                      |

### 시스템

| 옵션             | 기본값 | 설명                                  |
| ---------------- | ------ | ------------------------------------- |
| `--device`       | `auto` | `auto` \| `cpu` \| `cuda` \| `cuda:N` |
| `--workers`      | `4`    | DataLoader worker 프로세스 수         |
| `--preload`      | off    | 학습 전 전체 세그먼트를 RAM에 로드    |
| `--no-normalize` | off    | PPG z-score 정규화 비활성화           |

### 데이터 증강 (기본값: 전체 활성화)

| 옵션             | 설명                                                  |
| ---------------- | ----------------------------------------------------- |
| `--no-aug-noise` | Gaussian noise 비활성화 (std=0.01)                    |
| `--no-aug-scale` | 진폭 스케일링 비활성화 (×0.8~1.2)                     |
| `--no-aug-shift` | 원형 시간축 이동 비활성화 (±50 샘플)                  |
| `--no-aug-mask`  | 랜덤 span 마스킹 비활성화 (5~10% 연속 구간, 최대 1초) |

### 환자 균형 샘플링 (기본값: 활성화)

| 옵션                   | 설명                                                          |
| ---------------------- | ------------------------------------------------------------- |
| `--no-patient-balance` | WeightedRandomSampler 비활성화. 기본적으로 케이스당 균등 기여 |

## 4. 출력 파일

학습 후 `data/models/<model>/` 아래에 다음 파일이 생성된다.

| 파일          | 내용                                                                 |
| ------------- | -------------------------------------------------------------------- |
| `config.json` | 이 실행에 사용된 전체 CLI 인자 (재현을 위해 저장)                    |
| `best.pt`     | 검증 손실이 가장 낮았던 에폭의 체크포인트                            |
| `last.pt`     | 마지막 에폭의 체크포인트 (재개 용도)                                 |
| `metrics.csv` | 에폭별 train/val 손실과 SBP/DBP MAE 기록                             |
| `runs.jsonl`  | 실행 요약 (모델명, 하이퍼파라미터, 최적 지표); 실행마다 한 줄씩 추가 |

### 체크포인트 구조 (`.pt`)

```python
{
    "epoch":                int,     # 해당 에폭 번호
    "model_state_dict":     dict,    # 모델 가중치
    "optimizer_state_dict": dict,    # 옵티마이저 상태 (last.pt에만)
    "val_loss":             float,   # 해당 에폭의 검증 손실
    "val_sbp_mae":          float,   # SBP MAE (mmHg)
    "val_dbp_mae":          float,   # DBP MAE (mmHg)
}
```

### metrics.csv 컬럼

```text
epoch, train_loss, val_loss, train_sbp_mae, train_dbp_mae, val_sbp_mae, val_dbp_mae, lr
```

## 5. 상세 설계

### 5.1 디바이스 선택 (`resolve_device`)

`--device auto`이면 CUDA가 가능한 경우 자동으로 GPU를 선택하고, 그렇지 않으면 CPU를 사용한다.

```python
def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)
```

### 5.2 재현성 (`set_seed`)

Python `random`, NumPy, PyTorch(CPU+CUDA) 시드를 모두 설정한다.
동일한 `--seed`와 동일한 데이터셋·하이퍼파라미터라면 같은 결과가 재현된다.

```python
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

### 5.3 데이터 증강 파이프라인

네 가지 증강이 기본적으로 모두 활성화된다. `--no-aug-*` 플래그로 각각 비활성화할 수 있다.

| 증강              | 클래스                                      | 동작                                                  |
| ----------------- | ------------------------------------------- | ----------------------------------------------------- |
| Gaussian noise    | `GaussianNoise(std=0.01)`                   | PPG 전 샘플에 N(0, 0.01) 잡음 추가                    |
| Amplitude scaling | `AmplitudeScaling(lo=0.8, hi=1.2)`          | 신호 전체를 [0.8, 1.2] 범위의 배율로 곱함             |
| Time shift        | `TimeShift(max_shift=50)`                   | 최대 ±50 샘플 원형 이동 (끝이 처음으로 연결)          |
| Random masking    | `RandomMasking(lo_frac=0.05, hi_frac=0.10)` | 길이 5~10%의 연속 span 하나를 0으로 마스킹 (최대 1초) |

활성화된 증강은 `PPGAugment` 래퍼로 합성되어 `PPGDataset`의 `transform`으로 전달된다.
증강은 **훈련 세트에만** 적용되며 검증 세트에는 적용되지 않는다.

증강 상세 설계는 [docs/data-augmentation.md](data-augmentation.md)를 참조한다.

### 5.4 데이터셋과 DataLoader

`PPGDataset`은 `data/dataset/{train,val}/` 아래의 NPZ 파일들을 읽는다.
각 NPZ는 한 케이스의 PPG 세그먼트(`x`)와 레이블(`y`)을 담는다.

```python
train_ds = PPGDataset(dataset_dir / "train", normalize=True, preload=False, augment=augment)
val_ds   = PPGDataset(dataset_dir / "val",   normalize=True, preload=False)
```

- `normalize=True`: 각 세그먼트를 개별적으로 z-score 정규화한다 (`--no-normalize`로 비활성화).
- `preload=True`: 전체 데이터를 RAM에 미리 로드한다. 빠르지만 메모리를 많이 사용한다.

#### 환자 균형 샘플링

기본적으로 `WeightedRandomSampler`를 사용해 세그먼트 수가 많은 케이스가 훈련을 지배하지
않도록 케이스당 균등한 기여를 보장한다.

```python
weights = train_ds.sample_weights()   # 각 세그먼트의 샘플링 가중치 (1 / n_segs_in_case)
sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
```

`--no-patient-balance`를 사용하면 일반 shuffle DataLoader로 대체된다.

### 5.5 모델 생성

`bpe.models.create_model(model_name)`으로 레지스트리에서 모델을 생성한다.
등록되지 않은 이름을 입력하면 `KeyError`가 발생하고 스크립트가 종료된다.

사용 가능한 모델 목록은 `uv run python scripts/train-model.py --help`에서 확인할 수 있다.

### 5.6 옵티마이저와 학습률 스케줄러

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 1e-2)
```

| 구성 요소             | 설명                                                                 |
| --------------------- | -------------------------------------------------------------------- |
| **AdamW**             | 가중치 감쇠(weight decay)를 그래디언트에서 분리해 적용하는 Adam 변형 |
| **CosineAnnealingLR** | 학습률을 코사인 함수로 `lr` → `lr × 0.01`까지 감소시킴               |

`T_max=epochs`로 설정하므로 스케줄러는 전체 학습 기간 동안 1회 코사인 주기를 완성한다.

### 5.7 손실 함수

```python
criterion = nn.HuberLoss(delta=5.0)
```

Huber 손실(delta=5 mmHg)을 사용한다.

- `|error| ≤ 5 mmHg`: MSE 방식 (부드러운 기울기)
- `|error| > 5 mmHg`: MAE 방식 (큰 오차에서 선형, 이상치에 강인)

순수 MSE보다 혈압 이상치에 강인하고, 순수 MAE보다 0 근처에서 안정적으로 수렴한다.

### 5.8 Trainer 클래스

`bpe.train.trainer.Trainer`가 실제 학습 루프를 담당한다.

```python
trainer = Trainer(model, optimizer, scheduler, criterion, device, run_dir)
result  = trainer.fit(train_loader, val_loader, epochs=epochs, patience=patience)
```

`Trainer.fit()`의 내부 동작:

1. 매 에폭마다 훈련 루프 → 검증 루프 실행
2. 검증 손실이 개선되면 `best.pt` 갱신
3. 매 에폭 후 `last.pt` 갱신 (재개 용도)
4. `metrics.csv`에 에폭 결과 추가
5. `patience` 에폭 동안 개선이 없으면 Early stopping 종료

반환값 `result`는 `best_epoch`, `best_val_loss`, `best_val_sbp_mae`, `best_val_dbp_mae`를 포함한다.

### 5.9 설정 저장 (`save_config`)

학습 시작 직전 `run_dir/config.json`에 전체 CLI 인자를 저장한다.

```json
{
  "model": "resnet1d",
  "dataset_dir": "data/dataset",
  "models_dir": "data/models",
  "epochs": 100,
  "batch_size": 256,
  "lr": 0.001,
  ...
}
```

이 파일은 `eval-model.py`가 모델 이름을 확인할 때도 사용한다.

### 5.10 학습 재개 (`--resume`)

```python
ckpt = torch.load(path, map_location=device, weights_only=True)
model.load_state_dict(ckpt["model_state_dict"])
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
start_epoch = ckpt.get("epoch", 0)
```

`last.pt`는 옵티마이저 상태를 포함하므로 중단된 학습을 정확히 재개할 수 있다.
`best.pt`는 평가 용도이므로 모델 가중치만 포함한다.

### 5.11 실행 요약 (`runs.jsonl`)

학습이 끝나면 `data/models/<model>/runs.jsonl`에 한 줄의 JSON을 추가한다.

```json
{"run_dir": "data/models/resnet1d", "best_epoch": 87, "best_val_loss": 4.231,
 "best_val_sbp_mae": 8.14, "best_val_dbp_mae": 5.22,
 "model": "resnet1d", "epochs": 100, "batch_size": 256, "lr": 0.001, "weight_decay": 0.0001, "seed": 42}
```

같은 모델을 여러 번 실행하면 줄이 누적되어 하이퍼파라미터 탐색 이력이 남는다.

## 6. 학습 결과 확인

학습 중 또는 완료 후 손실/MAE 그래프를 확인하려면 `generate-train-status.py`를 사용한다.

```bash
bin/generate-train-status     data/models/resnet1d   # Linux / macOS
bin\generate-train-status.bat data\models\resnet1d   # Windows
```

학습된 모델을 테스트 세트로 평가하려면 `eval-model.py`를 사용한다.

```bash
bin/eval-model     data/models/resnet1d   # Linux / macOS
bin\eval-model.bat data\models\resnet1d   # Windows
```

## 7. 관련 모듈

| 모듈                      | 역할                                                          |
| ------------------------- | ------------------------------------------------------------- |
| `bpe/models/__init__.py`  | 모델 레지스트리 (`create_model`, `list_models`)               |
| `bpe/train/dataset.py`    | `PPGDataset` — NPZ 로더, z-score 정규화, 샘플 가중치          |
| `bpe/train/augment.py`    | 데이터 증강 클래스들 (`GaussianNoise`, `AmplitudeScaling`, …) |
| `bpe/train/trainer.py`    | `Trainer` — 학습/검증 루프, 체크포인트, metrics.csv           |
| `scripts/generate-train-status.py` | 학습 곡선 시각화                                              |
| `scripts/eval-model.py`   | 테스트 세트 평가 (MAE, RMSE, BHS, AAMI)                       |
