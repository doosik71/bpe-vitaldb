# `generate-train-status.py` 사용 및 상세 설계

작성일: 2026-06-22  
관련 코드: [scripts/generate-train-status.py](../scripts/generate-train-status.py)  
관련 문서: [docs/train-model.md](train-model.md), [README.md](../README.md)

## 1. 목적

`scripts/generate-train-status.py`는 학습 실행 디렉터리에 저장된 `metrics.csv`를 읽어
손실/MAE 그래프 PNG를 생성하고 콘솔에 요약 표를 출력하는 시각화 스크립트다.

학습 도중(에폭 진행 중)에도 실행할 수 있으며, 학습 완료 후 결과 확인에도 사용한다.

## 2. 사용 방법

### 기본 실행

```bash
uv run python scripts/generate-train-status.py data/models/resnet1d
```

또는 제공 런처를 사용한다.

```bash
bin/generate-train-status     data/models/resnet1d        # Linux / macOS
bin\generate-train-status.bat data\models\resnet1d        # Windows
```

### 자주 쓰는 예시

```bash
# PNG 저장 없이 요약 출력만
bin/generate-train-status data/models/resnet1d --no-save

# 별도 모델 디렉터리 지정
bin/generate-train-status data/models-v1/st_resnet
bin/generate-train-status data/models-v2/mtae
```

## 3. CLI 옵션

| 옵션        | 설명                                             |
| ----------- | ------------------------------------------------ |
| `run_dir`   | 실행 디렉터리 경로 (필수 위치 인자, `data/models/<model>`) |
| `--no-save` | PNG 파일을 저장하지 않고 콘솔 요약만 출력        |

## 4. 출력

### 콘솔 요약 (항상 출력)

```text
Run directory : data/models/resnet1d
Epochs logged : 87  (best epoch: 72)

Metric                   Last      Best
--------------------------------------------
  train_loss             4.3521    3.9847
  val_loss               4.7813    4.2031
  train_sbp_mae          6.2140    5.8321
  train_dbp_mae          3.9870    3.7102
  val_sbp_mae            7.4523    6.9811
  val_dbp_mae            4.8210    4.4302
```

### PNG 파일 (--no-save 미지정 시 저장)

실행 디렉터리 안에 두 개의 PNG가 저장된다.

| 파일             | 내용                                                      |
| ---------------- | --------------------------------------------------------- |
| `loss_graph.png` | `train_loss` vs `val_loss` — 에폭별 Huber 손실 비교      |
| `mae_graph.png`  | SBP / DBP MAE — `train_sbp_mae`, `train_dbp_mae`, `val_sbp_mae`, `val_dbp_mae` |

`loss_graph.png` 예시 (파란색 = 훈련, 빨간색 = 검증):

```
Loss
  │  ╲
  │   ╲___
  │       ╲___________  ← train_loss
  │            ╲______  ← val_loss
  └──────────────────── Epoch
```

## 5. 입력 파일 형식 (`metrics.csv`)

`train-model.py`가 매 에폭 후 자동으로 기록하는 CSV 파일이다.

```text
epoch,train_loss,val_loss,train_sbp_mae,train_dbp_mae,val_sbp_mae,val_dbp_mae,lr
1,12.3145,13.4201,9.8321,6.2140,10.1423,6.8712,0.001
2,11.2341,12.3102,9.1234,5.9821,9.8321,6.5432,0.000999
...
```

`metrics.csv`가 없으면 오류 메시지를 출력하고 종료한다.

## 6. 상세 설계

### 6.1 데이터 로드 (`load_metrics`)

```python
def load_metrics(csv_path: Path) -> dict[str, list]:
    """Return columns as lists of floats."""
```

CSV 전체를 열별 float 리스트로 읽는다. 에폭이 하나도 없으면 오류로 종료한다.

### 6.2 손실 그래프 (`plot_loss`)

- X축: 에폭 번호
- Y축: Huber 손실
- `train_loss` → 파란색 실선
- `val_loss` → 빨간색 실선

### 6.3 MAE 그래프 (`plot_mae`)

- X축: 에폭 번호
- Y축: MAE (mmHg)
- `train_sbp_mae` → 파란색 실선
- `train_dbp_mae` → 파란색 점선
- `val_sbp_mae` → 빨간색 실선
- `val_dbp_mae` → 빨간색 점선

### 6.4 요약 출력 (`print_summary`)

best epoch는 `val_loss`가 최소인 에폭으로 결정한다.
각 지표에 대해 마지막 에폭 값(`Last`)과 전체 에폭 중 최솟값(`Best`)을 함께 출력한다.

## 7. 관련 모듈

| 모듈                              | 역할                                               |
| --------------------------------- | -------------------------------------------------- |
| `scripts/train-model.py`          | `metrics.csv` 생성; `best.pt`, `last.pt` 저장       |
| `bpe/train/trainer.py`            | 에폭별 지표를 `metrics.csv`에 기록하는 `Trainer`   |
| `scripts/eval-model.py`           | 학습 완료 후 테스트 세트 평가                      |
