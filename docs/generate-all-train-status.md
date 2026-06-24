# `generate-all-train-status.py` 사용 및 상세 설계

작성일: 2026-06-22  
관련 코드: [scripts/generate-all-train-status.py](../scripts/generate-all-train-status.py)  
관련 문서: [docs/generate-train-status.md](generate-train-status.md), [README.md](../README.md)

## 1. 목적

`scripts/generate-all-train-status.py`는 `bpe.models.list_models()`에 등록된 모든 모델을
순회하면서 `scripts/generate-train-status.py`를 모델별로 실행하는 배치 실행기다.

`metrics.csv`가 없는 모델(학습 미완료)은 경고를 출력하고 건너뛴다.

## 2. 사용 방법

### 기본 실행

```bash
uv run python scripts/generate-all-train-status.py
```

또는 제공 런처를 사용한다.

```bash
bin/generate-all-train-status         # Linux / macOS
bin\generate-all-train-status.bat     # Windows
```

### 자주 쓰는 예시

```bash
# 기본 실행 (data/models 아래 전체 모델)
bin/generate-all-train-status

# 별도 모델 디렉터리 지정
bin/generate-all-train-status --models-dir data/models-v1
bin/generate-all-train-status --models-dir data/models-v2

# PNG 저장 없이 콘솔 요약만 출력
bin/generate-all-train-status --no-save

# 실제 실행 없이 생성될 명령어만 확인
bin/generate-all-train-status --dry-run
bin/generate-all-train-status --dry-run --models-dir data/models-v1
```

## 3. CLI 옵션

### 전용 옵션

| 옵션            | 기본값        | 설명                                         |
| --------------- | ------------- | -------------------------------------------- |
| `--models-dir`  | `data/models` | 학습된 모델 루트 디렉터리                    |
| `--dry-run`     | off           | 실제 실행 없이 생성될 명령어만 출력          |

### 전달 옵션

`--models-dir`와 `--dry-run` 외의 나머지 옵션은 `generate-train-status.py`에 그대로 전달된다.

| 옵션        | 전달 대상                    | 설명                              |
| ----------- | ---------------------------- | --------------------------------- |
| `--no-save` | `generate-train-status.py`   | PNG 저장 없이 콘솔 요약만 출력    |

## 4. 동작 방식

### 4.1 모델 목록 수집

`bpe.models.list_models()`를 호출해 등록된 모델 이름 전체를 읽는다.

### 4.2 스킵 판별

`<models-dir>/<model>/metrics.csv`가 없는 모델은 학습이 완료되지 않은 것으로 간주하고
건너뛴다. 건너뛴 모델은 경고 로그에 한 줄로 표시된다.

```text
09:00:01 [WARNING] Skipping (no metrics.csv): naive, pulsewoq_resnet1d
```

### 4.3 순차 실행

각 모델에 대해 아래 형식의 명령어를 순차적으로 실행한다.

```bash
uv run python scripts/generate-train-status.py <models-dir>/<model> [forward-args...]
```

예:

```bash
uv run python scripts/generate-train-status.py data/models-v1/resnet1d
uv run python scripts/generate-train-status.py data/models-v1/st_resnet
...
```

`generate-train-status.py`의 출력(콘솔 요약)은 터미널에 직접 표시된다.

### 4.4 종료 코드 집계

모든 모델 처리가 끝난 뒤 성공/실패 요약을 출력한다.
하나라도 실패하면 전체 종료 코드는 `1`이 된다.

```text
09:05:23 [INFO] ============================================================
09:05:23 [INFO] resnet1d                   ok
09:05:23 [INFO] st_resnet                  ok
09:05:23 [INFO] minception                 FAILED
```

## 5. 출력 파일

각 모델 디렉터리 안에 `generate-train-status.py`가 저장하는 파일들이 생성된다.

```text
data/models/
├── resnet1d/
│   ├── metrics.csv       ← 입력
│   ├── loss_graph.png    ← 생성
│   └── mae_graph.png     ← 생성
├── st_resnet/
│   ├── metrics.csv
│   ├── loss_graph.png
│   └── mae_graph.png
...
```

`--no-save`를 전달하면 PNG 저장 없이 콘솔 요약만 출력한다.

## 6. 관련 모듈

| 모듈                                       | 역할                                                   |
| ------------------------------------------ | ------------------------------------------------------ |
| `scripts/generate-train-status.py`         | 단일 모델 학습 상태 그래프 생성                        |
| `scripts/train-model.py`                   | `metrics.csv` 생성; `best.pt`, `last.pt` 저장          |
| `bpe/train/trainer.py`                     | 에폭별 지표를 `metrics.csv`에 기록하는 `Trainer`       |
| `scripts/eval-all-model.py`                | 전체 모델 테스트 세트 평가 (유사한 배치 실행 패턴)     |
