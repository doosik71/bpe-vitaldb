# `collect-result.py` 사용 및 상세 설계

작성일: 2026-06-22  
관련 코드: [scripts/collect-result.py](../scripts/collect-result.py)  
관련 문서: [docs/generate-overview.md](generate-overview.md), [docs/eval-all-model.md](eval-all-model.md)

## 1. 목적

`scripts/collect-result.py`는 개별 모델 디렉터리(`data/models/<model>/`)에 흩어져 있는
학습·평가 결과 파일을 단일 출력 디렉터리(`data/results/`)로 모아 정리한다.

학습·평가 완료 후 아래 용도로 사용된다.

- `generate-overview.py`로 모델 간 비교 그래프를 그리기 위한 전처리
- 결과물을 버전별 디렉터리(`data/results-v1`, `data/results-v2`)에 따로 보관

## 2. 사용 방법

### 기본 실행

```bash
uv run python scripts/collect-result.py
```

또는 제공 런처를 사용한다.

```bash
bin/collect-result         # Linux / macOS
bin\collect-result.bat     # Windows
```

### 자주 쓰는 예시

```bash
# 기본 실행 (data/models → data/results)
bin/collect-result

# 버전별 모델/결과 디렉터리 지정
bin/collect-result --models-dir data/models-v1 --results-dir data/results-v1
bin/collect-result --models-dir data/models-v2 --results-dir data/results-v2
```

## 3. CLI 옵션

| 옵션            | 기본값          | 설명                                         |
| --------------- | --------------- | -------------------------------------------- |
| `--models-dir`  | `data/models`   | 학습·평가 결과가 저장된 모델 루트 디렉터리   |
| `--results-dir` | `data/results`  | 수집 결과를 저장할 출력 루트 디렉터리        |

## 4. 수집 파일 목록

| 소스 파일                     | 수집 대상 경로                                      |
| ----------------------------- | --------------------------------------------------- |
| `loss_graph.png`              | `results/loss_graph/<model>.png`                    |
| `mae_graph.png`               | `results/mae_graph/<model>.png`                     |
| `error_hist.png`              | `results/error_hist/<model>.png`                    |
| `eval_plot.png`               | `results/eval_plot/<model>.png`                     |
| `bland_altman.png`            | `results/bland_altman/<model>.png`                  |
| `bland_altman_all.png`        | `results/bland_altman_all/<model>.png`              |
| `bland_altman_accepted.png`   | `results/bland_altman_accepted/<model>.png`         |
| `eval_results.json`           | `results/eval_results/<model>.json`                 |
| `metrics.csv`                 | `results/metrics/<model>.csv`                       |

소스 파일이 모델 디렉터리에 없으면 해당 항목은 조용히 건너뛴다.

## 5. 출력 디렉터리 구조

```text
data/results/
├── loss_graph/
│   ├── acfa.png
│   ├── ae_lstm.png
│   └── ...
├── mae_graph/
│   └── ...
├── error_hist/
│   └── ...
├── eval_plot/
│   └── ...
├── bland_altman/
│   └── ...
├── eval_results/
│   ├── acfa.json
│   ├── ae_lstm.json
│   └── ...
└── metrics/
    ├── acfa.csv
    ├── ae_lstm.csv
    └── ...
```

`--results-dir` 아래 서브디렉터리는 필요 시 자동 생성된다.

## 6. 동작 방식

1. `models_dir` 아래에서 서브디렉터리를 이름순으로 열거한다.
2. 각 모델 디렉터리에서 수집 대상 파일을 하나씩 확인한다.
3. 파일이 존재하면 `results_dir/<subdir>/<model>.<ext>`로 복사한다.  
   (`shutil.copy2` 사용 — mtime 등 메타데이터 보존)
4. 완료 후 복사된 파일 수를 출력한다.

## 7. 관련 모듈

| 모듈                                   | 역할                                                     |
| -------------------------------------- | -------------------------------------------------------- |
| `scripts/eval-model.py`               | `eval_results.json`, `eval_plot.png`, `bland_altman.png` 등 생성 |
| `scripts/generate-train-status.py`    | `loss_graph.png`, `mae_graph.png` 생성                   |
| `scripts/eval-all-model.py`           | 전체 모델 평가 배치 실행                                  |
| `scripts/generate-overview.py`        | `results/eval_results/` 등을 읽어 모델 간 비교 그래프 생성 |
