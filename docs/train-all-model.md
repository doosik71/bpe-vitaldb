# `train-all-model.py` 사용 및 상세 설계

작성일: 2026-06-12  
관련 코드: [scripts/train-all-model.py](../scripts/train-all-model.py)  
관련 문서: [docs/train-model.md](../docs/train-model.md), [README.md](../README.md)

## 1. 목적

`scripts/train-all-model.py`는 `bpe.models.list_models()`에 등록된 모든 모델을 자동으로 순회하면서
`scripts/train-model.py`를 모델별로 실행하는 학습 오케스트레이터다.

이 스크립트의 역할은 다음과 같다.

- 모델 레지스트리에서 전체 모델 이름 목록을 읽는다.
- 현재 CUDA 디바이스 수를 확인한다.
- GPU가 있으면 디바이스 수만큼 병렬 학습 슬롯을 만든다.
- 각 모델에 대해 `train-model.py`를 별도 프로세스로 실행한다.
- `--model`, `--device`는 내부에서 자동 지정하고, 나머지 옵션은 그대로 전달한다.
- 각 모델의 표준 출력/에러를 `<output-dir>/<model>/train-all.log`에 저장한다.
- 모든 작업 종료 후 성공/실패를 요약한다.

즉, 이 도구는 단일 모델 학습기가 아니라
"현재 장비에서 가능한 만큼 여러 모델을 자동 분산 실행하는 배치 실행기"다.

## 2. 사용 방법

### 기본 실행

```bash
uv run python scripts/train-all-model.py
```

또는 제공 런처를 사용할 수 있다.

```bash
bin/train-all-model
bin\train-all-model.bat
```

### 자주 쓰는 예시

```bash
# data/dataset 전체 모델 학습
uv run python scripts/train-all-model.py

# v1 데이터셋과 별도 모델 출력 디렉터리 사용
uv run python scripts/train-all-model.py \
  --dataset-dir data/dataset-v1 \
  --models-dir data/models-v1

# 짧은 테스트 학습
uv run python scripts/train-all-model.py \
  --dataset-dir data/dataset-v1 \
  --models-dir data/models-v1 \
  --epochs 5 \
  --batch-size 64

# 실제 실행 없이 모델/디바이스 배치만 확인
uv run python scripts/train-all-model.py \
  --dry-run \
  --dataset-dir data/dataset-v1 \
  --models-dir data/models-v1
```

## 3. CLI 옵션

### 전용 옵션

| 옵션         | 기본값 | 설명                                             |
| ------------ | ------ | ------------------------------------------------ |
| `--poll-sec` | `2.0`  | 실행 중인 학습 프로세스 상태를 확인하는 주기(초) |
| `--dry-run`  | off    | 실제 실행 없이 생성될 명령어만 출력              |

### 전달 옵션

`train-all-model.py`는 `--model`, `--device`를 직접 받지 않는다.
이 두 옵션은 내부에서 자동 지정한다.

그 외의 옵션은 `scripts/train-model.py`에 그대로 전달된다.
대표적으로 다음과 같은 옵션을 사용할 수 있다.

| 옵션                   | 전달 대상        | 설명                      |
| ---------------------- | ---------------- | ------------------------- |
| `--dataset-dir`        | `train-model.py` | 입력 데이터셋 루트        |
| `--models-dir`         | `train-model.py` | 모델 출력 루트            |
| `--epochs`             | `train-model.py` | 최대 에폭 수              |
| `--batch-size`         | `train-model.py` | 배치 크기                 |
| `--lr`                 | `train-model.py` | 초기 학습률               |
| `--weight-decay`       | `train-model.py` | AdamW weight decay        |
| `--patience`           | `train-model.py` | early stopping patience   |
| `--seed`               | `train-model.py` | 난수 시드                 |
| `--workers`            | `train-model.py` | DataLoader worker 수      |
| `--preload`            | `train-model.py` | 전체 세그먼트 RAM preload |
| `--no-normalize`       | `train-model.py` | z-score 정규화 비활성화   |
| `--resume`             | `train-model.py` | 체크포인트 재개 경로      |
| `--no-aug-*`           | `train-model.py` | 개별 증강 비활성화        |
| `--no-patient-balance` | `train-model.py` | 환자 균형 샘플링 비활성화 |

중요한 제약:

- `--model` 전달 금지
- `--device` 전달 금지

이 둘을 넘기면 스케줄러가 종료한다.

## 4. 동작 방식

### 4.1 모델 목록 수집

`bpe.models.list_models()`를 호출해서 현재 레지스트리에 등록된 모델 이름을 읽는다.

즉, 새로운 모델을 학습 대상에 넣으려면
별도 하드코딩이 아니라 모델 레지스트리에 등록되어 있어야 한다.

### 4.2 디바이스 탐지

현재 구현은 다음 정책을 사용한다.

- CUDA 사용 가능 + GPU 개수 `N > 0`
  - `cuda:0`, `cuda:1`, ..., `cuda:N-1` 슬롯 생성
- CUDA 불가
  - `cpu` 슬롯 1개 생성

즉, GPU가 4개면 최대 4개 모델을 동시에 학습한다.
CPU만 있으면 순차 실행이다.

### 4.3 스케줄링

모델 큐를 만들고, 비어 있는 디바이스 슬롯에 순서대로 배치한다.

예를 들어 모델이 10개이고 GPU가 4개면:

- 처음 4개 모델을 `cuda:0~3`에 실행
- 어떤 GPU 슬롯이 끝나면 다음 대기 모델을 그 슬롯에 배치
- 모든 모델이 끝날 때까지 반복

즉, 정적인 1회 분배가 아니라
"작업 큐 + 빈 슬롯 재사용" 방식이다.

### 4.4 실행 명령 생성

모델별 실행은 내부적으로 아래 형식의 명령어를 만든다.

```bash
uv run python scripts/train-model.py \
  --model <model-name> \
  --device <assigned-device> \
  <forwarded-args...>
```

예:

```bash
uv run python scripts/train-model.py \
  --model resnet1d \
  --device cuda:2 \
  --dataset-dir data/dataset-v1 \
  --models-dir data/models-v1 \
  --epochs 100
```

### 4.5 로그 저장

각 모델은 자기 출력 디렉터리 아래에 `train-all.log`를 남긴다.

```text
data/models-v1/
├── resnet1d/
│   ├── train-all.log
│   ├── best.pt
│   ├── last.pt
│   └── ...
├── minception/
│   ├── train-all.log
│   └── ...
```

`train-all.log`에는 다음이 기록된다.

- 실행 시각
- 배정된 디바이스
- 실제 실행 명령어
- `train-model.py`의 표준 출력/에러

### 4.6 종료 코드 집계

모든 모델 학습이 끝난 뒤 스크립트는 다음을 요약한다.

- 모델명
- 사용 디바이스
- 종료 상태 (`ok` 또는 `exit N`)
- 실행 시간
- 로그 파일 경로

하나라도 실패하면 `train-all-model.py` 전체 종료 코드는 `1`이 된다.
모두 성공하면 `0`이다.

## 5. 내부 설계

### 5.1 `parse_args`

스케줄러 전용 옵션만 직접 파싱하고,
나머지 미인식 인자는 모두 `train-model.py` 전달 인자로 취급한다.

이 방식 덕분에 `train-model.py` 옵션이 늘어나도
`train-all-model.py`를 매번 크게 수정할 필요가 없다.

### 5.2 `detect_devices`

```python
def detect_devices() -> list[str]:
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count > 0:
            return [f"cuda:{i}" for i in range(count)]
    return ["cpu"]
```

매우 단순한 정책이지만 예측 가능하고 안정적이다.

### 5.3 `build_command`

모델명과 디바이스를 조합해 실제 `subprocess.Popen` 명령 리스트를 만든다.
명령 문자열을 직접 셸로 해석하지 않고 리스트로 넘기므로 quoting 오류 가능성이 낮다.

### 5.4 `launch_job`

각 모델에 대해:

- `<output-dir>/<model>` 디렉터리를 만들고
- `train-all.log`를 열고
- `subprocess.Popen`으로 `train-model.py`를 실행한다.

출력은 콘솔로 직접 흘리지 않고 로그 파일에 합친다.
여러 모델이 병렬 실행될 때 로그가 섞이는 문제를 피하려는 설계다.

### 5.5 메인 스케줄 루프

메인 루프는 다음 단계를 반복한다.

1. 비어 있는 디바이스 슬롯을 찾는다.
2. 대기 중인 모델이 있으면 그 슬롯에 배치한다.
3. `poll_sec`만큼 대기한다.
4. 종료된 프로세스를 수거하고 결과를 기록한다.

즉, 복잡한 작업 큐 프레임워크를 쓰지 않고
가벼운 polling 기반 스케줄러로 구현했다.

## 6. CPU / GPU 사용 정책

### GPU가 있을 때

- GPU 하나당 학습 프로세스 하나만 띄운다.
- 같은 GPU에 여러 학습을 동시에 겹쳐 올리지 않는다.
- 모델 개수가 GPU 개수보다 많으면 남은 모델은 대기열에 들어간다.

이 방식은 다음 장점이 있다.

- VRAM 충돌 가능성을 낮춘다.
- 스케줄링 규칙이 명확하다.
- 디버깅이 쉽다.

### GPU가 없을 때

- CPU 슬롯 하나로 순차 실행한다.
- 결과적으로 `train-model.py`를 모델 수만큼 차례대로 호출하는 효과다.

## 7. 주의할 점

### 7.1 `--resume` 전달 시 동작

`--resume`은 그대로 모든 모델 실행에 전달된다.
따라서 특정 모델별 resume 경로를 따로 다르게 주는 기능은 현재 없다.

즉, 다음 같은 사용은 부적절하다.

```bash
uv run python scripts/train-all-model.py --resume data/models/resnet1d/last.pt
```

이 경우 모든 모델에 같은 resume 경로가 전달되므로 잘못된 체크포인트 로드가 발생할 수 있다.

현재 구조에서는 `resume` 없이 두고,
각 모델이 자기 `<output-dir>/<model>/last.pt`를 자동 탐색하게 두는 것이 안전하다.

### 7.2 출력 디렉터리 충돌

`train-model.py`는 모델별로 `<output-dir>/<model>`을 사용한다.
따라서 같은 `output-dir`에서 병렬 실행해도 서로 다른 모델끼리는 충돌하지 않는다.

다만 같은 모델을 동시에 두 번 실행하는 구조는 아니므로,
이 스크립트 내부에서는 동일 모델 충돌이 생기지 않는다.

### 7.3 키보드 인터럽트

실행 중 `Ctrl+C`가 들어오면 활성 학습 프로세스들을 종료하려고 시도한다.
먼저 `terminate()`를 보내고, 필요하면 `kill()`까지 수행한다.

즉, 중단 요청 시 orphan process가 남지 않도록 배려했다.

## 8. 추천 사용 흐름

### 데이터셋 v1 전체 모델 학습

```bash
bin/train-all-model \
  --dataset-dir data/dataset-v1 \
  --models-dir data/models-v1
```

### 먼저 배치만 확인

```bash
bin/train-all-model \
  --dry-run \
  --dataset-dir data/dataset-v1 \
  --models-dir data/models-v1
```

### 짧은 smoke run

```bash
bin/train-all-model \
  --dataset-dir data/dataset-v1 \
  --models-dir data/models-v1-smoke \
  --epochs 1 \
  --batch-size 8
```

## 9. 한계와 향후 확장

현재 구현은 단순함을 우선한다.

현재 한계:

- 모델별 개별 하이퍼파라미터 지정 불가
- 모델별 개별 resume 경로 지정 불가
- GPU 메모리 사용량을 보고 동적으로 동시 실행 수를 조절하지 않음
- 학습 진행률을 중앙 대시보드로 보여주지 않음

하지만 지금 목적에는 충분하다.

- 전체 모델 일괄 학습
- GPU 자동 분배
- 실패 모델 식별
- 로그 분리 저장

이 네 가지를 가장 작은 구현으로 제공한다.
