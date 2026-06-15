# `eval-all-model.py` 사용 및 상세 설계

작성일: 2026-06-15  
관련 코드: [scripts/eval-all-model.py](../scripts/eval-all-model.py)  
관련 문서: [docs/eval-model.md](eval-model.md), [docs/train-all-model.md](train-all-model.md), [README.md](../README.md)

## 1. 목적

`scripts/eval-all-model.py`는 `bpe.models.list_models()`에 등록된 모든 모델을 자동으로 순회하면서
`scripts/eval-model.py`를 모델별로 실행하는 평가 오케스트레이터다.

이 스크립트의 역할은 다음과 같다.

- 모델 레지스트리에서 전체 모델 이름 목록을 읽는다.
- `best.pt` 체크포인트가 없는 모델은 건너뛰고 경고를 출력한다.
- 현재 CUDA 디바이스 수를 확인하고 병렬 평가 슬롯을 구성한다.
- 각 모델에 대해 `eval-model.py`를 별도 프로세스로 실행한다.
- `--device`는 내부에서 자동 지정하고, 나머지 옵션은 그대로 전달한다.
- 각 모델의 표준 출력/에러를 `<models-dir>/<model>/eval-all.log`에 저장한다.
- 모든 작업 종료 후 성공/실패를 요약한다.

즉, 이 도구는 단일 모델 평가기가 아니라
"현재 장비에서 가능한 만큼 여러 모델을 자동 분산 실행하는 배치 실행기"다.

## 2. 사용 방법

### 기본 실행

```bash
uv run python scripts/eval-all-model.py
```

또는 제공 런처를 사용할 수 있다.

```bash
bin/eval-all-model
bin\eval-all-model.bat
```

### 자주 쓰는 예시

```bash
# v1 데이터셋, v1 모델 디렉터리 사용
bin/eval-all-model --dataset-dir data/dataset-v1 --models-dir data/models-v1

# 배치 크기 축소 (GPU 메모리 부족 시)
bin/eval-all-model --models-dir data/models-v1 --batch-size 128

# 정규화 비활성화 (학습 시 --no-normalize를 사용했을 때)
bin/eval-all-model --models-dir data/models-v1 --no-normalize

# 실제 실행 없이 모델/디바이스 배치만 확인
bin/eval-all-model --models-dir data/models-v1 --dry-run

# 폴링 간격 단축 (단기 평가에서 슬롯 대기 최소화)
bin/eval-all-model --models-dir data/models-v1 --poll-sec 0.5
```

## 3. CLI 옵션

### 전용 옵션

| 옵션            | 기본값       | 설명                                                  |
| --------------- | ------------ | ----------------------------------------------------- |
| `--models-dir`  | `data/models`| 학습된 모델 서브디렉터리가 있는 루트 디렉터리         |
| `--poll-sec`    | `2.0`        | 실행 중인 평가 프로세스 상태를 확인하는 주기(초)      |
| `--dry-run`     | off          | 실제 실행 없이 생성될 명령어만 출력                   |

### 전달 옵션

`eval-all-model.py`는 `--device`를 직접 받지 않는다.
이 옵션은 내부에서 자동 지정한다.
`--duo` 역시 금지된다. 이 스크립트는 단일 모델 평가만 지원한다.

그 외의 옵션은 `scripts/eval-model.py`에 그대로 전달된다.
대표적으로 다음과 같은 옵션을 사용할 수 있다.

| 옵션             | 전달 대상       | 설명                                         |
| ---------------- | --------------- | -------------------------------------------- |
| `--dataset-dir`  | `eval-model.py` | NPZ 데이터셋 루트 디렉터리 (default: `data/dataset`) |
| `--batch-size`   | `eval-model.py` | 추론 배치 크기 (default: `512`)              |
| `--no-normalize` | `eval-model.py` | PPG z-score 정규화 비활성화                  |

중요한 제약:

- `--device` 전달 금지 (스케줄러가 자동 지정)
- `--duo` 전달 금지 (단일 모델 평가만 지원)

이 둘을 넘기면 스케줄러가 즉시 오류를 출력하고 종료한다.

## 4. 동작 방식

### 4.1 모델 목록 수집

`bpe.models.list_models()`를 호출해서 현재 레지스트리에 등록된 모델 이름을 읽는다.

이후 각 모델에 대해 `<models-dir>/<model>/best.pt`가 존재하는지 확인한다.
체크포인트가 없는 모델은 평가 대상에서 제외되고 경고가 출력된다.
등록된 모델이 하나도 없거나, 체크포인트가 있는 모델이 하나도 없으면 `sys.exit(1)`로 종료한다.

### 4.2 디바이스 탐지

현재 구현은 다음 정책을 사용한다.

- CUDA 사용 가능 + GPU 개수 `N > 0`
  - `cuda:0`, `cuda:1`, ..., `cuda:N-1` 슬롯 생성
- CUDA 불가
  - `cpu` 슬롯 1개 생성

즉, GPU가 4개면 최대 4개 모델을 동시에 평가한다.
CPU만 있으면 순차 실행이다.

### 4.3 스케줄링

모델 큐(deque)를 만들고, 빈 디바이스 슬롯에 순서대로 배치한다.

예를 들어 모델이 10개이고 GPU가 4개면:

- 처음 4개 모델을 `cuda:0~3`에 실행
- 어떤 GPU 슬롯이 끝나면 다음 대기 모델을 그 슬롯에 배치
- 모든 모델이 끝날 때까지 반복

즉, 정적인 1회 분배가 아니라
"작업 큐 + 빈 슬롯 재사용" 방식이다.

### 4.4 실행 명령 생성

모델별 실행은 내부적으로 아래 형식의 명령어를 만든다.

```bash
uv run python scripts/eval-model.py \
  <models-dir>/<model> \
  --device <assigned-device> \
  <forwarded-args...>
```

예:

```bash
uv run python scripts/eval-model.py \
  data/models-v1/resnet1d \
  --device cuda:2 \
  --dataset-dir data/dataset-v1 \
  --batch-size 256
```

### 4.5 로그 저장

각 모델은 자기 런 디렉터리 아래에 `eval-all.log`를 남긴다.
`best.pt`와 같은 위치다.

```text
data/models-v1/
├── resnet1d/
│   ├── best.pt
│   ├── eval-all.log       ← eval-all-model.py 실행 로그
│   ├── eval_results.json  ← eval-model.py 결과
│   └── ...
├── minception/
│   ├── eval-all.log
│   └── ...
```

`eval-all.log`에는 다음이 기록된다.

- 실행 시각
- 배정된 디바이스
- 실제 실행 명령어
- `eval-model.py`의 표준 출력/에러

로그 파일은 추가(`a`) 모드로 열리므로, 같은 모델을 재실행하면 이전 로그가 누적된다.
구분선(`=== eval-all launch at ... ===`)으로 각 실행을 구별할 수 있다.

### 4.6 종료 코드 집계

모든 모델 평가가 끝난 뒤 스크립트는 다음을 요약한다.

- 모델명
- 사용 디바이스
- 종료 상태 (`ok` 또는 `exit N`)
- 실행 시간
- 로그 파일 경로

하나라도 실패하거나 완료되지 않은 모델이 있으면 전체 종료 코드는 `1`이다.
모두 성공하면 `0`이다.

## 5. 내부 설계

### 5.1 `parse_args`

`--models-dir`, `--poll-sec`, `--dry-run` 세 옵션만 직접 파싱하고,
나머지 미인식 인자는 모두 `eval-model.py` 전달 인자(`forward_args`)로 취급한다.

이 방식 덕분에 `eval-model.py` 옵션이 늘어나도
`eval-all-model.py`를 매번 크게 수정할 필요가 없다.

```python
args, forward_args = parser.parse_known_args()
```

인자 목록 앞의 구분자 `--`는 자동으로 제거된다.
`forward_args`에 `--device` 또는 `--duo`가 포함되면 즉시 오류를 출력하고 종료한다.

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

모델 런 디렉터리와 디바이스를 조합해 실제 `subprocess.Popen` 명령 리스트를 만든다.
명령 문자열을 직접 셸로 해석하지 않고 리스트로 넘기므로 인용(quoting) 오류 가능성이 낮다.

```python
def build_command(run_dir: Path, device: str, forward_args: list[str]) -> list[str]:
    return ["uv", "run", "python", str(EVAL_SCRIPT), str(run_dir), "--device", device, *forward_args]
```

### 5.4 `ActiveJob`

실행 중인 평가 프로세스 하나를 추적하는 데이터 클래스다.

| 필드         | 타입                 | 내용                             |
| ------------ | -------------------- | -------------------------------- |
| `model`      | `str`                | 모델 이름                        |
| `device`     | `str`                | 배정된 디바이스 (`cuda:N`, `cpu`) |
| `process`    | `subprocess.Popen`   | 실행 중인 프로세스 핸들          |
| `log_path`   | `Path`               | `eval-all.log` 경로              |
| `started_at` | `float`              | `time.time()` 기준 시작 시각     |

로그 파일 핸들은 `process._eval_all_log_file` 속성으로 연결해 두고,
`close_job_log()`에서 프로세스 종료 후 닫는다.

### 5.5 `launch_job`

각 모델에 대해:

1. `<models-dir>/<model>/eval-all.log`를 추가 모드로 열고 헤더를 기록한다.
2. `subprocess.Popen`으로 `eval-model.py`를 실행한다.
   - `stdout`과 `stderr`를 모두 로그 파일로 리다이렉트한다.
   - `cwd`는 프로젝트 루트(`ROOT`)로 지정한다.
3. `ActiveJob` 인스턴스를 반환한다.

출력을 콘솔로 직접 흘리지 않고 로그 파일에 합치는 이유는
여러 모델이 병렬 실행될 때 로그가 뒤섞이는 문제를 피하려는 설계다.

### 5.6 메인 스케줄 루프

`main()`의 스케줄 루프는 다음 단계를 반복한다.

1. 비어 있는 디바이스 슬롯을 순서대로 확인한다.
2. 대기 중인 모델이 있으면 빈 슬롯에 배치한다 (`launch_job`).
3. `poll_sec`만큼 대기한다 (`time.sleep`).
4. 각 슬롯의 프로세스를 폴링(`process.poll()`)해 종료 여부를 확인한다.
5. 종료된 프로세스를 수거하고, 결과를 `completed` 목록에 기록한다.
6. 해당 슬롯을 `active` 딕셔너리에서 제거한다.

복잡한 작업 큐 프레임워크를 쓰지 않고
가벼운 polling 기반 스케줄러로 구현했다.

## 6. CPU / GPU 사용 정책

### GPU가 있을 때

- GPU 하나당 평가 프로세스 하나만 띄운다.
- 같은 GPU에 여러 평가를 동시에 겹쳐 올리지 않는다.
- 모델 개수가 GPU 개수보다 많으면 남은 모델은 대기열에 들어간다.

이 방식의 장점:

- VRAM 충돌 가능성을 낮춘다.
- 스케줄링 규칙이 명확하다.
- 디버깅이 쉽다.

### GPU가 없을 때

- CPU 슬롯 하나로 순차 실행한다.
- 결과적으로 `eval-model.py`를 모델 수만큼 차례대로 호출하는 효과다.

## 7. 주의할 점

### 7.1 `best.pt` 없는 모델 건너뜀

체크포인트가 없는 모델은 평가 대상에서 제외된다.
경고 메시지 예시:

```
WARNING Skipping (no best.pt): ae_lstm, conv_reg_nas
```

학습이 완료되지 않은 모델이 있어도 나머지는 정상적으로 평가가 계속된다.

### 7.2 eval-all.log 누적

로그 파일은 추가 모드로 열린다.
같은 모델을 여러 번 실행하면 이전 결과가 남아 있다.
`eval_results.json`은 `eval-model.py`가 덮어쓰므로 최신 결과만 유지된다.

### 7.3 `--no-normalize` 일관성

학습 시 `--no-normalize`를 사용한 모델은 평가 시에도 반드시 동일하게 지정해야 한다.
`eval-all-model.py`는 모든 모델에 동일한 `forward_args`를 전달하므로,
모델별로 정규화 여부가 다른 경우 이 스크립트를 사용할 수 없다.

### 7.4 `--duo` 사용 불가

`eval-all-model.py`는 단일 모델 평가만 지원한다.
Duo 앙상블 평가는 `eval-model.py --duo`를 직접 사용해야 한다.

### 7.5 키보드 인터럽트

실행 중 `Ctrl+C`가 들어오면 활성 평가 프로세스들을 종료하려고 시도한다.
먼저 `terminate()`를 보내고, 10초 안에 끝나지 않으면 `kill()`까지 수행한다.
이후 각 로그 파일을 닫고 예외를 다시 던진다.
중단 요청 시 orphan process가 남지 않도록 배려했다.

## 8. 추천 사용 흐름

### v1 데이터셋 전체 모델 평가

```bash
bin/eval-all-model \
  --dataset-dir data/dataset-v1 \
  --models-dir data/models-v1
```

### 먼저 배치만 확인

```bash
bin/eval-all-model \
  --dry-run \
  --dataset-dir data/dataset-v1 \
  --models-dir data/models-v1
```

### 평가 완료 후 결과 수집

```bash
bin/collect-result \
  --models-dir data/models-v1 \
  --images-dir data/images-v1 \
  --logs-dir data/logs-v1

bin/generate-overview \
  --models-dir data/models-v1 \
  --output-dir data/images-v1
```

## 9. 한계와 향후 확장

현재 구현은 단순함을 우선한다.

현재 한계:

- 모델별 개별 `forward_args` 지정 불가 (모든 모델에 동일한 옵션 전달)
- GPU 메모리 사용량을 보고 동적으로 동시 실행 수를 조절하지 않음
- 평가 진행률을 중앙 대시보드로 보여주지 않음
- 같은 GPU에서 VRAM 용량이 남아도 두 모델을 동시에 올리지 않음

하지만 지금 목적에는 충분하다.

- 전체 모델 일괄 평가
- GPU 자동 분배
- 실패 모델 식별
- 로그 분리 저장

이 네 가지를 가장 작은 구현으로 제공한다.
