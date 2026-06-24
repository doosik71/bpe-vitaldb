# `pipeline.py` 사용 및 상세 설계

작성일: 2026-06-12  
관련 코드: [scripts/pipeline.py](../scripts/pipeline.py)  
관련 런처: [bin/pipeline](../bin/pipeline), [bin/pipeline.bat](../bin/pipeline.bat)  
관련 문서: [README.md](../README.md)

## 1. 목적

`scripts/pipeline.py`는 이 저장소의 주요 스크립트들을 한 화면에서 실행할 수 있게 해 주는
Tkinter 기반 GUI 오케스트레이터다.

이 모듈의 역할은 다음과 같다.

- 데이터 다운로드, 데이터셋 생성, 학습, 평가, 분석 스크립트를 한 곳에 모은다.
- 각 스크립트별 파라미터 입력 폼을 자동으로 구성한다.
- `uv run python <script>` 형식의 실행 명령을 내부에서 생성한다.
- 비GUI 스크립트의 표준 출력과 표준 에러를 Output Console에 실시간으로 표시한다.
- 장시간 실행 중에도 GUI가 마우스와 키보드 입력을 계속 처리하도록 유지한다.
- GUI 브라우저 계열 스크립트는 별도 프로세스로 즉시 실행한다.

즉, 이 도구는 단순한 버튼 모음이 아니라
"프로젝트 전체 실험 절차를 순서대로 실행하고 추적하기 위한 통합 실행 패널"이다.

## 2. 실행 방법

### 기본 실행

```bash
uv run python scripts/pipeline.py
```

또는 런처를 사용할 수 있다.

```bash
bin/pipeline
bin\pipeline.bat
```

## 3. 화면 구성

`pipeline.py`는 크게 좌측 사이드바, 우측 설정 패널, 하단 콘솔의 3영역으로 구성된다.

### 3.1 좌측 사이드바

좌측 패널에는 전체 파이프라인 단계가 카테고리별로 정렬되어 표시된다.

카테고리:

- `Environment`
- `Data Acquisition`
- `Dataset`
- `Model`
- `Training`
- `Evaluation`
- `Analysis`

각 항목에는 상태 표시 기호가 함께 표시된다.

- `.`: 아직 실행되지 않음
- `*`: 현재 실행 중
- `o`: 정상 종료
- `ERROR:`: 비정상 종료 또는 실행 실패

사용자는 항목을 클릭해 해당 스크립트의 설명과 파라미터 입력 폼을 볼 수 있다.

### 3.2 우측 설정 패널

선택한 단계에 대해 다음 정보가 표시된다.

- 단계 이름
- GUI 앱 여부
- 단계 설명
- 실행 파라미터 입력 폼
- `Run` 또는 `Launch` 버튼
- 필요 시 `Stop` 버튼
- 현재 실행 상태 라벨

GUI 스크립트는 `Launch` 버튼이 보이고,
일반 CLI 스크립트는 `Run`과 `Stop` 버튼이 함께 제공된다.

### 3.3 Output Console

하단 콘솔은 비GUI 스크립트의 실행 로그를 보여준다.

표시 내용:

- 구분선
- 실행한 단계 이름
- 실제 생성된 명령어
- 스크립트의 표준 출력/에러
- 종료 코드 기반 성공/실패 메시지

`Clear` 버튼으로 콘솔을 비울 수 있다.

## 4. 현재 제공 단계

현재 `PIPELINE` 정의에는 다음 단계가 등록되어 있다.

| Category         | Label                | Script                       | Type |
| ---------------- | -------------------- | ---------------------------- | ---- |
| Environment      | Check CUDA           | `check-cuda.py`              | CLI  |
| Data Acquisition | Download VitalDB     | `download-vitaldb.py`        | CLI  |
| Data Acquisition | Browse VitalDB       | `vitaldb-browser.py`         | GUI  |
| Dataset          | Build Dataset        | `construct-dataset.py`       | CLI  |
| Dataset          | Browse Dataset       | `dataset-browser.py`         | GUI  |
| Dataset          | PSD Browser          | `psd-browser.py`             | GUI  |
| Dataset          | Spectrogram Browser  | `spectro-browser.py`         | GUI  |
| Dataset          | Dataset Statistics   | `dataset-statistic.py`       | CLI  |
| Dataset          | Share Data           | `share-data.py`              | CLI  |
| Model            | Print Model          | `print-model.py`             | CLI  |
| Training         | Train Model          | `train-model.py`             | CLI  |
| Training         | Training Status      | `generate-train-status.py`            | CLI  |
| Evaluation       | Eval Model           | `eval-model.py`              | CLI  |
| Evaluation       | Eval PulseWoQ        | `eval-model-pulsewoq.py`     | CLI  |
| Evaluation       | Browse BPE Results   | `bpe-browser.py`             | GUI  |
| Evaluation       | Browse Pulse Results | `pulse-browser.py`           | GUI  |
| Analysis         | Collect Results      | `collect-result.py`          | CLI  |
| Analysis         | Overview Graph       | `generate-overview-graph.py` | CLI  |

새 단계를 추가하려면 `scripts/pipeline.py`의 `PIPELINE` 리스트에 항목 하나를 추가하면 된다.

## 5. 파라미터 폼 설계

각 단계는 `params` 리스트를 통해 입력 폼을 선언한다.
각 파라미터는 다음 형태를 따른다.

```python
(flag, widget_type, default, help_text[, choices])
```

지원하는 `widget_type`은 다음과 같다.

| 타입             | 의미                                                           |
| ---------------- | -------------------------------------------------------------- |
| `entry`          | 일반 문자열 입력. 공백으로 여러 토큰을 넣으면 여러 인자로 분리 |
| `int`            | 정수 입력                                                      |
| `float`          | 실수 입력                                                      |
| `bool`           | 체크박스. 체크 시 `--flag` 추가                                |
| `dir`            | 디렉터리 입력 + browse 버튼                                    |
| `file`           | 파일 입력 + browse 버튼                                        |
| `positional_dir` | 위치 인자 입력 + browse 버튼                                   |
| `combo`          | 읽기 전용 드롭다운                                             |
| `combo_free`     | 직접 입력 가능한 콤보박스                                      |

### 5.1 상대 경로 처리

`dir`, `file`, `positional_dir` 타입의 기본값이 상대 경로이면
GUI에 표시할 때 프로젝트 루트 기준 절대 경로로 확장한다.

이 덕분에 사용자는 현재 작업 디렉터리를 의식하지 않고 경로를 확인할 수 있다.

### 5.2 모델/디바이스 목록 자동 반영

모델 관련 폼은 가능하면 하드코딩하지 않고 동적으로 채운다.

- 모델 목록: `bpe.models.list_models()` 호출 결과 사용
- GPU 목록: `torch.cuda.is_available()`와 `torch.cuda.device_count()` 기반 생성

따라서 새 모델이 등록되거나 GPU 수가 달라져도 GUI 선택지가 자동으로 따라간다.

## 6. 실행 명령 생성 방식

실행 명령은 `_build_cmd()`에서 구성한다.
기본 형식은 다음과 같다.

```bash
uv run python scripts/<target>.py [positionals...] [--flag value ...]
```

예:

```bash
uv run python scripts/train-model.py   --model resnet1d   --dataset-dir /abs/path/data/dataset   --models-dir /abs/path/data/models   --epochs 100
```

명령 생성 규칙은 다음과 같다.

- `bool`
  - 체크된 경우에만 `--flag`를 추가한다.
- `positional_dir`
  - `--flag` 없이 위치 인자로 추가한다.
- `entry`
  - 공백 기준으로 분리해서 여러 토큰을 그대로 전달한다.
  - 예: `0.7 0.1 0.2` -> `--split 0.7 0.1 0.2`
- 그 외 타입
  - 값이 비어 있지 않으면 `--flag value` 형태로 전달한다.

## 7. GUI 단계와 CLI 단계의 실행 차이

### 7.1 GUI 단계

`step.get("gui") == True`인 경우:

- `subprocess.Popen()`으로 즉시 실행한다.
- 표준 출력을 추적하지 않는다.
- 실행 성공 시 현재 런처 GUI에서는 즉시 `done` 상태로 표시한다.
- 실제 자식 GUI 앱 내부 동작 성공 여부까지 감시하지는 않는다.

즉, `pipeline.py`는 GUI 브라우저를 "실행해 주는 런처" 역할에 집중한다.

### 7.2 CLI 단계

비GUI 스크립트는 다음 방식으로 실행한다.

- `stdout=PIPE`
- `stderr=STDOUT`
- `PYTHONUNBUFFERED=1`

이렇게 하면 표준 출력과 에러가 한 스트림으로 합쳐져
Output Console에 시간 지연 없이 표시된다.

## 8. 출력 콘솔 처리 방식

장시간 실행 스크립트는 tqdm 스타일 진행 로그를 많이 출력할 수 있다.
`pipeline.py`는 이 상황에서 Tk 이벤트 루프가 멈추지 않도록 별도 설계를 사용한다.

### 8.1 백그라운드 reader thread

실행 중인 프로세스의 출력을 별도 스레드에서 1바이트씩 읽는다.
그 후 내부 큐 `self._out_q`에 이벤트를 넣는다.

이벤트 종류:

- `("line", text)`
- `("done", step_id, rc)`

현재 구현에서는 `\r`도 `\n`처럼 처리하여 줄 단위 로그로 바꾼다.
이 방식은 완전한 in-place progress 렌더링은 포기하지만,
Tk 콘솔에서 줄 경계가 깨지는 문제를 줄이고 가독성을 높인다.

### 8.2 메인 스레드 polling

`_poll_q()`는 `after(100, ...)`로 100 ms마다 큐를 비운다.
한 번에 처리하는 최대 아이템 수도 제한해,
출력이 아주 많더라도 버튼과 스크롤, 키보드 입력이 굶지 않도록 했다.

이 설계는 특히 다음 문제를 완화하기 위한 것이다.

- 다운로드 진행 중 GUI freeze
- Stop 버튼 미반응
- 마우스/키보드 입력 지연

### 8.3 콘솔 스타일

콘솔은 태그 기반 색상 표시를 사용한다.

- `ts`: 구분선/타임스탬프 계열 색상
- `cmd`: 실행 명령어 색상
- `info`: 단계 정보 색상
- `ok`: 성공 메시지 색상
- `err`: 오류 메시지 색상

## 9. Stop 동작

`Stop` 버튼은 현재 실행 중인 비GUI 프로세스에 대해 `terminate()`를 호출한다.

설계상 특징:

- 한 번에 하나의 비GUI 작업만 실행 가능
- 실행 중에는 다른 단계의 `Run` 버튼이 비활성화됨
- 중지 후 프로세스 종료 코드가 비0이면 해당 단계는 `error` 상태가 됨

제약:

- 현재 구현은 자식 프로세스 트리 전체 강제 종료보다 "주 프로세스 terminate"에 가깝다.
- 스크립트 내부에서 추가 하위 프로세스를 만들었다면 종료 전파는 해당 스크립트 구현에 의존한다.

## 10. 내부 클래스와 주요 메서드

### 10.1 `BPEApp`

메인 Tk 애플리케이션 클래스다.

핵심 상태:

- `_selected_id`: 현재 선택된 단계 ID
- `_running_id`: 현재 실행 중인 단계 ID
- `_process`: 현재 실행 중인 `Popen` 객체
- `_out_q`: reader thread와 메인 스레드 사이의 로그 큐
- `_statuses`: 단계별 상태 저장
- `_step_map`: ID -> 단계 정의 매핑
- `_param_vars`: 현재 폼 위젯 값 저장

### 10.2 주요 메서드

- `_build_ui()`
  - 전체 레이아웃 생성
- `_populate_sidebar()`
  - 단계 목록 생성
- `_select_step()`
  - 현재 단계 선택 및 설정 패널 갱신
- `_build_config()`
  - 단계별 파라미터 폼 렌더링
- `_add_param_row()`
  - 파라미터 타입에 맞는 위젯 생성
- `_build_cmd()`
  - 실제 실행 명령 생성
- `_run()`
  - GUI/CLI 단계별 실행 분기
- `_stop()`
  - 현재 비GUI 프로세스 종료 요청
- `_poll_q()`
  - 출력 큐 polling 및 상태 업데이트
- `_set_status()`
  - 사이드바 상태 기호와 색상 변경

## 11. 사용 예시

### 11.1 데이터 다운로드

1. `Download VitalDB` 선택
2. `output-dir`, `start-case`, `end-case`, `workers` 입력
3. `Run` 클릭
4. Output Console에서 진행 상황 확인

### 11.2 데이터셋 브라우저 실행

1. `Browse Dataset` 선택
2. `dataset-dir`와 `target-hz` 확인
3. `Launch` 클릭
4. 별도 브라우저 창에서 세그먼트 탐색

### 11.3 모델 학습 실행

1. `Train Model` 선택
2. 모델, 데이터셋 경로, 출력 경로, 에폭 수, 디바이스 설정
3. `Run` 클릭
4. Output Console에서 학습 로그 확인
5. 필요 시 `Stop`으로 종료 요청

## 12. 확장 방법

새 스크립트를 파이프라인에 추가하는 가장 단순한 방법은 `PIPELINE` 리스트에 새 항목을 넣는 것이다.

예:

```python
{
    "id": "new_step",
    "label": "New Step",
    "category": "Analysis",
    "script": "new-step.py",
    "desc": "Short description.",
    "gui": False,
    "params": [
        ("input-dir", "dir", "data/input", "Input directory"),
        ("verbose", "bool", False, "Enable verbose output"),
    ],
}
```

이 구조의 장점은 다음과 같다.

- 별도 라우팅 코드 없이 자동으로 사이드바에 나타남
- 파라미터 폼도 자동 생성됨
- 실행 명령도 공통 로직으로 처리됨

즉, 작은 스크립트 추가 비용이 낮다.

## 13. 제약 사항

현재 구현의 의도적 제약은 다음과 같다.

- 동시에 여러 CLI 작업을 실행하지 않는다.
- 단계 간 의존성 검증은 하지 않는다.
  - 예: 데이터셋이 없는데 학습을 눌러도 사전 차단하지 않는다.
- 파라미터 유효성 검사는 최소 수준이다.
  - 대부분은 실제 하위 스크립트가 검증한다.
- GUI 단계는 실행만 담당하고 종료 상태까지 추적하지 않는다.
- 콘솔은 풍부한 터미널 에뮬레이터가 아니라 단순 로그 뷰어다.

이 제약은 구현을 단순하고 유지보수 가능하게 유지하기 위한 선택이다.

## 14. 권장 사용 흐름

이 저장소의 목적이 VitalDB 기반 PPG -> BP 추정 파이프라인 구축이라는 점을 고려하면,
보통 다음 순서로 사용하는 것이 자연스럽다.

1. `Check CUDA`
2. `Download VitalDB`
3. `Browse VitalDB`
4. `Build Dataset`
5. `Browse Dataset` / `PSD Browser` / `Spectrogram Browser`
6. `Dataset Statistics`
7. `Train Model`
8. `Training Status`
9. `Eval Model` 또는 `Eval PulseWoQ`
10. `Browse BPE Results` 또는 `Browse Pulse Results`
11. `Collect Results`
12. `Overview Graph`

## 15. 요약

`pipeline.py`는 이 프로젝트의 여러 스크립트를 일관된 방식으로 실행하기 위한 통합 GUI 런처다.

핵심 특징은 다음과 같다.

- 선언형 `PIPELINE` 리스트 기반 단계 구성
- 파라미터 폼 자동 생성
- `uv run python` 기반 일관된 실행
- 비GUI 스크립트의 실시간 로그 표시
- 장시간 출력 중에도 GUI 반응성 유지
- 실험 절차 전체를 한 화면에서 탐색 가능

기존 개별 스크립트 사용법을 몰라도,
이 GUI 하나만으로 프로젝트의 전체 작업 흐름을 순서대로 따라갈 수 있게 하는 것이
이 모듈의 가장 큰 목적이다.
