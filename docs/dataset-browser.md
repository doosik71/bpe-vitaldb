# `dataset-browser.py` 사용 및 개발 설명

작성일: 2026-06-08  
관련 코드: [scripts/dataset-browser.py](/home/doosik/work/bpe-vitaldb/scripts/dataset-browser.py:1)  
관련 문서: [docs/construct-dataset.md](/home/doosik/work/bpe-vitaldb/docs/construct-dataset.md:1)

---

## 1. 목적

`scripts/dataset-browser.py`는 `construct-dataset.py`가 생성한 `NPZ` 데이터셋을
시각적으로 점검하는 GUI 브라우저다.

이 스크립트의 역할은 다음과 같다.

- `train / val / test` split별 케이스 목록을 보여준다.
- 각 케이스의 세그먼트 개수와 파일 크기를 표시한다.
- 선택한 케이스의 PPG 세그먼트를 하나씩 그려준다.
- 세그먼트의 `SBP`, `DBP` 레이블을 함께 표시한다.
- 슬라이더, 버튼, 키보드로 세그먼트와 케이스를 빠르게 이동할 수 있게 한다.

즉, 이 도구는 모델 학습 전에 "데이터셋이 예상대로 만들어졌는지"를 확인하는
가벼운 품질 점검 브라우저다.

---

## 2. 입력과 출력

### 입력

- 루트 디렉터리: 기본값 `data/dataset`
- 기대 구조
  - `data/dataset/train/*.npz`
  - `data/dataset/val/*.npz`
  - `data/dataset/test/*.npz`

각 `npz` 파일은 다음 배열을 포함해야 한다.

```text
x  float32  (N, samples)   PPG segment array
y  float32  (N, 2)         [SBP, DBP]
```

이 형식은 `construct-dataset.py`의 출력 형식과 맞춰져 있다.

### 출력

- 별도의 파일을 저장하지 않는다.
- Tkinter + Matplotlib 기반의 데스크톱 GUI 창을 띄운다.

---

## 3. 실행 방법

### 기본 실행

```bash
uv run python scripts/dataset-browser.py
```

### 옵션 포함 실행

```bash
uv run python scripts/dataset-browser.py \
  --dataset-dir data/dataset \
  --target-hz 125
```

옵션 설명:

- `--dataset-dir`
  - `train / val / test` 하위 폴더를 포함하는 데이터셋 루트
  - 기본값: `data/dataset`
- `--target-hz`
  - 데이터셋 생성 시 사용한 PPG 샘플링 주파수
  - 기본값: `125`
  - x축 시간 길이 계산에 사용된다.

### 자주 쓰는 예시

다른 경로의 데이터셋 열기:

```bash
uv run python scripts/dataset-browser.py --dataset-dir data/dataset-125hz
```

125 Hz가 아닌 데이터셋 열기:

```bash
uv run python scripts/dataset-browser.py --target-hz 100
```

---

## 4. UI 구성

브라우저는 좌우 2패널 구조다.

### 왼쪽 패널

- split 선택 버튼
  - `Train`
  - `Val`
  - `Test`
- 케이스 수 / 세그먼트 수 요약 라벨
- 케이스 목록 `Treeview`
  - `Case ID`
  - `Segments`
  - `Size`

초기에는 세그먼트 수와 파일 크기를 `...`로 보여주고,
백그라운드 인덱싱이 진행되면서 실제 값으로 갱신한다.

### 오른쪽 패널

- 상단 정보 바
  - 현재 케이스 ID
  - 현재 세그먼트의 `SBP`
  - 현재 세그먼트의 `DBP`
- 중앙 플롯 영역
  - PPG 파형
  - 플롯 내부 `SBP`, `DBP` 주석 박스
- 하단 내비게이션 바
  - `Prev`
  - `Next`
  - 세그먼트 인덱스 표시
  - 세그먼트 슬라이더
  - Jump 입력창

처음에는 "Select a case from the list" 플레이스홀더를 보여주고,
케이스를 처음 열 때 Matplotlib 캔버스를 표시한다.

---

## 5. 내비게이션

### 마우스 조작

- 리스트에서 케이스 선택
- `Prev` / `Next` 버튼으로 세그먼트 이동
- 슬라이더로 원하는 세그먼트로 즉시 이동
- `Jump` 입력창에 1-based 세그먼트 번호를 넣고 `Enter`

### 키보드 단축키

- `←` / `→`
  - 이전 / 다음 세그먼트
- `↑` / `↓`
  - 이전 / 다음 케이스

Jump 입력 역시 1-based 번호를 사용한다.
즉 `1`을 입력하면 첫 번째 세그먼트로 이동한다.

---

## 6. 처리 흐름

### 6.1 파일 탐색

`_discover_files()`는 각 split 디렉터리에서 `*.npz`를 찾는다.
파일명 stem이 숫자면 숫자 기준으로 정렬한다.

각 파일에 대해 즉시 전체 데이터를 읽지는 않고,
먼저 placeholder row만 만든다.

### 6.2 메타데이터 백그라운드 로딩

브라우저 시작 시 `_start_metadata_worker()`가 별도 스레드를 띄운다.

이 스레드는 각 `npz` 파일에 대해:

- `x` 배열 길이로 세그먼트 수 계산
- 파일 크기 계산

을 수행한 뒤 `queue.Queue`를 통해 메인 스레드로 결과를 넘긴다.

메인 스레드는 `_drain_metadata_queue()`에서 주기적으로 큐를 비우면서:

- 해당 행의 `Segments`, `Size` 값을 갱신하고
- 상태 표시줄에 인덱싱 진행 상황을 표시한다.

이 방식 덕분에 파일 수가 많아도 UI가 바로 뜨고, 목록 정보가 점진적으로 채워진다.

### 6.3 split 전환

`_select_split()`은 현재 split 상태를 바꾸고:

- 버튼 색상을 갱신하고
- 리스트를 다시 채우고
- 현재 열려 있던 케이스/세그먼트를 초기화하고
- 오른쪽 캔버스를 비운다.

즉 split 전환은 단순 필터링이 아니라 우측 뷰도 함께 리셋하는 동작이다.

### 6.4 케이스 로드

사용자가 케이스를 선택하면 `_load_case()`가 실행된다.

이 함수는:

- `np.load(path)`로 파일을 열고
- `x`, `y` 배열을 메모리에 올리고
- 현재 세그먼트 인덱스를 `0`으로 초기화한 뒤
- 첫 세그먼트를 바로 그린다.

로드 실패 시 상태 바에 에러 메시지를 보여주고 종료한다.

### 6.5 세그먼트 표시

`_show_segment(idx)`는 현재 세그먼트의 핵심 렌더링 함수다.

이 함수는:

- `ppg = self._x[idx]`
- `sbp = self._y[idx, 0]`
- `dbp = self._y[idx, 1]`

를 읽고, `target_hz`를 이용해 x축 시간을 계산한다.

플롯은 다음 특징을 가진다.

- 단일 PPG 라인 플롯
- 밝은 배경, 연한 그리드
- 우상단 `SBP`, `DBP` 값 주석 박스
- 상태 바에 현재 세그먼트 번호, 샘플 수, 샘플링 주파수 표시

### 6.6 세그먼트 이동

세그먼트 이동은 네 가지 경로를 지원한다.

- `_prev_seg()`
- `_next_seg()`
- `_on_segment_slider()`
- `_on_jump()`

슬라이더는 내부적으로 1-based 값을 쓰지만, 실제 배열 인덱스는 0-based다.
이 차이는 `_set_segment_slider()`와 `_on_segment_slider()`가 흡수한다.

---

## 7. 함수별 설명

### `parse_args()`

CLI 인자를 정의한다.

### `DatasetBrowser.__init__()`

애플리케이션 상태 초기화, 파일 탐색, UI 생성, 초기 split 선택,
메타데이터 워커 시작을 담당한다.

### `_discover_files()`

split별 `npz` 파일 목록과 placeholder row를 준비한다.

### `_start_metadata_worker()` / `_metadata_worker()` / `_drain_metadata_queue()`

메타데이터를 비동기로 읽고 UI를 점진적으로 갱신한다.

### `_build_list_panel()` / `_build_canvas_panel()`

좌측 리스트 패널과 우측 플롯 패널을 구성한다.

### `_load_case()`

하나의 `npz` 케이스 파일을 메모리에 로드한다.

### `_show_segment()`

현재 세그먼트의 파형과 레이블을 렌더링한다.

### `_prev_case()` / `_next_case()`

현재 split의 정렬된 케이스 목록 기준으로 이전/다음 케이스를 연다.

---

## 8. 상태 표시와 사용자 피드백

상태 바는 다음 정보를 보여준다.

- 초기 안내 메시지
- 메타데이터 인덱싱 진행률
- 파일 로딩 중 메시지
- 로딩 실패 메시지
- 현재 케이스 / 세그먼트 / 샘플 수 / 레이블 정보

내비게이션 버튼은 현재 위치에 따라 자동으로 활성화/비활성화된다.

- 첫 세그먼트면 `Prev` 비활성화
- 마지막 세그먼트면 `Next` 비활성화

---

## 9. 개발 시 알아둘 제약과 주의점

### 9.1 정렬 기능은 없다

리스트 컬럼 헤더는 존재하지만, `dataset-browser.py`는 컬럼 클릭 정렬을 구현하지 않았다.
현재는 항상 `case` 기준 오름차순 정렬이다.

### 9.2 파일 구조를 강하게 가정한다

`npz` 내부에 `x`, `y` 배열이 있다고 가정한다.
다른 포맷의 실험용 데이터에는 바로 사용할 수 없다.

### 9.3 `target_hz`는 표시용이다

이 값은 파형의 시간축 계산에 사용된다.
즉, 실제 데이터 생성 시의 샘플링 주파수와 다르게 주면 x축 시간만 잘못 표시된다.

### 9.4 메타데이터 읽기와 실제 케이스 로딩은 분리되어 있다

목록에 세그먼트 수가 보이더라도, 실제 `npz` 내부 데이터 로딩은
케이스 선택 시점에 다시 이루어진다.

### 9.5 대용량 배열은 메모리로 통째로 읽는다

케이스를 선택하면 `x`, `y`를 메모리에 올린다.
대형 `npz` 파일이 많거나 매우 큰 경우 메모리 사용량에 주의해야 한다.

---

## 10. 검증 포인트

문서화 기준으로 점검할 때는 아래를 우선 보면 된다.

- `uv run python scripts/dataset-browser.py --help`가 문서와 일치하는가
- `train / val / test` 폴더가 있을 때 split 버튼이 정상 동작하는가
- 리스트에 세그먼트 수와 파일 크기가 점진적으로 채워지는가
- 케이스 선택 시 PPG 플롯과 `SBP`, `DBP`가 표시되는가
- `← → ↑ ↓` 키가 정상 동작하는가
- Jump 입력이 정상 동작하는가

---

## 11. 요약

`dataset-browser.py`는 학습용 `NPZ` 데이터셋을 사람 눈으로 빠르게 검수하기 위한 브라우저다.

현재 구현의 핵심 특징은 다음과 같다.

- split별 케이스 목록을 한 창에서 탐색할 수 있다.
- 메타데이터 인덱싱을 백그라운드에서 처리해 UI 반응성을 유지한다.
- 선택한 세그먼트의 PPG와 `SBP`, `DBP`를 즉시 확인할 수 있다.
- 슬라이더, Jump, 방향키로 세그먼트 탐색이 빠르다.

전처리 결과가 올바른지, 레이블이 상식적인지, 세그먼트 품질이 어떤지 확인할 때
가장 먼저 열어볼 수 있는 기본 브라우저다.
