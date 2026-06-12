# `psd-browser.py` 사용 및 개발 설명

작성일: 2026-06-12  
관련 코드: [scripts/psd-browser.py](../scripts/psd-browser.py)  
관련 문서: [docs/dataset-browser.md](../docs/dataset-browser.md), [notebook/psd-analysis.ipynb](../notebook/psd-analysis.ipynb)

## 1. 목적

`scripts/psd-browser.py`는 `NPZ` 데이터셋 세그먼트의 주파수 특성을 시각적으로 점검하는 GUI 브라우저다.

이 스크립트의 역할은 다음과 같다.

- `train / val / test` split별 케이스를 탐색한다.
- 선택한 세그먼트의 PPG waveform을 보여준다.
- Welch PSD를 계산해 주파수 스펙트럼을 그린다.
- `power_ratio = Power(0.67–3.0 Hz) / Power(0.5–10.0 Hz)`를 계산해 표시한다.
- `power_ratio` 범위별 세그먼트 목록을 제공해 빠르게 점프할 수 있게 한다.

즉, 이 도구는 단순 파형 브라우저가 아니라
"어떤 세그먼트가 심박 대역 에너지 중심인지"를 빠르게 확인하는 품질 분석 브라우저다.

## 2. 입력과 출력

### 입력

- 데이터셋 루트: 기본값 `data/dataset`
- 기대 구조
  - `data/dataset/train/*.npz`
  - `data/dataset/val/*.npz`
  - `data/dataset/test/*.npz`

각 `npz` 파일은 다음 배열을 포함해야 한다.

```text
x  float32  (N, samples)
y  float32  (N, 2)
```

### 출력

- 파일을 저장하지 않는다.
- Tkinter + Matplotlib 기반 GUI를 띄운다.
- 중앙 패널에 waveform과 PSD를 동시에 표시한다.
- 우측 패널에 `power_ratio` 범위 기반 세그먼트 목록을 표시한다.

## 3. 실행 방법

### 기본 실행

```bash
uv run python scripts/psd-browser.py
```

### 옵션 포함 실행

```bash
uv run python scripts/psd-browser.py \
  --dataset-dir data/dataset \
  --target-hz 125 \
  --nperseg 256
```

옵션 설명:

- `--dataset-dir`
  - 데이터셋 루트
  - 기본값: `data/dataset`
- `--target-hz`
  - PPG 샘플링 주파수
  - 기본값: `125`
- `--nperseg`
  - Welch PSD 계산 구간 길이
  - 기본값: `256`

## 4. UI 구성

브라우저는 3열 구조다.

### 왼쪽 패널

- split 선택 버튼
- 케이스 수 / 세그먼트 수 상태 라벨
- 케이스 목록
  - `Case ID`
  - `Segments`
  - `Size`

### 중앙 패널

- 상단 정보 바
  - 케이스 ID
  - `SBP`, `DBP`
  - 현재 세그먼트 `power_ratio`
  - 현재 케이스의 ratio 요약
- 상단 플롯
  - 선택한 세그먼트의 PPG waveform
- 하단 플롯
  - Welch PSD
  - `0.5–10.0 Hz` / `0.67–3.0 Hz` 강조 영역
  - 밴드 파워와 `power_ratio` 요약 박스
- 하단 내비게이션 바
  - `Prev`, `Next`
  - 슬라이더
  - Jump 입력창

### 오른쪽 패널

- `power_ratio` 구간 선택 라디오 버튼
- 현재 구간에 속한 세그먼트 목록
- 세그먼트 개수 상태 라벨

현재 구간 목록 항목을 클릭하면 중앙 그래프가 해당 세그먼트로 즉시 이동한다.

## 5. `power_ratio` 정의

이 브라우저는 다음 값을 핵심 지표로 사용한다.

```text
power_ratio = Power(0.67–3.0 Hz) / Power(0.5–10.0 Hz)
```

의미:

- 분자: 대략 정상 심박 성분이 많이 위치하는 주파수 대역
- 분모: 저주파 drift와 고주파 잡음을 제외한 전체 유효 대역

`power_ratio`가 높을수록 대체로 심박성 주기 성분이 강하고,
낮을수록 artifact 또는 비심박성 에너지가 상대적으로 많을 가능성이 있다.

## 6. 처리 흐름

### 6.1 파일 인덱싱

`dataset-browser.py`와 같은 방식으로 split별 `NPZ`를 탐색하고,
백그라운드 스레드에서 세그먼트 수와 파일 크기를 읽어 리스트를 갱신한다.

### 6.2 케이스 로드

케이스 선택 시 `x`, `y`를 메모리에 로드한다.
로드 후 다음 작업을 수행한다.

- 현재 세그먼트 인덱스를 0으로 초기화
- 케이스 내 모든 세그먼트의 `power_ratio`를 계산해 캐시
- 우측 패널의 현재 ratio 구간 목록 갱신
- 첫 세그먼트 렌더링

### 6.3 PSD 계산

각 세그먼트에 대해 `scipy.signal.welch`를 사용한다.

핵심 설정:

- window: `hann`
- scaling: `density`
- `nperseg = min(len(signal), user_nperseg)`
- 적분: `np.trapezoid`

이후 밴드 파워를 적분해 `power_ratio`를 계산한다.

### 6.4 빠른 세그먼트 탐색

우측 패널은 현재 케이스의 cached `power_ratio` 배열을 이용한다.

선택 가능한 구간:

- `0.0 ~ 0.4`
- `0.4 ~ 0.5`
- `0.5 ~ 0.6`
- `0.6 ~ 0.7`
- `0.7 ~ 0.8`
- `0.8 ~ 0.9`
- `0.9 ~ 1.0`

구간 선택 시 조건에 맞는 세그먼트 번호와 ratio가 리스트로 갱신된다.

## 7. 내비게이션

### 마우스 조작

- 케이스 목록에서 케이스 선택
- `Prev` / `Next` 버튼으로 세그먼트 이동
- 슬라이더 이동
- Jump 입력 후 `Enter`
- 우측 `power_ratio` 목록에서 세그먼트 선택

### 키보드 단축키

- `←` / `→`: 이전 / 다음 세그먼트
- `↑` / `↓`: 이전 / 다음 케이스

## 8. 사용 목적과 해석

이 브라우저는 특히 다음 상황에서 유용하다.

- 잡음이 많은 세그먼트 찾기
- 심박 대역 에너지가 약한 세그먼트 찾기
- `power_ratio >= 0.6` 같은 필터 기준을 눈으로 점검하기
- 특정 케이스에서 좋은/나쁜 세그먼트가 어떻게 섞여 있는지 보기

즉, `construct-dataset-v1.py` 같은 ratio 기반 필터링의 타당성을
시각적으로 확인하는 데 적합하다.
