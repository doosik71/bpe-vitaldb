# `spectro-browser.py` 사용 및 개발 설명

작성일: 2026-06-12  
관련 코드: [scripts/spectro-browser.py](../scripts/spectro-browser.py)  
관련 문서: [docs/psd-browser.md](../docs/psd-browser.md), [docs/dataset-browser.md](../docs/dataset-browser.md)

## 1. 목적

`scripts/spectro-browser.py`는 PPG 세그먼트의 시간-주파수 변화를 보는 GUI 브라우저다.

이 스크립트의 역할은 다음과 같다.

- `NPZ` 데이터셋 세그먼트를 탐색한다.
- 선택한 세그먼트의 waveform을 보여준다.
- spectrogram을 계산해 시간에 따라 주파수 에너지가 어떻게 바뀌는지 시각화한다.
- dominant frequency trace를 함께 그린다.
- `power_ratio`를 계산하고, ratio 구간별 빠른 세그먼트 이동 기능을 제공한다.

즉, 이 도구는 PSD 브라우저의 확장판으로,
"전체 주파수 분포"뿐 아니라 "주파수 성분이 시간에 따라 흔들리는 양상"을 보려는 목적에 맞는다.

## 2. 입력과 출력

### 입력

- 데이터셋 루트: 기본값 `data/dataset`
- 기대 구조
  - `data/dataset/train/*.npz`
  - `data/dataset/val/*.npz`
  - `data/dataset/test/*.npz`

각 `npz`는 다음 배열을 포함해야 한다.

```text
x  float32  (N, samples)
y  float32  (N, 2)
```

### 출력

- 파일을 저장하지 않는다.
- Tkinter + Matplotlib GUI를 띄운다.
- 중앙 패널에 waveform과 spectrogram을 표시한다.
- 우측 패널에 `power_ratio` 구간 기반 빠른 이동 리스트를 표시한다.

## 3. 실행 방법

### 기본 실행

```bash
uv run python scripts/spectro-browser.py
```

### 옵션 포함 실행

```bash
uv run python scripts/spectro-browser.py \
  --dataset-dir data/dataset \
  --target-hz 125 \
  --nperseg 128 \
  --noverlap 64
```

옵션 설명:

- `--dataset-dir`
  - 데이터셋 루트
  - 기본값: `data/dataset`
- `--target-hz`
  - PPG 샘플링 주파수
  - 기본값: `125`
- `--nperseg`
  - spectrogram / Welch 분석 창 길이
  - 기본값: `128`
- `--noverlap`
  - 인접 spectrogram 창 사이 overlap 길이
  - 기본값: `nperseg // 2`

## 4. UI 구성

기본 레이아웃은 `psd-browser.py`와 동일한 3열 구조다.

### 왼쪽 패널

- split 선택
- 케이스 목록
- 세그먼트 수 / 파일 크기 정보

### 중앙 패널

- 상단 정보 바
  - 케이스 ID
  - `SBP`, `DBP`
  - 현재 세그먼트 `power_ratio`
  - 케이스 ratio 요약
- 상단 플롯
  - PPG waveform
- 하단 플롯
  - spectrogram (`0.5–10.0 Hz` 대역 중심)
  - dominant frequency trace
  - colorbar
  - 밴드 파워, `power_ratio`, dominant frequency 평균/표준편차 요약 박스
- 하단 내비게이션 바
  - `Prev`, `Next`, 슬라이더, Jump 입력

### 오른쪽 패널

- `power_ratio` 구간 선택
- 해당 구간에 속하는 세그먼트 목록
- 상태 라벨

## 5. 왜 spectrogram이 필요한가

PSD는 세그먼트 전체를 하나의 평균 스펙트럼으로 요약한다.
따라서 다음 정보는 잘 드러나지 않는다.

- 시간에 따라 심박 주파수가 이동하는지
- 특정 구간에서만 artifact가 생기는지
- 세그먼트 앞부분/뒷부분의 품질 차이
- 일시적인 주파수 burst나 dropout

spectrogram은 시간축과 주파수축을 동시에 보여주기 때문에,
이런 비정상 패턴을 더 쉽게 찾을 수 있다.

## 6. 계산 방식

### 6.1 `power_ratio`

`power_ratio` 계산은 PSD 브라우저와 동일하다.

```text
Power(0.67–3.0 Hz) / Power(0.5–10.0 Hz)
```

즉, 빠른 세그먼트 탐색 기준은 spectrogram 전용 지표가 아니라
기존 PSD 기반 지표를 그대로 재사용한다.

### 6.2 spectrogram

`spectrogram()`은 `scipy.signal.spectrogram`을 사용한다.

핵심 설정:

- window: `hann`
- mode: `psd`
- scaling: `density`
- `nperseg = min(len(signal), user_nperseg)`
- `noverlap = min(user_noverlap, nperseg - 1)`

출력은 다음과 같다.

- `freqs`: 주파수 축
- `times`: 시간 축
- `sxx`: 시간-주파수 PSD 행렬

시각화는 `0.5–10.0 Hz` 대역만 잘라서 그린다.

### 6.3 dominant frequency trace

각 시간 프레임에서 `0.5–10.0 Hz` 대역 안의 최대 에너지를 갖는 주파수를 선택한다.
이 값들을 시간축으로 이은 것이 dominant frequency trace다.

이 값은 다음 해석에 유용하다.

- 심박 관련 주파수가 안정적인지
- 갑작스러운 흔들림이 있는지
- artifact 때문에 peak 주파수가 튀는지

## 7. colorbar 처리

초기 구현에서는 세그먼트를 바꿀 때 colorbar를 제거 후 재생성했다.
하지만 Matplotlib axes 관리와 충돌해 예외가 발생할 수 있었다.

현재 구현은 다음 정책을 사용한다.

- 첫 렌더링 시 colorbar 생성
- 이후 세그먼트 변경 시 `update_normal(mesh)`로 재사용

이 방식은 더 안정적이고,
세그먼트 연속 이동 시 Tkinter callback 오류를 피한다.

## 8. 내비게이션

### 마우스 조작

- 케이스 목록 선택
- `Prev` / `Next`
- 세그먼트 슬라이더
- Jump 입력
- 우측 ratio 목록 클릭

### 키보드 단축키

- `←` / `→`: 이전 / 다음 세그먼트
- `↑` / `↓`: 이전 / 다음 케이스

즉, `psd-browser.py`와 거의 같은 사용감을 유지한다.

## 9. 추천 사용 시나리오

다음 상황에서 특히 유용하다.

- 평균 PSD는 괜찮아 보이는데 일부 시간 구간만 이상한 경우
- artifact가 연속 구간으로 들어오는 경우
- dominant frequency가 시간에 따라 불안정하게 흔들리는 세그먼트를 찾고 싶은 경우
- `power_ratio`는 비슷하지만 시간적 안정성이 다른 세그먼트를 비교하고 싶은 경우

즉, `psd-browser.py`가 정적 주파수 요약 도구라면,
`spectro-browser.py`는 동적 주파수 분석 도구라고 볼 수 있다.
