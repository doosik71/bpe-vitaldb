# `pulse-browser.py` 사용 및 개발 설명

작성일: 2026-06-08  
관련 코드: [scripts/pulse-browser.py](/home/doosik/work/bpe-vitaldb/scripts/pulse-browser.py:1)  
관련 문서: [docs/dataset-browser.md](/home/doosik/work/bpe-vitaldb/docs/dataset-browser.md:1)

---

## 1. 목적

`scripts/pulse-browser.py`는 `dataset-browser.py`를 확장한 모델 해석 브라우저다.

이 스크립트의 역할은 다음과 같다.

- `NPZ` 데이터셋 세그먼트를 탐색한다.
- `pulsewo_resnet1d`의 최신 `best.pt` 체크포인트를 자동 탐색한다.
- 각 PPG 세그먼트에 대해 `SBP`, `DBP` 예측을 수행한다.
- 모델이 내부적으로 계산한 quality weight를 시각화한다.
- 최종 예측값과 정답값의 차이를 한 화면에서 보여준다.

즉, 이 도구는 단순 데이터 브라우저가 아니라
"모델이 이 세그먼트에서 무엇을 얼마나 신뢰했는지"를 함께 보는 분석용 브라우저다.

---

## 2. 입력과 출력

### 입력

- 데이터셋 루트: 기본값 `data/dataset`
- 모델 루트: 기본값 `data/models`
- 기대 모델 경로:
  - `data/models/pulsewo_resnet1d/best.pt`

데이터셋 파일은 `dataset-browser.py`와 동일하게 다음 형식을 가정한다.

```text
x  float32  (N, samples)
y  float32  (N, 2)
```

### 출력

- 파일을 저장하지 않는다.
- Tkinter + Matplotlib 기반 GUI를 띄운다.
- PPG 파형, quality-weight 시각화, 세그먼트별 예측 분포를 한 화면에 표시한다.

---

## 3. 실행 방법

### 기본 실행

```bash
uv run python scripts/pulse-browser.py
```

### 옵션 포함 실행

```bash
uv run python scripts/pulse-browser.py \
  --dataset-dir data/dataset \
  --models-dir data/models \
  --device cuda \
  --target-hz 125
```

옵션 설명:

- `--dataset-dir`
  - 데이터셋 루트
  - 기본값: `data/dataset`
- `--models-dir`
  - 모델 체크포인트 루트
  - 기본값: `data/models`
- `--device`
  - 추론 디바이스
  - 예: `cpu`, `cuda`, `cuda:0`
  - 기본값: 비우면 `cuda` 가능 시 `cuda`, 아니면 `cpu`
- `--target-hz`
  - 데이터셋의 PPG 샘플링 주파수
  - 기본값: `125`

### 자주 쓰는 예시

CPU로 강제 실행:

```bash
uv run python scripts/pulse-browser.py --device cpu
```

다른 모델 저장소 루트 사용:

```bash
uv run python scripts/pulse-browser.py --models-dir data/models-experiment
```

---

## 4. UI 구성

기본 레이아웃은 `dataset-browser.py`와 비슷하지만,
오른쪽 분석 패널이 훨씬 풍부하다.

### 왼쪽 패널

- split 선택 버튼
- 케이스 수 / 세그먼트 수 상태 라벨
- 케이스 목록
  - `Case ID`
  - `Segments`
  - `Size`
- 모델 상태 블록
  - 체크포인트 탐색 결과
  - 로딩 중 상태
  - 사용 중 run 이름
  - 사용 디바이스 정보

### 오른쪽 패널

- 상단 정보 바
  - 케이스 ID
  - 정답 `SBP`, `DBP`
  - 예측 `SBP`, `DBP`
  - 오차 `ΔS`, `ΔD`
- 중앙 3개 플롯
  - PPG 파형 + quality shading
  - quality-weight triangle chart
  - per-segment `SBP`, `DBP` prediction panel
- 하단 내비게이션 바
  - `Prev`, `Next`
  - 세그먼트 슬라이더
  - Jump 입력

---

## 5. 모델 로딩 방식

### 최신 run 탐색

`find_best_pt(models_dir)`는 다음 경로를 찾는다.

```text
<models-dir>/pulsewo_resnet1d/best.pt
```

`pulsewo_resnet1d` 디렉터리에서 `best.pt`를 직접 찾는다.

### 비동기 로딩

모델 로딩은 `_load_model_async()`가 별도 스레드에서 수행한다.

이 방식의 목적은:

- GUI 시작 지연을 줄이고
- 무거운 `torch.load()` 때문에 창이 멈추는 것을 피하기 위함이다.

로딩이 끝나면:

- 모델 상태 라벨을 갱신하고
- 이미 케이스가 열려 있으면 현재 세그먼트를 다시 렌더링해서
  예측 결과를 즉시 반영한다.

### 체크포인트 복원

`load_model()`은 `PulseWOResNet1D`를 생성한 뒤,
체크포인트에서 다음 우선순위로 state를 찾는다.

- `model_state_dict`
- `model`
- `state_dict`
- 없으면 체크포인트 객체 자체

즉, 학습 저장 포맷이 약간 달라도 어느 정도 흡수하도록 작성돼 있다.

---

## 6. 추론 방식

### 6.1 표준 `forward()`를 직접 쓰지 않는 이유

이 스크립트는 단순히 최종 예측값만 보여주지 않고,
모델 내부의 quality score와 세그먼트별 예측값도 함께 보여줘야 한다.

그래서 `infer_with_weights()`는 모델 내부 계산 그래프를 단계별로 다시 실행한다.

### 6.2 정규화

입력 PPG 세그먼트는 먼저 세그먼트 단위 z-score 정규화를 한다.

```text
x_norm = (ppg - mean) / std
```

`std < 1e-6`이면 0으로 나누지 않도록 `1e-6`으로 보정한다.

### 6.3 내부 세그먼트 분할

모델은 전체 PPG 세그먼트를 다시 더 작은 overlapping sub-segment들로 나눈다.

현재 코드 기준:

- `seg_len = model.seg_len`
- `stride = model.stride`

기본 모델 설명 주석상 대표 값은:

- `seg_len = 125`
- `stride = 62`

즉, 1초 길이 조각을 약 50% overlap으로 펼쳐서 backbone에 넣는다.

### 6.4 quality weight 계산

backbone 출력은 각 sub-segment마다:

- 혈압 예측값 2개
- quality score 1개

를 포함한다.

그 뒤:

- quality score에 `softmax`
- softmax weight로 각 sub-segment 예측을 가중 평균

하여 최종 `SBP`, `DBP`를 계산한다.

즉, 최종 예측은 단순 평균이 아니라
"모델이 더 믿는 sub-segment에 더 큰 가중치를 준 가중 평균"이다.

---

## 7. 플롯 설명

### 7.1 PPG waveform panel

첫 번째 플롯은 원본 PPG 세그먼트를 보여준다.

특징:

- 녹색 PPG 라인
- 파란 shading으로 quality weight 강도 표현
- 가장 큰 weight를 가진 sub-segment를 x축 아래 굵은 파란 막대로 강조
- 우측 상단에 정답과 예측 요약 박스 표시

shading의 `alpha`는 상대 weight 비율에 따라 정해지며,
가장 큰 구간일수록 더 진하게 보인다.

### 7.2 Quality-weight triangle chart

두 번째 플롯은 sub-segment별 softmax weight를 삼각형 모양으로 그린다.

특징:

- 각 삼각형의 높이 = weight percent
- 가장 높은 weight 구간은 더 진한 파란색으로 강조
- apex 위에 raw quality score 숫자 표시
- uniform weight 기준선을 점선으로 표시

이 패널을 통해 모델이 특정 구간에 과도하게 집중하는지,
아니면 전체 구간을 고르게 보는지 확인할 수 있다.

### 7.3 Per-segment SBP / DBP predictions

세 번째 플롯은 sub-segment별 혈압 예측값을 보여준다.

표시 요소:

- 각 sub-segment 중심 시점에서의 `SBP`, `DBP` 점/선
- 각 점 옆 숫자 라벨
- 정답 `SBP`, `DBP` 수평 점선
- 최종 가중 평균 예측값 수평 점선

이 패널은 다음 해석에 유용하다.

- sub-segment별 예측 분산이 큰지
- quality weight가 최종 평균을 어디로 끌고 가는지
- 오차가 특정 몇 개 구간 때문인지

---

## 8. 내비게이션

조작 방식은 `dataset-browser.py`와 동일하다.

### 마우스

- 리스트에서 케이스 선택
- `Prev` / `Next`
- 세그먼트 슬라이더
- Jump 입력

### 키보드

- `←` / `→`: 이전 / 다음 세그먼트
- `↑` / `↓`: 이전 / 다음 케이스

---

## 9. 처리 흐름

### 9.1 데이터셋 메타데이터 인덱싱

`dataset-browser.py`와 동일하게 별도 스레드에서 `npz` 메타데이터를 읽는다.
세그먼트 수와 파일 크기를 먼저 목록에 채워 넣는다.

### 9.2 케이스 로드

케이스를 선택하면 `x`, `y`를 메모리에 읽고,
첫 세그먼트를 즉시 렌더링한다.

### 9.3 세그먼트 렌더링

`_show_segment(idx)`는:

1. 현재 PPG 세그먼트와 정답 혈압을 읽고
2. 모델이 준비돼 있으면 `infer_with_weights()`를 호출하고
3. 정보 바와 3개 플롯을 모두 갱신하고
4. 상태 바와 버튼 상태를 갱신한다.

### 9.4 모델이 없을 때의 동작

모델 체크포인트가 없거나 로딩 실패 시에도 브라우저는 동작한다.

이 경우:

- 정답 파형 탐색은 계속 가능
- 예측값은 비우거나 `no model`로 표시
- 두 번째, 세 번째 패널에 `No model loaded` 메시지 표시

즉, 모델 의존성이 강제되지 않는 분석 도구다.

---

## 10. 함수별 설명

### `find_best_pt()`

최신 run에서 `best.pt`를 찾는다.

### `load_model()`

체크포인트를 `PulseWOResNet1D`에 복원한다.

### `infer_with_weights()`

세그먼트 정규화, 내부 unfold, backbone 실행, softmax weight 계산,
최종 예측값 계산까지 담당하는 핵심 추론 함수다.

### `PulseBrowser.__init__()`

파일 탐색, UI 생성, 비동기 모델 로딩, split 초기화, 메타데이터 로더 시작을 담당한다.

### `_load_model_async()` / `_model_loader_thread()`

백그라운드 모델 로딩을 담당한다.

### `_show_segment()`

정답/예측/weight 시각화를 모두 처리하는 중심 렌더링 함수다.

---

## 11. 상태 표시와 사용자 피드백

상태 바는 다음 정보를 보여준다.

- 메타데이터 인덱싱 진행 상황
- 케이스 로딩 메시지
- 추론 오류 메시지
- 현재 케이스, 세그먼트, 샘플 수, 정답 혈압, 예측 오차

오차 표시 색상 규칙:

- 절대 오차 `<= 5`: 녹색
- 절대 오차 `<= 10`: 주황색
- 그보다 크면 빨간색

즉, 정보 바만 봐도 현재 샘플의 예측 품질을 빠르게 읽을 수 있다.

---

## 12. 개발 시 알아둘 제약과 주의점

### 12.1 모델 이름이 고정돼 있다

현재 구현은 `MODEL_NAME = "pulsewo_resnet1d"`로 고정돼 있다.
다른 모델 브라우징에는 그대로 재사용할 수 없다.

### 12.2 `target_hz`와 모델 내부 `seg_len/stride`의 조합을 암묵적으로 가정한다

시각화는 `target_hz`와 모델 내부 sub-segment 길이가 시간축에 맞는다고 가정한다.
훈련과 다른 샘플링 주파수의 데이터셋에 그대로 쓰면 해석이 어긋날 수 있다.

### 12.3 현재 세그먼트 전체를 메모리에서 즉시 추론한다

한 세그먼트 추론은 가볍지만, 케이스를 넘길 때마다 새 세그먼트마다 추론이 다시 돈다.
GPU가 없거나 느린 환경에서는 탐색 반응성이 떨어질 수 있다.

### 12.4 `sys.path`를 직접 수정한다

스크립트는 `bpe` 패키지 import를 위해 저장소 루트를 `sys.path` 앞에 넣는다.
스크립트형 도구로는 실용적이지만, 패키징 관점에서는 강한 결합이다.

### 12.5 quality 시각화는 모델 구조에 의존적이다

이 브라우저는 backbone 출력 형식과 `out_features`, quality head 위치를 알고 있다는 전제 위에 서 있다.
모델 구조가 바뀌면 `infer_with_weights()`도 함께 수정해야 한다.

---

## 13. 검증 포인트

문서화 기준으로 점검할 때는 아래를 우선 보면 된다.

- `uv run python scripts/pulse-browser.py --help`가 문서와 일치하는가
- 모델 체크포인트가 있을 때 자동 로딩되는가
- 체크포인트가 없을 때도 브라우저가 죽지 않고 열리는가
- 세그먼트 선택 시 예측값과 오차가 정보 바에 표시되는가
- quality shading과 triangle chart가 함께 갱신되는가
- `← → ↑ ↓`와 Jump 입력이 정상 동작하는가

---

## 14. 요약

`pulse-browser.py`는 데이터셋 브라우저에 모델 예측과 내부 quality-weight 해석을 추가한 분석 도구다.

현재 구현의 핵심 특징은 다음과 같다.

- 최신 `pulsewo_resnet1d` 체크포인트를 자동 탐색한다.
- 세그먼트 단위 정답과 예측을 즉시 비교한다.
- softmax quality weight를 시간축 위에 직관적으로 시각화한다.
- sub-segment별 예측 분포까지 한 화면에서 확인할 수 있다.

모델이 틀린 이유를 정성적으로 파악하거나,
quality weighting 메커니즘이 실제로 어떻게 작동하는지 확인할 때 가장 유용한 브라우저다.
