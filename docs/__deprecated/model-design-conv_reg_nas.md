# ConvReg NAS 모델 설계서

## 1. 개요

ConvReg NAS는 기존 [`bpe/models/conv_reg.py`](../bpe/models/conv_reg.py)의
단순한 1D CNN 회귀 구조를 유지하면서, 일부 아키텍처 선택을 학습으로 자동화하기
위한 NAS(Network Architecture Search) 확장 모델이다.

- **구현 대상 파일**: `bpe/models/conv_reg_nas.py`
- **예상 등록명**: `conv_reg_nas`
- **입력**: PPG 세그먼트 `(B, 1000)` 또는 `(B, 1, 1000)`
- **출력**: `[SBP, DBP]` 형태의 `(B, 2)`
- **목표**:
  - 기존 학습 및 평가 CLI 인터페이스를 유지한다.
  - `conv_reg`의 stage 수(6개)와 전체 데이터 흐름은 유지한다.
  - 커널 크기, 채널 폭, 컨볼루션 타입 선택을 학습 가능한 구조로 바꾼다.
  - 최종 평가 시에는 supernet 전체가 아니라 하나의 best subnet만 활성화한다.

이 모델은 전형적인 DARTS-style micro search 하나만 사용하는 구조가 아니다.
대신 아래 두 수준의 탐색을 결합한다.

1. **Micro-level search**
각 stage 내부에서 `kernel size = 3 / 5 / 7` 후보를 선택한다.

2. **Architecture-level search**
`channel multiplier`와 `conv type` 조합으로 정의되는 완성형 backbone 후보들 중
가장 적합한 구조를 선택한다.


## 2. 설계 목표

### 2.1 기존 인터페이스 유지

현재 학습/평가 파이프라인은 모델 이름만으로 모델을 생성한다.

- 학습: `create_model(args.model)`
- 평가: `config.json`의 `"model"` 값을 읽고 동일한 이름으로 재생성
- 체크포인트: `state_dict` 기반 저장/복원

따라서 `conv_reg_nas` 역시 **인자 없는 생성자**로 생성 가능해야 하며,
모델 내부에 architecture parameter를 포함한 전체 상태가 저장되어야 한다.

### 2.2 최소 변경

다음은 유지한다.

- stage 개수: 6
- 각 stage 뒤의 `AvgPool1d(2)`
- 마지막 전역 feature 축소 후 2층 regressor
- 기존 `train-model.py`, `eval-model.py` 사용 방식

다음은 확장한다.

- stage별 kernel 선택
- backbone별 channel width 선택
- backbone별 convolution type 선택

### 2.3 평가 시 빠른 실행

학습 시에는 supernet으로 여러 후보를 동시에 고려하되,
평가 및 추론 시에는 학습된 architecture parameter로부터
**하나의 best subnet architecture만 hard selection**하여 실행한다.

이로써 기존 `best.pt` 저장 방식은 유지하면서도,
평가 속도는 일반 단일 모델에 가깝게 만들 수 있다.


## 3. 전체 구조 요약

ConvReg NAS는 다음 두 계층으로 구성된다.

```text
입력 PPG
   │
   ▼
Architecture-level Supernet
   ├─ Backbone A: standard conv, 1.0x channels
   ├─ Backbone B: standard conv, 1.5x channels
   ├─ Backbone C: standard conv, 2.0x channels
   ├─ Backbone D: depthwise separable conv, 1.0x channels
   ├─ Backbone E: depthwise separable conv, 1.5x channels
   └─ Backbone F: depthwise separable conv, 2.0x channels
            │
            ▼
각 backbone 내부 6개 stage
   └─ 각 stage는 kernel choice {3, 5, 7}
            │
            ▼
학습 시:
   backbone 혼합 + stage별 kernel 혼합

평가 시:
   backbone 하나 선택 + stage별 kernel 하나씩 선택
            │
            ▼
최종 회귀 출력 (SBP, DBP)
```


## 4. 탐색 공간 정의

## 4.1 Stage 수

기존 `conv_reg`와 동일하게 6개 stage를 유지한다.

시간 축 변화도 동일하다.

```text
1000 → 500 → 250 → 125 → 62 → 31 → 15
```

각 stage는 다음 순서를 따른다.

```text
Conv block → BatchNorm1d → ReLU → AvgPool1d(2)
```

마지막에는 전역 요약을 위해 `AdaptiveAvgPool1d(1)`을 사용한다.


## 4.2 Kernel size 탐색

각 stage 내부에서 아래 세 후보 중 하나를 선택한다.

- `k = 3`
- `k = 5`
- `k = 7`

패딩은 시간 길이 보존을 위해 다음 규칙을 사용한다.

```text
padding = kernel_size // 2
```

즉, 각 candidate conv는 pooling 전까지 동일한 시간 길이를 유지한다.

### 설계 의도

- `k=3`: 더 국소적인 파형 변화 반영
- `k=5`: 기존 `conv_reg`의 기준 설정과 유사
- `k=7`: 더 넓은 문맥 반영


## 4.3 Channel multiplier 탐색

기존 `conv_reg`의 base 채널 구성은 다음과 같다.

```text
[8, 16, 32, 64, 64, 64]
```

Backbone 단위에서 이 base 채널에 multiplier를 적용한다.

- `1.0x`
- `1.5x`
- `2.0x`

예시:

| base | 1.0x | 1.5x | 2.0x |
| ---- | ---- | ---- | ---- |
| 8    | 8    | 12   | 16   |
| 16   | 16   | 24   | 32   |
| 32   | 32   | 48   | 64   |
| 64   | 64   | 96   | 128  |

본 설계에서는 multiplier 결과를 backbone 정의 시 고정한다.
즉, stage 간 채널 mismatch를 supernet 내부에서 동적으로 해소하지 않고,
각 backbone이 처음부터 끝까지 일관된 채널 구성을 갖는다.

이 점이 본 설계가 구현 난이도를 낮추는 핵심이다.


## 4.4 Convolution type 탐색

Backbone 단위에서 아래 두 가지 conv block 타입을 후보로 둔다.

1. **Standard Conv1d**
일반 `nn.Conv1d`

2. **Depthwise Separable Conv1d**
depthwise conv + pointwise conv 조합

depthwise separable conv는 이미
[`bpe/models/conv_reg_ds.py`](../bpe/models/conv_reg_ds.py)에서 사용 중인
개념을 재사용할 수 있다.


## 4.5 Architecture-level backbone 후보

`conv type × channel multiplier` 조합으로 backbone 후보를 정의한다.

총 후보 수는 6개다.

| Backbone ID | Conv type | Channel multiplier |
| ----------- | --------- | ------------------ |
| A           | standard  | 1.0x               |
| B           | standard  | 1.5x               |
| C           | standard  | 2.0x               |
| D           | depthwise separable | 1.0x    |
| E           | depthwise separable | 1.5x    |
| F           | depthwise separable | 2.0x    |

각 backbone은 이미 “완성된 모델 구조”에 해당한다.
따라서 본 설계는 stage 내부 연산 선택만 하는 순수 micro-search보다,
**architecture-level supernet** 성격이 더 강하다.


## 5. 왜 hybrid search를 택하는가

## 5.1 순수 micro-search의 문제

만약 `kernel size`, `channel`, `conv type`을 모두 stage 내부 후보로 두면
다음 문제가 생긴다.

- stage 간 채널 연결이 복잡해진다.
- depthwise separable conv와 standard conv를 같은 텐서 흐름에서
  공유하려면 구현 복잡도가 커진다.
- supernet이 지나치게 세밀해지고 shape 관리가 어려워진다.

## 5.2 순수 architecture-level search의 한계

반대로 모든 kernel 조합까지 backbone 단위 후보로 만들면,
후보 수가 과도하게 증가한다.

stage가 6개이고 stage별 kernel 후보가 3개이므로

```text
3^6 = 729
```

conv type 2개, multiplier 3개까지 곱하면 총 후보 수는

```text
729 × 2 × 3 = 4,374
```

이 수는 완성형 backbone 후보로 다루기에는 비효율적이다.

## 5.3 Hybrid search의 장점

따라서 아래 방식이 가장 균형이 좋다.

- `kernel size`: stage 내부 micro search
- `channel multiplier`, `conv type`: backbone 수준 architecture search

이렇게 하면

- kernel 탐색의 유연성은 유지하고
- 채널/conv 구조 변화에 따른 shape 복잡성은 backbone 내부로 격리하며
- supernet 구현을 상대적으로 단순하게 유지할 수 있다.


## 6. 제안 모듈 구조

아래와 같은 내부 모듈 구성을 제안한다.

### 6.1 `KernelChoiceConvBlock`

한 stage 내부에서 `k=3/5/7` 후보 conv를 관리하는 블록.

역할:

- 동일한 in/out channel 조건 하에서 kernel 후보 3개 보유
- architecture parameter를 이용해 soft selection 또는 hard selection 수행
- 이후 `BatchNorm1d`, `ReLU`, `AvgPool1d(2)` 적용

지원 모드:

- `soft`: 학습용, 후보 출력의 가중합
- `hard`: 평가용, 최고 점수 후보만 실행

### 6.2 `ConvRegNasBackbone`

특정 `(conv type, channel multiplier)` 조합에 해당하는 완성형 backbone.

역할:

- 6개 stage를 가짐
- 각 stage는 `KernelChoiceConvBlock`
- 마지막 `AdaptiveAvgPool1d(1)` 수행
- 채널 수는 backbone 전체에서 일관되게 고정

즉 backbone은 다음 정보를 고정한다.

- conv type
- channel plan

다음 정보는 stage별로 학습한다.

- kernel choice

### 6.3 `ConvRegNas`

전체 architecture-level supernet.

역할:

- 6개 backbone 후보 보유
- backbone selection용 architecture parameter 보유
- 학습 시 backbone 출력들을 soft selection으로 혼합
- 평가 시 최고 score backbone 하나만 선택
- 공통 regressor 또는 backbone별 regressor 전략 중 하나 선택

본 설계에서는 **backbone별 feature 차원이 달라질 수 있으므로**,
최종 `AdaptiveAvgPool1d(1)` 후의 feature dimension을 공통으로 맞추는 장치가 필요하다.

가장 단순한 방법은 각 backbone의 마지막 feature를
별도 projection으로 공통 차원으로 사상한 뒤, 공통 regressor에 넣는 것이다.

예시:

```text
backbone feature -> Linear/1x1 conv projection -> shared regressor
```

또는 backbone마다 regressor를 따로 둘 수도 있으나,
최소 변경과 비교 용이성을 위해 공통 regressor가 더 적합하다.


## 7. 텐서 흐름 예시

학습 시 `soft` 모드의 개념적 흐름은 다음과 같다.

```text
x
 ├─ backbone A -> feature A -> projection A -> zA
 ├─ backbone B -> feature B -> projection B -> zB
 ├─ backbone C -> feature C -> projection C -> zC
 ├─ backbone D -> feature D -> projection D -> zD
 ├─ backbone E -> feature E -> projection E -> zE
 └─ backbone F -> feature F -> projection F -> zF

architecture softmax over backbones = [wA, ..., wF]

z = wA·zA + wB·zB + ... + wF·zF
pred = shared_regressor(z)
```

각 backbone 내부 stage의 kernel choice도 같은 방식으로 soft selection한다.

평가 시 `hard` 모드는 다음과 같다.

```text
x
 -> best backbone only
 -> each stage uses best kernel only
 -> projection
 -> shared_regressor
 -> pred
```


## 8. 학습 방식

## 8.1 기본 원칙

- train set만 사용해 weight parameter와 architecture parameter를 함께 학습한다.
- 별도의 bilevel optimization은 사용하지 않는다.
- 기존 `Trainer` 흐름을 최대한 유지한다.

즉, 본 모델은 “full NAS search phase + retrain phase” 방식이 아니라,
**one-shot joint training supernet**에 가깝다.

## 8.2 Architecture parameter

학습 대상 파라미터는 크게 두 종류다.

1. **Weight parameter**
conv, batchnorm, projection, regressor 등의 일반 가중치

2. **Architecture parameter**
- backbone 선택 logits
- 각 backbone 각 stage의 kernel 선택 logits

## 8.3 Forward 모드

학습 중:

- backbone: soft selection
- kernel: soft selection

평가 중:

- backbone: hard argmax
- kernel: hard argmax

## 8.4 Loss

기본 예측 loss는 기존과 동일하게 회귀 손실을 사용한다.

- 현재 프로젝트 기준: `HuberLoss(delta=5.0)`

필요하면 architecture regularization을 추가할 수 있다.
예를 들어:

- entropy penalty
- FLOPs/parameter penalty
- large-model preference 억제용 complexity penalty

하지만 1차 구현에서는 최소 변경 원칙에 따라
**예측 loss만 사용하는 단순 버전**이 적합하다.


## 9. 평가 및 추론 방식

## 9.1 목표

평가 단계에서는 supernet 전체를 다 돌리지 않고,
가장 점수가 높은 subnet architecture만 활성화한다.

## 9.2 Best subnet 추출 기준

평가 시점에 architecture logits에 대해 argmax를 취한다.

- backbone logits -> best backbone 1개 선택
- 각 stage kernel logits -> best kernel 1개씩 선택

## 9.3 실행 방식

방법은 두 가지가 있다.

1. `forward()` 내부에서 `self.training` 또는 별도 플래그를 기준으로
   soft/hard 실행을 분기
2. checkpoint 로드 후 `freeze_best_subnet()` 같은 메서드로
   내부 선택을 고정

최소 변경 관점에서는 `forward()`가 평가 모드에서 자동으로 hard selection을 수행하는
구조가 가장 단순하다.

즉:

- `model.train()` 상태: soft supernet
- `model.eval()` 상태: hard best subnet

이 방식이면 기존 `eval-model.py`를 거의 수정하지 않아도 된다.


## 10. 기존 인터페이스와의 호환성

## 10.1 학습

기존과 동일하게 실행 가능해야 한다.

```bash
bin/train-model --model conv_reg_nas
```

또는

```bash
uv run python scripts/train-model.py --model conv_reg_nas
```

## 10.2 평가

기존과 동일하게 실행 가능해야 한다.

```bash
bin/eval-model data/models/conv_reg_nas
```

또는

```bash
uv run python scripts/eval-model.py data/models/conv_reg_nas
```

`config.json`에는 기존처럼 `"model": "conv_reg_nas"`만 저장하면 된다.
architecture parameter와 weight parameter는 모두 model state에 포함된다.

## 10.3 체크포인트

현재 trainer는 `model.state_dict()`를 그대로 저장한다.
따라서 architecture logits가 `nn.Parameter`로 등록되어 있다면
별도 포맷 확장 없이 저장/복원이 가능하다.


## 11. 장점

### 11.1 구현 단순성

- 채널 변화와 conv 방식 변화를 backbone 후보로 분리해 shape 관리가 쉬워진다.
- 기존 registry, trainer, evaluator 구조를 크게 바꾸지 않아도 된다.

### 11.2 기존 baseline과 비교 용이

- stage 수와 pooling 구조를 그대로 유지한다.
- `conv_reg`, `conv_reg_ds`, `conv_reg_nas`를 비교하기 쉽다.

### 11.3 빠른 평가 가능

- 학습 후에는 단일 subnet만 실행하므로 추론 비용을 줄일 수 있다.


## 12. 한계와 주의점

### 12.1 학습 비용 증가

학습 시 여러 backbone 후보와 kernel 후보를 함께 고려하므로,
기존 `conv_reg` 대비 메모리와 연산량이 증가한다.

### 12.2 Train-only architecture learning

architecture parameter를 validation set이 아닌 train set만으로 학습하므로,
탐색 편향이나 과적합 위험이 있다.

### 12.3 후보 간 weight sharing의 한계

완성형 backbone 후보를 병렬로 두는 구조이므로,
micro-level NAS보다 weight sharing 효율은 낮을 수 있다.

### 12.4 최적 subnet 보장 아님

one-shot supernet에서 argmax로 뽑은 subnet이
진정한 독립 최적 구조와 항상 일치한다고 보장할 수는 없다.


## 13. 구현 우선순위

1. `conv_reg_nas.py`에 backbone 후보 6개와 stage별 kernel choice 구현
2. 학습 시 soft selection, 평가 시 hard selection 동작 구현
3. 기존 `train-model.py` / `eval-model.py`에서 `conv_reg_nas` 호환 확인
4. 필요 시 architecture summary 출력 유틸리티 추가


## 14. 결론

본 설계는 다음 요구사항을 만족한다.

- 기존 CLI 인터페이스 유지
- `conv_reg`의 구조적 단순성 유지
- kernel, channel, conv type 탐색 가능
- 평가 시 best subnet만 활성화하는 빠른 실행 가능

특히 이 설계의 핵심은 다음 한 문장으로 요약된다.

```text
kernel search는 stage 내부 micro-search로,
channel/conv-type search는 완성형 backbone 후보를 고르는
architecture-level supernet으로 구현한다.
```

이는 구현 복잡도, 기존 코드와의 호환성, 탐색 유연성 사이에서
가장 현실적인 절충안이다.
