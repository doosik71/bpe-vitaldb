# MInception 모델 상세 설계서

## 1. 개요

MInception(Multi-scale Inception 1D CNN)은 Inception 아키텍처의 핵심 아이디어인
**병렬 다중 스케일 합성곱**을 1D PPG 시계열에 적용한 혈압 회귀 모델이다.

- **구현 파일**: [`bpe/models/minception.py`](../bpe/models/minception.py)
- **모델 이름**: `minception` (레지스트리 등록)
- **설계 동기**: PPG 파형에는 심박 주기(~0.1 s), 맥파 형태(~0.05 s), 저주파 추세
  (~수 초) 등 서로 다른 시간 규모의 특징이 공존한다. 단일 커널 크기 합성곱은 한
  가지 규모만 포착하므로, 크기가 다른 여러 커널을 **병렬**로 적용해 다중 스케일
  특징을 동시에 추출한다.

## 2. 전체 아키텍처

```text
입력: PPG 세그먼트
                  (B, 1000)  또는  (B, 1, 1000)
                            │
                            ▼  ensure_3d
                       (B, 1, 1000)
                            │
┌───────────────────────────┴──────────────────────────────┐
│  MInceptionBackbone                                      │
│                                                          │
│  InceptionBlock1D(1→48)                                  │
│   └─ MaxPool1d(k=3, stride=2)  → (B, 48, 500)            │
│                                                          │
│  InceptionBlock1D(48→64)                                 │
│   └─ MaxPool1d(k=3, stride=2)  → (B, 64, 250)            │
│                                                          │
│  InceptionBlock1D(64→96)                                 │
│   └─ MaxPool1d(k=3, stride=2)  → (B, 96, 125)            │
│                                                          │
│  InceptionBlock1D(96→128)                                │
│   └─ MaxPool1d(k=3, stride=2)  → (B, 128, 63)            │
└───────────────────────────┬──────────────────────────────┘
                       (B, 128, 63)
                            │
┌───────────────────────────┴──────────────────────────────┐
│  RegressionHead                                          │
│  AdaptiveAvgPool1d(1) → (B, 128)                         │
│  Dropout(0.15) → Linear(128→2)                           │
└───────────────────────────┬──────────────────────────────┘
                         (B, 2)
                       [SBP, DBP] (mmHg)
```

## 3. 텐서 흐름 요약

| 단계 | 처리                           | 입력 shape    | 출력 shape    |
| ---- | ------------------------------ | ------------- | ------------- |
| 0    | ensure_3d                      | (B, 1000)     | (B, 1, 1000)  |
| 1    | InceptionBlock1D(1→48)         | (B, 1, 1000)  | (B, 48, 1000) |
| 2    | MaxPool1d(k=3, s=2)            | (B, 48, 1000) | (B, 48, 500)  |
| 3    | InceptionBlock1D(48→64)        | (B, 48, 500)  | (B, 64, 500)  |
| 4    | MaxPool1d(k=3, s=2)            | (B, 64, 500)  | (B, 64, 250)  |
| 5    | InceptionBlock1D(64→96)        | (B, 64, 250)  | (B, 96, 250)  |
| 6    | MaxPool1d(k=3, s=2)            | (B, 96, 250)  | (B, 96, 125)  |
| 7    | InceptionBlock1D(96→128)       | (B, 96, 125)  | (B, 128, 125) |
| 8    | MaxPool1d(k=3, s=2)            | (B, 128, 125) | (B, 128, 63)  |
| 9    | AdaptiveAvgPool1d(1) + flatten | (B, 128, 63)  | (B, 128)      |
| 10   | Dropout + Linear(128→2)        | (B, 128)      | (B, 2)        |

> MaxPool1d(k=3, stride=2, padding=1)의 출력 길이: `ceil(L / 2)`

## 4. 모듈별 상세 설계

### 4.1 ConvBnAct1d (공유 블록)

**위치**: [`bpe/models/blocks.py`](../bpe/models/blocks.py)

`Conv1d → BatchNorm1d → Activation`의 표준 삼중 조합. 패딩은
`((kernel_size - 1) * dilation) // 2`로 자동 계산하여 출력 길이를 입력과
동일하게 유지한다 (same-padding).

```text
ConvBnAct1d(in, out, k):
    Conv1d(in, out, k, padding=(k-1)//2, bias=False)
    BatchNorm1d(out)
    ReLU (기본값)
```

Conv에는 `bias=False`를 사용한다. BatchNorm이 자체 편향(β)을 학습하므로
Conv 편향은 중복이다.

### 4.2 InceptionBlock1D

**역할**: 서로 다른 수용 영역(receptive field)의 합성곱 브랜치를 병렬 실행 후
합산·투영하고, 잔차 연결로 마무리하는 핵심 특징 추출 블록.

#### 처리 흐름

```text
입력 x : (B, C_in, L)
    │
    ├─ [병목 변환]
    │   bottleneck = ConvBnAct1d(C_in → C_bn, k=1)       (B, C_bn, L)
    │
    ├─ [커널 브랜치] (kernel_sizes = (9, 19, 39))
    │   branch_0 = ConvBnAct1d(C_bn → c₀, k=9)           (B, c₀, L)
    │   branch_1 = ConvBnAct1d(C_bn → c₁, k=19)          (B, c₁, L)
    │   branch_2 = ConvBnAct1d(C_bn → c₂, k=39)          (B, c₂, L)
    │   c₀ + c₁ + c₂ = C_out  (균등 분배, 나머지는 앞 브랜치에 +1씩)
    │
    ├─ [풀링 브랜치]
    │   MaxPool1d(k=3, stride=1, padding=1)               (B, C_in, L)
    │   ConvBnAct1d(C_in → c_pool, k=1)                   (B, c_pool, L)
    │   c_pool = C_out // len(kernel_sizes)
    │
    ├─ concat([branch_0, branch_1, branch_2, pool_branch])
    │   = (B, C_out + c_pool, L)
    │
    ├─ project = ConvBnAct1d(C_out + c_pool → C_out, k=1) (B, C_out, L)
    │
    ├─ shortcut(x):
    │   C_in == C_out → Identity
    │   C_in ≠  C_out → Conv1d(C_in→C_out, k=1, bias=False) + BN
    │
    ▼  ReLU(project + shortcut(x))
출력 : (B, C_out, L)
```

#### 채널 분배 규칙

```text
branch_channels = C_out // len(kernel_sizes)
remainder       = C_out  - branch_channels * len(kernel_sizes)
branch_i 채널  = branch_channels + (1 if i < remainder else 0)
```

예시: `C_out=64, kernel_sizes=(9,19,39)` → `branch_channels=21, remainder=1`
→ 브랜치 채널 수: (22, 21, 21)

이 규칙으로 `C_out`이 브랜치 수로 나누어 떨어지지 않아도 채널 총합이 항상
정확히 `C_out`이 된다.

#### 병목(bottleneck) 채널 수 결정

```python
bottleneck_channels = bottleneck_channels or max(out_channels // 2, 8)
```

기본값을 `C_out // 2`로 설정하되 하한 8을 두어 초소형 블록도 동작하도록 한다.
병목 `k=1` 합성곱은 채널 수를 줄여 이후 큰 커널 합성곱의 연산량을 절감한다.

#### 잔차 연결

입력 채널 수와 출력 채널 수가 같으면 `nn.Identity()`를 사용해 파라미터 추가 없이
항등 연결을 구성한다. 채널 수가 다르면 `Conv1d(k=1) + BN`으로 차원을 맞춘다.

### 4.3 MInceptionBackbone

**역할**: `InceptionBlock1D`와 `MaxPool1d`를 교대로 적층한 특징 추출 백본.

```python
for out_channels in channels:          # (48, 64, 96, 128)
    layers += [InceptionBlock1D(current_ch, out_channels, kernel_sizes)]
    layers += [MaxPool1d(k=3, stride=2, padding=1)]
    current_ch = out_channels
```

총 4개 스테이지, 각 스테이지에서 공간 해상도를 절반으로 줄인다.

| 스테이지 | InceptionBlock1D | MaxPool1d 이후 length |
| -------- | ---------------- | --------------------- |
| 1        | (1→48)           | 1000 → 500            |
| 2        | (48→64)          | 500 → 250             |
| 3        | (64→96)          | 250 → 125             |
| 4        | (96→128)         | 125 → 63              |

`out_channels` 속성으로 최종 채널 수(128)를 `RegressionHead`에 전달한다.

### 4.4 RegressionHead (공유 블록)

**위치**: [`bpe/models/blocks.py`](../bpe/models/blocks.py)

```text
입력 x : (B, C, L)
    │
    ▼  AdaptiveAvgPool1d(1)
(B, C, 1)
    │
    ▼  flatten → (B, C)
    │
    ▼  Dropout(p)
    │
    ▼  Linear(C → out_features)
출력 : (B, out_features)   ← [SBP, DBP]
```

`AdaptiveAvgPool1d(1)`은 임의의 시퀀스 길이에서 (B, C, 1)로 압축하므로
입력 길이 변화에 자동 대응한다.

### 4.5 MInception (최상위 조합)

```python
self.backbone = MInceptionBackbone(in_channels, channels, kernel_sizes)
self.head     = RegressionHead(self.backbone.out_channels, out_features, dropout)

def forward(self, x):
    return self.head(self.backbone(x))   # ensure_3d는 Backbone 내부에서 호출
```

전체 구조는 Backbone → Head의 2단계로 단순히 구성되어 있어, Backbone만
단독으로 재사용하거나 헤드를 교체하기 쉽다.

## 5. 각 스테이지의 InceptionBlock1D 세부 파라미터

기본 설정 `channels=(48,64,96,128)`, `kernel_sizes=(9,19,39)` 기준.

### Stage 1: InceptionBlock1D(1→48)

| 구성요소    | in→out 채널 | 커널          |
| ----------- | ----------- | ------------- |
| bottleneck  | 1 → 24      | k=1           |
| branch_0    | 24 → 16     | k=9           |
| branch_1    | 24 → 16     | k=19          |
| branch_2    | 24 → 16     | k=39          |
| pool branch | 1 → 16      | MaxPool + k=1 |
| project     | 64 → 48     | k=1           |
| shortcut    | 1 → 48      | k=1 + BN      |

### Stage 2: InceptionBlock1D(48→64)

| 구성요소    | in→out 채널 | 커널          |
| ----------- | ----------- | ------------- |
| bottleneck  | 48 → 32     | k=1           |
| branch_0    | 32 → 22     | k=9           |
| branch_1    | 32 → 21     | k=19          |
| branch_2    | 32 → 21     | k=39          |
| pool branch | 48 → 21     | MaxPool + k=1 |
| project     | 85 → 64     | k=1           |
| shortcut    | 48 → 64     | k=1 + BN      |

### Stage 3: InceptionBlock1D(64→96)

| 구성요소    | in→out 채널 | 커널          |
| ----------- | ----------- | ------------- |
| bottleneck  | 64 → 48     | k=1           |
| branch_0    | 48 → 32     | k=9           |
| branch_1    | 48 → 32     | k=19          |
| branch_2    | 48 → 32     | k=39          |
| pool branch | 64 → 32     | MaxPool + k=1 |
| project     | 128 → 96    | k=1           |
| shortcut    | 64 → 96     | k=1 + BN      |

### Stage 4: InceptionBlock1D(96→128)

| 구성요소    | in→out 채널 | 커널          |
| ----------- | ----------- | ------------- |
| bottleneck  | 96 → 64     | k=1           |
| branch_0    | 64 → 43     | k=9           |
| branch_1    | 64 → 43     | k=19          |
| branch_2    | 64 → 42     | k=39          |
| pool branch | 96 → 42     | MaxPool + k=1 |
| project     | 170 → 128   | k=1           |
| shortcut    | 96 → 128    | k=1 + BN      |

## 6. 하이퍼파라미터 참조표

| 파라미터       | 기본값            | 역할                                          |
| -------------- | ----------------- | --------------------------------------------- |
| `in_channels`  | 1                 | 입력 채널 수 (단일 PPG)                       |
| `out_features` | 2                 | 출력 차원 ([SBP, DBP])                        |
| `channels`     | (48, 64, 96, 128) | 각 스테이지의 출력 채널 수 (스테이지 수 결정) |
| `kernel_sizes` | (9, 19, 39)       | 병렬 브랜치 커널 크기 (수용 영역 다양성)      |
| `dropout`      | 0.15              | RegressionHead Dropout 비율                   |

### 커널 크기와 수용 영역

125 Hz PPG 기준 각 커널이 포착하는 시간 범위:

| 커널 크기    | 시간 범위 | 포착 대상                |
| ------------ | --------- | ------------------------ |
| k=9          | ~72 ms    | 맥파 상승부, 중첩 반사파 |
| k=19         | ~152 ms   | 심박 주기 내 전체 파형   |
| k=39         | ~312 ms   | 인접 심박 간 관계        |
| MaxPool(k=3) | ~24 ms    | 국소 피크 추출           |

### 스테이지 수와 채널 수 조정

```bash
# 경량 버전 (빠른 실험)
bin\train-model.bat --model minception \
    --model-kwargs "channels=(32,48,64)"

# 대형 버전 (표현력 증가)
bin\train-model.bat --model minception \
    --model-kwargs "channels=(64,96,128,192)"

# 추가 스케일 (저주파 포착)
bin\train-model.bat --model minception \
    --model-kwargs "kernel_sizes=(9,19,39,79)"
```

## 7. 설계 결정 사항

### 7.1 병목(bottleneck) k=1 합성곱

큰 커널(k=39)을 `C_in` 채널 전체에 직접 적용하면 파라미터가 `C_in × C_out × 39`개
필요하다. 먼저 k=1 병목으로 채널을 `C_in // 2`로 줄인 뒤 대형 커널을 적용하면
파라미터와 연산량이 약 절반으로 감소한다.

### 7.2 풀링 브랜치

`MaxPool1d(k=3) + ConvBnAct1d(k=1)` 브랜치는 국소 최댓값 기반 불변 특징을
제공한다. GoogLeNet/Inception-v1의 설계를 따른 것으로, 맥파 피크 감지에
유효하다. 이 브랜치는 병목을 거치지 않고 원본 `x`에 직접 적용한다는 점에
주의한다.

### 7.3 `project` Conv로 채널 통일

브랜치 채널 합(`C_out + c_pool`)과 목표 채널 수(`C_out`)가 다르므로
k=1 합성곱으로 투영(projection)한 뒤 잔차와 더한다. 이 투영에도 BN + ReLU가
적용되며, 이후 shortcut과 더한 결과에 최종 ReLU를 적용한다.

### 7.4 잔차 연결의 활성화 순서

PyTorch 관례인 Post-BN 잔차 구조를 따른다:

```text
output = ReLU(project(concat_branches) + shortcut(x))
```

`project` 내부에도 BN + ReLU가 있어 활성화가 두 번 적용되는 것처럼 보이나,
`project`의 ReLU는 내부 중간 표현에, 최종 ReLU는 잔차 합산 후에 적용된다.

### 7.5 MInceptionBackbone 분리

헤드(`RegressionHead`)와 백본(`MInceptionBackbone`)을 분리한 이유는 재사용성
때문이다. 백본을 독립 모듈로 제공하면 다른 태스크(분류, 사전 학습 등)에서
동일 특징 추출기를 임포트해 쓸 수 있다.

## 8. 훈련 방법

### 기본 훈련

```bash
bin\train-model.bat --model minception
```

### 하이퍼파라미터 조정 예시

```bash
# 큰 배치 + 높은 학습률
bin\train-model.bat --model minception \
    --batch-size 256 \
    --lr 1e-3 \
    --epochs 100
```

## 9. 모델 검사

```bash
# 레이어 구조와 파라미터 수 출력
bin\print-model.bat --model minception
```

출력 예시:

```text
MInception
  (backbone): MInceptionBackbone
    (net): Sequential
      (0): InceptionBlock1D    ...
      (1): MaxPool1d            ...
      ...
  (head): RegressionHead
    (pool): AdaptiveAvgPool1d  ...
    (dropout): Dropout(p=0.15)
    (fc): Linear(in=128, out=2)

Total params    : <N>
Trainable params: <N>
Input shape     : (1, 1000)
```
