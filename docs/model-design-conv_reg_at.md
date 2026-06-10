# ConvRegAt 모델 상세 설계서

## 1. 개요

ConvRegAt는 ConvReg의 전역 평균 풀링(AdaptiveAvgPool1d) 대신 **학습 가능한 시간적
어텐션 풀링(temporal attention pooling)**을 적용한 변형 모델이다. 어텐션 가중치가
15개 시간 위치 각각의 기여도를 학습하여, 혈압 예측에 중요한 PPG 파형 구간을
선택적으로 강조할 수 있다.

- **구현 파일**: [`bpe/models/conv_reg_at.py`](../bpe/models/conv_reg_at.py)
- **등록명**: `conv_reg_at`
- **파라미터 수**: 39.0 K
- **기반 모델**: ConvReg (`conv_reg`) — feature_extractor 6스테이지 동일
- **차이점**: AdaptiveAvgPool1d(1) → 학습 가능한 어텐션 풀링

## 2. 전체 아키텍처

```text
입력: PPG 세그먼트
           (B, 1000)  또는  (B, 1, 1000)
                    │
                    ▼  ensure_3d
               (B, 1, 1000)
                    │
┌───────────────────┴─────────────────────────────────────────┐
│  feature_extractor  (ConvReg와 동일, AdaptiveAvgPool 없음)  │
│                                                             │
│  Conv1d(1→8,  k=5) + BN + ReLU + AvgPool(2) → (B,8,500)     │
│  Conv1d(8→16, k=5) + BN + ReLU + AvgPool(2) → (B,16,250)    │
│  Conv1d(16→32,k=5) + BN + ReLU + AvgPool(2) → (B,32,125)    │
│  Conv1d(32→64,k=3) + BN + ReLU + AvgPool(2) → (B,64, 62)    │
│  Conv1d(64→64,k=3) + BN + ReLU + AvgPool(2) → (B,64, 31)    │
│  Conv1d(64→64,k=3) + BN + ReLU + AvgPool(2) → (B,64, 15)    │
└───────────────────┬─────────────────────────────────────────┘
               (B, 64, 15)
                    │
        ┌───────────┴───────────┐
        │ [특징 경로]           │ [어텐션 경로]
        │ x: (B, 64, 15)        │
        │                       │
        │              ┌────────┴──────────────────────────┐
        │              │  attention                        │
        │              │  Conv1d(64→32, k=1) + Tanh        │
        │              │      → (B, 32, 15)                │
        │              │  Conv1d(32→1, k=1)                │
        │              │      → (B, 1, 15)  (attn_score)   │
        │              │  Softmax(dim=-1)                  │
        │              │      → (B, 1, 15)  (attn_weight)  │
        │              └────────┬──────────────────────────┘
        │                       │ attn_weight
        └───────────────────────┘
                    │ element-wise multiply + sum(dim=-1, keepdim=True)
                    ▼  (B, 64, 15) * (B, 1, 15) → sum → (B, 64, 1)
               (B, 64, 1)
                    │
┌───────────────────┴─────────────────────────────────────┐
│  regressor  (ConvReg와 동일)                            │
│  Flatten                                    → (B, 64)   │
│  Linear(64→32) + ReLU + Dropout(0.2)        → (B, 32)   │
│  Linear(32→2)                               → (B,  2)   │
└───────────────────┬─────────────────────────────────────┘
                  (B, 2)
              [SBP, DBP] (mmHg)
```

## 3. 텐서 흐름 요약

| 단계 | 처리                                          | 입력 shape   | 출력 shape   |
| ---- | --------------------------------------------- | ------------ | ------------ |
| 0    | ensure_3d                                     | (B, 1000)    | (B, 1, 1000) |
| 1–12 | feature_extractor (ConvReg와 동일, 6스테이지) | (B, 1, 1000) | (B, 64, 15)  |
| 13   | Conv1d(64→32, k=1) + Tanh                     | (B, 64, 15)  | (B, 32, 15)  |
| 14   | Conv1d(32→1, k=1)                             | (B, 32, 15)  | (B, 1, 15)   |
| 15   | Softmax(dim=-1)                               | (B, 1, 15)   | (B, 1, 15)   |
| 16   | 가중합: sum(x * attn_weight, dim=-1)          | (B, 64, 15)  | (B, 64, 1)   |
| 17   | Flatten                                       | (B, 64, 1)   | (B, 64)      |
| 18   | Linear(64→32) + ReLU + Dropout(0.2)           | (B, 64)      | (B, 32)      |
| 19   | Linear(32→2)                                  | (B, 32)      | (B, 2)       |

## 4. 모듈별 상세 설계

### 4.1 feature_extractor

ConvReg와 완전히 동일한 6스테이지 구성이나, 마지막 `AdaptiveAvgPool1d(1)`이
**없다**. 6번의 AvgPool1d(2)만 적용하여 `(B, 64, 15)`를 출력한다.

- 1000 → 500 → 250 → 125 → 62 → 31 → 15 (6번 반분)
- 15개 시간 위치 각각은 약 533 ms(= 8000 ms / 15)의 PPG 구간을 나타냄

### 4.2 attention

1×1 합성곱 두 개로 구성된 경량 어텐션 모듈이다. 각 시간 위치에 스칼라 가중치를
할당하여 정보량이 많은 구간을 강조한다.

```text
입력: (B, 64, 15)
    │
    ▼  Conv1d(64→32, kernel_size=1)  [채널 방향 압축]
(B, 32, 15)
    │
    ▼  Tanh()
(B, 32, 15)
    │
    ▼  Conv1d(32→1, kernel_size=1)   [스칼라 점수 생성]
(B, 1, 15)  ← attn_score
    │
    ▼  Softmax(dim=-1)               [시간 방향 정규화]
(B, 1, 15)  ← attn_weight  (합 = 1.0)
```

- `kernel_size=1`(1×1 합성곱)이므로 **채널 방향 혼합만** 수행하고 시간 방향
  패턴은 건드리지 않는다
- Tanh는 중간 표현을 [-1, 1]로 제한하여 최종 점수 스케일 폭발을 방지한다
- Softmax는 시간 축(`dim=-1`) 전체를 정규화하므로 15개 가중치의 합이 항상 1이다

### 4.3 어텐션 가중합

```text
x           : (B, 64, 15)   — feature_extractor 출력
attn_weight : (B,  1, 15)   — attention 출력 (브로드캐스트 가능)

x * attn_weight             : (B, 64, 15)  [element-wise, 브로드캐스트]
torch.sum(..., dim=-1, keepdim=True) : (B, 64,  1)
```

각 채널 차원(64)에 대해 15개 위치의 가중 평균을 독립적으로 계산한다.
단순 평균(`AdaptiveAvgPool1d`)은 15개 위치에 동일한 가중치(1/15)를 부여하는
어텐션의 특수 케이스다.

### 4.4 return_attention 인터페이스

`forward(x, return_attention=True)` 호출 시 예측값과 어텐션 가중치를 함께 반환한다.

```python
out, attn = model(x, return_attention=True)
# out  : (B, 2)   — [SBP, DBP]
# attn : (B, 15)  — 15개 시간 위치의 어텐션 가중치 (합 = 1.0)
```

이 인터페이스는 학습 후 어느 시간 구간이 혈압 예측에 중요했는지 시각화하는 데
사용할 수 있다.

## 5. 파라미터 수 분석

| 모듈                          | 파라미터 수         | 비고           |
| ----------------------------- | ------------------- | -------------- |
| feature_extractor (6스테이지) | 34,704              | ConvReg와 동일 |
| attention: Conv1d(64→32, k=1) | 64×32 + 32 = 2,080  | bias=True      |
| attention: Conv1d(32→1, k=1)  | 32×1 + 1 = 33       | bias=True      |
| regressor: Linear(64→32)      | 64×32 + 32 = 2,080  |                |
| regressor: Linear(32→2)       | 32×2 + 2 = 66       |                |
| **합계**                      | **38,963 ≈ 39.0 K** |                |

ConvReg 대비 추가 파라미터: 2,113개 (어텐션 모듈) — 전체의 5.4%

## 6. ConvReg와의 비교

| 항목              | ConvReg                       | ConvRegAt                                    |
| ----------------- | ----------------------------- | -------------------------------------------- |
| feature_extractor | 동일                          | 동일                                         |
| 풀링 방식         | `AdaptiveAvgPool1d(1)` (고정) | 학습 가능한 어텐션 가중합                    |
| 추가 모듈         | 없음                          | `attention` (1×1 Conv × 2)                   |
| 파라미터 수       | 36.9 K                        | 39.0 K (+2.1 K)                              |
| return_attention  | 미지원                        | 지원 (`forward(..., return_attention=True)`) |
| 해석 가능성       | 없음                          | 시간 어텐션 가중치 시각화 가능               |

## 7. 설계 결정 사항

### 7.1 1×1 합성곱으로 어텐션 점수 계산

1×1 합성곱은 위치별로 독립적인 비선형 변환을 적용하므로, 시간 위치 간 상호작용
없이 각 위치의 채널 정보만으로 어텐션 점수를 결정한다. 이는 각 시간 슬라이스가
얼마나 정보량이 있는지를 해당 슬라이스의 특징만으로 판단하게 한다.

### 7.2 Tanh 활성화 함수

중간 어텐션 레이어에 ReLU 대신 Tanh를 사용한다. ReLU는 음수 입력을 0으로
잘라내어 특정 채널의 신호가 완전히 억제될 수 있다. Tanh는 양·음수 방향 모두
유지하므로 어텐션 점수 계산의 표현력이 더 넓다.

### 7.3 Softmax 정규화

어텐션 가중치의 합을 1로 강제하는 Softmax는 학습 안정성을 높인다.
Sigmoid 기반 정규화 없이 raw 점수를 사용하면 가중치 스케일이 학습 중 불안정해질 수 있다.

### 7.4 ConvReg feature_extractor 재사용

어텐션 모듈을 제외하면 ConvReg와 완전히 동일한 구조를 유지함으로써,
어텐션 풀링의 효과를 독립적으로 측정할 수 있다.

## 8. 훈련 방법

```bash
# 기본 훈련
bin/train-model --model conv_reg_at

# ConvReg와 비교 실험
bin/train-model --model conv_reg
bin/train-model --model conv_reg_at
```

## 9. 어텐션 시각화

학습 완료 후 어텐션 가중치를 시각화하여 모델이 중요하게 생각하는 PPG 구간을
확인할 수 있다.

```python
import torch
from bpe.models.conv_reg_at import ConvRegAt

model = ConvRegAt()
model.load_state_dict(torch.load("data/models/conv_reg_at/best.pt")["model"])
model.eval()

with torch.no_grad():
    pred, attn = model(ppg_segment, return_attention=True)
    # attn: (B, 15) — 각 시간 위치(약 533 ms 간격)의 중요도
```

## 10. 모델 검사

```bash
bin/print-model --model conv_reg_at
```
