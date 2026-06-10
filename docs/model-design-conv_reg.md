# ConvReg 모델 상세 설계서

## 1. 개요

ConvReg는 PPG → SBP/DBP 회귀를 위한 단순한 1D CNN 기준 모델(baseline)이다.
6단계 Conv1d 스테이지로 시계열 특징을 점진적으로 추출한 뒤, 전역 평균 풀링과
2층 MLP로 혈압 값을 예측한다.

- **구현 파일**: [`bpe/models/conv_reg.py`](../bpe/models/conv_reg.py)
- **등록명**: `conv_reg`
- **파라미터 수**: 36.9 K
- **설계 목적**: 복잡한 구조 없이 CNN 회귀의 기본 성능을 측정하는 단순 기준선

## 2. 전체 아키텍처

```text
입력: PPG 세그먼트
           (B, 1000)  또는  (B, 1, 1000)
                    │
                    ▼  ensure_3d
               (B, 1, 1000)
                    │
┌───────────────────┴───────────────────────────────────────┐
│  feature_extractor                                        │
│                                                           │
│  Conv1d(1→8,  k=5) + BN + ReLU + AvgPool(2) → (B,8,500)   │
│  Conv1d(8→16, k=5) + BN + ReLU + AvgPool(2) → (B,16,250)  │
│  Conv1d(16→32,k=5) + BN + ReLU + AvgPool(2) → (B,32,125)  │
│  Conv1d(32→64,k=3) + BN + ReLU + AvgPool(2) → (B,64, 62)  │
│  Conv1d(64→64,k=3) + BN + ReLU + AvgPool(2) → (B,64, 31)  │
│  Conv1d(64→64,k=3) + BN + ReLU + AvgPool(2) → (B,64, 15)  │
│                                                           │
│  AdaptiveAvgPool1d(1)                       → (B,64,  1)  │
└───────────────────┬───────────────────────────────────────┘
               (B, 64, 1)
                    │
┌───────────────────┴───────────────────────────────────────┐
│  regressor                                                │
│  Flatten                                    → (B, 64)     │
│  Linear(64→32) + ReLU + Dropout(0.2)        → (B, 32)     │
│  Linear(32→2)                               → (B,  2)     │
└───────────────────┬───────────────────────────────────────┘
                  (B, 2)
              [SBP, DBP] (mmHg)
```

## 3. 텐서 흐름 요약

| 단계 | 처리                                | 입력 shape   | 출력 shape   |
| ---- | ----------------------------------- | ------------ | ------------ |
| 0    | ensure_3d                           | (B, 1000)    | (B, 1, 1000) |
| 1    | Conv1d(1→8, k=5, p=2) + BN + ReLU   | (B, 1, 1000) | (B, 8, 1000) |
| 2    | AvgPool1d(2)                        | (B, 8, 1000) | (B, 8, 500)  |
| 3    | Conv1d(8→16, k=5, p=2) + BN + ReLU  | (B, 8, 500)  | (B, 16, 500) |
| 4    | AvgPool1d(2)                        | (B, 16, 500) | (B, 16, 250) |
| 5    | Conv1d(16→32, k=5, p=2) + BN + ReLU | (B, 16, 250) | (B, 32, 250) |
| 6    | AvgPool1d(2)                        | (B, 32, 250) | (B, 32, 125) |
| 7    | Conv1d(32→64, k=3, p=1) + BN + ReLU | (B, 32, 125) | (B, 64, 125) |
| 8    | AvgPool1d(2)                        | (B, 64, 125) | (B, 64, 62)  |
| 9    | Conv1d(64→64, k=3, p=1) + BN + ReLU | (B, 64, 62)  | (B, 64, 62)  |
| 10   | AvgPool1d(2)                        | (B, 64, 62)  | (B, 64, 31)  |
| 11   | Conv1d(64→64, k=3, p=1) + BN + ReLU | (B, 64, 64)  | (B, 64, 31)  |
| 12   | AvgPool1d(2)                        | (B, 64, 31)  | (B, 64, 15)  |
| 13   | AdaptiveAvgPool1d(1)                | (B, 64, 15)  | (B, 64, 1)   |
| 14   | Flatten                             | (B, 64, 1)   | (B, 64)      |
| 15   | Linear(64→32) + ReLU + Dropout(0.2) | (B, 64)      | (B, 32)      |
| 16   | Linear(32→2)                        | (B, 32)      | (B, 2)       |

## 4. 모듈별 상세 설계

### 4.1 feature_extractor

6개 스테이지로 구성된 순차 모듈이다. 각 스테이지는 Conv1d + BatchNorm1d + ReLU +
AvgPool1d(2)의 4개 레이어로 이루어진다.

**스테이지 구성표**

| 스테이지 | Conv 입/출력 채널 | 커널 크기 | 패딩 | AvgPool 후 시간 길이 |
| -------- | ----------------- | --------- | ---- | -------------------- |
| 1        | 1 → 8             | 5         | 2    | 500                  |
| 2        | 8 → 16            | 5         | 2    | 250                  |
| 3        | 16 → 32           | 5         | 2    | 125                  |
| 4        | 32 → 64           | 3         | 1    | 62                   |
| 5        | 64 → 64           | 3         | 1    | 31                   |
| 6        | 64 → 64           | 3         | 1    | 15                   |

**패딩 규칙**: `padding = kernel_size // 2`로 설정되어 풀링 전 시간 길이가 보존된다.

- 스테이지 1–3: k=5로 저주파 PPG 형태(맥파 상승·하강·반사파) 추출
- 스테이지 4–6: k=3으로 상위 추상 특징 정제
- 각 AvgPool1d(2)가 시간 해상도를 절반씩 줄여 수용 영역을 2배씩 확장

마지막 `AdaptiveAvgPool1d(1)`은 남은 15개 위치를 전역 평균으로 단일 벡터로 압축한다.

### 4.2 regressor

전역 특징 벡터(64-d)를 혈압 값 2개로 매핑하는 2층 MLP다.

```text
(B, 64, 1)
    │
    ▼  Flatten
(B, 64)
    │
    ▼  Linear(64→32) + ReLU
(B, 32)
    │
    ▼  Dropout(0.2)
(B, 32)
    │
    ▼  Linear(32→2)
(B, 2)  ← [SBP, DBP] in mmHg
```

Dropout(0.2)는 과적합 방지를 위해 첫 번째 선형 변환 후에 적용된다.

### 4.3 ensure_3d

`bpe/models/blocks.py`에 정의된 유틸리티 함수로, 2D 입력 `(B, L)`을
`(B, 1, L)`로 변환한다. 3D 입력은 그대로 통과시킨다.

## 5. 파라미터 수 분석

| 모듈                    | 파라미터 수           | 구성           |
| ----------------------- | --------------------- | -------------- |
| Conv1d(1→8, k=5) + BN   | 48 + 16 = 64          | Conv bias=True |
| Conv1d(8→16, k=5) + BN  | 656 + 32 = 688        |                |
| Conv1d(16→32, k=5) + BN | 2,592 + 64 = 2,656    |                |
| Conv1d(32→64, k=3) + BN | 6,208 + 128 = 6,336   |                |
| Conv1d(64→64, k=3) + BN | 12,352 + 128 = 12,480 |                |
| Conv1d(64→64, k=3) + BN | 12,352 + 128 = 12,480 |                |
| Linear(64→32)           | 2,080                 |                |
| Linear(32→2)            | 66                    |                |
| **합계**                | **36,850 ≈ 36.9 K**   |                |

## 6. 설계 결정 사항

### 6.1 채널 수 증가 전략

1 → 8 → 16 → 32 → 64로 시간 해상도가 절반이 될 때마다 채널을 2배씩 늘린다.
4단계부터는 64로 고정되는데, 이는 더 이상 채널을 늘리지 않아도 64-d 표현이
충분하다는 경험적 판단이다.

### 6.2 AvgPool vs MaxPool

PPG 신호는 노이즈가 섞인 연속 파형이므로 MaxPool의 첨두값 선택보다
AvgPool의 국소 평균이 더 안정적인 다운샘플링을 제공한다.

### 6.3 AdaptiveAvgPool1d(1)과 Global Average Pooling

`AvgPool1d(2)`를 6번 적용하면 1000 → 500 → 250 → 125 → 62 → 31 → 15로
길이가 줄어든다. 1000이 2의 거듭제곱이 아니므로 62(홀수 나눗셈), 31, 15로
불균등하게 감소한다. `AdaptiveAvgPool1d(1)`은 이 불확실한 잔여 길이를 흡수하여
항상 정확히 64-d 벡터를 생성한다.

### 6.4 Dropout 위치

Dropout은 첫 번째 선형 레이어(64→32) 직후, 두 번째 선형 레이어(32→2) 직전에
위치한다. 최종 출력 레이어 이후에는 Dropout을 적용하지 않는다.

## 7. 훈련 방법

```bash
# 기본 훈련
bin/train-model --model conv_reg

# 학습률·배치 조정
bin/train-model --model conv_reg --lr 5e-4 --batch-size 512
```

## 8. 모델 검사

```bash
bin/print-model --model conv_reg
```

출력 예시:

```text
Layer (name)                    Type                Output shape        Params
--------------------------------------------------------------------------------------------
feature_extractor               Sequential          (1, 64, 1)
feature_extractor.0             Conv1d              (1, 8, 1000)            48
feature_extractor.1             BatchNorm1d         (1, 8, 1000)            16
feature_extractor.2             ReLU                (1, 8, 1000)             0
feature_extractor.3             AvgPool1d           (1, 8, 500)              0
...
regressor                       Sequential          (1, 2)
regressor.0                     Flatten             (1, 64)                  0
regressor.1                     Linear              (1, 32)              2,080
regressor.2                     ReLU                (1, 32)                  0
regressor.3                     Dropout             (1, 32)                  0
regressor.4                     Linear              (1, 2)                  66
--------------------------------------------------------------------------------------------
  Total params    : 36,850  (36.9 K)
  Trainable params: 36,850  (36.9 K)
  Input shape     : (1, 1000)
```
