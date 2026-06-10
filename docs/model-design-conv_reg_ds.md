# ConvRegDs 모델 상세 설계서

## 1. 개요

ConvRegDs는 ConvReg의 표준 Conv1d를 **깊이별 분리 합성곱(depthwise-separable
convolution)**으로 교체한 경량 변형 모델이다. 동일한 6스테이지 구조를 유지하면서
파라미터 수를 36.9 K에서 14.1 K로 약 62% 줄인다.

- **구현 파일**: [`bpe/models/conv_reg_ds.py`](../bpe/models/conv_reg_ds.py)
- **등록명**: `conv_reg_ds`
- **파라미터 수**: 14.1 K
- **기반 모델**: ConvReg (`conv_reg`) — 구조 동일, Conv1d만 교체
- **차이점**: 각 Conv1d → DepthwiseSeparableConv1d (depthwise + pointwise)

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
│  DSConv(1→8,  k=5) + BN + ReLU + AvgPool(2) → (B,8,500)   │
│  DSConv(8→16, k=5) + BN + ReLU + AvgPool(2) → (B,16,250)  │
│  DSConv(16→32,k=5) + BN + ReLU + AvgPool(2) → (B,32,125)  │
│  DSConv(32→64,k=3) + BN + ReLU + AvgPool(2) → (B,64, 62)  │
│  DSConv(64→64,k=3) + BN + ReLU + AvgPool(2) → (B,64, 31)  │
│  DSConv(64→64,k=3) + BN + ReLU + AvgPool(2) → (B,64, 15)  │
│                                                           │
│  AdaptiveAvgPool1d(1)                       → (B,64,  1)  │
└───────────────────┬───────────────────────────────────────┘
               (B, 64, 1)
                    │
┌───────────────────┴───────────────────────────────────────┐
│  regressor  (ConvReg와 동일)                              │
│  Flatten                                    → (B, 64)     │
│  Linear(64→32) + ReLU + Dropout(0.2)        → (B, 32)     │
│  Linear(32→2)                               → (B,  2)     │
└───────────────────┬───────────────────────────────────────┘
                  (B, 2)
              [SBP, DBP] (mmHg)
```

## 3. DepthwiseSeparableConv1d 상세

### 3.1 구조

표준 Conv1d를 두 단계로 분해한다.

```text
입력: (B, C_in, L)
    │
    ▼  Depthwise Conv1d(C_in → C_in, k, groups=C_in, bias=False)
       [채널당 독립적으로 시간 패턴 추출]
(B, C_in, L)
    │
    ▼  Pointwise Conv1d(C_in → C_out, k=1, bias=False)
       [채널 간 정보 혼합, 채널 수 변경]
(B, C_out, L)
```

### 3.2 Depthwise Conv1d

```python
nn.Conv1d(
    in_channels=C_in,
    out_channels=C_in,    # 채널 수 유지
    kernel_size=k,
    padding=p,
    groups=C_in,          # 채널당 독립 필터
    bias=False,
)
```

- `groups=in_channels`로 설정하여 각 입력 채널이 자신만의 1개 필터를 가진다
- 채널 간 정보 혼합 없이 **시간 방향 패턴만** 추출한다
- 파라미터 수: `C_in × 1 × k` (표준 대비 `C_out / C_in`배 절감)

### 3.3 Pointwise Conv1d

```python
nn.Conv1d(
    in_channels=C_in,
    out_channels=C_out,
    kernel_size=1,        # 시간 방향 수용 영역 없음
    bias=False,
)
```

- `kernel_size=1`이므로 **채널 방향 선형 결합만** 수행한다
- 시간 위치별로 독립적인 채널 혼합을 적용한다
- 파라미터 수: `C_in × C_out`

### 3.4 파라미터 비교 (스테이지별)

Conv1d(C_in → C_out, k)의 파라미터 수: `C_in × C_out × k + C_out`  
DSConv(C_in → C_out, k)의 파라미터 수: `C_in × k + C_in × C_out` (bias=False)

| 스테이지 | C_in→C_out | k   | Conv1d 파라미터 | DSConv 파라미터     | 절감 비율 |
| -------- | ---------- | --- | --------------- | ------------------- | --------- |
| 1        | 1→8        | 5   | 48              | 5 + 8 = 13          | 73%       |
| 2        | 8→16       | 5   | 656             | 40 + 128 = 168      | 74%       |
| 3        | 16→32      | 5   | 2,592           | 80 + 512 = 592      | 77%       |
| 4        | 32→64      | 3   | 6,208           | 96 + 2,048 = 2,144  | 65%       |
| 5        | 64→64      | 3   | 12,352          | 192 + 4,096 = 4,288 | 65%       |
| 6        | 64→64      | 3   | 12,352          | 192 + 4,096 = 4,288 | 65%       |

## 4. 텐서 흐름 요약

| 단계 | 처리                                          | 입력 shape   | 출력 shape   |
| ---- | --------------------------------------------- | ------------ | ------------ |
| 0    | ensure_3d                                     | (B, 1000)    | (B, 1, 1000) |
| 1    | DSConv(1→8, k=5, p=2) — depthwise + pointwise | (B, 1, 1000) | (B, 8, 1000) |
| 2    | BN(8) + ReLU + AvgPool1d(2)                   | (B, 8, 1000) | (B, 8, 500)  |
| 3    | DSConv(8→16, k=5, p=2)                        | (B, 8, 500)  | (B, 16, 500) |
| 4    | BN(16) + ReLU + AvgPool1d(2)                  | (B, 16, 500) | (B, 16, 250) |
| 5    | DSConv(16→32, k=5, p=2)                       | (B, 16, 250) | (B, 32, 250) |
| 6    | BN(32) + ReLU + AvgPool1d(2)                  | (B, 32, 250) | (B, 32, 125) |
| 7    | DSConv(32→64, k=3, p=1)                       | (B, 32, 125) | (B, 64, 125) |
| 8    | BN(64) + ReLU + AvgPool1d(2)                  | (B, 64, 125) | (B, 64, 62)  |
| 9    | DSConv(64→64, k=3, p=1)                       | (B, 64, 62)  | (B, 64, 62)  |
| 10   | BN(64) + ReLU + AvgPool1d(2)                  | (B, 64, 62)  | (B, 64, 31)  |
| 11   | DSConv(64→64, k=3, p=1)                       | (B, 64, 31)  | (B, 64, 31)  |
| 12   | BN(64) + ReLU + AvgPool1d(2)                  | (B, 64, 31)  | (B, 64, 15)  |
| 13   | AdaptiveAvgPool1d(1)                          | (B, 64, 15)  | (B, 64, 1)   |
| 14   | Flatten                                       | (B, 64, 1)   | (B, 64)      |
| 15   | Linear(64→32) + ReLU + Dropout(0.2)           | (B, 64)      | (B, 32)      |
| 16   | Linear(32→2)                                  | (B, 32)      | (B, 2)       |

## 5. 파라미터 수 분석

| 모듈                    | Depthwise | Pointwise  | BN      | 소계                |
| ----------------------- | --------- | ---------- | ------- | ------------------- |
| DSConv(1→8, k=5) + BN   | 5         | 8          | 16      | 29                  |
| DSConv(8→16, k=5) + BN  | 40        | 128        | 32      | 200                 |
| DSConv(16→32, k=5) + BN | 80        | 512        | 64      | 656                 |
| DSConv(32→64, k=3) + BN | 96        | 2,048      | 128     | 2,272               |
| DSConv(64→64, k=3) + BN | 192       | 4,096      | 128     | 4,416               |
| DSConv(64→64, k=3) + BN | 192       | 4,096      | 128     | 4,416               |
| Linear(64→32)           | —         | —          | —       | 2,080               |
| Linear(32→2)            | —         | —          | —       | 66                  |
| **합계**                | **605**   | **10,888** | **496** | **14,135 ≈ 14.1 K** |

> Depthwise와 Pointwise Conv는 모두 `bias=False`. MLP Linear는 bias=True (기본값).

## 6. ConvReg 패밀리 비교

| 항목                     | ConvReg         | ConvRegAt       | ConvRegDs                       |
| ------------------------ | --------------- | --------------- | ------------------------------- |
| 등록명                   | `conv_reg`      | `conv_reg_at`   | `conv_reg_ds`                   |
| feature_extractor 합성곱 | Conv1d          | Conv1d          | DepthwiseSeparableConv1d        |
| 풀링 방식                | AdaptiveAvgPool | 학습 어텐션     | AdaptiveAvgPool                 |
| 추가 모듈                | 없음            | attention (2층) | DepthwiseSeparableConv1d 클래스 |
| 파라미터 수              | 36.9 K          | 39.0 K          | **14.1 K**                      |
| 파라미터 절감율          | 기준            | +5.7%           | **-62%**                        |
| Conv bias                | True            | True            | **False**                       |
| 연산 효율                | 중간            | 중간            | 높음                            |

## 7. 설계 결정 사항

### 7.1 bias=False

`DepthwiseSeparableConv1d`의 depthwise와 pointwise Conv1d 모두 `bias=False`로
설정된다. 이는 MobileNet 등의 표준 경량 CNN 관례를 따른 것으로, 뒤이어 오는
BatchNorm1d가 편향(bias)과 동일한 이동(shift) 효과를 제공하기 때문이다.

### 7.2 BatchNorm은 DSConv 전체(depthwise + pointwise) 이후에 적용

`feature_extractor`에서 BN은 DSConv 두 단계가 모두 끝난 뒤 pointwise 출력에
적용된다. depthwise 직후에 BN을 삽입하는 MobileNetV2 스타일도 가능하지만,
이 모델은 단순성을 위해 하나의 BN만 사용한다.

### 7.3 분리 합성곱의 수용 영역

시간 방향 수용 영역은 depthwise Conv1d에서만 결정된다. k=5 또는 k=3의 패딩 규칙
(`padding = k // 2`)은 ConvReg와 동일하므로 각 스테이지의 수용 영역 크기도
ConvReg와 동일하다.

### 7.4 파라미터 절감 이유

표준 Conv1d(C_in → C_out, k)의 파라미터는 `C_in × C_out × k`에 비례한다.
DSConv는 이를 `C_in × k + C_in × C_out`으로 분해하며, 채널 수가 클수록
이 분해의 효율이 높아진다.

절감 비율: `(C_in × k + C_in × C_out) / (C_in × C_out × k)`
= `(1/C_out + 1/k)`

k=3, C_in=C_out=64일 경우: `1/64 + 1/3 ≈ 35%` → 65% 절감.

## 8. 훈련 방법

```bash
# 기본 훈련
bin/train-model --model conv_reg_ds

# ConvReg 패밀리 비교 실험
bin/train-model --model conv_reg
bin/train-model --model conv_reg_at
bin/train-model --model conv_reg_ds
```

## 9. 모델 검사

```bash
bin/print-model --model conv_reg_ds
```

출력 예시:

```text
Layer (name)                       Type                       Output shape        Params
------------------------------------------------------------------------------------------------
feature_extractor                  Sequential                 (1, 64, 1)
feature_extractor.0                DepthwiseSeparableConv1d   (1, 8, 1000)
feature_extractor.0.depthwise      Conv1d                     (1, 1, 1000)             5
feature_extractor.0.pointwise      Conv1d                     (1, 8, 1000)             8
feature_extractor.1                BatchNorm1d                (1, 8, 1000)            16
feature_extractor.2                ReLU                       (1, 8, 1000)             0
feature_extractor.3                AvgPool1d                  (1, 8, 500)              0
...
--------------------------------------------------------------------------------------------
  Total params    : 14,135  (14.1 K)
  Trainable params: 14,135  (14.1 K)
  Input shape     : (1, 1000)
```
