# SpectroTemporalResNet(ST-ResNet) 모델 상세 설계서

## 1. 개요

SpectroTemporalResNet(이하 ST-ResNet)은 단일 PPG 채널에서 속도 맥파(VPG)와
가속도 맥파(APG)를 수치 미분으로 유도하고, 세 신호 각각을 독립된 잔차 네트워크
브랜치로 처리한 뒤 융합하는 **3-브랜치 멀티채널 회귀 모델**이다.

- **구현 파일**: [`bpe/models/st_resnet.py`](../bpe/models/st_resnet.py)
- **모델 등록명**: `st_resnet` (별칭: `spectro_temporal_resnet`)
- **공유 블록**: `DerivativeChannels`, `BasicBlock1D`
  ([`bpe/models/blocks.py`](../bpe/models/blocks.py),
  [`bpe/models/resnet1d.py`](../bpe/models/resnet1d.py))

### 설계 동기

PPG의 1차 미분(VPG)은 맥파의 기울기 변화를 포착해 **맥파 전달 속도**와 관련된
타이밍 정보를 강조한다. 2차 미분(APG)은 맥파의 변곡점을 강조해 **동맥 탄성도
및 혈관 저항**과 관련된 파형 형태 정보를 제공한다. 세 신호는 진폭 범위와 스펙트럼
특성이 크게 다르므로, 하나의 브랜치로 공동 처리하면 PPG의 높은 진폭이 VPG/APG의
약한 신호를 압도한다. **채널별 독립 브랜치** + **채널별 정규화**로 이 문제를
해결한다.

## 2. 전체 아키텍처

```text
입력: PPG 세그먼트
     (B, 1000) 또는 (B, 1, 1000)
                │
                ▼  ensure_3d
           (B, 1, 1000)
                │
                ▼  DerivativeChannels
           (B, 3, 1000)
 [ch 0: PPG, ch 1: VPG, ch 2: APG]
                │
    ┌───────────┼───────────┐
    │           │           │
    ▼           ▼           ▼
SignalBranch  SignalBranch  SignalBranch
 (PPG)         (VPG)         (APG)
    │           │           │
 (B, 96)     (B, 96)     (B, 96)
    │           │           │
    └───────────┴───────────┘
                │
                ▼  torch.cat(dim=1)
           (B, 288)  [= 96 × 3]
                │
  ┌─────────────┴──────────────┐
  │  Fusion Head               │
  │  LayerNorm(288)            │
  │  Dropout(0.2)              │
  │  Linear(288 → 144) + ReLU  │
  │  Dropout(0.2)              │
  │  Linear(144 → 2)           │
  └─────────────┬──────────────┘
             (B, 2)
           [SBP, DBP] (mmHg)
```

## 3. 텐서 흐름 요약

### 3.1 최상위 흐름

| 단계 | 처리                             | 입력 shape        | 출력 shape      |
| ---- | -------------------------------- | ----------------- | --------------- |
| 0    | ensure_3d                        | (B, 1000)         | (B, 1, 1000)    |
| 1    | DerivativeChannels               | (B, 1, 1000)      | (B, 3, 1000)    |
| 2    | 채널 분리                        | (B, 3, 1000)      | 3× (B, 1, 1000) |
| 3    | SignalBranch (×3, 병렬)          | (B, 1, 1000) 각각 | (B, 96) 각각    |
| 4    | torch.cat                        | 3× (B, 96)        | (B, 288)        |
| 5    | LayerNorm(288)                   | (B, 288)          | (B, 288)        |
| 6    | Dropout + Linear(288→144) + ReLU | (B, 288)          | (B, 144)        |
| 7    | Dropout + Linear(144→2)          | (B, 144)          | (B, 2)          |

### 3.2 SignalBranch 내부 흐름 (base_channels=24)

| 단계 | 처리                           | 입력 shape   | 출력 shape   |
| ---- | ------------------------------ | ------------ | ------------ |
| 0    | ConvBnAct1d(1→24, k=15, s=2)   | (B, 1, 1000) | (B, 24, 500) |
| 1    | BasicBlock1D(24→24, s=1)       | (B, 24, 500) | (B, 24, 500) |
| 2    | BasicBlock1D(24→48, s=2)       | (B, 24, 500) | (B, 48, 250) |
| 3    | BasicBlock1D(48→96, s=2)       | (B, 48, 250) | (B, 96, 125) |
| 4    | AdaptiveAvgPool1d(1) + Flatten | (B, 96, 125) | (B, 96)      |
| 5    | Linear(96→96) + ReLU           | (B, 96)      | (B, 96)      |

## 4. 모듈별 상세 설계

### 4.1 DerivativeChannels

**역할**: 단일 PPG 입력에서 VPG(1차 미분)와 APG(2차 미분)를 수치적으로 계산하고,
세 채널 각각을 독립적으로 z-score 정규화한다.

```text
입력 x : (B, 1, L)
    │
    ├─ ppg = x                                         (B, 1, L)
    │
    ├─ vpg:  ppg[:,:,1:] - ppg[:,:,:-1]  → (B, 1, L-1)
    │        F.pad(..., (1, 0))           → (B, 1, L)   [좌측 0 패딩]
    │        vpg[t] = ppg[t] - ppg[t-1]  (t≥1), vpg[0] = 0
    │
    ├─ apg:  vpg[:,:,1:] - vpg[:,:,:-1]  → (B, 1, L-1)
    │        F.pad(..., (1, 0))           → (B, 1, L)
    │        apg[t] = vpg[t] - vpg[t-1]  (t≥1), apg[0] = 0
    │
    ├─ out = cat([ppg, vpg, apg], dim=1)               (B, 3, L)
    │
    ▼  _normalize: (x - mean) / std  (채널별 독립, dim=-1 기준)
출력 : (B, 3, L)
```

#### 정규화 세부

```python
mean = x.mean(dim=-1, keepdim=True)         # (B, 3, 1)
std  = x.std(dim=-1, keepdim=True).clamp_min(eps=1e-6)  # (B, 3, 1)
return (x - mean) / std
```

각 채널을 독립적으로 z-score 정규화하므로 PPG/VPG/APG의 진폭 차이(수십 배)가
각 브랜치 입력에서 균등화된다.

#### 이미 3채널인 입력 처리

`derive_channels=True`여도 입력이 이미 3채널이면 (`x.size(1)==3`) 미분을
건너뛰고 정규화만 수행한다. 사전 계산된 VPG/APG가 있는 데이터셋에 활용할 수 있다.

#### `derive_channels=False` 옵션

`nn.Identity()`로 대체되어 미분·정규화가 모두 스킵된다. 이 경우 3채널 입력을
직접 넣어야 한다.

### 4.2 SignalBranch

**역할**: 단일 신호 채널(PPG, VPG, 또는 APG)에서 `embedding_dim` 크기의 고정
길이 임베딩 벡터를 추출하는 소형 잔차 네트워크.

```text
입력 : (B, 1, 1000)
    │
    ▼  ConvBnAct1d(1 → base_ch, k=15, stride=2)
(B, base_ch, 500)
    │
    ▼  BasicBlock1D(base_ch → base_ch, stride=1)
(B, base_ch, 500)                ← 채널 유지, 해상도 유지
    │
    ▼  BasicBlock1D(base_ch → base_ch×2, stride=2)
(B, base_ch×2, 250)              ← 채널 ×2, 해상도 ÷2
    │
    ▼  BasicBlock1D(base_ch×2 → base_ch×4, stride=2)
(B, base_ch×4, 125)              ← 채널 ×2, 해상도 ÷2
    │
    ▼  AdaptiveAvgPool1d(1)
(B, base_ch×4, 1)
    │
    ▼  Flatten
(B, base_ch×4)
    │
    ▼  Linear(base_ch×4 → embedding_dim)
    ▼  ReLU
출력 : (B, embedding_dim)
```

기본값 `base_channels=24`에서 채널 진행: 1 → 24 → 24 → 48 → 96

#### ResNet1D Stem과의 차이

| 항목          | ResNet1D Stem                | SignalBranch                        |
| ------------- | ---------------------------- | ----------------------------------- |
| 초기 합성곱   | ConvBnAct1d(k=15, s=2)       | ConvBnAct1d(k=15, s=2)              |
| 뒤따르는 풀링 | MaxPool1d(k=3, s=2)          | 없음 (스테이지 stride로만 다운샘플) |
| 스테이지 수   | 4                            | 3 (첫 스테이지 stride=1)            |
| 최종 풀링     | AdaptiveAvgPool1d(1) in Head | AdaptiveAvgPool1d(1) in Branch      |
| 출력          | (B, 256, 32) → 헤드로 전달   | (B, embedding_dim) 직접 생성        |

세 브랜치의 모든 가중치는 **공유되지 않는다**. `ppg_branch`, `vpg_branch`,
`apg_branch`는 별도의 `SignalBranch` 인스턴스이며 각각 독립적으로 학습된다.

### 4.3 Fusion Head

**역할**: 세 브랜치의 임베딩을 연결(concatenate)하고 MLP로 SBP/DBP를 회귀한다.

```text
입력 : (B, embedding_dim × 3)  = (B, 288)  [기본값]
    │
    ▼  LayerNorm(288)
    ▼  Dropout(0.2)
    ▼  Linear(288 → 144)  [= fused // 2]
    ▼  ReLU
    ▼  Dropout(0.2)
    ▼  Linear(144 → 2)
출력 : (B, 2)
```

#### LayerNorm 선택 이유

`RegressionHead`(ResNet1D)의 BatchNorm 대신 **LayerNorm**을 사용한다.
BatchNorm은 배치 통계에 의존해 배치 크기가 작거나 추론 시 통계가 다를 때
불안정해질 수 있다. 세 브랜치 임베딩의 연결 벡터는 특징 차원이 288로
비교적 크므로 LayerNorm이 안정적이다.

## 5. 하이퍼파라미터 참조표

| 파라미터          | 기본값 | 역할                                                         |
| ----------------- | ------ | ------------------------------------------------------------ |
| `out_features`    | 2      | 출력 차원 ([SBP, DBP])                                       |
| `base_channels`   | 24     | SignalBranch 초기 채널 수 (채널 진행: ×1 → ×2 → ×4)          |
| `embedding_dim`   | 96     | 각 브랜치 출력 임베딩 차원                                   |
| `dropout`         | 0.2    | Fusion Head Dropout 비율                                     |
| `derive_channels` | `True` | `True`: 단일 PPG → 3채널 자동 유도, `False`: 3채널 직접 입력 |

### 채널·임베딩 규모 조정

```bash
# 경량 (빠른 실험)
bin\train-model.bat --model st_resnet \
    --model-kwargs "base_channels=16,embedding_dim=64"
# 브랜치 채널: 1→16→32→64, 융합 벡터: 192

# 대형 (표현력 증가)
bin\train-model.bat --model st_resnet \
    --model-kwargs "base_channels=32,embedding_dim=128"
# 브랜치 채널: 1→32→64→128, 융합 벡터: 384
```

### `embedding_dim`과 `base_channels×4`의 관계

기본값에서 `base_channels×4 = 96 = embedding_dim`으로 같다.
`embedding_dim`을 `base_channels×4`보다 크게 설정하면 Linear가 **업프로젝션**,
작게 설정하면 **다운프로젝션** 역할을 한다.

## 6. 설계 결정 사항

### 6.1 VPG·APG 수치 미분 vs 학습 가능 미분 필터

학습 가능한 Conv1d(k=3)로 미분을 근사하는 방법도 있다.
이 구현에서 고정 차분 연산자를 사용하는 이유:

- **물리적 해석 가능성**: VPG와 APG는 임상적으로 정의된 신호다. 고정 연산자는
  생리학적 의미를 유지한다.
- **파라미터 절감**: 학습 가능 미분 필터를 추가하면 3채널 × Conv 파라미터가
  늘어난다.
- **정규화로 보완**: 수치 미분은 진폭 범위가 불안정하지만, 뒤따르는 채널별
  z-score 정규화가 이를 완전히 해소한다.

### 6.2 독립 브랜치 vs 공유 브랜치

PPG/VPG/APG는 통계적 특성(진폭 분포, 주파수 성분)이 다르므로 별도의 가중치를
학습하는 것이 자연스럽다. 공유 브랜치를 사용하면 모델 크기는 1/3이 되지만
채널별 특화 필터를 학습할 수 없다. 독립 브랜치를 선택한 배경이다.

### 6.3 Stem MaxPool 생략

ResNet1D의 Stem은 `ConvBnAct1d(stride=2) → MaxPool1d(stride=2)`로
총 4× 다운샘플한다. SignalBranch는 MaxPool 없이 `ConvBnAct1d(stride=2)` 후
바로 BasicBlock1D 스테이지로 이어진다. 이유:

- SignalBranch는 ResNet1D보다 **레이어가 적어** (3스테이지 vs 4스테이지) 덜
  공격적인 다운샘플이 적합하다.
- 최종 특징 맵 길이: 1000 → 500(Stem s=2) → 500(Stage1 s=1) →
  250(Stage2 s=2) → 125(Stage3 s=2) → 1(AdaptivePool)
  충분한 시계열 정보를 압축에 이용할 수 있다.

### 6.4 두 등록명 제공

```python
@register_model("st_resnet")
@register_model("spectro_temporal_resnet")
```

짧은 이름(`st_resnet`)은 CLI 사용성을 위해, 긴 이름(`spectro_temporal_resnet`)은
코드 가독성을 위해 제공한다. 두 이름 모두 동일한 클래스 인스턴스를 생성한다.

## 7. 훈련 방법

### 기본 훈련

```bash
bin\train-model.bat --model st_resnet
```

### 사전 계산된 3채널 입력 사용

VPG/APG를 데이터 파이프라인에서 미리 계산한 경우:

```bash
bin\train-model.bat --model st_resnet \
    --model-kwargs "derive_channels=False"
```

입력 데이터가 이미 (B, 3, 1000) 형태의 정규화된 PPG/VPG/APG여야 한다.

## 8. 모델 검사

```bash
bin\print-model.bat --model st_resnet
```

출력 예시:

```text
SpectroTemporalResNet
  (derive): DerivativeChannels
  (ppg_branch): SignalBranch
    (net): Sequential
      (0): ConvBnAct1d(1→24, k=15, s=2)
      (1): BasicBlock1D(24→24)
      (2): BasicBlock1D(24→48, s=2)
      (3): BasicBlock1D(48→96, s=2)
      (4): AdaptiveAvgPool1d(1)
      (5): Flatten
      (6): Linear(96→96)
      (7): ReLU
  (vpg_branch): SignalBranch  [동일 구조]
  (apg_branch): SignalBranch  [동일 구조]
  (head): Sequential
    (0): LayerNorm(288)
    (1): Dropout(p=0.2)
    (2): Linear(288→144)
    (3): ReLU
    (4): Dropout(p=0.2)
    (5): Linear(144→2)

Total params    : <N>
Trainable params: <N>
Input shape     : (1, 1000)
```

## 9. 참고 문헌

- Elgendi, M. (2012). "On the Analysis of Fingertip Photoplethysmogram Signals."
  *Current Cardiology Reviews*, 8(1), pp. 14–25.
  (APG의 임상적 의미 및 혈압 관련 파라미터 정의)

- He, K., Zhang, X., Ren, S., and Sun, J. (2016). "Deep Residual Learning for
  Image Recognition." *CVPR 2016*, pp. 770–778.
