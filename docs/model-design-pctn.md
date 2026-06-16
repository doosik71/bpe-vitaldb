# PCTN 모델 상세 설계서

## 1. 개요

PCTN(Parallel CNN-Transformer Network)은 PPG 신호에서 CNN의 지역 특징(local features)과
Transformer의 전역 특징(global features)을 병렬로 추출하고, CBAM(Convolutional Block
Attention Module) 기반 융합 블록으로 두 특징을 결합하여 수축기·이완기 혈압을 회귀하는
모델이다.

- **구현 파일**: [`bpe/models/pctn.py`](../bpe/models/pctn.py)
- **모델 등록명**: `pctn`
- **설계 기반**: Tian et al., "A paralleled CNN and Transformer network for PPG-based
  cuff-less blood pressure estimation," *Biomed. Signal Process. Control* 99 (2025) 106741

| 항목               | 값                                         |
| ------------------ | ------------------------------------------ |
| 입력               | (B, 1000) 또는 (B, 1, 1000) — 8 s @ 125 Hz |
| 출력               | (B, 2) — \[SBP, DBP\] (mmHg)               |
| 총 파라미터        | 5,126,750 (5.13 M)                         |
| 학습 가능 파라미터 | 5,126,750 (5.13 M)                         |

### 핵심 아이디어

기존 CNN 기반 방법은 PPG의 지역 파형 형태만 포착하고, RNN 계열 방법은 장기 의존성
병렬 계산에 제약이 있다. PCTN은 두 가지 특징 추출기를 **병렬**로 배치함으로써
어느 한 쪽이 보지 못하는 정보를 상호 보완한다.

```text
CNN 브랜치     → 지역 특징: 파형 형태, 피크, 노치 등 국소 패턴
Transformer    → 전역 특징: 8초 전 구간의 장거리 시간 의존성
Fusion (CBAM)  → 어느 위치·어느 채널이 혈압 예측에 중요한지 선택적 강조
```

## 2. 전체 아키텍처

```text
입력: (B, 1000)  또는  (B, 1, 1000)
              │
   ┌──────────┴────────────────────────────────┐
   │         Stem                              │
   │  Conv(k=15,s=2)+BN+ReLU  → (B, 64, 500)   │
   │  MaxPool(k=3,s=2)        → (B, 64, 250)   │
   └──────────┬────────────────────────────────┘
              │   s = (B, 64, 250)
       ┌──────┴───────┐
       │              │
  ┌────┴─────┐   ┌────┴────────────────────────────┐
  │CNN 브랜치│   │       Transformer 브랜치        │
  │          │   │                                 │
  │Bottleneck│   │ tr_proj  Conv(k=1)              │
  │×3 블록   │   │ → (B, 256, 250)                 │
  │          │   │ permute → (B, 250, 256)         │
  │64→256ch  │   │ + tr_pos(학습 가능 위치 임베딩) │
  │          │   │ TransformerEncoder(6레이어)     │
  │          │   │ → (B, 250, 256)                 │
  │          │   │ permute → (B, 256, 250)         │
  └────┬─────┘   └────────────────┬────────────────┘
       │(B, 256, 250)             │(B, 256, 250)
       │                          │
  ┌────┴──────────────────────────┴────┐
  │           Fusion Block             │
  │                                    │
  │  SpatialAtt(CNN)  SpatialAtt(TR)   │  ← 각 브랜치 독립 적용
  │  (B,256,250)      (B,256,250)      │
  │        cat → (B, 512, 250)         │
  │  ChannelAtt (수정 SE, 1×1 Conv)    │
  │        → (B, 512, 250)             │
  └────────────────┬───────────────────┘
                   │(B, 512, 250)
   ┌───────────────┴────────────────┐
   │           Regressor            │
   │  AdaptiveAvgPool1d(1)          │
   │  → (B, 512)                    │
   │  FC(512→256) + ReLU + Dropout  │
   │  FC(256→2)                     │
   └───────────────┬────────────────┘
                   │
             (B, 2)  [SBP, DBP]
```

## 3. 텐서 흐름 요약

### 3.1 forward(x) — 추론 경로

| 단계 | 모듈        | 처리                         | 입력 shape    | 출력 shape    |
| ---- | ----------- | ---------------------------- | ------------- | ------------- |
| 0    | —           | ensure_3d                    | (B, 1000)     | (B, 1, 1000)  |
| 1    | stem.0      | Conv(k=15,s=2)+BN+ReLU       | (B, 1, 1000)  | (B, 64, 500)  |
| 2    | stem.1      | MaxPool(k=3,s=2,pad=1)       | (B, 64, 500)  | (B, 64, 250)  |
| 3    | cnn_branch  | Bottleneck1D × 3             | (B, 64, 250)  | (B, 256, 250) |
| 4    | tr_proj     | Conv1d(64→256, k=1)          | (B, 64, 250)  | (B, 256, 250) |
| 5    | —           | permute(0,2,1)               | (B, 256, 250) | (B, 250, 256) |
| 6    | tr_pos      | + 학습 가능 위치 임베딩      | (B, 250, 256) | (B, 250, 256) |
| 7    | tr_encoder  | TransformerEncoder(6 layers) | (B, 250, 256) | (B, 250, 256) |
| 8    | —           | permute(0,2,1)               | (B, 250, 256) | (B, 256, 250) |
| 9    | cnn_spatial | SpatialAttention1d (CNN 측)  | (B, 256, 250) | (B, 256, 250) |
| 10   | tr_spatial  | SpatialAttention1d (TR 측)   | (B, 256, 250) | (B, 256, 250) |
| 11   | —           | cat(dim=1)                   | 2×(B,256,250) | (B, 512, 250) |
| 12   | channel_att | ChannelAttention1d           | (B, 512, 250) | (B, 512, 250) |
| 13   | pool        | AdaptiveAvgPool1d(1)         | (B, 512, 250) | (B, 512, 1)   |
| 14   | —           | flatten(1)                   | (B, 512, 1)   | (B, 512)      |
| 15   | fc1 + act   | Linear(512→256) + ReLU       | (B, 512)      | (B, 256)      |
| 16   | drop        | Dropout(0.1)                 | (B, 256)      | (B, 256)      |
| 17   | fc2         | Linear(256→2)                | (B, 256)      | (B, 2)        |

### 3.2 Stem 출력 크기 계산

```text
입력 길이: L = 1000

Conv1d(k=15, s=2, padding=7):
  출력 = floor((1000 + 2×7 − 15) / 2) + 1 = floor(999/2) + 1 = 500

MaxPool1d(k=3, s=2, padding=1):
  출력 = floor((500 + 2×1 − 3) / 2) + 1 = floor(499/2) + 1 = 250

∴ Stem 출력: (B, 64, 250) — 이후 모든 모듈의 시퀀스 길이 L_s = 250
```

## 4. 모듈별 상세 설계

### 4.1 Stem 모듈

**역할**: 원시 PPG 신호에서 얕은 특징(shallow features)을 추출하고 입력 길이를 1/4로
축소하여 이후 CNN·Transformer 브랜치의 계산량을 줄인다.

```text
입력: (B, 1, 1000)
   │
   ▼  ConvBnAct1d(in=1, out=64, k=15, stride=2, padding=7)
(B, 64, 500)
   │
   ▼  MaxPool1d(kernel=3, stride=2, padding=1)
(B, 64, 250)
```

**대형 커널(k=15) 사용 이유**: 논문 Section 3.3.1은 "large convolution kernels can
increase the perceptual field while preserving more original information"이라고
명시한다. 125 Hz PPG에서 k=15는 약 120 ms의 수용 영역에 해당하며, 한 심박 주기
(~600–1000 ms)의 주요 파형 요소(수축기 피크, 중절흔, 이완기 파)를 커버한다.

**밴드패스 필터 부재**: 논문은 Stem에 0.5–10 Hz 밴드패스 필터를 포함하지만,
이 프로젝트에서는 `construct-dataset.py` 전처리 파이프라인에서 이미 적용되므로
모델 내부에서 중복 구현하지 않는다.

### 4.2 CNN 브랜치 — `_Bottleneck1D`

**역할**: ResNet-50의 첫 번째 스테이지(conv2_x)에 해당하는 3개 bottleneck 블록으로
지역 패턴을 계층적으로 추출한다.

#### 단일 Bottleneck 블록 구조

```text
입력 x : (B, C_in, L)
   │
   ├─ [주 경로]
   │   ▼  Conv1d(C_in → C_mid, k=1) + BN + ReLU   ← 채널 압축
   │  (B, C_mid, L)
   │   ▼  Conv1d(C_mid → C_mid, k=3) + BN + ReLU  ← 지역 공간 특징 추출
   │  (B, C_mid, L)
   │   ▼  Conv1d(C_mid → C_out, k=1) + BN          ← 채널 확장
   │  (B, C_out, L)
   │
   ├─ [단축 경로 (shortcut)]
   │   C_in ≠ C_out  →  Conv1d(C_in → C_out, k=1, bias=False) + BN
   │   C_in == C_out →  Identity
   │
   ▼  ReLU(주 경로 + 단축 경로)
출력 : (B, C_out, L)
```

`C_out = C_mid × expansion = C_mid × 4`

#### 3블록 채널 변화

| 블록 | C_in | C_mid | C_out | shortcut       |
| ---- | ---- | ----- | ----- | -------------- |
| 0    | 64   | 64    | 256   | Conv1d(64→256) |
| 1    | 256  | 64    | 256   | Identity       |
| 2    | 256  | 64    | 256   | Identity       |

블록 0 이후 채널이 64→256으로 확장되므로 이후 블록은 단축 경로가 Identity가 된다.

#### CNN 브랜치 파라미터 수

| 레이어                      | 파라미터 수 |
| --------------------------- | ----------- |
| Block 0 conv1 (64→64, k=1)  | 4,096       |
| Block 0 conv1 BN            | 128         |
| Block 0 conv2 (64→64, k=3)  | 12,288      |
| Block 0 conv2 BN            | 128         |
| Block 0 conv3 (64→256, k=1) | 16,384      |
| Block 0 conv3 BN            | 512         |
| Block 0 shortcut (64→256)   | 16,384      |
| Block 0 shortcut BN         | 512         |
| **Block 0 소계**            | **50,432**  |
| Block 1 conv1 (256→64, k=1) | 16,384      |
| Block 1 conv1 BN            | 128         |
| Block 1 conv2 (64→64, k=3)  | 12,288      |
| Block 1 conv2 BN            | 128         |
| Block 1 conv3 (64→256, k=1) | 16,384      |
| Block 1 conv3 BN            | 512         |
| **Block 1 소계**            | **45,824**  |
| Block 2 (Block 1과 동일)    | **45,824**  |
| **CNN 브랜치 합계**         | **142,080** |

### 4.3 Transformer 브랜치

**역할**: 250 토큰의 시퀀스에 Self-Attention을 적용해 8초 전 구간에 걸친 장거리
시간 의존성을 포착한다.

#### 브랜치 내부 처리 흐름

```text
Stem 출력 s : (B, 64, 250)
   │
   ▼  tr_proj: Conv1d(64→256, k=1, bias=False)
(B, 256, 250)
   │
   ▼  permute(0, 2, 1)
(B, 250, 256)   ← (batch, seq_len, d_model) 형태로 변환
   │
   ▼  + tr_pos[:, :250, :]    [학습 가능 위치 임베딩 (1, 250, 256)]
(B, 250, 256)
   │
   ▼  TransformerEncoder(6 layers, num_heads=4, d_ff=1024)
(B, 250, 256)
   │
   ▼  permute(0, 2, 1)
(B, 256, 250)   ← CNN 브랜치와 동일한 형태로 복원
```

#### 위치 임베딩 (`tr_pos`)

```text
shape : (1, 250, 256) — nn.Parameter, 0으로 초기화
        ↑     ↑    ↑
        배치  시퀀스 위치  임베딩 차원

tr_pos[0, t, :] : t번째 시간 스텝(125 Hz 기준 2×4 = 8 ms 간격)의 위치 임베딩

적용: tr = tr + tr_pos[:, :tr.size(1), :]
  → slicing을 통해 입력 길이 변동 시 안전하게 대응 (프로젝트에서는 항상 250)
```

Transformer는 입력 순서 정보가 없으므로 위치 임베딩 없이는 시간 250의 토큰과
시간 0의 토큰을 동일하게 취급한다. 0으로 초기화된 학습 가능 위치 임베딩은
학습 초기에 방해 없이 시작하면서 데이터에서 최적 위치 표현을 학습한다.

#### TransformerEncoderLayer 내부 구조

```text
입력 x : (B, 250, 256)   [batch_first=True]
   │
   ├─ [Multi-Head Self-Attention]
   │    head_dim = d_model / num_heads = 256 / 4 = 64
   │
   │    Q = x · W_Q,  K = x · W_K,  V = x · W_V    각 (B, 250, 256)
   │    분할 → 4개 헤드 각 (B, 250, 64)
   │
   │    Attn_h = softmax( Q_h · K_hᵀ / √64 ) · V_h   (B, 250, 64)
   │    병합   → (B, 250, 256) → 출력 투영 W_O        (B, 250, 256)
   │    Dropout(0.1)
   │
   ├─ residual add + LayerNorm  [Post-Norm, PyTorch 기본]
   │    x = LayerNorm(x + MHSA_out)
   │
   ├─ [Feed-Forward Network]
   │    Linear(256 → 1024) + ReLU + Dropout(0.1)
   │    Linear(1024 → 256)  + Dropout(0.1)
   │
   ▼  residual add + LayerNorm
   x = LayerNorm(x + FFN_out)
출력 : (B, 250, 256)
```

#### TransformerEncoderLayer 파라미터 수 (d_model=256, num_heads=4, d_ff=1024)

| 구성요소                            | 파라미터 수                 |
| ----------------------------------- | --------------------------- |
| MHSA in_proj (Q,K,V) weight+bias    | 3×256×256 + 3×256 = 197,376 |
| MHSA out_proj weight+bias           | 256×256 + 256 = 65,792      |
| FFN linear1 weight+bias             | 256×1024 + 1024 = 263,168   |
| FFN linear2 weight+bias             | 1024×256 + 256 = 262,400    |
| LayerNorm × 2 (weight+bias, 각 256) | 2 × 512 = 1,024             |
| **레이어 합계**                     | **789,760**                 |

6개 레이어 합계: 6 × 789,760 = **4,738,560**

Transformer 브랜치 전체:

| 구성요소                     | 파라미터 수      |
| ---------------------------- | ---------------- |
| tr_proj                      | 64×256 = 16,384  |
| tr_pos                       | 250×256 = 64,000 |
| TransformerEncoder (6레이어) | 4,738,560        |
| **브랜치 합계**              | **4,818,944**    |

### 4.4 Fusion 블록

**역할**: CNN(지역)과 Transformer(전역) 두 종류의 특징이 뒤섞이지 않도록 각각 공간
어텐션으로 먼저 정제한 뒤, 채널 방향으로 연결하고 채널 어텐션으로 혈압 예측에
중요한 채널을 선택적으로 강조한다.

논문 설계 원칙: *"Given that local and global features belong to two distinct types,
it is essential to apply spatial attention to each feature individually, ensuring
their separation to prevent amalgamation. Subsequently, a concatenation operation is
performed prior to channel attention to preserve the distinctiveness of the features."*

```text
CNN 특징   : (B, 256, 250)
TR 특징    : (B, 256, 250)

     ↓ SpatialAtt (독립 적용)        ↓ SpatialAtt (독립 적용)
(B, 256, 250) ────────────────── (B, 256, 250)
          cat(dim=1) ───────────────────────────
                          (B, 512, 250)
                    ChannelAtt ↓
                          (B, 512, 250)
```

#### `_SpatialAttention1d` — 1D CBAM 공간 어텐션

```text
입력 x : (B, C, L)   예: (B, 256, 250)
   │
   ├─ avg = x.mean(dim=1, keepdim=True)   (B, 1, L)  ← 채널 평균
   ├─ mx  = x.max(dim=1).values           (B, 1, L)  ← 채널 최댓값
   │
   ▼  cat([avg, mx], dim=1)               (B, 2, L)
   ▼  Conv1d(2→1, k=7, padding=3)        (B, 1, L)
   ▼  Sigmoid                             (B, 1, L)
   │
출력: x × attn                            (B, C, L)
```

채널 방향 평균·최댓값을 동시에 활용하는 것은 CBAM(Woo et al., ECCV 2018)의 설계를
그대로 따른 것이다. 평균 풀링은 전체적 활성화 강도를, 최댓값 풀링은 두드러진 국소
반응을 포착하여 상호 보완적으로 어텐션 맵을 형성한다.

파라미터: `Conv1d(2, 1, 7, bias=False)` = 2 × 7 = **14**

#### `_ChannelAttention1d` — 수정된 SE 채널 어텐션

```text
입력 x : (B, C, L)   C=512, L=250
   │
   ▼  AdaptiveAvgPool1d(1)   → (B, C, 1)   ← 전역 평균 풀링 (squeeze)
   │
   ▼  Conv1d(C → C//16, k=1, bias=False) + ReLU   → (B, 32, 1)   ← 채널 압축
   ▼  Conv1d(C//16 → C, k=1, bias=False) + Sigmoid → (B, C, 1)   ← 채널 복원 (excite)
   │
출력: x × attn                              (B, C, L)
```

**논문의 SE 수정점**: 원래 Squeeze-and-Excitation(Hu et al., CVPR 2018)은 FC 레이어를
사용하지만, PCTN 논문은 이를 `kernel=1` Conv1d로 교체한다. 논문 설명: *"This kind of
operation can get better local abstraction, smaller parameter space, and smaller
overfitting."* 구현 차원에서 Linear(C, mid)와 Conv1d(C, mid, 1)는 수학적으로 동일하나,
1D Conv 형태를 유지하면 입력 형태 변환(reshape) 없이 처리할 수 있어 구현이 간결하다.

파라미터 (C=512, reduction=16):

| 구성요소             | 파라미터 수     |
| -------------------- | --------------- |
| Conv1d(512→32, k=1)  | 512×32 = 16,384 |
| Conv1d(32→512, k=1)  | 32×512 = 16,384 |
| **채널 어텐션 합계** | **32,768**      |

Fusion 블록 전체 파라미터:

| 구성요소             | 파라미터 수 |
| -------------------- | ----------- |
| cnn_spatial (Conv×1) | 14          |
| tr_spatial  (Conv×1) | 14          |
| channel_att          | 32,768      |
| **Fusion 합계**      | **32,796**  |

### 4.5 Regressor 모듈

**역할**: 공간 전체에 걸쳐 특징을 집약하고 두 층의 완전 연결 레이어로 혈압 수치를
회귀한다.

```text
입력: (B, 512, 250)
   │
   ▼  AdaptiveAvgPool1d(1)       → (B, 512, 1)   ← 글로벌 평균 풀링
   ▼  flatten(1)                 → (B, 512)
   │
   ▼  Linear(512 → 256) + ReLU  → (B, 256)
   ▼  Dropout(0.1)
   │
   ▼  Linear(256 → 2)            → (B, 2)   ← [SBP, DBP]
```

파라미터:

| 구성요소             | 파라미터 수             |
| -------------------- | ----------------------- |
| fc1 (512→256 + bias) | 512×256 + 256 = 131,328 |
| fc2 (256→2 + bias)   | 256×2 + 2 = 514         |
| **Regressor 합계**   | **131,842**             |

## 5. 학습 가능 파라미터 전체 목록

기본값 `stem_channels=64, cnn_mid=64, cnn_blocks=3, d_model=256, num_heads=4,
num_tr_layers=6, ffn_ratio=4` 기준.

| 모듈               | 구성요소                     | 파라미터 수   |
| ------------------ | ---------------------------- | ------------- |
| Stem               | Conv1d(1→64, k=15)           | 960           |
|                    | BatchNorm1d(64)              | 128           |
|                    | **소계**                     | **1,088**     |
| CNN 브랜치         | Bottleneck Block 0           | 50,432        |
|                    | Bottleneck Block 1           | 45,824        |
|                    | Bottleneck Block 2           | 45,824        |
|                    | **소계**                     | **142,080**   |
| Transformer 브랜치 | tr_proj Conv1d(64→256, k=1)  | 16,384        |
|                    | tr_pos (1, 250, 256)         | 64,000        |
|                    | TransformerEncoder (6레이어) | 4,738,560     |
|                    | **소계**                     | **4,818,944** |
| Fusion             | cnn_spatial Conv1d(2→1, k=7) | 14            |
|                    | tr_spatial Conv1d(2→1, k=7)  | 14            |
|                    | channel_att Conv1d × 2       | 32,768        |
|                    | **소계**                     | **32,796**    |
| Regressor          | fc1 Linear(512→256)          | 131,328       |
|                    | fc2 Linear(256→2)            | 514           |
|                    | **소계**                     | **131,842**   |
| **전체 합계**      |                              | **5,126,750** |

## 6. 하이퍼파라미터 참조표

| 파라미터        | 기본값 | 역할                                             |
| --------------- | ------ | ------------------------------------------------ |
| `in_channels`   | 1      | 입력 PPG 채널 수                                 |
| `out_features`  | 2      | 출력 차원 (SBP, DBP)                             |
| `stem_channels` | 64     | Stem Conv 출력 채널; 이후 브랜치 공통 입력 차원  |
| `cnn_mid`       | 64     | Bottleneck 중간 채널; 출력 = cnn_mid × 4 = 256   |
| `cnn_blocks`    | 3      | Bottleneck 블록 수 (ResNet-50 conv2_x에 대응)    |
| `d_model`       | 256    | Transformer 임베딩 차원 및 Fusion 특징 차원      |
| `num_heads`     | 4      | MHSA 헤드 수; `d_model % num_heads == 0` 필수    |
| `num_tr_layers` | 6      | TransformerEncoder 레이어 수 (논문 Section 4.3)  |
| `ffn_ratio`     | 4      | FFN 확장 비율; d_ff = d_model × ffn_ratio = 1024 |
| `dropout`       | 0.1    | Transformer dropout 및 Regressor dropout         |

### 유효성 제약

```text
d_model % num_heads == 0    ← PyTorch TransformerEncoderLayer 내부에서 강제
cnn_mid × 4 == d_model      ← 이 조건이 불성립 시 추가 1×1 Conv 투영 삽입됨
                               (기본값 64×4=256 에서는 불필요)
```

### 규모 조정 예시

```bash
# 경량 (빠른 실험, ~1.5 M 파라미터)
bin/train-model --model pctn \
    --model-kwargs "d_model=128,num_heads=4,num_tr_layers=3"

# 기본값 (~5.13 M 파라미터)
bin/train-model --model pctn

# 대형 (표현력 증가, ~19 M 파라미터)
bin/train-model --model pctn \
    --model-kwargs "d_model=512,num_heads=8,num_tr_layers=6,cnn_mid=128"
```

## 7. 논문과의 대응 및 해석

### 7.1 주요 대응 항목

| 논문 설계                                              | 구현                                                 |
| ------------------------------------------------------ | ---------------------------------------------------- |
| Stem: 1D Conv + BN + MaxPool, large kernel             | `ConvBnAct1d(k=15, s=2)` + `MaxPool1d(k=3, s=2)`     |
| CNN branch: ResNet-50 bottleneck pyramid (conv 1→3→1)  | `_Bottleneck1D` × 3 (expansion=4)                    |
| Transformer branch: embedding + MHSA + MLP + residual  | `tr_proj` + `tr_pos` + `TransformerEncoderLayer` × 6 |
| 공간 어텐션 (Fig. 2(b)): avg+max pool → conv → sigmoid | `_SpatialAttention1d`                                |
| 채널 어텐션 (Fig. 2(a)): SE with 1×1 Conv              | `_ChannelAttention1d`                                |
| Regressor: 1D Global Pooling + 2× FC                   | `AdaptiveAvgPool1d(1)` + `fc1` + `fc2`               |
| 출력: SBP, DBP                                         | `fc2` out_features=2                                 |

### 7.2 논문 해석이 필요했던 부분

**Transformer 깊이**: 논문 Section 4.3은 `num_heads=4, depth=6`을 명시한다.
절제 실험(Section 5.4.1)에서 C|TTT(CNN 1개 + Transformer 3개 블록) 표기를 사용하는데,
이때 `depth`는 각 "블록" 당 레이어 수가 아닌 TransformerEncoder 전체 레이어 수로
해석했다. 즉 `num_tr_layers=6`이 논문 Section 4.3의 직접 대응이다.

**"1D Global Pooling with stride 2"**: 논문이 Regressor에서 "stride 2인 1D 글로벌
풀링"을 언급하지만 구체적 수식 없이 서술한다. "글로벌 풀링"의 본질(시퀀스 전체를
단일 벡터로 압축)을 따라 `AdaptiveAvgPool1d(1)`로 구현했다.

**Transformer의 "embedding module"**: 논문은 각 Transformer 블록 내에 "embedding
module"을 포함한다고 기술한다. `tr_proj`(채널 투영)와 `tr_pos`(위치 임베딩)를
조합한 사전 임베딩 단계가 이에 해당하며, PyTorch의 `TransformerEncoderLayer`는
이를 내부적으로 처리하지 않으므로 별도 구현했다.

### 7.3 프로젝트 적용 차이점

| 항목             | 논문                      | 본 구현                         |
| ---------------- | ------------------------- | ------------------------------- |
| 데이터셋         | MIMIC-III (808명, 125 Hz) | VitalDB (~3,000 케이스, 125 Hz) |
| 입력 길이        | 1024 샘플 (8.192 s)       | 1000 샘플 (8.0 s)               |
| 출력             | SBP, DBP, MAP             | SBP, DBP (MAP 제외)             |
| 밴드패스 필터    | Stem 모듈 포함            | 데이터셋 전처리에서 적용        |
| 파라미터 규모    | ~27.81 M                  | 5.13 M                          |
| Stem 시퀀스 길이 | 256 (1024/4)              | 250 (1000/4)                    |

**파라미터 규모 차이 원인**: 논문의 CNN backbone은 ResNet-50 전체(4 스테이지,
~25M)에 가까운 것으로 추정되나, 논문에서 채널 수 등 세부 수치를 공개하지 않는다.
본 구현은 첫 번째 스테이지(3 bottleneck, 64→256 채널)만 사용하여 모델을 적정
규모로 유지한다.

## 8. 설계 결정 사항

### 8.1 Spatial Attention: 브랜치별 독립 적용

CNN과 Transformer 특징에 **별도의** `SpatialAttention1d` 인스턴스를 적용한다.
단순히 가중치를 공유하지 않는 이유는 논문의 설계 의도와 일치한다:

> *"local and global features belong to two distinct types, it is essential to apply
spatial attention to each feature individually, ensuring their separation"*

공유 어텐션을 사용하면 두 종류의 특징이 같은 시간 위치를 강조하게 되어, 서로 다른
정보를 담은 두 브랜치의 보완성이 사라진다.

### 8.2 Channel Attention: FC 대신 1×1 Conv

표준 SE 블록은 전역 풀링 후 `Linear → ReLU → Linear → Sigmoid`를 사용한다.
PCTN은 이를 `Conv1d(k=1) → ReLU → Conv1d(k=1) → Sigmoid`로 교체한다.

수학적으로 `Linear(C, mid)`와 `Conv1d(C, mid, 1)` (입력이 (B, C, 1)인 경우)은
동치다. 이 구현에서 Conv1d를 선택한 이유:

- 채널 어텐션 전 과정(풀링 → excite → multiply)을 모두 (B, C, 1) 텐서 형태로
  처리하므로 `reshape`/`view` 없이 연결된다.
- 논문의 표현("1D convolution layer whose kernel size is 1")을 직역한다.

### 8.3 tr_pos: 학습 가능 위치 임베딩 vs 고정 Sinusoidal

두 가지 선택지가 있었다:

| 방식             | 장점                      | 단점                             |
| ---------------- | ------------------------- | -------------------------------- |
| 고정 Sinusoidal  | 임의 길이 일반화          | PPG 도메인 최적 표현 부재        |
| 학습 가능 (채택) | 데이터에서 최적 표현 학습 | 최대 길이 고정(250), 초기화 민감 |

입력 길이가 프로젝트 전체에서 항상 1000으로 고정되므로 학습 가능 위치 임베딩을
선택했다. 0 초기화는 학습 초기에 위치 정보의 방해 없이 Self-Attention이 내용
기반 의존성을 먼저 학습하도록 유도한다.

### 8.4 Post-Norm vs Pre-Norm

PyTorch의 `TransformerEncoderLayer` 기본값(`norm_first=False`)인 Post-Norm을 사용한다.
Post-Norm은 원 Transformer(Vaswani et al., 2017) 및 논문의 구현 환경과 일치한다.
Pre-Norm은 깊은 Transformer에서 학습 안정성이 높다고 알려져 있으나, 논문이
Post-Norm을 채용한 것으로 판단하여 원 설계를 따른다.

### 8.5 CNN branch: 1 스테이지 선택 (C|TTT 대응)

논문 절제 실험(Table 7)의 최고 성능 구성인 C|TTT(CNN 1블록 + Transformer 3블록)를
근거로, CNN 브랜치를 ResNet-50 첫 번째 스테이지(3 bottleneck)로 구현했다.
Transformer가 장거리 의존성 처리에 더 효과적이므로 CNN은 얕게 유지하는 것이
최적으로 나타났다는 논문의 결론을 반영한다.

## 9. 훈련 방법

### 기본 훈련

```bash
bin/train-model --model pctn
```

### 일반적인 설정 예시

```bash
# 배치 크기와 학습률 조정 (파라미터 수가 많아 작은 lr 권장)
bin/train-model --model pctn --batch-size 128 --lr 3e-4

# 더 긴 학습
bin/train-model --model pctn --epochs 150 --patience 20

# 체크포인트 재개
bin/train-model --model pctn --resume data/models/pctn/last.pt
```

### 규모 축소 실험

```bash
# Transformer 레이어 수 감소 (빠른 실험)
bin/train-model --model pctn --model-kwargs "num_tr_layers=3"

# 임베딩 차원 축소
bin/train-model --model pctn --model-kwargs "d_model=128,cnn_mid=32,num_heads=4"
```

## 10. 모델 검사

```bash
bin/print-model --model pctn
```

주요 출력 확인 항목:

| 레이어       | 예상 출력 shape | 의미                       |
| ------------ | --------------- | -------------------------- |
| stem         | (1, 64, 250)    | 4배 다운샘플링 완료        |
| cnn_branch   | (1, 256, 250)   | Bottleneck 확장 (64→256)   |
| tr_encoder   | (1, 250, 256)   | 시퀀스 방향 Attention 출력 |
| channel_att  | (1, 512, 250)   | 두 브랜치 채널 concat 후   |
| fc2          | (1, 2)          | [SBP, DBP] 예측            |
| Total params | 5,126,750       |                            |

## 11. 참고 문헌

- Tian, Z., Liu, A., Zhu, G., and Chen, X. (2025). "A paralleled CNN and Transformer
  network for PPG-based cuff-less blood pressure estimation." *Biomedical Signal
  Processing and Control*, 99, 106741. https://doi.org/10.1016/j.bspc.2024.106741
  (PCTN 원 논문: 전체 아키텍처, Fusion 블록, 실험 설정)

- He, K., Zhang, X., Ren, S., and Sun, J. (2016). "Deep Residual Learning for Image
  Recognition." *CVPR 2016*, pp. 770–778.
  (ResNet bottleneck: 1×1 → 3×3 → 1×1 + shortcut connection)

- Woo, S., Park, J., Lee, J.-Y., and Kweon, I. S. (2018). "CBAM: Convolutional Block
  Attention Module." *ECCV 2018*, pp. 3–19.
  (Spatial Attention: 채널 avg+max pool → conv → sigmoid)

- Hu, J., Shen, L., and Sun, G. (2018). "Squeeze-and-Excitation Networks." *CVPR
  2018*, pp. 7132–7141.
  (Channel Attention: squeeze-excitation, FC 기반 원 설계)

- Vaswani, A. et al. (2017). "Attention Is All You Need." *NeurIPS 2017*.
  (Transformer: MHSA, FFN, Post-Norm, residual connection)
