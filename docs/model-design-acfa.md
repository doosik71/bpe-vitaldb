# ACFA 모델 상세 설계서

## 1. 개요

ACFA(Adaptive Cross-domain Fusion Architecture)는 Li et al. (IEEE Access, 2026)이
제안한 하이브리드 딥러닝 모델을 이 프로젝트(VitalDB 기반 혈압 추정)에 맞게
구현한 것이다.

- **논문**: "ACFA: A Hybrid Deep Learning Framework for Cuffless Continuous Blood
  Pressure Estimation Using Time–Frequency Adaptive PPG Features"
  DOI: 10.1109/ACCESS.2026.3657471
- **구현 파일**: [`bpe/models/acfa.py`](../bpe/models/acfa.py)
- **모델 이름**: `acfa` (레지스트리 등록)

논문의 핵심 주장은 PPG 신호에서 혈압을 추정할 때 **시간-주파수 이중 도메인
특징 추출 + 장기 시계열 모델링 + 비선형 회귀**를 결합해야 한다는 것이다.

## 2. 전체 아키텍처

```text
입력: PPG 세그먼트
               (B, 1000)  또는  (B, 1, 1000)
                            │
                            ▼  ensure_3d
                       (B, 1, 1000)
                            │
    ┌───────────────────────┴──────────────────────────┐
    │  DyCASNet  (CASB × 2 + DCB × 2, 2단계)           │
    │  - Stem Conv1d(1→64, k=7)                        │
    │  - Stage 1: CASB₁ (res) → DCB₁ (res) → BN        │
    │  - Stage 2: CASB₂ (res) → DCB₂ (res) → BN        │
    └───────────────────────┬──────────────────────────┘
                     (B, 64, 1000)
                            │
                    AvgPool1d(stride=4)
                            │
                      (B, 64, 250)
                            │
                        transpose
                            │
                      (B, 250, 64)
                            │
    ┌───────────────────────┴──────────────────────────┐
    │  xLSTMStack  (4 layers, 교대 sLSTM / mLSTM)      │
    │  Layer 0: sLSTMBlock  (causal conv + BiLSTM)     │
    │  Layer 1: mLSTMBlock  (causal conv + causal MHA) │
    │  Layer 2: sLSTMBlock                             │
    │  Layer 3: mLSTMBlock                             │
    └────────────────────────┬─────────────────────────┘
                       (B, 250, 64)
                             │
    ┌────────────────────────┴─────────────────────────┐
    │  TransformerBranch  (2 layers, 4 heads)          │
    │  pos_embed(learnable) + pre-LN TransformerEncoder│
    └────────────────────────┬─────────────────────────┘
                        (B, 250, 64)
                             │
                    AdaptiveAvgPool1d(1)
                             │
                          (B, 64)
                             │
    ┌────────────────────────┴─────────────────────────┐
    │  FKAN  (FastKANLayer(64→128) + Dropout + Linear) │
    └────────────────────────┬─────────────────────────┘
                          (B, 2)
                        [SBP, DBP] (mmHg)
```

## 3. 텐서 흐름 요약

| 단계 | 처리                           | 입력 shape    | 출력 shape    |
| ---- | ------------------------------ | ------------- | ------------- |
| 0    | ensure_3d                      | (B, 1000)     | (B, 1, 1000)  |
| 1    | DyCASNet.stem                  | (B, 1, 1000)  | (B, 64, 1000) |
| 2    | DyCASNet CASB₁+DCB₁ Stage 1    | (B, 64, 1000) | (B, 64, 1000) |
| 3    | DyCASNet CASB₂+DCB₂ Stage 2    | (B, 64, 1000) | (B, 64, 1000) |
| 4    | AvgPool1d(stride=4)            | (B, 64, 1000) | (B, 64, 250)  |
| 5    | transpose                      | (B, 64, 250)  | (B, 250, 64)  |
| 6    | xLSTMStack (4 layers)          | (B, 250, 64)  | (B, 250, 64)  |
| 7    | TransformerBranch (2 layers)   | (B, 250, 64)  | (B, 250, 64)  |
| 8    | AdaptiveAvgPool1d(1) + squeeze | (B, 250, 64)  | (B, 64)       |
| 9    | FKAN                           | (B, 64)       | (B, 2)        |

## 4. 모듈별 상세 설계

### 4.1 CASB (Channel-Aware Spectral Block)

**역할**: 주파수 도메인에서 적응적 잡음 제거와 채널별 중요도 재조정을 수행한다.

**수식 참조**: 논문 Eq. 2–6

#### CASB 처리 흐름

```text
입력 x : (B, C, L)
    │
    ▼  rfft (dim=-1)
F_x : (B, C, L//2+1)  [complex64]
    │
    ├─ P = |F_x|²       (B, C, L//2+1) [power spectrum]
    │      │
    │      ▼  P > |θ|   [학습 가능한 채널별 임계값, shape (1,C,1)]
    │  mask : (B, C, L//2+1) [float]
    │      │
    │      ▼
    │  F_filt = F_x ⊙ mask     [noise band 제거]
    │
    ├─ F_ref = F_x × gw + F_filt × lw
    │          ↑ gw = complex(global_r, global_i)  (1, C, 1)
    │          ↑ lw = complex(local_r, local_i)    (1, C, 1)
    │          [학습 가능한 복소수 가중치]
    │
    ├─ energy = mean(|F_ref|, dim=-1)  (B, C)
    │      │
    │      ▼  SE(energy) [Linear(C→C//4)→ReLU→Linear(C//4→C)→Sigmoid]
    │  scale : (B, C, 1)
    │      │
    │      ▼
    │  F_att = F_ref × scale
    │
    ▼  irfft(F_att, n=L)
출력 : (B, C, L)
```

#### 학습 가능 파라미터

| 파라미터               | shape          | 초기값 | 역할                            |
| ---------------------- | -------------- | ------ | ------------------------------- |
| `threshold`            | (1, C, 1)      | 0      | 채널별 주파수 마스킹 임계값 θ   |
| `global_r`, `global_i` | (1, C, 1) 각각 | 1, 0   | 전역 스펙트럼 복소수 가중치     |
| `local_r`, `local_i`   | (1, C, 1) 각각 | 1, 0   | 필터링된 스펙트럼 복소수 가중치 |
| SE Linear(C, C//4)     | (C, C//4)      | 기본   | 채널 중요도 압축                |
| SE Linear(C//4, C)     | (C//4, C)      | 기본   | 채널 중요도 복원                |

> **구현 메모**: 복소수 파라미터는 실수부/허수부 두 개의 `nn.Parameter`로 분리
> 저장한다. PyTorch의 일부 구버전에서 complex dtype을 `nn.Parameter`로 직접
> 사용하면 직렬화(serialize)가 불안정하기 때문이다.
> Forward 시 `torch.complex(global_r, global_i)`로 즉석에서 조합한다.

### 4.2 DCB (Dynamic Convolution Block)

**역할**: 소수용체(local) 및 대수용체(long-range) 1D 합성곱의 출력을 입력 의존적
동적 가중치로 융합하고, SE 채널 어텐션으로 마무리한다.

**수식 참조**: 논문 Eq. 7–10

#### DCB 처리 흐름

```text
입력 x : (B, C, L)
    │
    ├─ A1 = GELU(Conv1d(k=3)(x))    (B, C, L)  [local 패턴]
    ├─ A2 = GELU(Conv1d(k=15)(x))   (B, C, L)  [long-range 패턴]
    │
    ├─ alpha_net(x):
    │    AdaptiveAvgPool1d(1) → Flatten → Linear(C, K) → Softmax
    │    alpha : (B, K)
    │
    ├─ W_mix = Σ_k  alpha_k × W_k        W: (K, C, 1), alpha: (B, K)
    │           ─────────────────→ (B, C, 1)
    │
    ├─ O = (A1 + A2) × W_mix            (B, C, L)
    │
    ├─ SE(O):
    │    AdaptiveAvgPool1d(1) → Flatten → Linear(C, C//4) → ReLU
    │    → Linear(C//4, C) → Sigmoid → unsqueeze(-1)
    │    scale : (B, C, 1)
    │
    ▼
출력 = O × scale  : (B, C, L)
```

#### 설계 메모

논문 Eq. 8의 `Σ_k α_k (A1 ⊙ W_k + A2 ⊙ W_k)` 에서 W_k는
채널별 스케일 파라미터 (shape `(K, C, 1)`)이다.  
구현에서는 이를 `(A1 + A2) × Σ_k(α_k W_k)` 형태로 수학적으로 동등하게
재정렬하여 브로드캐스트 연산 한 번으로 처리한다.

### 4.3 DyCASNet

**역할**: CASB(주파수 도메인)와 DCB(시간 도메인)를 결합한 이중 도메인 특징 추출기.

```text
입력: (B, 1, L)
    │
    ▼  Stem: Conv1d(1→d, k=7, pad=3) + BN + GELU
(B, d, L)
    │
    ▼  CASB₁(x) + x  [주파수 도메인 잔차]
(B, d, L)
    │
    ▼  DCB₁(·) + x   [시간 도메인 잔차]
(B, d, L)
    │
    ▼  BN
(B, d, L)  ← Stage 1 완료
    │
    ▼  CASB₂(x) + x
(B, d, L)
    │
    ▼  DCB₂(·) + x
(B, d, L)
    │
    ▼  BN
출력: (B, d, L)  ← Stage 2 완료
```

논문은 CASB와 DCB 각 한 단계를 묘사하나, 이 구현에서는 표현력 향상을 위해
**2단계 직렬 적층**을 채택한다. 각 단계는 CASB와 DCB를 순차 적용하며
각각 잔차 연결을 갖는다.

### 4.4 sLSTMBlock (Scalar LSTM)

**역할**: 논문 xLSTM의 sLSTM 변형에 해당한다. 인과적(causal) 합성곱 전처리 후
BiLSTM으로 지역 시계열 의존성을 포착한다.

```text
입력 x : (B, L, D)
    │
    ├─ 인과적 뎁스와이즈 Conv1d(k=4, pad=3, groups=D)
    │    패딩: 좌측에 3 패드, 우측은 슬라이스로 제거 → 출력 길이 = L (인과 보장)
    │  xc : (B, L, D)
    │
    ├─ LayerNorm(x + xc)   [사전 정규화 잔차]
    │
    ├─ BiLSTM(D → 2D)
    │    out : (B, L, 2D)
    │
    ├─ Linear(2D → D)  [proj]
    │
    ▼  LayerNorm(x + proj(out))
출력 : (B, L, D)
```

#### 인과적 합성곱 구현 방법

`padding=3` (대칭)으로 Conv1d를 적용하면 출력 길이가 `L+3`이 된다.
`[:, :, :L]`로 슬라이스하면 각 위치 t의 출력이 입력 위치 `[t−3, t−2, t−1, t]`
(모두 ≤ t)만 참조하므로 인과성이 보장된다.

### 4.5 mLSTMBlock (Matrix LSTM)

**역할**: 논문 xLSTM의 mLSTM 변형에 해당한다. 인과적 멀티헤드 셀프 어텐션으로
고차 비선형 시계열 상호작용을 모델링하고, mLSTM 스타일 출력 게이트를 적용한다.

```text
입력 x : (B, L, D)
    │
    ├─ 인과적 뎁스와이즈 Conv1d(k=4)  → xc : (B, L, D)
    │
    ├─ x_in = LayerNorm(x + xc)
    │
    ├─ Q, K, V = Linear(D → 3D)(x_in).split(D, dim=-1)
    │    reshape to (B, H, L, D/H)
    │
    ├─ h = scaled_dot_product_attention(Q, K, V, is_causal=True)
    │    [Flash Attention — CUDA에서 O(L) 메모리]
    │    reshape back to (B, L, D)
    │    Linear(D → D)  → h
    │
    ├─ o = Sigmoid(Linear(D → D)(x_in))   [출력 게이트]
    │
    ▼  LayerNorm(x + o ⊙ h)
출력 : (B, L, D)
```

#### 논문 mLSTM과의 차이

| 논문 mLSTM                                      | 이 구현                                                         |
| ----------------------------------------------- | --------------------------------------------------------------- |
| 행렬 메모리 C ∈ ℝ^{D×D}, outer-product 업데이트 | 인과적 멀티헤드 셀프 어텐션으로 근사                            |
| 순차 recurrence (O(L) 처리)                     | PyTorch `F.scaled_dot_product_attention` — CUDA Flash Attention |
| 입력 / 망각 게이트 (i, f)                       | 단순화: 출력 게이트 (o)만 유지                                  |

원 논문의 행렬 메모리 outer-product 업데이트를 Python 수준에서 구현하면
L=1000 길이 시퀀스에서 `O(L × D²)` 순차 루프가 필요해 훈련이 수십 배 느려진다.
Flash Attention 기반 인과적 어텐션으로 근사하면 CUDA에서 `O(L)` 메모리,
`O(L²)` 연산의 효율적 구현이 가능하다.

### 4.6 xLSTMStack

**역할**: sLSTMBlock(짝수 인덱스)과 mLSTMBlock(홀수 인덱스)을 교대로 적층한다.

```text
index 0 → sLSTMBlock  (causal conv + BiLSTM)
index 1 → mLSTMBlock  (causal conv + causal attn)
index 2 → sLSTMBlock
index 3 → mLSTMBlock
```

각 블록이 자체적으로 LayerNorm과 잔차 연결을 포함하므로 스택 자체는
단순한 순차 실행이다.

### 4.7 TransformerBranch

**역할**: 전체 시퀀스에 걸친 전역 컨텍스트를 멀티헤드 셀프 어텐션으로 포착한다.

```text
입력 x : (B, L, D)
    │
    ├─ pos_embed : (1, L, D)   [학습 가능 위치 임베딩, trunc_normal 초기화]
    │    L 불일치 시 F.interpolate로 자동 보간
    │
    ▼  x + pos_embed
(B, L, D)
    │
    ▼  TransformerEncoderLayer × num_layers
       [pre-LN, nhead=4, dim_ff=D×4, activation=GELU]
출력 : (B, L, D)
```

**Pre-LN(norm_first=True) 선택 이유**: post-LN보다 학습 초기 안정성이 높고
잔차 스케일이 레이어별로 고르게 유지된다.

### 4.8 FastKANLayer

**역할**: RBF(Radial Basis Function) 기저 함수 기반 비선형 변환을 구현한다.

**수식 참조**: 논문 Eq. 15

```text
입력 x : (N, F)  [F = in_features]
    │
    ├─ [메인 경로]
    │    x_exp = x.unsqueeze(-1)                       (N, F, 1)
    │    rbf = exp(-(x_exp - centers)² / (2σ²))        (N, F, M)
    │           ↑ centers : (F, M) 학습 가능 (균등 격자 초기화)
    │           ↑ σ = exp(log_widths) : (F, M) 학습 가능
    │    rbf_flat = rbf.reshape(N, F×M)
    │    main = Linear(F×M → out)(rbf_flat)            (N, out)
    │
    ├─ [보조 경로]
    │    aux = Linear(F → out)(x)                      (N, out)
    │
    ▼  LayerNorm(main + aux)
출력 : (N, out)
```

RBF 기저는 지역적 감응성(local sensitivity)을 가져, 입력 공간의 특정 영역에서
선택적으로 활성화된다. 이를 통해 PPG-혈압 간 복잡한 비선형 매핑을 표현한다.

### 4.9 FKAN

**역할**: `FastKANLayer`를 쌓고 마지막에 선형 출력층을 붙인 회귀 헤드.

기본 설정 (`fkan_layers=2`, `fkan_hidden=128`):

```text
(B, 64)
    │
    ▼  FastKANLayer(64 → 128)
(B, 128)
    │
    ▼  Dropout(0.1)
(B, 128)
    │
    ▼  Linear(128 → 2)
(B, 2)  ← [SBP, DBP]
```

## 5. 하이퍼파라미터 참조표

| 파라미터       | 기본값 | 역할                                          |
| -------------- | ------ | --------------------------------------------- |
| `d_model`      | 64     | 전 레이어 공통 채널/임베딩 차원               |
| `xlstm_layers` | 4      | xLSTM 총 레이어 수 (sLSTM: 짝수, mLSTM: 홀수) |
| `xlstm_heads`  | 4      | mLSTMBlock 내 어텐션 헤드 수                  |
| `tr_layers`    | 2      | Transformer 인코더 레이어 수                  |
| `tr_nhead`     | 4      | Transformer 어텐션 헤드 수                    |
| `fkan_hidden`  | 128    | FKAN FastKANLayer 히든 차원                   |
| `fkan_layers`  | 2      | FKAN 레이어 수 (마지막은 Linear)              |
| `num_basis`    | 8      | FastKANLayer 입력 차원당 RBF 기저 수          |
| `num_kernels`  | 4      | DCB 동적 가중치 커널 셋 수 (K)                |
| `reduction`    | 4      | CASB·DCB SE 병목 축소 비율                    |
| `pool_stride`  | 4      | DyCASNet 후 시간 축 다운샘플 보폭             |
| `dropout`      | 0.1    | Transformer · FKAN Dropout 비율               |
| `input_length` | 1000   | PPG 입력 샘플 수 (위치 임베딩 크기 결정)      |
| `out_features` | 2      | 출력 차원 ([SBP, DBP])                        |

### 파라미터 수 예시

| 설정             | d_model          | 총 파라미터 |
| ---------------- | ---------------- | ----------- |
| 기본값 (default) | 64               | 542.6 K     |
| 중형             | 128              | ~3.8 M      |
| 논문 원형 근사   | 128 + depth 증가 | ~4.6 M      |

논문과 파라미터 수를 맞추려면:

```bash
bin\train-model.bat --model acfa \
    --model-kwargs d_model=128 xlstm_layers=6 tr_layers=4 fkan_hidden=256
```

## 6. 논문과의 차이점 및 설계 결정 근거

### 6.1 입력 길이: 789 → 1000 샘플

|              | 논문                     | 이 구현                            |
| ------------ | ------------------------ | ---------------------------------- |
| 데이터셋     | MIMIC-III (6 s 세그먼트) | VitalDB (8 s, 125 Hz = 1 000 샘플) |
| 입력 샘플 수 | 789                      | 1 000                              |

`input_length` 인자로 변경 가능하며, 위치 임베딩은 `F.interpolate`로 자동 보간된다.

### 6.2 DyCASNet 후 시간 축 다운샘플 (pool_stride=4)

논문은 DyCASNet 이후 원본 시계열 해상도(789 타임스텝)를 유지한 채 xLSTM과
Transformer를 적용한다고 기술한다.

이 구현에서 1 000 타임스텝을 그대로 Transformer에 넣으면 배치 크기 256 기준
어텐션 행렬 메모리가 `256 × 4 × 1000 × 1000 × 4 bytes ≈ 4 GB`에 달한다.
Flash Attention을 사용해도 연산량이 매우 크다.

`AvgPool1d(stride=4)`로 250 타임스텝으로 축소하면:

- 어텐션 메모리: `256 × 4 × 250 × 250 × 4 bytes ≈ 256 MB` (관리 가능)
- 250 타임스텝 = 2초 분량 (125 Hz 기준) → 2~3 심박 사이클 포함
- 시계열 패턴을 포착하기에 충분한 해상도

### 6.3 mLSTM: 행렬 메모리 outer-product → 인과 어텐션 근사

논문 mLSTM은 `C_t = f_t C_{t-1} + i_t v_t k_t^T`의 행렬 메모리 재귀식을 사용한다.
L=250 시퀀스에서 Python 수준 recurrence 루프는 배치당 `250 × D²` 연산이
순차 실행되어 GPU 병렬화가 불가능하다.

`F.scaled_dot_product_attention(is_causal=True)`은 수학적으로 유사한
시퀀스 내 위치 간 의존성을 모델링하면서 CUDA Flash Attention 커널을 활용해
훈련 효율성을 유지한다.

### 6.4 출력: [SBP, DBP] (MAP 제외)

논문은 SBP, DBP, MAP 세 가지를 예측한다.
VitalDB 데이터셋 레이블은 [SBP, DBP] 2개이므로 `out_features=2`를 기본값으로 한다.

### 6.5 DyCASNet 2단계 적층

논문 그림 1에는 CASB와 DCB를 각 1회 적용하는 단일 DyCASNet이 묘사된다.
이 구현에서는 특징 표현력을 높이기 위해 CASB+DCB 쌍을 2단계 직렬 적용한다.
추가 파라미터 대비 표현력 향상 효과가 ablation에서도 확인된 DyCASNet의
중요도(논문: 제거 시 SBP MAE 2.21 → 6.53 mmHg)를 감안한 결정이다.

## 7. 훈련 방법

### 기본 훈련

```bash
bin\train-model.bat --model acfa
```

### 논문에 가까운 설정

```bash
bin\train-model.bat --model acfa \
    --epochs 100 \
    --batch-size 128 \
    --lr 3e-4
```

논문 실험 조건 (참고용):

| 항목       | 논문            | 권장 설정                        |
| ---------- | --------------- | -------------------------------- |
| 손실 함수  | MAE             | MAE (기본값)                     |
| 옵티마이저 | Adam, lr=3×10⁻⁴ | AdamW, lr=1×10⁻³ (기본) → 3×10⁻⁴ |
| 배치 크기  | 128             | 128–256                          |
| 최대 에폭  | 100             | 100                              |
| 조기 종료  | 적용            | `--patience 15` (기본)           |

### 아키텍처 크기 조정

```bash
# 경량 (빠른 실험)
bin\train-model.bat --model acfa   # 기본값: d_model=64, 542 K 파라미터

# 논문 규모 근사 (약 3.8 M 파라미터)
bin\train-model.bat --model acfa --model-kwargs "d_model=128"
```

## 8. 모델 검사

```bash
# 레이어 구조와 파라미터 수 출력
bin\print-model.bat --model acfa

# 특정 입력 길이로 검사
bin\print-model.bat --model acfa --input-length 1000
```

출력 예시:

```text
Total params    : 542,602  (542.6 K)
Trainable params: 542,602  (542.6 K)
Input shape     : (1, 1000)
```

## 9. 참고 문헌

- Li, F., Li, L., Wang, L., and Xu, H. (2026). "ACFA: A Hybrid Deep Learning
  Framework for Cuffless Continuous Blood Pressure Estimation Using
  Time–Frequency Adaptive PPG Features." *IEEE Access*, vol. 14, pp. 20839–20855.
  DOI: 10.1109/ACCESS.2026.3657471

- Beck, M. et al. (2024). "xLSTM: Extended Long Short-Term Memory."
  *Proc. 1st Workshop Long-Context Found. Models @ ICML 2024.*

- Hu, J., Shen, L., and Sun, G. (2018). "Squeeze-and-Excitation Networks."
  *CVPR 2018.*

- Chen, Y. et al. (2020). "Dynamic Convolution: Attention Over Convolution Kernels."
  *CVPR 2020.*
