# CNN-BiLSTM-AT 모델 상세 설계서

## 1. 개요

CNN-BiLSTM-AT는 Mohammadi et al. (Scientific Reports, 2025)이 제안한 하이브리드
딥러닝 모델을 이 프로젝트(VitalDB 기반 혈압 추정)에 맞게 구현한 것이다.

- **논문**: "Cuff-less blood pressure monitoring via PPG signals using a hybrid
  CNN-BiLSTM deep learning model with attention mechanism"  
  Scientific Reports, vol. 15, p. 22229, 2025.  
  DOI: 10.1038/s41598-025-07087-2
- **구현 파일**: [`bpe/models/cnn_bilstm_at.py`](../bpe/models/cnn_bilstm_at.py)
- **모델 이름**: `cnn_bilstm_at` (레지스트리 등록)

논문의 핵심 주장은 **CNN으로 PPG 신호의 공간적(지역) 특징을 먼저 추출하고,
BiLSTM으로 시계열 의존성을 포착한 뒤, 어텐션 메커니즘이 혈압 추정에 가장
관련 있는 시간 구간에 집중**하면 종래 방법보다 우월한 정확도를 달성할 수 있다는
것이다. 논문은 MIMIC-II 데이터셋 2064명 환자에 대해 5-fold 교차검증으로
SBP MAE 1.88 mmHg, DBP MAE 1.34 mmHg를 보고하였다.

## 2. 전체 아키텍처

```text
입력: PPG 세그먼트
           (B, 1000)  또는  (B, 1, 1000)
                        │
                        ▼  ensure_3d
                   (B, 1, 1000)
                        │
    ┌───────────────────┴──────────────────────────┐
    │  CNN 블록 (3층)                              │
    │  Layer 1: Conv1d(1→32, k=3, pad=1) + ReLU    │
    │           MaxPool1d(size=2, stride=1)        │
    │  Layer 2: Conv1d(32→64, k=3, pad=1) + ReLU   │
    │           MaxPool1d(size=2, stride=1)        │
    │  Layer 3: Conv1d(64→128, k=3, pad=1) + ReLU  │
    │           MaxPool1d(size=2, stride=1)        │
    └───────────────────┬──────────────────────────┘
                 (B, 128, 997)
                        │
                    transpose
                        │
                 (B, 997, 128)
                        │
    ┌───────────────────┴──────────────────────────┐
    │  BiLSTM 1층: hidden=128, bidirectional=True  │
    │  Dropout(0.2)                                │
    └───────────────────┬──────────────────────────┘
                 (B, 997, 256)
                        │
    ┌───────────────────┴──────────────────────────┐
    │  BiLSTM 2층: hidden=128, bidirectional=True  │
    │  Dropout(0.2)                                │
    └───────────────────┬──────────────────────────┘
                 (B, 997, 256)
                        │
    ┌───────────────────┴──────────────────────────┐
    │  Additive Attention                          │
    │  e_t = tanh(W_a·h_t + b_a)   (Eq. 2)         │
    │  α_t = softmax({e_t})          (Eq. 3)       │
    │  c   = Σ_t α_t·h_t            (Eq. 4)        │
    └───────────────────┬──────────────────────────┘
                      (B, 256)
                        │
                    Linear(256→2)
                        │
                      (B, 2)
                   [SBP, DBP] (mmHg)
```

## 3. 텐서 흐름 요약

| 단계 | 처리                        | 입력 shape    | 출력 shape    |
| ---- | --------------------------- | ------------- | ------------- |
| 0    | ensure_3d                   | (B, 1000)     | (B, 1, 1000)  |
| 1    | Conv1d(1→32, k=3) + ReLU    | (B, 1, 1000)  | (B, 32, 1000) |
| 2    | MaxPool1d(size=2, stride=1) | (B, 32, 1000) | (B, 32, 999)  |
| 3    | Conv1d(32→64, k=3) + ReLU   | (B, 32, 999)  | (B, 64, 999)  |
| 4    | MaxPool1d(size=2, stride=1) | (B, 64, 999)  | (B, 64, 998)  |
| 5    | Conv1d(64→128, k=3) + ReLU  | (B, 64, 998)  | (B, 128, 998) |
| 6    | MaxPool1d(size=2, stride=1) | (B, 128, 998) | (B, 128, 997) |
| 7    | transpose                   | (B, 128, 997) | (B, 997, 128) |
| 8    | BiLSTM 1층 + Dropout        | (B, 997, 128) | (B, 997, 256) |
| 9    | BiLSTM 2층 + Dropout        | (B, 997, 256) | (B, 997, 256) |
| 10   | Additive Attention          | (B, 997, 256) | (B, 256)      |
| 11   | Linear(256→2)               | (B, 256)      | (B, 2)        |

## 4. 모듈별 상세 설계

### 4.1 CNN 특징 추출 블록

**역할**: PPG 신호에서 국소적(local) 파형 패턴을 계층적으로 추출한다. 각 층의
필터가 심박수 관련 리듬, 파형 형태, 맥박 특성 등을 자동으로 학습한다.

**수식 참조**: 논문 Eq. 1

```text
f(x) = σ( Σᵢ wᵢ · x_{t-i+1} + b )
```

여기서 `wᵢ`는 필터 가중치, `b`는 편향, `σ`는 ReLU 활성화 함수이다.

#### 각 층의 구성

```text
Layer 1
  Conv1d(in=1, out=32, kernel=3, padding=1)
    ─ 패딩=1: 합성곱 후 길이 보존 (1000 → 1000)
    ─ bias=True (기본값)
  ReLU (inplace)
  MaxPool1d(kernel=2, stride=1)
    ─ stride=1: 길이 1 감소 (1000 → 999)

Layer 2
  Conv1d(in=32, out=64, kernel=3, padding=1)
  ReLU (inplace)
  MaxPool1d(kernel=2, stride=1)   (999 → 998)

Layer 3
  Conv1d(in=64, out=128, kernel=3, padding=1)
  ReLU (inplace)
  MaxPool1d(kernel=2, stride=1)   (998 → 997)
```

#### 설계 메모

논문 Table 4의 최적 하이퍼파라미터:

| 파라미터    | 탐색 범위     | 선택값        |
| ----------- | ------------- | ------------- |
| Filters     | [32, 64, 128] | [32, 64, 128] |
| Kernel size | [3, 5]        | 3             |
| Stride      | [1, 2]        | 1             |
| Pool method | Max, Average  | Max           |
| Pool size   | [2, 3]        | 2             |
| Pool stride | [1, 2, 4]     | 1             |

**MaxPool stride=1 선택 이유**: Pool stride를 1로 선택하면 길이 감소가
층당 1에 그쳐(1000 → 997) BiLSTM에 넘겨지는 시계열의 시간 해상도가
최대한 보존된다. 심박수 60~100 bpm일 때 8초 창에 8~13 사이클이 포함되므로
시간 해상도 손실 없이 BiLSTM이 개별 심박 구조를 인식할 수 있다.

**BatchNorm 미적용**: 논문은 "Conv + ReLU" 구성만 기술하며 배치 정규화를
명시하지 않는다. 논문의 기술을 충실히 따르기 위해 BatchNorm을 생략한다.

### 4.2 BiLSTM 시계열 의존성 포착

**역할**: CNN이 추출한 특징 맵의 시계열 의존성을 양방향으로 모델링한다.
BiLSTM은 순방향(forward)과 역방향(backward) LSTM을 병렬로 실행해 각 시간
스텝에서 과거와 미래 컨텍스트를 동시에 활용한다.

#### BiLSTM 2층 구성

```text
입력 x : (B, 997, 128)   [997 타임스텝, 128 특징]
    │
    ▼  LSTM-forward  (128 → 128)
    │  LSTM-backward (128 → 128)  [역순 입력]
    │  concat → (B, 997, 256)     [순방향 + 역방향 은닉 상태 연결]
    ▼  Dropout(0.2)
(B, 997, 256)   ← BiLSTM 1층 출력
    │
    ▼  LSTM-forward  (256 → 128)
    │  LSTM-backward (256 → 128)
    │  concat → (B, 997, 256)
    ▼  Dropout(0.2)
(B, 997, 256)   ← BiLSTM 2층 출력
```

#### LSTM 셀 내부 게이트 (참고)

```text
fₜ = σ(W_f [hₜ₋₁, xₜ] + b_f)      ← 망각 게이트
iₜ = σ(W_i [hₜ₋₁, xₜ] + b_i)      ← 입력 게이트
C̃ₜ = tanh(W_C [hₜ₋₁, xₜ] + b_C)   ← 후보 셀 상태
Cₜ = fₜ⊙Cₜ₋₁ + iₜ⊙C̃ₜ              ← 셀 상태 업데이트
oₜ = σ(W_o [hₜ₋₁, xₜ] + b_o)      ← 출력 게이트
hₜ = oₜ⊙tanh(Cₜ)                  ← 은닉 상태
```

#### 파라미터 수 산출

| 층       | 입력 크기 | 은닉 크기 | 방향   | LSTM 파라미터 수                        |
| -------- | --------- | --------- | ------ | --------------------------------------- |
| BiLSTM 1 | 128       | 128       | 양방향 | 4 × (128+128+1) × 128 × 2 = **264,192** |
| BiLSTM 2 | 256       | 128       | 양방향 | 4 × (256+128+1) × 128 × 2 = **395,264** |

> LSTM 파라미터: 4(게이트) × (input_size + hidden_size + 1(bias)) × hidden_size

#### Dropout 배치 근거

논문 Table 4의 Dropout ratio = 0.2를 BiLSTM 출력 직후에 적용한다.
이는 BiLSTM 층 간 및 BiLSTM → Attention 전이 시 과적합을 방지하는
가장 일반적인 배치 방식이다.

### 4.3 Additive Attention

**역할**: BiLSTM 전체 시퀀스의 은닉 상태 중 혈압 추정에 가장 유용한
시간 구간에 높은 가중치를 부여해 맥락 벡터(context vector) `c`를 생성한다.
PPG 신호에서 수축기 피크, 이완기 골, 중절 노치(dicrotic notch) 등
특정 구간이 혈압과 강한 상관관계를 가지므로 어텐션이 이를 자동으로 선택할 수 있다.

**수식 참조**: 논문 Eq. 2–4

#### 처리 흐름

```text
입력 h : (B, T, H)    T = 997 타임스텝, H = 256 (BiLSTM 출력 차원)
    │
    ▼  Linear(H → 1)   [W_a ∈ ℝ^{H×1}, b_a ∈ ℝ]
(B, T, 1)
    │
    ▼  tanh              (Eq. 2: e_t = tanh(W_a · h_t + b_a))
(B, T, 1)
    │
    ▼  softmax(dim=1)    (Eq. 3: α_t = exp(eₜ) / Σₜ' exp(eₜ'))
(B, T, 1)
    │
    ▼  weighted sum      (Eq. 4: c = Σ_t α_t · h_t)
       = (alpha * h).sum(dim=1)
(B, H)   ← 맥락 벡터 c
```

#### 수식 구현 대응

| 논문 수식                   | 코드                        | 설명                              |
| --------------------------- | --------------------------- | --------------------------------- |
| `e_t = tanh(W_a·h_t + b_a)` | `torch.tanh(self.score(h))` | `score = Linear(H, 1, bias=True)` |
| `α_t = softmax({e_t})`      | `torch.softmax(e, dim=1)`   | 시간 축(dim=1) 방향으로 정규화    |
| `c = Σ_t α_t · h_t`         | `(alpha * h).sum(dim=1)`    | 브로드캐스트 곱 후 시간 축 합산   |

#### 쿼리 벡터 없는 단일 어텐션

논문의 어텐션은 Bahdanau(2014) 어텐션의 간략화된 형태로,
쿼리(query) 벡터 없이 각 시간 스텝의 은닉 상태를 직접 스칼라 점수로
변환한다. 출력 맥락 벡터의 차원이 은닉 상태 차원(H=256)과 동일하게
유지되어 다음 선형 회귀 층으로 바로 전달된다.

#### 학습 가능 파라미터

| 파라미터 | shape    | 초기값 | 역할                    |
| -------- | -------- | ------ | ----------------------- |
| `W_a`    | (256, 1) | 기본   | 은닉 상태 → 어텐션 점수 |
| `b_a`    | (1,)     | 0      | 어텐션 점수 편향        |

총 257개 파라미터 — 모델 전체 대비 0.04%.

### 4.4 회귀 헤드

**역할**: 어텐션 맥락 벡터 `c`를 [SBP, DBP] 예측값으로 변환한다.

```text
입력 c : (B, 256)
    │
    ▼  Linear(256 → 2)
(B, 2)  ← [SBP, DBP] in mmHg
```

논문은 "맥락 벡터를 사용해 최종 혈압 추정값을 생성한다"고만 기술하므로
단일 선형 층을 사용하는 것이 가장 충실한 해석이다.

## 5. 파라미터 수 분석

| 모듈       | 파라미터 수 |
| ---------- | ----------- |
| CNN 3층    | 31,040      |
| BiLSTM 1층 | 264,192     |
| BiLSTM 2층 | 395,264     |
| Attention  | 257         |
| 회귀 헤드  | 514         |
| **합계**   | **691,267** |

```text
print-model 출력:
  Total params    : 691,267  (691.3 K)
  Trainable params: 691,267  (691.3 K)
  Input shape     : (1, 1000)
```

CNN이 전체의 4.5%에 불과한 반면, BiLSTM 2층이 95% 이상을 차지한다.
이는 시계열 시간 의존성 모델링에 주된 표현력이 집중됨을 반영한다.

## 6. 하이퍼파라미터 참조표

| 파라미터       | 기본값        | 논문 탐색 범위       | 역할                         |
| -------------- | ------------- | -------------------- | ---------------------------- |
| `filters`      | (32, 64, 128) | [32, 64, 128]        | 각 CNN 층의 출력 채널 수     |
| `kernel_size`  | 3             | [3, 5]               | 합성곱 커널 너비             |
| `pool_size`    | 2             | [2, 3]               | MaxPool1d 커널 크기          |
| `lstm_units`   | 128           | [64, 128, 256]       | BiLSTM 방향당 은닉 상태 차원 |
| `dropout`      | 0.2           | [0.1, 0.2, 0.3, 0.4] | BiLSTM 출력 후 드롭아웃 비율 |
| `out_features` | 2             | —                    | 출력 차원 ([SBP, DBP])       |

### 훈련 하이퍼파라미터 (논문 Table 4)

| 파라미터      | 탐색 범위                  | 선택값 |
| ------------- | -------------------------- | ------ |
| Learning rate | [0.0001, 0.001, 0.01, 0.1] | 0.001  |
| Optimizer     | Adam, SGD                  | Adam   |
| Patience      | [30, 50, 60]               | 30     |
| Epochs        | [300, 500, 700]            | 500    |
| Batch size    | [64, 128]                  | 64     |

## 7. 논문과의 차이점 및 설계 결정 근거

### 7.1 입력 길이: 1024 → 1000 샘플

|               | 논문                       | 이 구현               |
| ------------- | -------------------------- | --------------------- |
| 데이터셋      | MIMIC-II, 8.192 s 세그먼트 | VitalDB, 8 s 세그먼트 |
| 샘플링 레이트 | 125 Hz                     | 125 Hz (동일)         |
| 입력 샘플 수  | 1024 (= 8.192 × 125)       | 1000 (= 8 × 125)      |
| CNN 후 길이   | 1021 (= 1024 − 3)          | 997 (= 1000 − 3)      |

LSTM과 Attention 모두 가변 길이 시퀀스를 처리할 수 있으므로 구조 변경이
필요 없다. 이 구현은 임의의 입력 길이에서 동작한다.

### 7.2 MaxPool stride=1 선택 근거

논문은 Pool stride {1, 2, 4} 중 1을 선택했다. Conv1d padding=kernel//2로
합성곱 후 길이를 보존하고, MaxPool(size=2, stride=1)로 층당 1씩만 줄여
시계열 해상도를 최대 보존한다.

stride=2 또는 stride=4를 선택하면 BiLSTM 입력 시퀀스가 짧아져 세밀한
파형 구조 포착이 어려워진다. 예를 들어 stride=4이면 3층 후 길이가
1000/4³ ≈ 16 타임스텝까지 줄어 개별 심박 파형이 소실된다.

### 7.3 배치 정규화(BatchNorm) 미적용

기존 프로젝트 모델(`resnet1d`, `st_resnet` 등)은 `ConvBnAct1d`를 통해
Conv + BN + ReLU 패턴을 기본으로 사용한다. 그러나 원 논문은 BN을 명시하지
않으므로, 논문 기술에 충실하게 Conv + ReLU만 사용한다.

BN을 추가해도 학습 안정성 개선 효과는 있을 수 있으나, 논문 재현성을
우선시해 생략한다.

### 7.4 출력: [SBP, DBP] (MAP 제외)

논문은 SBP, DBP 두 값을 모두 예측한다. VitalDB 데이터셋 레이블도 동일하게
[SBP, DBP] 두 값이므로 `out_features=2`를 기본값으로 한다.

### 7.5 어텐션 쿼리 벡터 없는 단순화

논문 Eq. 2의 어텐션 스코어는 `e_t = tanh(W_a·h_t + b_a)` 형태로,
Bahdanau(2014)의 전형적인 어텐션과 달리 별도의 컨텍스트/쿼리 벡터 `v`를
사용하지 않는다. 즉 디코더 은닉 상태나 글로벌 쿼리 없이 BiLSTM 출력 자체만으로
점수를 계산한다. 구현에서도 `Linear(H, 1)` 하나로 이를 그대로 반영했다.

## 8. 훈련 방법

### 기본 훈련

```bash
bin\train-model.bat --model cnn_bilstm_at
```

### 논문에 가까운 설정

```bash
bin\train-model.bat --model cnn_bilstm_at ^
    --epochs 500 ^
    --batch-size 64 ^
    --lr 1e-3 ^
    --patience 30
```

논문 실험 조건 대비:

| 항목       | 논문           | 권장 설정                           |
| ---------- | -------------- | ----------------------------------- |
| 손실 함수  | MSE + MAE 평가 | MSE 기본값 (프로젝트 표준)          |
| 옵티마이저 | Adam, lr=0.001 | AdamW, lr=1×10⁻³                    |
| 배치 크기  | 64             | 64–256 (메모리 여유에 따라)         |
| 최대 에폭  | 500            | 100 (기본값)로 시작 후 연장         |
| 조기 종료  | patience=30    | `--patience 30`                     |
| 검증 분할  | 훈련 10%       | 별도 val split (데이터셋 레벨 분리) |

> **참고**: 논문은 5-fold 교차검증을 사용하지만, 이 프로젝트는 케이스 레벨로
> train/val/test를 사전 분할해 둔다. 교차검증 대신 고정 분할로 평가한다.

### 경량 실험 (빠른 검증)

```bash
bin\train-model.bat --model cnn_bilstm_at --epochs 30 --batch-size 256
```

BiLSTM은 시퀀스 길이(997)에 순차적 의존성이 있어 CNN 전용 모델보다
배치당 처리 시간이 길다. 배치 크기를 키워 GPU 활용률을 높이면 효과적이다.

## 9. 모델 검사

```bash
# 레이어 구조와 파라미터 수 출력
bin\print-model.bat --model cnn_bilstm_at

# 특정 입력 길이로 검사
bin\print-model.bat --model cnn_bilstm_at --input-length 1000
```

출력 예시:

```text
==========================================================
  Model: cnn_bilstm_at
==========================================================
Layer (name)          Type                 Output shape     Params
----------------------------------------------------------
cnn                   Sequential           (1, 128, 997)
cnn.0                 Conv1d               (1, 32, 1000)       128
cnn.1                 ReLU                 (1, 32, 1000)
cnn.2                 MaxPool1d            (1, 32, 999)
cnn.3                 Conv1d               (1, 64, 999)      6.2 K
cnn.4                 ReLU                 (1, 64, 999)
cnn.5                 MaxPool1d            (1, 64, 998)
cnn.6                 Conv1d               (1, 128, 998)    24.7 K
cnn.7                 ReLU                 (1, 128, 998)
cnn.8                 MaxPool1d            (1, 128, 997)
bilstm1               LSTM                 (1, 997, 256)   264.2 K
drop1                 Dropout              (1, 997, 256)
bilstm2               LSTM                 (1, 997, 256)   395.3 K
drop2                 Dropout              (1, 997, 256)
attention             _AdditiveSelfAttention (1, 256)
attention.score       Linear               (1, 997, 1)         257
head                  Linear               (1, 2)              514
----------------------------------------------------------
  Total params    : 691,267  (691.3 K)
  Trainable params: 691,267  (691.3 K)
  Input shape     : (1, 1000)
```

## 10. 다른 모델과의 비교

| 모델            | 특징 추출   | 시계열 모델링       | 파라미터 | 복잡도 |
| --------------- | ----------- | ------------------- | -------- | ------ |
| `resnet1d`      | ResNet CNN  | 없음 (글로벌 풀)    | 2.18 M   | 낮음   |
| `cnn_bilstm_at` | CNN 3층     | BiLSTM + Attention  | 691.3 K  | 중간   |
| `acfa`          | DyCASNet    | xLSTM + Transformer | 542.6 K  | 높음   |
| `xresnet1d`     | XResNet-101 | 없음 (글로벌 풀)    | 9.47 M   | 높음   |

`cnn_bilstm_at`는 파라미터 수는 적지만 BiLSTM의 순차적 처리 특성상
순수 CNN 모델보다 배치당 추론 시간이 길다.

## 11. 참고 문헌

- Mohammadi, H., Tarvirdizadeh, B., Alipour, K., and Ghamari, M. (2025).
  "Cuff-less blood pressure monitoring via PPG signals using a hybrid
  CNN-BiLSTM deep learning model with attention mechanism."
  *Scientific Reports*, vol. 15, p. 22229.
  DOI: 10.1038/s41598-025-07087-2

- Hochreiter, S. and Schmidhuber, J. (1997).
  "Long Short-Term Memory."
  *Neural Computation*, vol. 9, no. 8, pp. 1735–1780.

- Schuster, M. and Paliwal, K. K. (1997).
  "Bidirectional Recurrent Neural Networks."
  *IEEE Transactions on Signal Processing*, vol. 45, no. 11, pp. 2673–2681.

- Bahdanau, D., Cho, K., and Bengio, Y. (2015).
  "Neural Machine Translation by Jointly Learning to Align and Translate."
  *ICLR 2015.* arXiv:1409.0473.
