# AE-LSTM 모델 상세 설계서

## 1. 개요

AE-LSTM(Autoencoder-LSTM)은 Vanithamani et al. (Measurement: Sensors, 2025)이
제안한 LSTM 기반 오토인코더 구조를 이 프로젝트(VitalDB 기반 혈압 추정)에 맞게
구현한 것이다.

- **논문**: R. Vanithamani, S. Sri Jayabharathi, S. Pavithra, and E. Smily Jeya Jothi,
  "Deep learning approaches for continuous blood pressure estimation from
  photoplethysmography signal,"
  *Measurement: Sensors*, vol. 39, p. 101866, June 2025.
  DOI: [10.1016/j.measen.2025.101866](https://doi.org/10.1016/j.measen.2025.101866)
- **구현 파일**: [`bpe/models/ae_lstm.py`](../bpe/models/ae_lstm.py)
- **모델 등록명**: `ae_lstm` (레지스트리 등록)

논문의 핵심 주장은 **LSTM 기반 오토인코더가 PPG 시계열의 시간적 의존성을 압축된
잠재 표현(latent representation)으로 포착하고, 이 표현에서 SBP와 DBP를 추정**할
수 있다는 것이다. 논문은 PhysioNet 손목 PPG 데이터셋에 대해
SBP MAE 1.05 mmHg (SD 1.89), DBP MAE 0.92 mmHg (SD 1.05)를 보고하였으며,
비교 대상인 TCN, LSTM, TCN-LSTM 모두를 능가하였다.

본 구현은 논문의 아키텍처를 최대한 충실히 반영하되, 데이터셋(PhysioNet →
VitalDB)과 출력 방식(BP 분류 → SBP/DBP 직접 회귀)을 이 프로젝트에 맞게 조정한다.
또한 재구성 손실(reconstruction loss)을 보조 학습 목표로 추가하는 멀티태스크
인터페이스를 `compute_loss()`로 제공하며, 이는 이 프로젝트의 MTAE 모델과 동일한
규약을 따른다.

## 2. 전체 아키텍처

```text
입력: PPG 세그먼트
          (B, 1000)  또는  (B, 1, 1000)
                      │
                      ▼  ensure_3d
                 (B, 1, 1000)
                      │
                      ▼  permute(0, 2, 1)
                 (B, 1000, 1)          ← 시계열로 변환 (각 샘플이 1차원 입력)
                      │
    ┌─────────────────┴──────────────────────────┐
    │  _LSTMEncoder                              │
    │  LSTM(input_size=1, hidden_size=64)        │
    │    → 출력: (B, 1000, 64)   [all timesteps] │
    │    → 은닉: h_n (1, B, 64) [최종 은닉 상태] │
    │  h_n.squeeze(0)  → (B, 64)                 │
    │  Dropout(0.2)    → (B, 64)                 │
    └─────────────────┬──────────────────────────┘
               (B, 64)   ← 잠재 벡터 z
        ┌──────────────┴───────────────────────┐
        │ BP Head (추론 + 훈련)                │  Decoder (훈련 시 compute_loss)
   Linear(64→2)                   ┌────────────┴─────────────────────────┐
        │                         │  _LSTMDecoder                        │
     (B, 2)                       │  z.unsqueeze(1)  → (B, 1, 64)        │
  [SBP, DBP]                      │  expand(-1, 1000, -1) → (B, 1000, 64)│
                                  │  LSTM(input_size=64, hidden_size=64) │
                                  │    → (B, 1000, 64)                   │
                                  │  Linear(64→1)   → (B, 1000, 1)       │
                                  │  permute(0,2,1) → (B, 1, 1000)       │
                                  └────────────┬─────────────────────────┘
                                          (B, 1, 1000)
                                          [재구성 PPG]
```

## 3. 텐서 흐름 요약

### _LSTMEncoder

| 단계 | 처리                                  | 입력 shape   | 출력 shape                          |
| ---- | ------------------------------------- | ------------ | ----------------------------------- |
| 0    | ensure_3d                             | (B, 1000)    | (B, 1, 1000)                        |
| 1    | permute(0, 2, 1)                      | (B, 1, 1000) | (B, 1000, 1)                        |
| 2    | LSTM(input=1, hidden=64, batch_first) | (B, 1000, 1) | out: (B, 1000, 64), h_n: (1, B, 64) |
| 3    | h_n.squeeze(0)                        | (1, B, 64)   | (B, 64)                             |
| 4    | Dropout(0.2)                          | (B, 64)      | (B, 64)                             |

### _LSTMDecoder

| 단계 | 처리                                   | 입력 shape    | 출력 shape    |
| ---- | -------------------------------------- | ------------- | ------------- |
| 0    | unsqueeze(1)                           | (B, 64)       | (B, 1, 64)    |
| 1    | expand(-1, 1000, -1).contiguous()      | (B, 1, 64)    | (B, 1000, 64) |
| 2    | LSTM(input=64, hidden=64, batch_first) | (B, 1000, 64) | (B, 1000, 64) |
| 3    | Linear(64→1)                           | (B, 1000, 64) | (B, 1000, 1)  |
| 4    | permute(0, 2, 1)                       | (B, 1000, 1)  | (B, 1, 1000)  |

### BP Head

| 단계 | 처리         | 입력 shape | 출력 shape |
| ---- | ------------ | ---------- | ---------- |
| 0    | Linear(64→2) | (B, 64)    | (B, 2)     |

## 4. 모듈별 상세 설계

### 4.1 LSTM 인코더

**역할**: PPG 파형 전체를 시계열로 처리해, 혈압과 관련된 시간적 특징
(맥박 진폭, 파형 형태, 상승/하강 기울기 등)을 64차원 잠재 벡터로 압축한다.

논문 Section 3.5.1:

> "The encoder's LSTM units allow it to learn dependencies over time and
> represent features like pulse waveform amplitude and shape."

LSTM은 각 시간 스텝에서 망각 게이트(forget gate), 입력 게이트(input gate),
출력 게이트(output gate)를 통해 장기 의존성을 유지하므로, 1000 샘플(8초)에
걸친 PPG 파형의 패턴을 단일 벡터로 요약할 수 있다.

#### LSTM 셀 내부 게이트

```text
fₜ = σ(W_f [hₜ₋₁, xₜ] + b_f)       ← 망각 게이트: 이전 정보 보존 여부
iₜ = σ(W_i [hₜ₋₁, xₜ] + b_i)       ← 입력 게이트: 새 정보 추가 여부
C̃ₜ = tanh(W_C [hₜ₋₁, xₜ] + b_C)    ← 후보 셀 상태
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ           ← 셀 상태 업데이트
oₜ = σ(W_o [hₜ₋₁, xₜ] + b_o)       ← 출력 게이트
hₜ = oₜ ⊙ tanh(Cₜ)                  ← 은닉 상태
```

#### 잠재 벡터 추출 방법

LSTM은 모든 타임스텝의 은닉 상태 `(h₁, h₂, ..., h₁₀₀₀)`를 출력하지만,
인코더는 **마지막 타임스텝의 은닉 상태 `h₁₀₀₀`만을 잠재 벡터로 사용**한다.
이는 LSTM의 셀 상태가 전체 시퀀스에 걸친 정보를 누적해 왔으므로,
마지막 은닉 상태가 전체 PPG 파형의 요약이 된다는 점에 근거한다.

```python
# _LSTMEncoder.forward() 핵심 부분
_, (h_n, _) = self.lstm(x.permute(0, 2, 1))
# h_n: (num_layers=1, B, hidden_size=64)
return self.drop(h_n.squeeze(0))   # (B, 64)
```

#### Dropout 위치

논문 Table 1의 Dropout 0.2를 인코더 LSTM 출력(잠재 벡터) 직후에 적용한다.
이는 잠재 표현이 BP 헤드와 디코더 양쪽으로 전달되기 전 과적합을 억제하는
가장 효과적인 위치이다.

#### 파라미터 수 산출

LSTM 파라미터 공식:

```text
4 × hidden_size × (input_size + hidden_size + 2)
```

> `+2`는 bias_ih와 bias_hh 각각의 게이트별 편향 항

| 항목        | 값                                               |
| ----------- | ------------------------------------------------ |
| input_size  | 1 (샘플 단위 스칼라 입력)                        |
| hidden_size | 64                                               |
| 파라미터 수 | 4 × 64 × (1 + 64 + 2) = 4 × 64 × 67 = **17,152** |

### 4.2 LSTM 디코더

**역할**: 잠재 벡터 `z`에서 원본 PPG 신호를 재구성한다. 이 재구성 경로는
훈련 시 보조 자기지도(self-supervised) 목표로만 사용된다. 즉 디코더가 잘
재구성하려면 인코더가 혈압 관련 파형 정보를 손실 없이 압축해야 하므로,
인코더 표현 품질을 높이는 정규화 역할을 한다.

논문 Section 3.5.1:

> "The decoder then recreates the original input from this compressed form.
> With the help of LSTM layers, the decoder can effectively capture the
> temporal correlations in the data."

#### 디코더 입력 구성 전략: 잠재 벡터 반복 확장

디코더 LSTM은 길이 `L`의 시퀀스를 출력해야 하지만, 입력으로는 1차원 벡터 `z`만
받는다. 이를 해결하기 위해 **`z`를 `L`번 반복(expand)해 `(B, L, H)` 시퀀스를
만들고, 이를 LSTM 입력으로 사용**한다.

```python
# _LSTMDecoder.forward() 핵심 부분
z_seq = z.unsqueeze(1).expand(-1, self.seq_len, -1).contiguous()
# z:     (B, 64)
# z_seq: (B, 1000, 64) — 같은 잠재 벡터를 1000번 반복
h, _ = self.lstm(z_seq)   # (B, 1000, 64)
```

모든 타임스텝에서 동일한 `z`를 입력받지만, LSTM의 순환 연결(recurrent connection)이
각 스텝에서 다른 은닉 상태를 생성하므로 결과적으로 다양한 시계열 패턴을 복원할 수 있다.

| 대안 전략                     | 설명                           | 미채택 이유                           |
| ----------------------------- | ------------------------------ | ------------------------------------- |
| 잠재 벡터 → 초기 은닉 상태    | `h₀ = z`, 입력은 zero sequence | 입력이 없어 `z`의 정보 활용이 제한적  |
| 선형 투영 후 reshape          | `Linear(H, L×H)` → reshape     | 파라미터 과다, 시계열 구조 무시       |
| **잠재 벡터 반복 확장(채택)** | `z`를 `L`번 복사해 시퀀스 구성 | 단순, LSTM 순환성 활용, 파라미터 최소 |

#### 파라미터 수 산출

| 항목          | 값                                                 |
| ------------- | -------------------------------------------------- |
| input_size    | 64 (잠재 벡터 차원)                                |
| hidden_size   | 64                                                 |
| LSTM 파라미터 | 4 × 64 × (64 + 64 + 2) = 4 × 64 × 130 = **33,280** |
| out_proj      | Linear(64→1): 64 × 1 + 1 = **65**                  |

### 4.3 BP 회귀 헤드

**역할**: 잠재 벡터 `z`를 [SBP, DBP] 두 스칼라 값으로 변환한다.

```text
입력 z : (B, 64)
    │
    ▼  Linear(64 → 2)     bias 포함
(B, 2)  ← [SBP(mmHg), DBP(mmHg)]
```

논문은 원래 3-클래스 BP 분류(정상/고혈압 전단계/고혈압)에 Softmax를 사용하였다.
이 구현에서는 VitalDB의 연속적 SBP/DBP 레이블에 맞게 **Softmax 없는 선형 회귀**로
변경한다.

파라미터 수: 64 × 2 + 2 = **130**

## 5. 파라미터 수 분석

| 모듈             | 구성          | 파라미터 수 |
| ---------------- | ------------- | ----------- |
| encoder.lstm     | LSTM(1, 64)   | 17,152      |
| encoder.drop     | Dropout(0.2)  | 0           |
| decoder.lstm     | LSTM(64, 64)  | 33,280      |
| decoder.out_proj | Linear(64, 1) | 65          |
| bp_head          | Linear(64, 2) | 130         |
| **합계**         |               | **50,627**  |

```text
print-model 출력:
  Total params    : 50,627  (50.6 K)
  Trainable params: 50,627  (50.6 K)
  Input shape     : (1, 1000)
```

디코더 LSTM이 전체의 65.7%를 차지하며, 인코더 LSTM이 33.9%를 차지한다.
BP 헤드와 출력 투영은 합계 0.4%에 불과하다.

### 논문의 파라미터 수와의 차이

논문 Table 1은 "No. of Hidden units-100, 53200 parameters"로 기술한다.
53,200은 hidden_size=100, input_size=31 일 때의 단일 LSTM 파라미터
(4 × 100 × (31 + 100 + 2) = 53,200)와 일치한다. 논문이 PPG 파형을 일정 길이
세그먼트 단위로 묶어 입력으로 사용했을 가능성이 있으나, 구체적인 전처리 방식이
명시되지 않아 정확한 재현은 불가능하다.

이 구현은 hidden_size=64로 설정해 각 PPG 샘플(스칼라)을 LSTM의 1차원 입력으로
직접 처리하며, 총 파라미터 수는 50,627(≈ 50.6 K)이다.

## 6. 멀티태스크 손실 및 훈련 인터페이스

### 6.1 멀티태스크 손실 공식

```text
loss = (1 - recon_weight) × bp_loss + recon_weight × recon_loss
```

- `bp_loss`: `criterion(pred, y)` — [SBP, DBP] 예측 오차
- `recon_loss`: `criterion(recon, x)` — PPG 재구성 오차
- 기본값 `recon_weight=0.5`: 두 손실의 동등 가중 합산

논문 Section 3.5.2:

> "This framework encourages the autoencoder to refine its feature extraction
> and reconstruction process until the estimated and actual blood pressure
> readings differ as little as possible by improving the model parameters
> using the mean squared error (MSE) loss function."

### 6.2 Trainer 연동 프로토콜

```python
# Trainer (bpe/train/trainer.py)
if hasattr(self.model, "compute_loss"):
    loss, pred = self.model.compute_loss(x, y, self.criterion)
else:
    pred = self.model(x)
    loss = self.criterion(pred, y)
```

`compute_loss`가 있으면 모델이 손실 계산을 직접 담당한다.
`forward()`는 인코더 → BP 헤드만 실행하므로 평가·추론 속도에 영향이 없다.

### 6.3 compute_loss 흐름

```python
def compute_loss(self, x, y, criterion):
    x3d  = ensure_3d(x)              # (B, 1, 1000)
    z    = self.encoder(x3d)         # (B, 64)
    pred = self.bp_head(z)           # (B, 2)
    recon = self.decoder(z)          # (B, 1, 1000)

    bp_loss    = criterion(pred, y)      # 스칼라
    recon_loss = criterion(recon, x3d)   # 스칼라

    loss = (1.0 - self.recon_weight) * bp_loss
         +         self.recon_weight  * recon_loss
    return loss, pred
```

### 6.4 forward()와 compute_loss()의 실행 경로 차이

| 상황                            | 실행 경로                   | 디코더 실행 |
| ------------------------------- | --------------------------- | ----------- |
| `model(x)` — 추론 / print-model | encoder → bp_head           | 없음        |
| `compute_loss` — 훈련           | encoder → bp_head + decoder | 있음        |
| `compute_loss` — 검증           | encoder → bp_head + decoder | 있음        |

디코더는 훈련과 검증 모두에서 실행된다. 검증 손실에도 재구성 오차가 포함되어야
올바른 `recon_weight` 튜닝이 가능하기 때문이다.

### 6.5 재구성 보조 학습의 효과

잠재 벡터 `z`에서 원본 PPG 파형을 복원하려면, 인코더가 파형의 세밀한 시간적
구조(수축기 피크, 이완기 피크, 중절 노치의 위치와 크기 등)를 손실 없이
압축해야 한다. 이 제약이 인코더를 혈압과 상관된 물리적 특징에 민감하게 만드는
암묵적 정규화 역할을 한다.

## 7. 하이퍼파라미터 참조표

### 모델 하이퍼파라미터

| 파라미터       | 기본값 | 논문 설정 | 역할                                      |
| -------------- | ------ | --------- | ----------------------------------------- |
| `hidden_size`  | 64     | ~100      | LSTM 인코더/디코더 은닉 상태 차원         |
| `dropout`      | 0.2    | 0.2       | 인코더 출력 드롭아웃 비율 (과적합 방지)   |
| `seq_len`      | 1000   | (가변)    | 입·출력 시퀀스 길이 (8 s × 125 Hz)        |
| `recon_weight` | 0.5    | 미명시    | 재구성 손실 가중치 (0: BP만, 1: 재구성만) |
| `out_features` | 2      | 3 (분류)  | 출력 차원 — 이 구현에서는 [SBP, DBP] 회귀 |

### 훈련 하이퍼파라미터 (논문 Table 1 및 프로젝트 표준)

| 파라미터       | 논문 설정                    | 이 프로젝트 표준     |
| -------------- | ---------------------------- | -------------------- |
| Learning rate  | 0.0001                       | 1×10⁻³ (기본값)      |
| Optimizer      | SGDM                         | AdamW                |
| Batch size     | 32 / 125 (두 값이 혼재)      | 256 (기본값)         |
| Epochs         | 30                           | 100 (기본값)         |
| Dropout        | 0.2                          | 0.2                  |
| Loss function  | Cross-entropy (분류)         | HuberLoss(δ=5.0)     |
| Early stopping | 있음 (조기 종료 기준 미명시) | patience=15 (기본값) |

## 8. 논문과의 차이점 및 설계 결정 근거

### 8.1 입력 데이터: PhysioNet → VitalDB

|               | 논문 (PhysioNet)                  | 이 구현 (VitalDB)               |
| ------------- | --------------------------------- | ------------------------------- |
| 출처          | PhysioNet 손목 PPG (운동 중 수집) | VitalDB 수술 중 PPG (안정 상태) |
| 신호 위치     | 손목                              | 손가락 (SNUADC/PLETH)           |
| 샘플링 레이트 | 256 Hz                            | 125 Hz (다운샘플링)             |
| 세그먼트 길이 | 명시 없음                         | 8 s → 1000 샘플                 |
| 레이블        | SBP, DBP (수치 회귀 또는 분류)    | SBP, DBP mean (mmHg)            |

### 8.2 출력 방식: 3-클래스 분류 → 2-값 회귀

논문은 BP를 저혈압/정상/고혈압 3개 클래스로 분류하고 Softmax + 교차엔트로피를
사용한다 (Table 1: "function – crossentropyex"). VitalDB 레이블은 연속적인
SBP/DBP 수치이므로, 이 구현은 **활성화 함수 없는 Linear(64→2) + MSE/Huber 손실**로
회귀를 수행한다.

### 8.3 옵티마이저: SGDM → AdamW

논문은 SGDM(Stochastic Gradient Descent with Momentum)을 사용한다. 이 프로젝트의
모든 모델은 AdamW를 표준 옵티마이저로 사용하므로, 기본 설정을 따른다. 논문의
lr=0.0001은 `--lr 1e-4`로 재현할 수 있다.

### 8.4 재구성 보조 손실 추가

논문은 오토인코더 구조를 사용하지만, 최종 훈련 손실에 재구성 오차를 명시적으로
포함하는지 기술하지 않는다. 이 구현은 MTAE 모델의 멀티태스크 손실 공식을 채용해
PPG 재구성을 보조 자기지도 목표로 추가한다. `recon_weight=0.0`으로 설정하면
재구성 없이 순수 BP 회귀만 수행한다.

### 8.5 디코더 존재 여부

논문은 오토인코더 구조를 BP **분류** 목적으로 사용하고, "decoder recreates the
original input"이라고 명시한다 (Section 3.5.1). 즉 잠재 벡터에서 입력 PPG를
복원하는 디코더가 존재함을 확인할 수 있다.

### 8.6 hidden_size: 100 → 64

논문 Table 1은 "No. of Hidden units-100"으로 기술한다. 이 구현은 64를 기본값으로
선택한다. 이유는 다음과 같다:

- hidden_size=64는 2의 거듭제곱이어서 GPU 연산 효율이 높다
- hidden_size=100으로 설정할 경우 총 파라미터가 약 122 K로 크게 증가한다
- `--model-kwargs "hidden_size=100"`으로 논문 설정을 그대로 재현할 수 있다

hidden_size별 총 파라미터 수:

| hidden_size | 총 파라미터 수 |
| ----------- | -------------- |
| 64 (기본값) | 50,627         |
| 100 (논문)  | 122,303        |
| 128         | 198,274        |

## 9. 훈련 방법

### 기본 훈련

```bash
# Linux / macOS
bin/train-model --model ae_lstm

# Windows
bin\train-model.bat --model ae_lstm
```

### 논문에 가까운 설정

```bash
bin/train-model --model ae_lstm \
    --lr 1e-4 \
    --batch-size 32 \
    --epochs 30
```

### hidden_size=100 (논문 설정)

```bash
bin/train-model --model ae_lstm \
    --model-kwargs "hidden_size=100"
```

### 재구성 보조 없이 순수 BP 회귀

```bash
# recon_weight=0.0 → compute_loss에서 재구성 손실 기여 없음
# (디코더는 forward만 호출되지 않으므로 파라미터만 유지됨)
bin/train-model --model ae_lstm \
    --model-kwargs "recon_weight=0.0"
```

### 재구성 위주 훈련 후 fine-tuning

```bash
# 1단계: 재구성에 집중해 인코더 표현 품질 향상
bin/train-model --model ae_lstm \
    --model-kwargs "recon_weight=0.8" \
    --epochs 30

# 2단계: BP 회귀에 집중 (별도 스크립트 필요 — 체크포인트 이어서 훈련)
bin/train-model --model ae_lstm \
    --model-kwargs "recon_weight=0.2" \
    --resume data/models/ae_lstm/last.pt
```

훈련 하이퍼파라미터 비교:

| 항목       | 논문          | 권장 설정                            |
| ---------- | ------------- | ------------------------------------ |
| 손실 함수  | Cross-entropy | HuberLoss(δ=5.0) (프로젝트 표준)     |
| 옵티마이저 | SGDM          | AdamW, lr=1×10⁻³ (기본값)            |
| 배치 크기  | 32 또는 125   | 256 (기본값); 32 (`--batch-size 32`) |
| 최대 에폭  | 30            | 100 (기본값)                         |
| 조기 종료  | 명시 없음     | `--patience 15` (기본값)             |

## 10. 모델 검사

```bash
# 레이어 구조와 파라미터 수 출력
bin/print-model --model ae_lstm

# 특정 입력 길이로 검사
bin/print-model --model ae_lstm --input-length 1000
```

출력 예시:

```text
======================================================================================
  Model: ae_lstm
======================================================================================
Layer (name)       Type           Output shape   Params
--------------------------------------------------------------------------------------
encoder            _LSTMEncoder   (1, 64)
encoder.lstm       LSTM           (1, 1000, 64)  17.2 K
encoder.drop       Dropout        (1, 64)
decoder            _LSTMDecoder   -
decoder.lstm       LSTM           -              33.3 K
decoder.out_proj   Linear         -                  65
bp_head            Linear         (1, 2)            130
--------------------------------------------------------------------------------------
  Total params    : 50,627  (50.6 K)
  Trainable params: 50,627  (50.6 K)
  Input shape     : (1, 1000)
```

> `decoder` 모듈의 출력 shape가 `-`로 표시되는 이유: `forward()`는 추론 경로
> (인코더 → BP 헤드)만 실행하므로 디코더의 forward hook이 등록되지 않는다.
> 디코더는 `compute_loss()` 경로에서만 실행된다.

## 11. 다른 모델과의 비교

| 모델            | 특징 추출    | 시계열 모델링       | 재구성 보조 | 파라미터 |
| --------------- | ------------ | ------------------- | ----------- | -------- |
| `resnet1d`      | ResNet CNN   | 없음                | 없음        | 2.18 M   |
| `mtae`          | CNN (3층)    | 없음                | 있음 (CNN)  | 119.5 K  |
| `ae_lstm`       | LSTM 인코더  | LSTM 디코더         | 있음 (LSTM) | 50.6 K   |
| `cnn_bilstm_at` | CNN 3층      | BiLSTM + Attention  | 없음        | 691.3 K  |
| `acfa`          | DyCASNet CNN | xLSTM + Transformer | 없음        | 542.6 K  |

`ae_lstm`은 **순수 LSTM만으로 인코더와 디코더를 구성**하는 유일한 모델로,
CNN이나 Attention 없이 시계열 의존성만으로 BP를 추정한다. 파라미터 수가 50.6 K로
이 프로젝트에서 가장 적은 축에 속하며, LSTM의 순차적 처리 특성상 순수 CNN 모델보다
배치당 처리 시간이 길다.

## 12. 참고 문헌

- Vanithamani, R., Sri Jayabharathi, S., Pavithra, S., and Smily Jeya Jothi, E. (2025).
  "Deep learning approaches for continuous blood pressure estimation from
  photoplethysmography signal."
  *Measurement: Sensors*, vol. 39, p. 101866.
  DOI: 10.1016/j.measen.2025.101866

- Hochreiter, S. and Schmidhuber, J. (1997).
  "Long Short-Term Memory."
  *Neural Computation*, vol. 9, no. 8, pp. 1735–1780.

- Cho, K. et al. (2014).
  "Learning Phrase Representations using RNN Encoder–Decoder for Statistical
  Machine Translation."
  *EMNLP 2014.* arXiv:1406.1078.
  (Seq2Seq 오토인코더 구조의 원형)
