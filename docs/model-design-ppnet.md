# PP-Net 모델 상세 설계서

## 1. 개요

PP-Net은 Panwar et al. (IEEE Sensors Journal, 2020)이 제안한 LRCN(Long-term
Recurrent Convolutional Network) 기반 하이브리드 딥러닝 모델을 이 프로젝트
(VitalDB 기반 혈압 추정)에 맞게 구현한 것이다.

- **논문**: "PP-Net: A Deep Learning Framework for PPG-Based Blood Pressure and
  Heart Rate Estimation"  
  IEEE Sensors Journal, vol. 20, no. 17, pp. 10000–10011, Sep. 2020.  
  DOI: 10.1109/JSEN.2020.2990864
- **구현 파일**: [`bpe/models/ppnet.py`](../bpe/models/ppnet.py)
- **모델 이름**: `ppnet` (레지스트리 등록)

논문의 핵심 주장은 **CNN이 PPG 신호에서 국소적(local) 파형 패턴을 추출하고,
LSTM이 시계열 순차 의존성을 포착**하는 LRCN 구조가 CNN 단독 또는 LSTM 단독
모델보다 혈압 및 심박수를 더 정확하게 추정할 수 있다는 것이다. 논문은
MIMIC-II 데이터셋 1,557명 환자에 대해 10-fold 교차검증으로 SBP 및 DBP 평균
MAE 3.14 ± 0.13 mmHg, AAMI 기준 통과, BHS 기준 Grade A를 달성하였다.

또한 논문은 1,000 샘플 입력을 4배 다운샘플하여 250 샘플로 줄이는 전처리가
CNN의 inherent stride 방식보다 성능과 복잡도 양쪽에서 더 우수함을 실험으로
검증하였다(논문 Table II).

## 2. 전체 아키텍처

```text
입력: PPG 세그먼트
           (B, 1000)  또는  (B, 1, 1000)
                        │
                        ▼  ensure_3d
                   (B, 1, 1000)
                        │
              ┌─────────┴─────────┐
              │  다운샘플 단계    │
              │  AvgPool1d(4,4)   │
              └─────────┬─────────┘
                   (B, 1, 250)
                        │
    ┌───────────────────┴──────────────────────────┐
    │  CNN 블록 1                                  │
    │  Conv1d(1→20, kernel=9, padding=4)           │
    │  ReLU                                        │
    │  MaxPool1d(kernel=4, stride=4)               │
    │  Dropout(0.1)                                │
    └───────────────────┬──────────────────────────┘
                   (B, 20, 62)
                        │
    ┌───────────────────┴──────────────────────────┐
    │  CNN 블록 2                                  │
    │  Conv1d(20→20, kernel=9, padding=4)          │
    │  ReLU                                        │
    │  MaxPool1d(kernel=4, stride=4)               │
    │  Dropout(0.1)                                │
    └───────────────────┬──────────────────────────┘
                   (B, 20, 15)
                        │
                    transpose
                        │
                   (B, 15, 20)  ← LSTM 입력: 15 타임스텝, 20 특징
                        │
    ┌───────────────────┴──────────────────────────┐
    │  LSTM 1층: hidden=64, tanh                   │
    │  Dropout(0.1)                                │
    └───────────────────┬──────────────────────────┘
                   (B, 15, 64)
                        │
    ┌───────────────────┴──────────────────────────┐
    │  LSTM 2층: hidden=128, tanh                  │
    │  Dropout(0.1)                                │
    └───────────────────┬──────────────────────────┘
                   (B, 15, 128)
                        │
                 최종 타임스텝 선택 [:, -1, :]
                        │
                      (B, 128)
                        │
                   Linear(128→2)
                        │
                      (B, 2)
                   [SBP, DBP] (mmHg)
```

## 3. 텐서 흐름 요약

| 단계 | 처리                            | 입력 shape   | 출력 shape   |
| ---- | ------------------------------- | ------------ | ------------ |
| 0    | ensure_3d                       | (B, 1000)    | (B, 1, 1000) |
| 1    | AvgPool1d(kernel=4, stride=4)   | (B, 1, 1000) | (B, 1, 250)  |
| 2    | Conv1d(1→20, k=9, pad=4)        | (B, 1, 250)  | (B, 20, 250) |
| 3    | ReLU                            | (B, 20, 250) | (B, 20, 250) |
| 4    | MaxPool1d(kernel=4, stride=4)   | (B, 20, 250) | (B, 20, 62)  |
| 5    | Dropout(0.1)                    | (B, 20, 62)  | (B, 20, 62)  |
| 6    | Conv1d(20→20, k=9, pad=4)       | (B, 20, 62)  | (B, 20, 62)  |
| 7    | ReLU                            | (B, 20, 62)  | (B, 20, 62)  |
| 8    | MaxPool1d(kernel=4, stride=4)   | (B, 20, 62)  | (B, 20, 15)  |
| 9    | Dropout(0.1)                    | (B, 20, 15)  | (B, 20, 15)  |
| 10   | transpose(1, 2)                 | (B, 20, 15)  | (B, 15, 20)  |
| 11   | LSTM 1층 + Dropout(0.1)         | (B, 15, 20)  | (B, 15, 64)  |
| 12   | LSTM 2층 + Dropout(0.1)         | (B, 15, 64)  | (B, 15, 128) |
| 13   | 최종 타임스텝 선택 `[:, -1, :]` | (B, 15, 128) | (B, 128)     |
| 14   | Linear(128→2)                   | (B, 128)     | (B, 2)       |

## 4. 모듈별 상세 설계

### 4.1 다운샘플 단계

**역할**: 1,000 샘플 입력을 250 샘플로 4배 압축하여 CNN-LSTM의 연산 복잡도를
줄인다. 논문 Section III-A에서 "PPG data are down-sampled with a scaling factor
of 4"로 명시된 전처리 단계이며, 본 구현에서는 모델 내부 첫 단계로 포함한다.

**구현**: `nn.AvgPool1d(kernel_size=4, stride=4)`

평균 풀링(AvgPool)을 사용하는 이유:
- 논문은 다운샘플링 방법을 명시하지 않는다.
- AvgPool은 4개 샘플의 평균을 취해 DC 성분과 저주파 심박 파형을 보존한다.
- MaxPool보다 에일리어싱(aliasing) 위험이 낮다.
- 단순 데시메이션(스트라이드만 적용)보다 노이즈 억제 효과가 있다.

**시퀀스 길이**: `floor(1000 / 4) = 250`

논문 Table II 재현성:
논문은 1000 샘플(미압축) 대비 250 샘플(4× 압축)에서 NMAE 차이 0.005로
성능 손실이 미미하면서 연산량이 약 4배 감소함을 보고하였다. 본 구현은
이 결과를 근거로 압축 방식을 선택한다.

### 4.2 CNN 블록 1

**역할**: 단일 채널 PPG 파형에서 국소적 파형 특징을 20개의 1D 필터로 추출한다.
각 필터는 크기 9 수용 영역(receptive field)으로 심박 파형의 피크, 기울기, 면적
등 저수준(low-level) 패턴을 학습한다.

```text
(B, 1, 250)
    │
    ▼  Conv1d(in=1, out=20, kernel=9, padding=4, bias=True)
       ─ padding=4 = (9-1)//2: 합성곱 후 길이 보존 (250 → 250)
(B, 20, 250)
    │
    ▼  ReLU(inplace)
(B, 20, 250)
    │
    ▼  MaxPool1d(kernel=4, stride=4)
       ─ 공간 해상도를 75% 감축 (250 → 62)
       ─ 위치 불변성 확보 + 연산량 감소
(B, 20, 62)
    │
    ▼  Dropout(0.1)
(B, 20, 62)
```

**MaxPool 설계 근거**: 논문은 "pooling is applied along the spatial dimensions
by 4 × 1 using the max operation. This progressively reduces the spatial size of
representation by 75%"로 기술한다. stride=4(= kernel=4)로 비중첩 풀링을 적용해
출력 길이를 정확히 1/4로 줄인다.

**논문 MAC 계산 검증**:
```text
Conv1d MACs = 출력길이 × kernel × in_ch × out_ch
            = 250 × 9 × 1 × 20 = 45,000  ← 논문 Table IX "Conv1: 45K" 일치
```

#### 파라미터 수

| 파라미터 | shape      | 수      |
| -------- | ---------- | ------- |
| weight   | (20, 1, 9) | 180     |
| bias     | (20,)      | 20      |
| **합계** |            | **200** |

### 4.3 CNN 블록 2

**역할**: CNN 블록 1의 출력(20채널 특징 맵)에서 고수준(high-level) 복합 패턴을
추가로 추출하고, 시퀀스 길이를 더 줄여 LSTM 입력 차원을 최적화한다.

```text
(B, 20, 62)
    │
    ▼  Conv1d(in=20, out=20, kernel=9, padding=4, bias=True)
       ─ padding=4: 합성곱 후 길이 보존 (62 → 62)
(B, 20, 62)
    │
    ▼  ReLU(inplace)
(B, 20, 62)
    │
    ▼  MaxPool1d(kernel=4, stride=4)
       ─ 길이 축소: 62 → 15  [floor(62/4) = 15]
(B, 20, 15)
    │
    ▼  Dropout(0.1)
(B, 20, 15)
```

**출력 시퀀스 길이 15**: 이후 LSTM의 타임스텝 수가 된다. 8초 창에서 심박수
60~100 bpm 기준으로 8~13 사이클이 포함되므로, 15 타임스텝은 개별 심박 구조를
포착하기에 충분한 해상도를 유지한다.

#### 파라미터 수

| 파라미터 | shape       | 수        |
| -------- | ----------- | --------- |
| weight   | (20, 20, 9) | 3,600     |
| bias     | (20,)       | 20        |
| **합계** |             | **3,620** |

### 4.4 시퀀스 재구성 (transpose)

CNN 출력 `(B, 20, 15)`를 LSTM 입력 형식인 `(B, 15, 20)`으로 전치한다.
PyTorch LSTM은 `batch_first=True` 설정 시 `(batch, time_steps, features)`를
기대하므로, 채널 축(20)이 특징(features)으로, 공간 축(15)이 타임스텝(time_steps)이
된다.

```text
(B, 20, 15)  →  transpose(1, 2)  →  (B, 15, 20)
     채널 축↑     공간 축↑              타임스텝 ↑    특징 ↑
```

### 4.5 LSTM 1층

**역할**: CNN에서 추출한 20차원 특징의 시계열 의존성을 64개 메모리 셀로 모델링한다.

```text
입력: (B, 15, 20)
    │
    ▼  nn.LSTM(input_size=20, hidden_size=64, batch_first=True)
       활성화: tanh (LSTM 기본값, 논문 명시: "each using tangent activation")
출력: (B, 15, 64)  + (h_n, c_n) [미사용]
    │
    ▼  Dropout(0.1)
(B, 15, 64)
```

#### LSTM 셀 내부 게이트

```text
fₜ = σ(W_if · xₜ + b_if + W_hf · hₜ₋₁ + b_hf)     ← 망각 게이트
iₜ = σ(W_ii · xₜ + b_ii + W_hi · hₜ₋₁ + b_hi)     ← 입력 게이트
gₜ = tanh(W_ig · xₜ + b_ig + W_hg · hₜ₋₁ + b_hg)  ← 셀 입력
oₜ = σ(W_io · xₜ + b_io + W_ho · hₜ₋₁ + b_ho)     ← 출력 게이트
cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ                          ← 셀 상태
hₜ = oₜ ⊙ tanh(cₜ)                                ← 은닉 상태
```

#### 파라미터 수 산출

| 행렬/편향 | shape      | 수         | 설명                                   |
| --------- | ---------- | ---------- | -------------------------------------- |
| weight_ih | (4×64, 20) | 5,120      | 입력 → 4게이트 (forget/input/cell/out) |
| weight_hh | (4×64, 64) | 16,384     | 은닉 → 4게이트                         |
| bias_ih   | (4×64,)    | 256        | 입력 편향                              |
| bias_hh   | (4×64,)    | 256        | 은닉 편향                              |
| **합계**  |            | **22,016** |                                        |

### 4.6 LSTM 2층

**역할**: 1층 LSTM(64차원)의 출력을 입력으로 받아 128개 메모리 셀로 더 높은
수준의 시계열 표현을 학습한다.

```text
입력: (B, 15, 64)
    │
    ▼  nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
       활성화: tanh
출력: (B, 15, 128)  + (h_n, c_n) [미사용]
    │
    ▼  Dropout(0.1)
(B, 15, 128)
    │
    ▼  최종 타임스텝 선택: x[:, -1, :]
(B, 128)
```

**최종 타임스텝 선택 근거**: 논문은 "two LSTM layers ... are united with CNN
model for use case of regression problem"으로만 기술하며 풀링 방법을 명시하지
않는다. 시계열의 최종 은닉 상태 `h_T`는 전체 15개 타임스텝의 누적된 문맥을
담고 있으므로 회귀 헤드 입력으로 적합하다.

#### 파라미터 수 산출

| 행렬/편향 | shape        | 수         |
| --------- | ------------ | ---------- |
| weight_ih | (4×128, 64)  | 32,768     |
| weight_hh | (4×128, 128) | 65,536     |
| bias_ih   | (4×128,)     | 512        |
| bias_hh   | (4×128,)     | 512        |
| **합계**  |              | **99,328** |

### 4.7 회귀 헤드 (FC)

**역할**: LSTM 최종 은닉 상태(128차원)를 SBP, DBP 두 예측값으로 선형 변환한다.

```text
입력: (B, 128)
    │
    ▼  Linear(128 → 2, bias=True)
(B, 2)  ← [SBP, DBP] in mmHg
```

논문은 "one fully connected layer with 3 output neurons are introduced to find
the final prediction scores using the **linear function**"으로 기술한다.
활성화 함수 없이 선형 변환만 적용하므로 연속값 회귀에 적합하다.

#### 파라미터 수

| 파라미터 | shape    | 수      |
| -------- | -------- | ------- |
| weight   | (2, 128) | 256     |
| bias     | (2,)     | 2       |
| **합계** |          | **258** |

## 5. 파라미터 수 분석

| 모듈       | 파라미터 수 | 비율    |
| ---------- | ----------- | ------- |
| CNN 블록 1 | 200         | 0.16 %  |
| CNN 블록 2 | 3,620       | 2.89 %  |
| LSTM 1층   | 22,016      | 17.56 % |
| LSTM 2층   | 99,328      | 79.20 % |
| 회귀 헤드  | 258         | 0.21 %  |
| **합계**   | **125,422** | 100 %   |

```text
print-model 출력:
  Total params    : 125,422  (125.4 K)
  Trainable params: 125,422  (125.4 K)
  Input shape     : (1, 1000)
```

CNN은 전체의 3%에 불과한 반면, LSTM 2층이 79%를 차지한다. 이는 혈압 추정에서
시계열 의존성 모델링에 주된 표현력이 필요함을 반영하며, 논문이 강조하는
"CNN suffers from the vanishing gradient problem … LSTM proved to be an effective
choice for time series data"의 설계 철학과 일치한다.

## 6. 하이퍼파라미터 참조표

| 파라미터       | 기본값    | 논문 값                  | 역할                             |
| -------------- | --------- | ------------------------ | -------------------------------- |
| `cnn_filters`  | 20        | 20 (Fig. 3 명시)         | 각 CNN 층의 출력 채널(필터) 수   |
| `kernel_size`  | 9         | 9 (Fig. 3 명시)          | Conv1d 커널 너비                 |
| `pool_size`    | 4         | 4 (Fig. 3 명시)          | MaxPool 커널 크기 및 스트라이드  |
| `lstm_units`   | (64, 128) | (64, 128) (Fig. 3 명시)  | 1층/2층 LSTM 은닉 상태 차원      |
| `dropout`      | 0.1       | 0.1 (Section III-B 명시) | 풀링 및 LSTM 층 후 드롭아웃 비율 |
| `out_features` | 2         | 3 (SBP, DBP, HR)         | 출력 차원                        |

### 훈련 하이퍼파라미터 (논문 Section III-C)

| 파라미터   | 논문              | 프로젝트 기본값          |
| ---------- | ----------------- | ------------------------ |
| 옵티마이저 | Adam              | AdamW (프로젝트 표준)    |
| 손실 함수  | MSE               | MSE (동일)               |
| 최대 에폭  | 50 (100에서 조기) | 100                      |
| 배치 크기  | 100               | 256                      |
| K-fold     | 10-fold CV        | 고정 train/val/test 분리 |

## 7. 논문과의 차이점 및 설계 결정 근거

### 7.1 다운샘플링 위치: 전처리 → 모델 내부

|               | 논문               | 이 구현                             |
| ------------- | ------------------ | ----------------------------------- |
| 다운샘플 위치 | 데이터 전처리 단계 | 모델 내부 첫 단계                   |
| 구현 방식     | 명시 없음          | `AvgPool1d(kernel=4, stride=4)`     |
| 입력 길이     | 250 (전처리 후)    | 1000 (모델이 내부에서 250으로 압축) |

이 프로젝트의 모든 모델은 1,000 샘플 입력 인터페이스를 공유한다.
논문의 전처리 단계를 모델 내부로 흡수하면 인터페이스 일관성을 유지하면서
논문이 검증한 4× 압축의 이점을 그대로 적용할 수 있다.

### 7.2 출력: [SBP, DBP] (HR 제외)

|        | 논문       | 이 구현    |
| ------ | ---------- | ---------- |
| 출력 1 | DBP (mmHg) | SBP (mmHg) |
| 출력 2 | SBP (mmHg) | DBP (mmHg) |
| 출력 3 | HR (bpm)   | — (미구현) |

이 프로젝트는 VitalDB ABP 신호 기반 혈압(SBP, DBP) 추정만을 목표로 한다.
HR 출력을 제거(`out_features=2`)하면 파라미터와 연산량이 약간 감소하고,
두 출력 간의 학습 간섭도 제거된다.

### 7.3 BatchNorm 미적용

기존 프로젝트 모델(`resnet1d`, `st_resnet` 등)은 `ConvBnAct1d`를 통해
Conv + BN + ReLU 패턴을 사용한다. 그러나 원 논문은 BN을 명시하지 않으므로,
논문 재현성 우선 원칙에 따라 CNN 블록에서 Conv + ReLU만 사용한다.

### 7.4 최종 타임스텝 vs. 글로벌 풀링

논문은 LSTM의 마지막 출력을 FC에 연결하는 것을 암시하나 명시하지 않는다.
`x[:, -1, :]`로 최종 타임스텝을 선택한다. 이는 LSTM의 단방향 구조에서
모든 이전 타임스텝의 정보가 마지막 은닉 상태에 누적된다는 특성과 일치한다.

### 7.5 다운샘플 방식: AvgPool vs. 스트라이드 합성곱

논문 Table II-B는 입력 압축 방식이 CNN inherent stride보다 우수함을 보인다.
AvgPool1d는 신호 에너지를 보존하며 에일리어싱 위험이 낮아 PPG 전처리에 적합하다.

## 8. 훈련 방법

### 기본 훈련

```bash
bin/train-model --model ppnet
```

### 논문에 가까운 설정

```bash
# Linux / macOS
bin/train-model --model ppnet --epochs 50 --batch-size 100 --lr 1e-3

# Windows
bin\train-model.bat --model ppnet --epochs 50 --batch-size 100 --lr 1e-3
```

논문 실험 조건 대비:

| 항목       | 논문                  | 권장 설정                             |
| ---------- | --------------------- | ------------------------------------- |
| 손실 함수  | MSE                   | MSE (프로젝트 기본값, 동일)           |
| 옵티마이저 | Adam, lr=0.001        | AdamW, lr=1×10⁻³                      |
| 배치 크기  | 100                   | 100–256 (메모리 여유에 따라)          |
| 최대 에폭  | 50 (100 중 조기 종료) | 100 기본값으로 시작 후 조정           |
| 조기 종료  | 없음(고정 50 에폭)    | `--patience 15` (기본값)              |
| 검증 방법  | 10-fold CV            | 케이스 레벨 고정 분리 (프로젝트 표준) |

> **참고**: 논문은 10-fold 교차검증을 사용하지만, 이 프로젝트는 케이스 레벨로
> train/val/test를 사전 분할한다. 교차검증 대신 고정 분할로 평가한다.

### 경량 실험 (빠른 검증)

```bash
bin/train-model --model ppnet --epochs 20 --batch-size 512
```

PP-Net은 125.4 K 파라미터로 가볍지만, LSTM 2층의 순차 처리 특성 때문에
배치당 처리 시간이 순수 CNN 모델보다 길 수 있다. 배치 크기를 키우면 GPU
활용률을 높일 수 있다.

## 9. 모델 검사

```bash
# 레이어 구조와 파라미터 수 출력
bin/print-model --model ppnet

# Windows
bin\print-model.bat --model ppnet
```

출력 예시:

```text
======================================================================================================================
  Model: ppnet
======================================================================================================================
Layer (name)        Type                Output shape     Params
----------------------------------------------------------------------------------------------------------------------
downsample          AvgPool1d           (1, 1, 250)
cnn1                Sequential          (1, 20, 62)
cnn1.0              Conv1d              (1, 20, 250)         200
cnn1.1              ReLU                (1, 20, 250)
cnn1.2              MaxPool1d           (1, 20, 62)
cnn1.3              Dropout             (1, 20, 62)
cnn2                Sequential          (1, 20, 15)
cnn2.0              Conv1d              (1, 20, 62)        3.6 K
cnn2.1              ReLU                (1, 20, 62)
cnn2.2              MaxPool1d           (1, 20, 15)
cnn2.3              Dropout             (1, 20, 15)
lstm1               LSTM                (1, 15, 64)       22.0 K
drop_lstm1          Dropout             (1, 15, 64)
lstm2               LSTM                (1, 15, 128)      99.3 K
drop_lstm2          Dropout             (1, 15, 128)
fc                  Linear              (1, 2)               258
----------------------------------------------------------------------------------------------------------------------
  Total params    : 125,422  (125.4 K)
  Trainable params: 125,422  (125.4 K)
  Input shape     : (1, 1000)
```

## 10. 다른 모델과의 비교

| 모델            | 특징 추출        | 시계열 모델링      | 파라미터    | 복잡도 |
| --------------- | ---------------- | ------------------ | ----------- | ------ |
| `conv_reg`      | CNN 6층          | 없음 (GlobalAvg)   | 36.9 K      | 낮음   |
| `ae_lstm`       | 없음 (직접 LSTM) | LSTM 1층           | 50.6 K      | 낮음   |
| `ppnet`         | CNN 2층          | LSTM 2층           | **125.4 K** | 낮음   |
| `cnn_bilstm_at` | CNN 3층          | BiLSTM + Attention | 691.3 K     | 중간   |
| `resnet1d`      | ResNet CNN 5단계 | 없음 (GlobalAvg)   | 2.18 M      | 중간   |

PP-Net은 `ae_lstm`보다 CNN 특징 추출 단계를 명시적으로 두어 구조적으로 더
명확하고, `cnn_bilstm_at`보다 5.5배 가볍다. 논문에서 주장하는 "light-weight,
generalized framework"의 특성을 파라미터 수에서 확인할 수 있다.

## 11. 참고 문헌

- Panwar, M., Gautam, A., Biswas, D., and Acharyya, A. (2020).
  "PP-Net: A Deep Learning Framework for PPG-Based Blood Pressure and
  Heart Rate Estimation."
  *IEEE Sensors Journal*, vol. 20, no. 17, pp. 10000–10011.
  DOI: 10.1109/JSEN.2020.2990864

- Hochreiter, S. and Schmidhuber, J. (1997).
  "Long Short-Term Memory."
  *Neural Computation*, vol. 9, no. 8, pp. 1735–1780.

- Biswas, D. et al. (2019).
  "CorNET: Deep learning framework for PPG-based heart rate estimation and
  biometric identification in ambulant environment."
  *IEEE Transactions on Biomedical Circuits and Systems*, vol. 13, no. 2,
  pp. 282–291.
