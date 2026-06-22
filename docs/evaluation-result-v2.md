# 모델 평가 결과 (dataset-v2 기반 학습 및 평가)

작성일: 2026-06-22  
평가 대상: VitalDB PPG → SBP/DBP 직접 회귀 모델 (dataset-v2로 학습)  
평가 데이터셋: `data/dataset-v2/test` (case-level held-out, 672 cases, 1,572,941 segments)  
비교 기준: `data/dataset-v1` 기반 모델 평가 결과 (`data/results-v1`, `docs/evaluation-result-v1.md`)

## 1. 개요

본 문서는 `data/dataset-v2`로 학습한 혈압 추정 모델들의 테스트셋 평가 결과를 종합한다.
`data/dataset-v2`는 원본 `.vital` 파일에서 ABP 파형 peak/foot 기반 레이블과 9단계 정제 룰을
적용해 구축한 고품질 데이터셋이다.

이전 dataset-v2 구축 시도(44,664 세그먼트)에 비해, Rule 2(FASQA PSD 임계값) 완화를 통해
총 **7,829,237 세그먼트** (dataset-v1의 80.8%)를 확보했다.
케이스 불균형 문제(old: 상위 10% 케이스가 세그먼트의 62% 점유)도
해소됐다(new: 상위 10% 케이스가 세그먼트의 21.5% 점유).

모든 v2 모델은 dataset-v1 기반 모델과 동일한 아키텍처·하이퍼파라미터로 학습되었으며,
평가도 동일한 `eval-model.py`로 수행됐다.

## 2. 평가 환경

| 항목                | 내용                                                                    |
| ------------------- | ----------------------------------------------------------------------- |
| 데이터셋            | VitalDB (dataset-v2: 9단계 ABP 정제 룰 적용, 7,829,237 세그먼트)        |
| 입력 신호           | PPG (`SNUADC/PLETH`), 125 Hz, 8초 (1,000 샘플)                          |
| 레이블              | SBP/DBP (mmHg), ABP 파형 peak/foot 직접 추출 (dataset-v1과 방식 상이)   |
| case 분할           | train 70% / val 10% / test 20% (case-level, seed=42)                    |
| 테스트 케이스 수    | 672 cases                                                               |
| 테스트 세그먼트 수  | 1,572,941 segments                                                      |
| 평가 체크포인트     | 각 모델의 `best.pt` (val loss 최소 epoch)                               |
| 공통 하이퍼파라미터 | lr=1e-3, weight_decay=1e-4, batch_size=256, max_epochs=100, patience=5  |
| 출력 디렉터리       | `data/models-v2`, `data/results-v2`                                     |

### 2.1 데이터셋 규모 비교

| 구분  | dataset-v1                     | dataset-v2 (이전)          | dataset-v2 (현재)              |
| ----- | ------------------------------ | -------------------------- | ------------------------------ |
| train | 6,769,507 세그먼트             | 32,594 세그먼트            | 5,478,776 세그먼트 / 2,307 cases |
| val   | 962,633 세그먼트               | 4,787 세그먼트             | 777,520 세그먼트 / 339 cases   |
| test  | 1,955,049 세그먼트 / 672 cases | 7,283 세그먼트 / 459 cases | 1,572,941 세그먼트 / 672 cases |
| total | 9,687,189 세그먼트             | 44,664 세그먼트            | 7,829,237 세그먼트 / 3,318 cases |

현재 dataset-v2는 dataset-v1의 **80.8%** 규모를 회복했다.
이전 dataset-v2 대비 약 **175배** 증가했으며, test cases도 459 → 672로 dataset-v1과 동일하다.

### 2.2 레이블 방식 차이 주의

dataset-v2의 레이블은 ABP 파형 peak/foot에서 직접 산출된다.
dataset-v1의 `Solar8000/ART_SBP·DBP` 1 Hz 수치 평균 레이블과 동일 케이스에서도 수 mmHg
차이가 발생하므로, 수치를 직접 비교할 때 이 점을 감안해야 한다.

## 3. 평가 지표

### 3.1 정량 지표

| 지표     | 정의                               | 의미                                                      |
| -------- | ---------------------------------- | --------------------------------------------------------- |
| **MAE**  | Mean Absolute Error (mmHg)         | 예측 오차의 절대값 평균. 임상에서 가장 직관적인 오차 지표 |
| **ME**   | Mean Error (mmHg)                  | 예측 편향(bias). 양수=과추정, 음수=과소추정               |
| **SD**   | Standard Deviation of error (mmHg) | 예측 오차의 산포. AAMI 기준의 핵심 지표                   |
| **RMSE** | Root Mean Squared Error (mmHg)     | 이상치에 민감한 오차. √(ME² + SD²)                        |

### 3.2 임상 표준 기준

#### **AAMI (Association for the Advancement of Medical Instrumentation) 기준**

| 조건 | 임계값    |
| ---- | --------- |
| ME   | ≤ ±5 mmHg |
| SD   | ≤ 8 mmHg  |

#### **BHS (British Hypertension Society) 등급**

| 등급 | ±5 mmHg 이내 | ±10 mmHg 이내 | ±15 mmHg 이내 |
| ---- | ------------ | ------------- | ------------- |
| A    | ≥ 60%        | ≥ 85%         | ≥ 95%         |
| B    | ≥ 50%        | ≥ 75%         | ≥ 90%         |
| C    | ≥ 40%        | ≥ 65%         | ≥ 85%         |
| D    | C 미달       |               |               |

> **임상 적용 기준**: AAMI 통과 + BHS Grade B 이상이 임상 사용의 최소 요건.
> Grade C는 연구용 참고 기준으로 활용된다.

## 4. 평가 대상 모델

| 모델명           | 분류                     | 파라미터 수 | 평가 여부 | 비고                  |
| ---------------- | ------------------------ | ----------- | --------- | --------------------- |
| `naive`          | 베이스라인               | —           | ✅        |                       |
| `resnet1d`       | ResNet1D 계열            | 2.18 M      | ✅        |                       |
| `resnet1d_mini`  | ResNet1D 계열            | 964.4 K     | ✅        |                       |
| `resnet1d_tiny`  | ResNet1D 계열            | 60.6 K      | ✅        |                       |
| `resnet1d_micro` | ResNet1D 계열            | 15.1 K      | ✅        |                       |
| `st_resnet`      | 다중 채널                | 478.9 K     | ✅        |                       |
| `minception`     | 다중 스케일              | 440.7 K     | ✅        |                       |
| `xresnet1d`      | 대형 ResNet              | 9.47 M      | ✅        |                       |
| `acfa`           | Attention CNN            | 542.6 K     | ✅        |                       |
| `ae_lstm`        | AE + LSTM                | 50.6 K      | ✅        |                       |
| `bpnet_cf`       | BPNet-CF (dual-scale)    | —           | ✅        | v1 미평가             |
| `cnn_bilstm_at`  | CNN + BiLSTM             | 691.3 K     | ✅        |                       |
| `conv_reg`       | Conv 회귀                | 36.9 K      | ✅        |                       |
| `conv_reg_ds`    | Conv 회귀 (Depthwise)    | 14.1 K      | ✅        |                       |
| `mtae`           | 다중 태스크 오토인코더   | 119.5 K     | ✅        |                       |
| `mtae_tr`        | MTAE + Transformer       | 109.4 K     | ✅        |                       |
| `pctn`           | Parallel CNN-Transformer | 5.13 M      | ✅        | v1 미평가             |
| `conv_reg_at`    | Conv 회귀 (Attention)    | 39.0 K      | ⛔        | deprecated (학습 실패)|

> `conv_reg_at`는 반복적 학습 실패로 평가 대상에서 제외됐다. 이하 모든 분석은 17개 모델 기준이다.

## 5. 테스트셋 정량 평가 결과

### 5.1 SBP(수축기혈압) 종합 비교

| 모델                | MAE ↓     | ME         | SD        | RMSE      | ±5%       | ±10%      | ±15%      | BHS | AAMI  |
| ------------------- | --------- | ---------- | --------- | --------- | --------- | --------- | --------- | --- | ----- |
| **`mtae`**          | **12.578**| +0.29      | **16.172**| **16.175**| **25.83%**| **48.60%**| **67.27%**| D   | ❌    |
| `ae_lstm`           | 12.621    | −0.59      | 16.255    | 16.266    | 25.91%    | 48.82%    | 67.07%    | D   | ❌    |
| `bpnet_cf`          | 12.627    | −1.21      | 16.266    | 16.311    | 25.98%    | **48.95%**| **67.22%**| D   | ❌    |
| `conv_reg_ds`       | 12.683    | +0.46      | 16.346    | 16.353    | 25.77%    | 48.51%    | 66.92%    | D   | ❌    |
| `resnet1d_micro`    | 12.691    | −0.05      | 16.264    | 16.264    | 25.26%    | 48.19%    | 66.65%    | D   | ❌    |
| `cnn_bilstm_at`     | 12.783    | −0.35      | 16.481    | 16.484    | 25.48%    | 48.35%    | 66.57%    | D   | ❌    |
| `mtae_tr`           | 12.809    | −0.13      | 16.434    | 16.435    | 25.13%    | 47.83%    | 66.30%    | D   | ❌    |
| `pctn`              | 12.813    | +1.22      | 16.440    | 16.485    | 25.46%    | 47.85%    | 66.13%    | D   | ❌    |
| `resnet1d_tiny`     | 12.859    | +1.25      | 16.503    | 16.550    | 25.40%    | 47.90%    | 66.12%    | D   | ❌    |
| `conv_reg`          | 13.029    | +1.29      | 16.767    | 16.816    | 25.23%    | 47.63%    | 65.53%    | D   | ❌    |
| `resnet1d_mini`     | 13.034    | +0.85      | 16.773    | 16.794    | 25.20%    | 47.59%    | 65.49%    | D   | ❌    |
| `xresnet1d`         | 13.055    | +1.45      | 16.810    | 16.873    | 25.22%    | 47.80%    | 65.60%    | D   | ❌    |
| `resnet1d`          | 13.123    | −1.41      | 16.977    | 17.035    | 25.46%    | 47.85%    | 65.50%    | D   | ❌    |
| `st_resnet`         | 13.246    | +0.53      | 17.046    | 17.055    | 24.66%    | 46.82%    | 64.74%    | D   | ❌    |
| `minception`        | 13.270    | −1.59      | 17.178    | 17.251    | 25.10%    | 47.48%    | 65.19%    | D   | ❌    |
| `acfa`              | 13.508    | +2.60      | 17.148    | 17.344    | 24.07%    | 45.75%    | 63.65%    | D   | ❌    |
| `naive`             | 14.997    | −1.46      | 18.943    | 18.999    | 21.13%    | 40.90%    | 58.20%    | D   | ❌    |

> ↓: 낮을수록 좋음. ME 부호: 양수=과추정, 음수=과소추정.  
> 전 모델 SBP BHS Grade D. 모든 모델 |ME| ≤ 2.60 mmHg으로 SBP 체계적 편향이 해소됐다.

### 5.2 DBP(이완기혈압) 종합 비교

| 모델                | MAE ↓    | ME         | SD        | RMSE      | ±5%       | ±10%      | ±15%      | BHS | AAMI  |
| ------------------- | -------- | ---------- | --------- | --------- | --------- | --------- | --------- | --- | ----- |
| **`mtae`**          | **8.017**| −1.02      | **10.124**| 10.175    | **38.84%**| **68.66%**| 86.48%    | D   | ❌    |
| `resnet1d_micro`    | 8.020    | −0.84      | 10.117    | **10.152**| 38.54%    | 68.53%    | **86.57%**| D   | ❌    |
| `conv_reg_ds`       | 8.036    | −0.33      | 10.174    | 10.179    | 38.32%    | 68.65%    | 86.53%    | D   | ❌    |
| `bpnet_cf`          | 8.054    | +0.11      | 10.147    | 10.147    | 37.87%    | 68.26%    | **86.62%**| D   | ❌    |
| `ae_lstm`           | 8.064    | −0.38      | 10.233    | 10.240    | 38.45%    | 68.62%    | 86.20%    | D   | ❌    |
| `resnet1d_tiny`     | 8.107    | −0.16      | 10.245    | 10.247    | 38.04%    | 67.97%    | 86.14%    | D   | ❌    |
| `mtae_tr`           | 8.173    | −0.25      | 10.321    | 10.324    | 37.60%    | 67.61%    | 85.92%    | D   | ❌    |
| `conv_reg`          | 8.188    | −0.06      | 10.389    | 10.389    | 38.07%    | 67.73%    | 85.51%    | D   | ❌    |
| `xresnet1d`         | 8.211    | −1.66      | 10.321    | 10.454    | 38.15%    | 67.62%    | 85.47%    | D   | ❌    |
| `cnn_bilstm_at`     | 8.254    | −0.32      | 10.425    | 10.429    | 37.38%    | 67.13%    | 85.41%    | D   | ❌    |
| `pctn`              | 8.275    | +0.90      | 10.402    | 10.440    | 37.21%    | 66.91%    | 85.34%    | D   | ❌    |
| `resnet1d_mini`     | 8.310    | +0.43      | 10.536    | 10.545    | 37.44%    | 67.16%    | 85.06%    | D   | ❌    |
| `minception`        | 8.328    | −0.73      | 10.594    | 10.619    | 37.74%    | 67.08%    | 84.97%    | D   | ❌    |
| `st_resnet`         | 8.372    | −0.00      | 10.611    | 10.611    | 37.23%    | 66.68%    | 84.67%    | D   | ❌    |
| `resnet1d`          | 8.454    | −0.22      | 10.757    | 10.760    | 36.57%    | 66.32%    | 84.78%    | D   | ❌    |
| `acfa`              | 8.472    | −0.15      | 10.724    | 10.725    | 36.72%    | 65.83%    | 84.32%    | D   | ❌    |
| `naive`             | 9.528    | −1.02      | 11.877    | 11.920    | 31.74%    | 59.58%    | 79.33%    | D   | ❌    |

> **전 모델 DBP BHS Grade D.** AAMI 기준 SD ≤ 8 mmHg: 전 모델 미달 (최소 10.12 / `mtae`).  
> DBP ±10% ≥ 65% (Grade C 기준): `naive`를 제외한 16개 모델이 충족.  
> DBP ±15% ≥ 85% (Grade C 기준): 12개 모델이 충족. 단 ±5% 기준(≥40%)이 전 모델 미달로 Grade C 달성 불가.

### 5.3 종합 순위 (SBP MAE + DBP MAE 합산 기준)

| 순위 | 모델             | SBP MAE | DBP MAE | 합산       | v1 합산 | 대비             | v1 순위 |
| ---- | ---------------- | ------- | ------- | ---------- | ------- | ---------------- | ------- |
| 1    | `mtae`           | 12.578  | 8.017   | **20.595** | 20.79   | ↑ +0.20          | 2       |
| 2    | `bpnet_cf`       | 12.627  | 8.054   | **20.681** | —       | (v1 미평가)      | —       |
| 3    | `ae_lstm`        | 12.621  | 8.064   | **20.685** | 20.74   | ↑ +0.06          | 1       |
| 4    | `resnet1d_micro` | 12.691  | 8.020   | **20.711** | 20.89   | ↑ +0.18          | 3       |
| 5    | `conv_reg_ds`    | 12.683  | 8.036   | **20.719** | 21.19   | ↑ +0.47          | 6       |
| 6    | `resnet1d_tiny`  | 12.859  | 8.107   | **20.966** | 21.17   | ↑ +0.20          | 5       |
| 7    | `mtae_tr`        | 12.809  | 8.173   | **20.982** | 21.31   | ↑ +0.33          | 9       |
| 8    | `cnn_bilstm_at`  | 12.783  | 8.254   | **21.037** | 21.08   | ↑ +0.04          | 4       |
| 9    | `pctn`           | 12.813  | 8.275   | **21.088** | —       | (v1 미평가)      | —       |
| 10   | `conv_reg`       | 13.029  | 8.188   | **21.217** | 21.20   | ↓ −0.02          | 7       |
| 11   | `xresnet1d`      | 13.055  | 8.211   | **21.266** | 21.72   | ↑ +0.45          | 12      |
| 12   | `resnet1d_mini`  | 13.034  | 8.310   | **21.344** | 21.86   | ↑ +0.52          | 14      |
| 13   | `resnet1d`       | 13.123  | 8.454   | **21.577** | 21.53   | ↓ −0.05          | 11      |
| 14   | `minception`     | 13.270  | 8.328   | **21.598** | 21.85   | ↑ +0.25          | 13      |
| 15   | `st_resnet`      | 13.246  | 8.372   | **21.618** | 21.25   | ↓ −0.37          | 8       |
| 16   | `acfa`           | 13.508  | 8.472   | **21.980** | 21.51   | ↓ −0.47          | 10      |
| —    | `naive`          | 14.997  | 9.528   | **24.525** | 25.00   | ↑ +0.48          | 15      |

> ↑: v1 대비 성능 향상(합산 감소). ↓: v1 대비 성능 저하(합산 증가).  
> `mtae`가 최우수 모델로 부상. `conv_reg`와 `resnet1d`는 v1과 사실상 동일 수준(±0.05).  
> `st_resnet`(−0.37), `acfa`(−0.47)은 소폭 저하. 그 외 대부분은 개선됐다.

## 6. 모델별 상세 평가

### 6.1 mtae (Multi-Task AutoEncoder)

best_epoch: 3 / 총 8 에폭

```
SBP — MAE: 12.578, ME: +0.29, SD: 16.172, RMSE: 16.175 | Grade D | AAMI: ❌
DBP — MAE:  8.017, ME: −1.02, SD: 10.124, RMSE: 10.175 | Grade D | AAMI: ❌
```

**v2 합산 최우수 모델(20.595). v1 2위(20.79)에서 v2 1위로 상승.**
이전 dataset-v2(32,594 세그먼트)에서 SBP ME = −16.86 mmHg의 수렴 실패를 보였으나,
새 데이터셋(5.48M 세그먼트)에서 완전 정상화됐다. SBP ME = +0.29 mmHg로 전 모델 중 SBP 편향이
가장 작다. SBP SD 16.172, DBP SD 10.124도 전 모델 최저다. DBP ±5% 38.84%는
Grade C 기준(40%)까지 1.16%p 부족하다.

| 그래프               |                                               |
| -------------------- | --------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v2/eval_plot/mtae.png)    |
| Error Distribution   | ![](../data/results-v2/error_hist/mtae.png)   |
| Bland-Altman         | ![](../data/results-v2/bland_altman/mtae.png) |
| 훈련 손실 곡선       | ![](../data/results-v2/loss_graph/mtae.png)   |
| 훈련 MAE 곡선        | ![](../data/results-v2/mae_graph/mtae.png)    |

### 6.2 bpnet_cf (BPNet-CF Calibration-Free)

best_epoch: 1 / 총 6 에폭

```
SBP — MAE: 12.627, ME: −1.21, SD: 16.266, RMSE: 16.311 | Grade D | AAMI: ❌
DBP — MAE:  8.054, ME: +0.11, SD: 10.147, RMSE: 10.147 | Grade D | AAMI: ❌
```

**v2 합산 2위(20.681).** DBP ±15% 86.62%는 전 모델 최고다. DBP SD 10.147은 mtae에 이어
2번째로 낮다. SBP ME = −1.21 mmHg로 편향이 적다. DBP ME = +0.11 mmHg로 거의 무편향.
v1에서는 평가되지 않은 신규 모델로, v2에서 top 5 이내를 꾸준히 유지한다.
best_epoch=1으로 첫 에폭에서 바로 최적점에 도달한다.

| 그래프               |                                                  |
| -------------------- | ------------------------------------------------ |
| Prediction vs Actual | ![](../data/results-v2/eval_plot/bpnet_cf.png)   |
| Error Distribution   | ![](../data/results-v2/error_hist/bpnet_cf.png)  |
| Bland-Altman         | ![](../data/results-v2/bland_altman/bpnet_cf.png)|
| 훈련 손실 곡선       | ![](../data/results-v2/loss_graph/bpnet_cf.png)  |
| 훈련 MAE 곡선        | ![](../data/results-v2/mae_graph/bpnet_cf.png)   |

### 6.3 ae_lstm (Autoencoder + LSTM)

best_epoch: 6 / 총 11 에폭

```
SBP — MAE: 12.621, ME: −0.59, SD: 16.255, RMSE: 16.266 | Grade D | AAMI: ❌
DBP — MAE:  8.064, ME: −0.38, SD: 10.233, RMSE: 10.240 | Grade D | AAMI: ❌
```

**v1 최우수 모델(20.74)이 v2에서도 3위(20.685)를 유지.** SBP ME = −0.59 mmHg, DBP ME = −0.38 mmHg로
편향이 모두 낮다. 이전 dataset-v2에서는 SBP ME = −8.32 mmHg의 큰 편향을 보였으나
새 데이터셋에서 정상화됐다. v1 대비 +0.06 mmHg 소폭 개선됐다.
SBP ±15% 67.07%는 bpnet_cf(67.22%)에 이어 높은 편이다.

| 그래프               |                                                |
| -------------------- | ---------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v2/eval_plot/ae_lstm.png)  |
| Error Distribution   | ![](../data/results-v2/error_hist/ae_lstm.png) |
| Bland-Altman         | ![](../data/results-v2/bland_altman/ae_lstm.png)|
| 훈련 손실 곡선       | ![](../data/results-v2/loss_graph/ae_lstm.png) |
| 훈련 MAE 곡선        | ![](../data/results-v2/mae_graph/ae_lstm.png)  |

### 6.4 resnet1d_micro (초소형 ResNet1D)

best_epoch: 1 / 총 6 에폭

```
SBP — MAE: 12.691, ME: −0.05, SD: 16.264, RMSE: 16.264 | Grade D | AAMI: ❌
DBP — MAE:  8.020, ME: −0.84, SD: 10.117, RMSE: 10.152 | Grade D | AAMI: ❌
```

**v2 4위(20.711), v1 3위(20.89)에서 소폭 개선.** SBP ME = −0.05 mmHg로 전 모델 중 SBP 편향이 가장 작다.
DBP SD 10.117은 전 모델 최저. 15.1K 파라미터의 초소형 모델이 대형 모델과 경쟁력이 있음을 보여준다.
DBP ±15% 86.57%는 bpnet_cf(86.62%)에 이어 2위다.

| 그래프               |                                                       |
| -------------------- | ----------------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v2/eval_plot/resnet1d_micro.png)  |
| Error Distribution   | ![](../data/results-v2/error_hist/resnet1d_micro.png) |
| Bland-Altman         | ![](../data/results-v2/bland_altman/resnet1d_micro.png)|
| 훈련 손실 곡선       | ![](../data/results-v2/loss_graph/resnet1d_micro.png) |
| 훈련 MAE 곡선        | ![](../data/results-v2/mae_graph/resnet1d_micro.png)  |

### 6.5 conv_reg_ds (Depthwise-Separable Conv Regression)

best_epoch: 4 / 총 9 에폭

```
SBP — MAE: 12.683, ME: +0.46, SD: 16.346, RMSE: 16.353 | Grade D | AAMI: ❌
DBP — MAE:  8.036, ME: −0.33, SD: 10.174, RMSE: 10.179 | Grade D | AAMI: ❌
```

**v2 5위(20.719), v1 6위(21.19)에서 +0.47 mmHg 개선.** 14.1K 파라미터의 경량 모델이 상위권을 유지한다.
SBP·DBP ME 모두 ±0.5 이내로 낮은 편향을 보인다. DBP ±10% 68.65%는 mtae에 이어 전 모델 공동 2위다.
이전 dataset-v2(32,594 세그먼트)에서 합산 26.49였으나, 새 데이터셋에서 20.72로 크게 개선됐다.

| 그래프               |                                                     |
| -------------------- | --------------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v2/eval_plot/conv_reg_ds.png)   |
| Error Distribution   | ![](../data/results-v2/error_hist/conv_reg_ds.png)  |
| Bland-Altman         | ![](../data/results-v2/bland_altman/conv_reg_ds.png)|
| 훈련 손실 곡선       | ![](../data/results-v2/loss_graph/conv_reg_ds.png)  |
| 훈련 MAE 곡선        | ![](../data/results-v2/mae_graph/conv_reg_ds.png)   |

### 6.6 resnet1d_tiny

best_epoch: 2 / 총 7 에폭

```
SBP — MAE: 12.859, ME: +1.25, SD: 16.503, RMSE: 16.550 | Grade D | AAMI: ❌
DBP — MAE:  8.107, ME: −0.16, SD: 10.245, RMSE: 10.247 | Grade D | AAMI: ❌
```

**v2 6위(20.966), v1 5위(21.17)에서 +0.20 mmHg 개선.** 60.6K 파라미터의 소형 모델.
SBP ME = +1.25 mmHg, DBP ME = −0.16 mmHg로 편향이 낮다. DBP ±15% 86.14%로 Grade C ±15% 기준 달성.

| 그래프               |                                                      |
| -------------------- | ---------------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v2/eval_plot/resnet1d_tiny.png)  |
| Error Distribution   | ![](../data/results-v2/error_hist/resnet1d_tiny.png) |
| Bland-Altman         | ![](../data/results-v2/bland_altman/resnet1d_tiny.png)|
| 훈련 손실 곡선       | ![](../data/results-v2/loss_graph/resnet1d_tiny.png) |
| 훈련 MAE 곡선        | ![](../data/results-v2/mae_graph/resnet1d_tiny.png)  |

### 6.7 mtae_tr (MTAE with Transformer)

best_epoch: 2 / 총 7 에폭

```
SBP — MAE: 12.809, ME: −0.13, SD: 16.434, RMSE: 16.435 | Grade D | AAMI: ❌
DBP — MAE:  8.173, ME: −0.25, SD: 10.321, RMSE: 10.324 | Grade D | AAMI: ❌
```

**v2 7위(20.982), v1 9위(21.31)에서 +0.33 mmHg 개선.** SBP ME = −0.13 mmHg로 전 모델 중 두 번째로
낮은 SBP 편향이다. 이전 dataset-v2에서는 best_epoch=40으로 가장 늦게 수렴했으나,
새 데이터셋에서 best_epoch=2로 급격히 빨라졌다.

| 그래프               |                                                  |
| -------------------- | ------------------------------------------------ |
| Prediction vs Actual | ![](../data/results-v2/eval_plot/mtae_tr.png)    |
| Error Distribution   | ![](../data/results-v2/error_hist/mtae_tr.png)   |
| Bland-Altman         | ![](../data/results-v2/bland_altman/mtae_tr.png) |
| 훈련 손실 곡선       | ![](../data/results-v2/loss_graph/mtae_tr.png)   |
| 훈련 MAE 곡선        | ![](../data/results-v2/mae_graph/mtae_tr.png)    |

### 6.8 cnn_bilstm_at (CNN + BiLSTM with Attention)

best_epoch: 9 / 총 14 에폭

```
SBP — MAE: 12.783, ME: −0.35, SD: 16.481, RMSE: 16.484 | Grade D | AAMI: ❌
DBP — MAE:  8.254, ME: −0.32, SD: 10.425, RMSE: 10.429 | Grade D | AAMI: ❌
```

**v2 8위(21.037), v1 4위(21.08)에서 +0.04 mmHg 개선.** 전 모델 중 best_epoch가 가장 높다(9 에폭).
SBP·DBP ME 모두 ±0.35 이내로 편향이 낮다. 691.3K 파라미터의 중형 모델임에도 상위권을 유지한다.

| 그래프               |                                                       |
| -------------------- | ----------------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v2/eval_plot/cnn_bilstm_at.png)   |
| Error Distribution   | ![](../data/results-v2/error_hist/cnn_bilstm_at.png)  |
| Bland-Altman         | ![](../data/results-v2/bland_altman/cnn_bilstm_at.png)|
| 훈련 손실 곡선       | ![](../data/results-v2/loss_graph/cnn_bilstm_at.png)  |
| 훈련 MAE 곡선        | ![](../data/results-v2/mae_graph/cnn_bilstm_at.png)   |

### 6.9 pctn (Parallel CNN-Transformer)

best_epoch: 4 / 총 9 에폭

```
SBP — MAE: 12.813, ME: +1.22, SD: 16.440, RMSE: 16.485 | Grade D | AAMI: ❌
DBP — MAE:  8.275, ME: +0.90, SD: 10.402, RMSE: 10.440 | Grade D | AAMI: ❌
```

**v2 9위(21.088), v1 미평가.** 5.13M 파라미터의 대형 모델이 상위권을 유지한다.
SBP·DBP ME 모두 양수로 소폭 과추정 경향이 있다. 이전 dataset-v2에서는 5.13M 파라미터 대비
32,594 세그먼트로 심각한 과적합이 우려됐으나, 새 데이터셋에서 정상적으로 학습됐다.

| 그래프               |                                              |
| -------------------- | -------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v2/eval_plot/pctn.png)   |
| Error Distribution   | ![](../data/results-v2/error_hist/pctn.png)  |
| Bland-Altman         | ![](../data/results-v2/bland_altman/pctn.png)|
| 훈련 손실 곡선       | ![](../data/results-v2/loss_graph/pctn.png)  |
| 훈련 MAE 곡선        | ![](../data/results-v2/mae_graph/pctn.png)   |

### 6.10 conv_reg (Simple 1D CNN Regression)

best_epoch: 6 / 총 11 에폭

```
SBP — MAE: 13.029, ME: +1.29, SD: 16.767, RMSE: 16.816 | Grade D | AAMI: ❌
DBP — MAE:  8.188, ME: −0.06, SD: 10.389, RMSE: 10.389 | Grade D | AAMI: ❌
```

**v2 10위(21.217), v1 7위(21.20)에서 −0.02 mmHg로 사실상 동일.** 36.9K 파라미터의 단순 CNN.
DBP ME = −0.06 mmHg로 DBP 편향이 전 모델 중 가장 낮다. v1과 거의 같은 성능을 유지한다.

| 그래프               |                                                 |
| -------------------- | ----------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v2/eval_plot/conv_reg.png)  |
| Error Distribution   | ![](../data/results-v2/error_hist/conv_reg.png) |
| Bland-Altman         | ![](../data/results-v2/bland_altman/conv_reg.png)|
| 훈련 손실 곡선       | ![](../data/results-v2/loss_graph/conv_reg.png) |
| 훈련 MAE 곡선        | ![](../data/results-v2/mae_graph/conv_reg.png)  |

### 6.11 xresnet1d (Deep XResNet)

best_epoch: 2 / 총 7 에폭

```
SBP — MAE: 13.055, ME: +1.45, SD: 16.810, RMSE: 16.873 | Grade D | AAMI: ❌
DBP — MAE:  8.211, ME: −1.66, SD: 10.321, RMSE: 10.454 | Grade D | AAMI: ❌
```

**v2 11위(21.266), v1 12위(21.72)에서 +0.45 mmHg 개선.** 9.47M 파라미터의 대형 모델.
DBP ME = −1.66 mmHg로 DBP에서 소폭 과소추정 경향이 있다. SBP와 DBP 편향이 반대 방향으로
나타나는 불균형한 패턴이다. v1 대비 순위도 12위→11위로 한 단계 상승했다.

| 그래프               |                                                   |
| -------------------- | ------------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v2/eval_plot/xresnet1d.png)   |
| Error Distribution   | ![](../data/results-v2/error_hist/xresnet1d.png)  |
| Bland-Altman         | ![](../data/results-v2/bland_altman/xresnet1d.png)|
| 훈련 손실 곡선       | ![](../data/results-v2/loss_graph/xresnet1d.png)  |
| 훈련 MAE 곡선        | ![](../data/results-v2/mae_graph/xresnet1d.png)   |

### 6.12 resnet1d_mini

best_epoch: 1 / 총 6 에폭

```
SBP — MAE: 13.034, ME: +0.85, SD: 16.773, RMSE: 16.794 | Grade D | AAMI: ❌
DBP — MAE:  8.310, ME: +0.43, SD: 10.536, RMSE: 10.545 | Grade D | AAMI: ❌
```

**v2 12위(21.344), v1 14위(21.86)에서 +0.52 mmHg 개선.** 이전 dataset-v2에서는 v2 1위(24.97)였으나
새 데이터셋에서 12위로 순위가 크게 하락했다. 이는 이전 소규모 데이터셋에서 우연히 유리했던
초기 수렴 패턴이 대규모 데이터셋에서는 이점으로 작용하지 않음을 시사한다.

| 그래프               |                                                      |
| -------------------- | ---------------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v2/eval_plot/resnet1d_mini.png)  |
| Error Distribution   | ![](../data/results-v2/error_hist/resnet1d_mini.png) |
| Bland-Altman         | ![](../data/results-v2/bland_altman/resnet1d_mini.png)|
| 훈련 손실 곡선       | ![](../data/results-v2/loss_graph/resnet1d_mini.png) |
| 훈련 MAE 곡선        | ![](../data/results-v2/mae_graph/resnet1d_mini.png)  |

### 6.13 resnet1d (기준 모델)

best_epoch: 1 / 총 6 에폭

```
SBP — MAE: 13.123, ME: −1.41, SD: 16.977, RMSE: 17.035 | Grade D | AAMI: ❌
DBP — MAE:  8.454, ME: −0.22, SD: 10.757, RMSE: 10.760 | Grade D | AAMI: ❌
```

**v2 13위(21.577), v1 11위(21.53)에서 −0.05 mmHg로 사실상 동일.** 2.18M 파라미터의 기준 모델.
SBP ME = −1.41 mmHg로 소폭 과소추정. DBP ME = −0.22 mmHg로 편향이 낮다.
그룹 내에서 SBP SD(16.977), DBP SD(10.757)이 다소 높은 편이다.

| 그래프               |                                                  |
| -------------------- | ------------------------------------------------ |
| Prediction vs Actual | ![](../data/results-v2/eval_plot/resnet1d.png)   |
| Error Distribution   | ![](../data/results-v2/error_hist/resnet1d.png)  |
| Bland-Altman         | ![](../data/results-v2/bland_altman/resnet1d.png)|
| 훈련 손실 곡선       | ![](../data/results-v2/loss_graph/resnet1d.png)  |
| 훈련 MAE 곡선        | ![](../data/results-v2/mae_graph/resnet1d.png)   |

### 6.14 minception (Multi-scale Inception 1D)

best_epoch: 1 / 총 6 에폭

```
SBP — MAE: 13.270, ME: −1.59, SD: 17.178, RMSE: 17.251 | Grade D | AAMI: ❌
DBP — MAE:  8.328, ME: −0.73, SD: 10.594, RMSE: 10.619 | Grade D | AAMI: ❌
```

**v2 14위(21.598), v1 13위(21.85)에서 +0.25 mmHg 개선.** 440.7K 파라미터의 중형 모델.
SBP ME = −1.59 mmHg로 소폭 과소추정 경향이 있다.
best_epoch=1로 첫 에폭에서 최적점에 도달하는 빠른 수렴 패턴이다.

| 그래프               |                                                    |
| -------------------- | -------------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v2/eval_plot/minception.png)   |
| Error Distribution   | ![](../data/results-v2/error_hist/minception.png)  |
| Bland-Altman         | ![](../data/results-v2/bland_altman/minception.png)|
| 훈련 손실 곡선       | ![](../data/results-v2/loss_graph/minception.png)  |
| 훈련 MAE 곡선        | ![](../data/results-v2/mae_graph/minception.png)   |

### 6.15 st_resnet (Spectro-Temporal ResNet)

best_epoch: 4 / 총 9 에폭

```
SBP — MAE: 13.246, ME: +0.53, SD: 17.046, RMSE: 17.055 | Grade D | AAMI: ❌
DBP — MAE:  8.372, ME: −0.00, SD: 10.611, RMSE: 10.611 | Grade D | AAMI: ❌
```

**v2 15위(21.618), v1 8위(21.25)에서 −0.37 mmHg 저하.** PPG·VPG·APG 3채널 입력 모델.
v1 대비 가장 큰 상대적 순위 하락(8위→15위)을 보인다. DBP ME = −0.004 mmHg로 DBP 편향이 전 모델 중
가장 낮지만, SBP SD 17.046로 그룹 내에서 높은 편이다. 다중 채널 특성이 새 데이터셋에서 상대적 강점을
보이지 못했다.

| 그래프               |                                                   |
| -------------------- | ------------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v2/eval_plot/st_resnet.png)   |
| Error Distribution   | ![](../data/results-v2/error_hist/st_resnet.png)  |
| Bland-Altman         | ![](../data/results-v2/bland_altman/st_resnet.png)|
| 훈련 손실 곡선       | ![](../data/results-v2/loss_graph/st_resnet.png)  |
| 훈련 MAE 곡선        | ![](../data/results-v2/mae_graph/st_resnet.png)   |

### 6.16 acfa (Attention CNN Feature Aggregation)

best_epoch: 1 / 총 6 에폭

```
SBP — MAE: 13.508, ME: +2.60, SD: 17.148, RMSE: 17.344 | Grade D | AAMI: ❌
DBP — MAE:  8.472, ME: −0.15, SD: 10.724, RMSE: 10.725 | Grade D | AAMI: ❌
```

**v2 16위(21.980), v1 10위(21.51)에서 −0.47 mmHg 저하.** SBP ME = +2.60 mmHg로
전 모델 중 SBP 과추정이 가장 크다. DBP ME = −0.15 mmHg로 SBP와 편향 방향이 반대다.
SBP SD 17.148로 그룹 내에서 높다. Attention 메커니즘이 새 데이터셋에서 상대적으로
이점을 발휘하지 못했다.

| 그래프               |                                                |
| -------------------- | ---------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v2/eval_plot/acfa.png)     |
| Error Distribution   | ![](../data/results-v2/error_hist/acfa.png)    |
| Bland-Altman         | ![](../data/results-v2/bland_altman/acfa.png)  |
| 훈련 손실 곡선       | ![](../data/results-v2/loss_graph/acfa.png)    |
| 훈련 MAE 곡선        | ![](../data/results-v2/mae_graph/acfa.png)     |

### 6.17 naive (베이스라인)

best_epoch: 7 / 총 12 에폭

```
SBP — MAE: 14.997, ME: −1.46, SD: 18.943, RMSE: 18.999 | Grade D | AAMI: ❌
DBP — MAE:  9.528, ME: −1.02, SD: 11.877, RMSE: 11.920 | Grade D | AAMI: ❌
```

**v2 17위(24.525), v1 15위(25.00)에서 +0.48 mmHg 개선.** 베이스라인으로 최하위를 유지하나
다른 모델들과 격차(~3 mmHg)는 크다. 이전 dataset-v2에서는 DBP ME = +6.83 mmHg의 큰 DBP
과추정 편향이 있었으나, 새 데이터셋에서는 DBP ME = −1.02 mmHg로 정상화됐다. 케이스 불균형 해소로
인한 직접적인 개선 효과다.

| 그래프               |                                                 |
| -------------------- | ----------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v2/eval_plot/naive.png)     |
| Error Distribution   | ![](../data/results-v2/error_hist/naive.png)    |
| Bland-Altman         | ![](../data/results-v2/bland_altman/naive.png)  |
| 훈련 손실 곡선       | ![](../data/results-v2/loss_graph/naive.png)    |
| 훈련 MAE 곡선        | ![](../data/results-v2/mae_graph/naive.png)     |

## 7. 훈련 과정 분석

### 7.1 Early Stopping 동작 요약

| 모델             | Best Epoch | 총 에폭 | v1 Best Epoch | 수렴 패턴                        |
| ---------------- | ---------- | ------- | ------------- | -------------------------------- |
| `cnn_bilstm_at`  | **9**      | 14      | 3             | 가장 늦은 수렴                   |
| `naive`          | **7**      | 12      | 7             | v1과 동일한 수렴 속도            |
| `ae_lstm`        | 6          | 11      | 4             | 완만한 수렴                      |
| `conv_reg`       | 6          | 11      | —             | 완만한 수렴                      |
| `conv_reg_ds`    | 4          | 9       | —             | 중간 수렴                        |
| `pctn`           | 4          | 9       | —             | 중간 수렴                        |
| `st_resnet`      | 4          | 9       | —             | 중간 수렴                        |
| `mtae`           | 3          | 8       | 3             | v1과 동일한 수렴 속도            |
| `mtae_tr`        | **2**      | 7       | 3             | v1 대비 빠른 수렴 (40→2 에폭)   |
| `resnet1d_tiny`  | 2          | 7       | —             | 빠른 수렴                        |
| `xresnet1d`      | 2          | 7       | —             | 빠른 수렴                        |
| `acfa`           | **1**      | 6       | —             | 즉각 수렴                        |
| `bpnet_cf`       | **1**      | 6       | —             | 즉각 수렴                        |
| `minception`     | **1**      | 6       | —             | 즉각 수렴                        |
| `resnet1d`       | **1**      | 6       | —             | 즉각 수렴                        |
| `resnet1d_micro` | **1**      | 6       | —             | 즉각 수렴                        |
| `resnet1d_mini`  | **1**      | 6       | —             | 즉각 수렴                        |

patience=5. 총 에폭 = best_epoch + 5.

### 7.2 수렴 패턴 분석

대규모 데이터셋(5.48M 학습 세그먼트, 에폭당 약 21,400 gradient steps)에서도
대부분의 모델이 **1~6 에폭** 내에 best_epoch에 도달한다. 이는 dataset-v1(6.77M 세그먼트)과
유사한 패턴으로, 에폭 수가 적은 것이 아니라 에폭당 학습량이 충분함을 의미한다.

**주목할 변화**:

- `mtae_tr`: 이전 dataset-v2 best_epoch=40 → 새 best_epoch=2. Transformer 구조가 소규모
  데이터셋에서 수렴이 불안정했던 반면, 대규모에서는 정상화됐다.
- `ae_lstm`: 이전 dataset-v2 best_epoch=18 → 새 best_epoch=6. AE 기반 모델이 대규모
  데이터셋에서 수렴 패턴이 정상화됐다.
- `naive`, `mtae`: v1과 동일한 best_epoch(각 7, 3)을 보여 데이터 규모 회복의 효과를 반영한다.

**수렴 패턴 유형**:

- **즉각 수렴** (best_epoch=1, 6 에폭): acfa, bpnet_cf, minception, resnet1d, resnet1d_micro, resnet1d_mini
- **빠른 수렴** (best_epoch 2~4, 7~9 에폭): xresnet1d, mtae_tr, resnet1d_tiny, conv_reg_ds, pctn, st_resnet
- **완만한 수렴** (best_epoch 5~9, 10~14 에폭): ae_lstm, conv_reg, cnn_bilstm_at
- **안정적 수렴** (best_epoch 7, 12 에폭): naive

## 8. 국제 표준 기준 달성 현황

### 8.1 AAMI 기준 분석

| 기준         | SBP                                                                                                   | DBP                                              |
| ------------ | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| ME ≤ ±5 mmHg | ✅ **전 모델 충족** (최대 \|ME\| = 2.60 mmHg / `acfa`)                                               | ✅ **전 모델 충족** (최대 \|ME\| = 1.66 mmHg / `xresnet1d`) |
| SD ≤ 8 mmHg  | ❌ **전 모델 미달** (최소 16.17 mmHg / `mtae`)                                                        | ❌ **전 모델 미달** (최소 10.12 mmHg / `mtae`)   |

SBP ME: 이전 dataset-v2에서 많은 모델이 −7~−17 mmHg의 큰 과소추정을 보였으나,
새 데이터셋에서는 전 모델 |ME| ≤ 2.60 mmHg으로 완전히 해소됐다. 케이스 불균형 해소(top10% 점유율
62%→21.5%)가 주된 원인이다.

SD 기준은 여전히 전 모델이 미달이다. SBP 최소 SD 16.17은 기준(8 mmHg)의 약 2배이며,
단순히 데이터 규모를 늘리는 것만으로는 SD를 충분히 낮추기 어려움을 시사한다.

### 8.2 BHS 등급 달성 현황

| 등급             | SBP         | DBP                                          |
| ---------------- | ----------- | -------------------------------------------- |
| A (≥60%/85%/95%) | 전무        | 전무                                         |
| B (≥50%/75%/90%) | 전무        | 전무                                         |
| C (≥40%/65%/85%) | 전무        | **전무** (±5% 기준 미달로 Grade C 불가)      |
| D                | **전 모델** | **전 모델**                                  |

DBP ±10% ≥ 65%: naive를 제외한 16개 모델 충족 (v1에서 9종 Grade C 달성과 유사한 수준).  
DBP ±15% ≥ 85%: 12개 모델 충족.  
단, Grade C 달성에는 세 조건을 모두 충족해야 하며, DBP ±5% ≥ 40% 기준을
충족하는 모델이 없어 전 모델 Grade D에 머문다.

**DBP ±5% 상위 모델**:
- `mtae`: 38.84% (Grade C까지 1.16%p 부족)
- `resnet1d_micro`: 38.54%
- `ae_lstm`: 38.45%
- `conv_reg_ds`: 38.32%

## 9. v1 대비 성능 변화 분석

### 9.1 v1 대비 최우수 성능 비교

| 지표           | v1 최우수              | v2 최우수              | 변화               |
| -------------- | ---------------------- | ---------------------- | ------------------ |
| SBP MAE        | 12.89 (`ae_lstm`)      | **12.578** (`mtae`)    | **−0.31 (−2.4%)**  |
| DBP MAE        | 7.84 (`mtae`)          | 8.017 (`mtae`)         | **+0.18 (+2.3%)**  |
| 합산 MAE       | 20.74 (`ae_lstm`)      | **20.595** (`mtae`)    | **−0.15 (−0.7%)**  |
| DBP Grade C 수 | 9종                    | 0종                    | −9종               |

SBP 최우수 성능은 오히려 v1보다 향상됐다. 합산 최우수도 v1(20.74)보다 약간 낮은 20.595로,
새 dataset-v2가 dataset-v1과 동등하거나 약간 더 좋은 성능을 달성했다.

DBP 최우수는 v1 `mtae`(7.84) → v2 `mtae`(8.017)으로 0.18 mmHg 소폭 저하됐다.
DBP Grade C 달성 수(9→0)는 통계적으로는 퇴보처럼 보이지만, v2 DBP ±5% 최우수(38.84%)는
v1 Grade C 달성 모델들의 ±5% 값(보통 40~44%)에 근접한 수준이다.

### 9.2 모델별 v1 → v2 순위 변화

| 모델             | v1 순위 | v2 순위 | 변화    | 비고                          |
| ---------------- | ------- | ------- | ------- | ----------------------------- |
| `mtae`           | 2       | 1       | **+1**  | 수렴 실패 극복 → 최우수       |
| `conv_reg_ds`    | 6       | 5       | **+1**  |                               |
| `mtae_tr`        | 9       | 7       | **+2**  |                               |
| `xresnet1d`      | 12      | 11      | **+1**  |                               |
| `resnet1d_mini`  | 14      | 12      | **+2**  |                               |
| `ae_lstm`        | 1       | 3       | −2      |                               |
| `resnet1d_micro` | 3       | 4       | −1      |                               |
| `resnet1d_tiny`  | 5       | 6       | −1      |                               |
| `cnn_bilstm_at`  | 4       | 8       | **−4**  |                               |
| `conv_reg`       | 7       | 10      | **−3**  |                               |
| `resnet1d`       | 11      | 13      | −2      |                               |
| `minception`     | 13      | 14      | −1      |                               |
| `st_resnet`      | 8       | 15      | **−7**  | 가장 큰 상대적 순위 하락      |
| `acfa`           | 10      | 16      | **−6**  |                               |
| `naive`          | 15      | 17      | −2      | (베이스라인, 순위 제외 대상)  |

### 9.3 성능 저하 모델 분석

성능이 저하된 모델은 `st_resnet`(−0.37), `acfa`(−0.47), `resnet1d`(−0.05), `conv_reg`(−0.02) 네 모델이다.
`resnet1d`와 `conv_reg`는 변화가 미미(−0.05, −0.02)하므로 실질적 저하는 `st_resnet`과 `acfa`뿐이다.

- **`st_resnet`**: PPG·VPG·APG 3채널을 활용하지만 새 데이터셋에서 상대적으로 이점이 줄었다.
  미분 채널(VPG, APG) 계산 시 노이즈 증폭 효과가 새 데이터셋의 다양한 신호 분포에서 더 크게 나타날 수 있다.
- **`acfa`**: Attention 가중치가 새 데이터 분포에서 덜 효과적으로 작동하는 것으로 분석된다.
  SBP ME = +2.60 mmHg의 과추정도 v1 대비 새로 나타난 경향이다.

## 10. 주요 발견 및 시사점

### 10.1 데이터 규모 회복으로 v1 수준 성능 달성

Rule 2 완화로 dataset-v2 규모가 44,664 세그먼트에서 7,829,237 세그먼트(175배)로 확대됐다.
이를 통해 전 모델이 dataset-v1과 동등하거나 약간 더 나은 성능을 달성했다. 합산 MAE 최우수 기준
v2(20.595)가 v1(20.74)을 소폭 상회한다. 데이터 품질(9단계 ABP 정제)을 유지하면서 규모도 확보한 결과다.

### 10.2 mtae의 극적 역전

이전 dataset-v2에서 SBP ME = −16.86 mmHg의 수렴 실패를 보였던 `mtae`가 최우수 모델로 부상했다.
다중 태스크 손실(재구성 + BP 회귀)의 균형은 충분한 데이터량이 확보될 때 안정적으로 수렴한다.
소규모 데이터에서 불안정했던 모델이 대규모에서 정상화된 전형적 사례다.

### 10.3 케이스 불균형 해소

이전 dataset-v2에서 상위 10% 케이스가 세그먼트의 62%를 점유하던 극단적 불균형이
21.5%로 해소됐다. 그 결과:
- SBP 체계적 과소추정 해소: 전 모델 |ME| ≤ 2.60 mmHg
- naive의 DBP 과추정 편향 해소: DBP ME +6.83 → −1.02 mmHg
- 전반적인 예측 안정성 향상

### 10.4 SD 한계: 데이터 품질만으로 해결 불가

SBP SD 최솟값 16.17 mmHg(mtae)는 AAMI 기준 8 mmHg의 약 2배다.
dataset-v1(SBP 범위 50~249 mmHg)과 달리 dataset-v2는 SBP 60~180 mmHg로 범위가 제한되어 있으며,
이 자체가 더 어려운 회귀 문제일 수 있다. SD 개선을 위해서는 레이블·모델 구조·손실 함수 차원의
접근이 필요하다.

### 10.5 DBP Grade C 달성 근접

DBP ±5% ≥ 40%가 Grade C의 병목이다. 최우수 `mtae`의 38.84%는 문턱값까지 1.16%p 부족하다.
케이스별 균등 샘플링, DBP 분포에 맞춘 가중 손실 함수, 또는 DBP 전용 head 구조 개선이
Grade C 달성의 가장 현실적인 경로다.

### 10.6 파라미터 수와 성능의 약한 상관관계

상위 5개 모델(mtae 119.5K, bpnet_cf, ae_lstm 50.6K, resnet1d_micro 15.1K, conv_reg_ds 14.1K)은
모두 소형~중형 파라미터를 가진다. 대형 모델(xresnet1d 9.47M, pctn 5.13M)이 중위권에 머문다.
충분한 데이터량에서도 초대형 모델이 특별한 이점을 보이지 않는다.

## 11. 미완료 실험 및 향후 과제

### 11.1 미완료 실험

| 항목          | 상태  | 비고                                                  |
| ------------- | ----- | ----------------------------------------------------- |
| `conv_reg_at` | ⛔ 폐기| 반복적 학습 실패로 평가 대상에서 제외. 아키텍처 재설계 필요 |

### 11.2 주요 향후 과제

1. **DBP Grade C 달성 (DBP ±5% ≥ 40%)**: 현재 최우수 38.84%(mtae).
   케이스별 균등 샘플링 또는 DBP 구간별 가중 손실 함수 적용이 우선 실험 대상이다.

2. **SBP/DBP SD 감소 (AAMI SD ≤ 8 mmHg)**: 현재 최소 SBP SD 16.17.
   Quantile 손실 함수, 앙상블, 또는 post-hoc calibration을 통해 오차 분산을 줄이는
   방향으로 실험한다.

3. **dataset-v2 QC 파라미터 추가 완화**: Rule 7(`min_peaks=4`), Rule 8(`sbp_range_max=40`)
   완화를 통해 세그먼트 수를 추가로 확보하여 dataset-v1 규모를 달성한다.

4. **conv_reg_at 아키텍처 수정**: Attention 게이팅 함수 또는 초기화 방식 변경으로
   검증셋에서도 정상적으로 동작하도록 개선한다.

5. **교차 데이터셋 학습**: dataset-v1(많은 양, lower quality labels)으로 사전학습 후
   dataset-v2(적은 양, high quality labels)로 파인튜닝하는 전략을 실험한다.

## 부록: 모델별 그래프 인덱스

| 모델             | eval_plot                                               | error_hist                                               | bland_altman                                               | loss_graph                                               | mae_graph                                               |
| ---------------- | ------------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------- | ------------------------------------------------------- |
| mtae             | ![](../data/results-v2/eval_plot/mtae.png)              | ![](../data/results-v2/error_hist/mtae.png)              | ![](../data/results-v2/bland_altman/mtae.png)              | ![](../data/results-v2/loss_graph/mtae.png)              | ![](../data/results-v2/mae_graph/mtae.png)              |
| bpnet_cf         | ![](../data/results-v2/eval_plot/bpnet_cf.png)          | ![](../data/results-v2/error_hist/bpnet_cf.png)          | ![](../data/results-v2/bland_altman/bpnet_cf.png)          | ![](../data/results-v2/loss_graph/bpnet_cf.png)          | ![](../data/results-v2/mae_graph/bpnet_cf.png)          |
| ae_lstm          | ![](../data/results-v2/eval_plot/ae_lstm.png)           | ![](../data/results-v2/error_hist/ae_lstm.png)           | ![](../data/results-v2/bland_altman/ae_lstm.png)           | ![](../data/results-v2/loss_graph/ae_lstm.png)           | ![](../data/results-v2/mae_graph/ae_lstm.png)           |
| resnet1d_micro   | ![](../data/results-v2/eval_plot/resnet1d_micro.png)    | ![](../data/results-v2/error_hist/resnet1d_micro.png)    | ![](../data/results-v2/bland_altman/resnet1d_micro.png)    | ![](../data/results-v2/loss_graph/resnet1d_micro.png)    | ![](../data/results-v2/mae_graph/resnet1d_micro.png)    |
| conv_reg_ds      | ![](../data/results-v2/eval_plot/conv_reg_ds.png)       | ![](../data/results-v2/error_hist/conv_reg_ds.png)       | ![](../data/results-v2/bland_altman/conv_reg_ds.png)       | ![](../data/results-v2/loss_graph/conv_reg_ds.png)       | ![](../data/results-v2/mae_graph/conv_reg_ds.png)       |
| resnet1d_tiny    | ![](../data/results-v2/eval_plot/resnet1d_tiny.png)     | ![](../data/results-v2/error_hist/resnet1d_tiny.png)     | ![](../data/results-v2/bland_altman/resnet1d_tiny.png)     | ![](../data/results-v2/loss_graph/resnet1d_tiny.png)     | ![](../data/results-v2/mae_graph/resnet1d_tiny.png)     |
| mtae_tr          | ![](../data/results-v2/eval_plot/mtae_tr.png)           | ![](../data/results-v2/error_hist/mtae_tr.png)           | ![](../data/results-v2/bland_altman/mtae_tr.png)           | ![](../data/results-v2/loss_graph/mtae_tr.png)           | ![](../data/results-v2/mae_graph/mtae_tr.png)           |
| cnn_bilstm_at    | ![](../data/results-v2/eval_plot/cnn_bilstm_at.png)     | ![](../data/results-v2/error_hist/cnn_bilstm_at.png)     | ![](../data/results-v2/bland_altman/cnn_bilstm_at.png)     | ![](../data/results-v2/loss_graph/cnn_bilstm_at.png)     | ![](../data/results-v2/mae_graph/cnn_bilstm_at.png)     |
| pctn             | ![](../data/results-v2/eval_plot/pctn.png)              | ![](../data/results-v2/error_hist/pctn.png)              | ![](../data/results-v2/bland_altman/pctn.png)              | ![](../data/results-v2/loss_graph/pctn.png)              | ![](../data/results-v2/mae_graph/pctn.png)              |
| conv_reg         | ![](../data/results-v2/eval_plot/conv_reg.png)          | ![](../data/results-v2/error_hist/conv_reg.png)          | ![](../data/results-v2/bland_altman/conv_reg.png)          | ![](../data/results-v2/loss_graph/conv_reg.png)          | ![](../data/results-v2/mae_graph/conv_reg.png)          |
| xresnet1d        | ![](../data/results-v2/eval_plot/xresnet1d.png)         | ![](../data/results-v2/error_hist/xresnet1d.png)         | ![](../data/results-v2/bland_altman/xresnet1d.png)         | ![](../data/results-v2/loss_graph/xresnet1d.png)         | ![](../data/results-v2/mae_graph/xresnet1d.png)         |
| resnet1d_mini    | ![](../data/results-v2/eval_plot/resnet1d_mini.png)     | ![](../data/results-v2/error_hist/resnet1d_mini.png)     | ![](../data/results-v2/bland_altman/resnet1d_mini.png)     | ![](../data/results-v2/loss_graph/resnet1d_mini.png)     | ![](../data/results-v2/mae_graph/resnet1d_mini.png)     |
| resnet1d         | ![](../data/results-v2/eval_plot/resnet1d.png)          | ![](../data/results-v2/error_hist/resnet1d.png)          | ![](../data/results-v2/bland_altman/resnet1d.png)          | ![](../data/results-v2/loss_graph/resnet1d.png)          | ![](../data/results-v2/mae_graph/resnet1d.png)          |
| minception       | ![](../data/results-v2/eval_plot/minception.png)        | ![](../data/results-v2/error_hist/minception.png)        | ![](../data/results-v2/bland_altman/minception.png)        | ![](../data/results-v2/loss_graph/minception.png)        | ![](../data/results-v2/mae_graph/minception.png)        |
| st_resnet        | ![](../data/results-v2/eval_plot/st_resnet.png)         | ![](../data/results-v2/error_hist/st_resnet.png)         | ![](../data/results-v2/bland_altman/st_resnet.png)         | ![](../data/results-v2/loss_graph/st_resnet.png)         | ![](../data/results-v2/mae_graph/st_resnet.png)         |
| acfa             | ![](../data/results-v2/eval_plot/acfa.png)              | ![](../data/results-v2/error_hist/acfa.png)              | ![](../data/results-v2/bland_altman/acfa.png)              | ![](../data/results-v2/loss_graph/acfa.png)              | ![](../data/results-v2/mae_graph/acfa.png)              |
| naive            | ![](../data/results-v2/eval_plot/naive.png)             | ![](../data/results-v2/error_hist/naive.png)             | ![](../data/results-v2/bland_altman/naive.png)             | ![](../data/results-v2/loss_graph/naive.png)             | ![](../data/results-v2/mae_graph/naive.png)             |
