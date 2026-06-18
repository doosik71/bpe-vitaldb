# 모델 평가 결과 (dataset-v2 기반 학습 및 평가)

작성일: 2026-06-18  
평가 대상: VitalDB PPG → SBP/DBP 직접 회귀 모델 (dataset-v2로 학습)  
평가 데이터셋: `data/dataset-v2/test` (case-level held-out, 459 cases, 7,283 segments)  
비교 기준: `data/dataset-v1` 기반 모델 평가 결과 (`data/logs-v1`, `docs/evaluation-result-v1.md`)

## 1. 개요

본 문서는 `data/dataset-v2`로 학습한 혈압 추정 모델들의 테스트셋 평가 결과를 종합한다.
`data/dataset-v2`는 원본 `.vital` 파일에서 ABP 파형 peak/foot 기반 레이블과 9단계 정제 룰을
적용해 구축한 고품질 소규모 데이터셋이다.

데이터셋 규모가 dataset-v1(9,687,189 세그먼트) 대비 약 **220분의 1**(44,664 세그먼트)로 극단적으로
축소되었으며, 이것이 모델 성능에 미치는 영향을 dataset·v1 결과와 비교하여 분석한다.

모든 v2 모델은 dataset-v1 기반 모델과 동일한 아키텍처·하이퍼파라미터로 학습되었으며,
평가도 동일한 `eval-model.py`로 수행됐다.

## 2. 평가 환경

| 항목                | 내용                                                                    |
| ------------------- | ----------------------------------------------------------------------- |
| 데이터셋            | VitalDB (dataset-v2: 9단계 ABP 정제 룰 적용, 44,664 세그먼트)           |
| 입력 신호           | PPG (`SNUADC/PLETH`), 125 Hz, 8초 (1,000 샘플)                          |
| 레이블              | SBP/DBP (mmHg), ABP 파형 peak/foot 직접 추출 (dataset-v1과 방식 상이)   |
| case 분할           | train 70% / val 10% / test 20% (case-level, seed=42)                    |
| 테스트 케이스 수    | 459 cases                                                               |
| 테스트 세그먼트 수  | 7,283 segments                                                          |
| 평가 체크포인트     | 각 모델의 `best.pt` (val loss 최소 epoch)                               |
| 공통 하이퍼파라미터 | lr=1e-3, weight_decay=1e-4, batch_size=256, max_epochs=100, patience=15 |
| 출력 디렉터리       | `data/models-v2`, `data/images-v2`, `data/logs-v2`                      |

### 2.1 데이터셋 규모 비교

| 구분  | dataset-v1 (이전)              | dataset-v2 (현재)          | 감소율              |
| ----- | ------------------------------ | -------------------------- | ------------------- |
| train | 6,769,507 세그먼트             | 32,594 세그먼트            | **−99.5%**          |
| val   | 962,633 세그먼트               | 4,787 세그먼트             | **−99.5%**          |
| test  | 1,955,049 세그먼트 / 672 cases | 7,283 세그먼트 / 459 cases | **−99.6% / −31.7%** |

테스트 케이스 수도 672 → 459로 31.7% 감소했다.
9단계 ABP 기반 정제 룰에서 1건 이상의 유효 세그먼트를 확보하지 못한 케이스가 제외된 결과다.

### 2.2 레이블 방식 차이 주의

dataset-v2의 레이블은 ABP 파형 peak/foot에서 직접 산출된다.
dataset·v1의 `Solar8000/ART_SBP·DBP` 1 Hz 수치 평균 레이블과 동일 케이스에서도 수 mmHg
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

| 모델명           | 분류                     | 파라미터 수 | 평가 여부 | 비고                                 |
| ---------------- | ------------------------ | ----------- | --------- | ------------------------------------ |
| `naive`          | 베이스라인               | —           | ✅        |                                      |
| `resnet1d`       | ResNet1D 계열            | 2.18 M      | ✅        |                                      |
| `resnet1d_mini`  | ResNet1D 계열            | 964.4 K     | ✅        |                                      |
| `resnet1d_tiny`  | ResNet1D 계열            | 60.6 K      | ✅        |                                      |
| `resnet1d_micro` | ResNet1D 계열            | 15.1 K      | ✅        |                                      |
| `st_resnet`      | 다중 채널                | 478.9 K     | ✅        |                                      |
| `minception`     | 다중 스케일              | 440.7 K     | ✅        |                                      |
| `xresnet1d`      | 대형 ResNet              | 9.47 M      | ✅        |                                      |
| `acfa`           | Attention CNN            | 542.6 K     | ✅        |                                      |
| `ae_lstm`        | AE + LSTM                | 50.6 K      | ✅        |                                      |
| `bpnet_cf`       | BPNet-CF (dual-scale)    | —           | ✅        | v2 신규 평가                         |
| `cnn_bilstm_at`  | CNN + BiLSTM             | 691.3 K     | ✅        |                                      |
| `conv_reg`       | Conv 회귀                | 36.9 K      | ✅        |                                      |
| `conv_reg_ds`    | Conv 회귀 (Depthwise)    | 14.1 K      | ✅        |                                      |
| `conv_reg_at`    | Conv 회귀 (Attention)    | 39.0 K      | ✅        | 학습 완전 실패 (near-zero 출력 수렴) |
| `mtae`           | 다중 태스크 오토인코더   | 119.5 K     | ✅        | SBP 수렴 실패                        |
| `mtae_tr`        | MTAE + Transformer       | 109.4 K     | ✅        |                                      |
| `pctn`           | Parallel CNN-Transformer | 5.13 M      | ✅        |                                      |

> `conv_reg_at`는 학습 완전 실패 모델로, 5절 이하 비교표에서는 일반 모델과 별도 행으로 표시한다.

## 5. 테스트셋 정량 평가 결과

### 5.1 SBP(수축기혈압) 종합 비교

| 모델                 | MAE ↓     | ME         | SD        | RMSE  | ±5%       | ±10%      | ±15%      | BHS | AAMI  |
| -------------------- | --------- | ---------- | --------- | ----- | --------- | --------- | --------- | --- | ----- |
| **`resnet1d_micro`** | **16.60** | −3.96      | **21.65** | 22.01 | **21.8%** | **41.7%** | **58.4%** | D   | ❌    |
| `bpnet_cf`           | 16.67     | −0.81      | 21.67     | 21.69 | 19.8%     | 40.5%     | 56.6%     | D   | ❌    |
| `resnet1d_mini`      | 16.75     | −4.51      | 22.17     | 22.62 | 22.8%     | 42.8%     | 58.7%     | D   | ❌    |
| `cnn_bilstm_at`      | 16.92     | −4.06      | 22.20     | 22.56 | 21.1%     | 40.6%     | 57.2%     | D   | ❌    |
| `st_resnet`          | 17.10     | −3.19      | 22.58     | 22.81 | 20.6%     | 40.2%     | 57.7%     | D   | ❌    |
| `conv_reg`           | 17.19     | −3.47      | 23.59     | 23.84 | 22.6%     | 42.6%     | 58.7%     | D   | ❌    |
| `acfa`               | 17.21     | −7.60      | 22.50     | 23.75 | 22.6%     | 43.1%     | 58.8%     | D   | ❌    |
| `minception`         | 17.23     | −2.13      | 23.31     | 23.40 | 20.7%     | 41.0%     | 58.1%     | D   | ❌    |
| `pctn`               | 17.29     | −1.42      | 23.39     | 23.43 | 21.1%     | 41.1%     | 56.6%     | D   | ❌    |
| `resnet1d_tiny`      | 17.31     | −0.99      | 23.50     | 23.52 | 21.3%     | 40.3%     | 57.0%     | D   | ❌    |
| `xresnet1d`          | 17.34     | −0.93      | 22.07     | 22.09 | 19.5%     | 37.2%     | 52.6%     | D   | ❌    |
| `naive`              | 17.41     | −4.47      | 21.83     | 22.28 | 20.0%     | 37.8%     | 53.9%     | D   | ❌    |
| `conv_reg_ds`        | 17.56     | −2.97      | 23.45     | 23.64 | 21.7%     | 40.9%     | 56.0%     | D   | ❌    |
| `resnet1d`           | 17.60     | −6.16      | 23.76     | 24.54 | 22.8%     | 42.6%     | 57.8%     | D   | ❌    |
| `mtae_tr`            | 17.67     | −9.61      | 21.63     | 23.67 | 22.5%     | 41.3%     | 55.9%     | D   | ❌    |
| `ae_lstm`            | 17.84     | −8.32      | 21.83     | 23.36 | 20.7%     | 39.2%     | 53.7%     | D   | ❌    |
| `mtae`               | 20.62     | **−16.86** | 21.65     | 27.44 | 19.0%     | 36.1%     | 50.8%     | D   | ❌    |

> ↓: 낮을수록 좋음. ME 부호: 양수=과추정, 음수=과소추정.  
> 전 모델 SBP BHS Grade D. 대부분 모델에서 ME < 0 (SBP 체계적 과소추정).  
> `mtae`의 SBP ME = −16.86 mmHg은 다중 태스크 손실과 초소량 데이터의 수렴 실패를 나타낸다.

### 5.2 DBP(이완기혈압) 종합 비교

| 모델                | MAE ↓     | ME         | SD        | RMSE      | ±5%       | ±10%      | ±15%      | BHS | AAMI  |
| ------------------- | --------- | ---------- | --------- | --------- | --------- | --------- | --------- | --- | ----- |
| **`resnet1d_mini`** | **8.22**  | +1.01      | **10.35** | **10.40** | **37.8%** | **67.9%** | **84.8%** | D   | ❌    |
| `cnn_bilstm_at`     | 8.42      | +1.11      | 10.34     | 10.40     | 36.3%     | 63.3%     | 84.1%     | D   | ❌    |
| `acfa`              | 8.43      | −0.66      | 10.52     | 10.54     | 36.5%     | 65.5%     | 84.0%     | D   | ❌    |
| `bpnet_cf`          | 8.47      | +3.26      | 10.05     | 10.56     | 36.4%     | 65.2%     | 83.1%     | D   | ❌    |
| `conv_reg`          | 8.55      | +1.00      | 11.10     | 11.15     | 37.6%     | 66.2%     | 84.1%     | D   | ❌    |
| `resnet1d_tiny`     | 8.63      | +0.90      | 10.97     | 11.01     | 36.4%     | 65.4%     | 84.0%     | D   | ❌    |
| `pctn`              | 8.64      | +1.77      | 10.88     | 11.03     | 35.8%     | 65.6%     | 84.0%     | D   | ❌    |
| `mtae`              | 8.66      | −2.15      | 10.70     | 10.91     | 35.8%     | 65.6%     | 83.0%     | D   | ❌    |
| `st_resnet`         | 8.66      | +1.26      | 10.69     | 10.77     | 35.0%     | 64.5%     | 83.0%     | D   | ❌    |
| `resnet1d_micro`    | 8.77      | +3.22      | 10.46     | 10.94     | 35.2%     | 63.5%     | 80.8%     | D   | ❌    |
| `mtae_tr`           | 8.78      | −1.52      | 10.94     | 11.04     | 35.1%     | 64.8%     | 82.7%     | D   | ❌    |
| `minception`        | 8.78      | +0.59      | 11.07     | 11.09     | 35.2%     | 63.9%     | 83.0%     | D   | ❌    |
| `conv_reg_ds`       | 8.93      | +1.82      | 11.19     | 11.34     | 35.2%     | 64.4%     | 81.8%     | D   | ❌    |
| `ae_lstm`           | 8.98      | −1.02      | 11.17     | 11.22     | 34.5%     | 62.9%     | 80.8%     | D   | ❌    |
| `resnet1d`          | 9.10      | −0.26      | 11.35     | 11.36     | 32.6%     | 61.9%     | 82.8%     | D   | ❌    |
| `xresnet1d`         | 9.41      | +2.71      | 11.34     | 11.66     | 32.9%     | 59.2%     | 78.3%     | D   | ❌    |
| `naive`             | 10.89     | **+6.83**  | 11.17     | 13.09     | 26.7%     | 50.3%     | 70.4%     | D   | ❌    |
| *(학습 실패)*       |           |            |           |           |           |           |           |     |       |
| `conv_reg_at`       | **57.87** | **−57.87** | 11.39     | 58.98     | 0.03%     | 0.07%     | 0.08%     | D   | ❌    |

> **전 모델 DBP BHS Grade D** — dataset-v1에서 9종이 Grade C를 달성했던 것과 대비된다.  
> AAMI 기준 SD ≤ 8 mmHg: 전 모델 미달(최소 10.05 mmHg / `bpnet_cf`).  
> `naive`의 DBP ME = +6.83 mmHg은 케이스 불균형으로 인한 훈련 분포 편향이 추론 편향으로 이어진 결과다.  
> `conv_reg_at`의 DBP ME ≈ −57.87 mmHg은 모델이 DBP ~3.7 mmHg 고정값을 출력하는 학습 완전 실패를 나타낸다.

### 5.3 종합 순위 (SBP MAE + DBP MAE 합산 기준)

| 순위 | 모델             | SBP MAE | DBP MAE | 합산       | v1 합산 | 변화             | DBP BHS |
| ---- | ---------------- | ------- | ------- | ---------- | ------- | ---------------- | ------- |
| 1    | `resnet1d_mini`  | 16.75   | 8.22    | **24.97**  | 21.86   | ↓ −3.11          | D       |
| 2    | `bpnet_cf`       | 16.67   | 8.47    | **25.14**  | —       | (v1 미평가)      | D       |
| 3    | `cnn_bilstm_at`  | 16.92   | 8.42    | **25.34**  | 21.08   | ↓ −4.26          | D       |
| 4    | `resnet1d_micro` | 16.60   | 8.77    | **25.37**  | 20.89   | ↓ −4.48          | D       |
| 5    | `acfa`           | 17.21   | 8.43    | **25.64**  | 21.51   | ↓ −4.13          | D       |
| 6    | `conv_reg`       | 17.19   | 8.55    | **25.74**  | 21.20   | ↓ −4.54          | D       |
| 7    | `st_resnet`      | 17.10   | 8.66    | **25.76**  | 21.25   | ↓ −4.51          | D       |
| 8    | `pctn`           | 17.29   | 8.64    | **25.93**  | —       | (v1 미평가)      | D       |
| 9    | `resnet1d_tiny`  | 17.31   | 8.63    | **25.94**  | 21.17   | ↓ −4.77          | D       |
| 10   | `minception`     | 17.23   | 8.78    | **26.01**  | 21.85   | ↓ −4.16          | D       |
| 11   | `mtae_tr`        | 17.67   | 8.78    | **26.45**  | 21.31   | ↓ −5.14          | D       |
| 12   | `conv_reg_ds`    | 17.56   | 8.93    | **26.49**  | 21.19   | ↓ −5.30          | D       |
| 13   | `resnet1d`       | 17.60   | 9.10    | **26.70**  | 21.53   | ↓ −5.17          | D       |
| 14   | `xresnet1d`      | 17.34   | 9.41    | **26.75**  | 21.72   | ↓ −5.03          | D       |
| 15   | `ae_lstm`        | 17.84   | 8.98    | **26.82**  | 20.74   | ↓ **−6.08**      | D       |
| —    | `naive`          | 17.41   | 10.89   | **28.30**  | 25.00   | ↓ −3.30          | D       |
| —    | `mtae`           | 20.62   | 8.66    | **29.28**  | 20.79   | ↓ **−8.49**      | D       |
| —    | *(학습 실패)*    |         |         |            |         |                  |         |
| —    | `conv_reg_at`    | 112.29  | 57.87   | **170.16** | —       | (학습 완전 실패) | D       |

> ↓: v1 대비 합산 MAE 증가(성능 하락). 전 모델 성능 저하.  
> `ae_lstm`은 v1 최우수(20.74)에서 v2 15위(26.82)로 가장 큰 낙폭을 보인다.  
> `mtae`는 SBP 수렴 실패로 합산 29.28을 기록해 naive(28.30)보다 낮다.  
> `conv_reg_at`는 학습 완전 실패로 순위 산정에서 제외하며, 참고용으로만 표시한다.

## 6. 모델별 상세 평가

### 6.1 naive (베이스라인)

best_epoch: 100 / 총 100 에폭

```
SBP — MAE: 17.41, ME: -4.47, SD: 21.83, RMSE: 22.28 | Grade D | AAMI: ❌
DBP — MAE: 10.89, ME: +6.83, SD: 11.17, RMSE: 13.09 | Grade D | AAMI: ❌
```

best_epoch=100은 patience=15 이내에 val_loss가 단 한 번도 개선되지 않은 상태를 의미하지 않는다.
epoch 100까지 실행됐다는 것은 epoch 85 이후로 개선이 없었음을 나타낸다.
DBP ME = +6.83 mmHg는 케이스 불균형(상위 10% 케이스가 세그먼트의 62%를 점유)으로 인해
훈련셋 분포가 DBP가 높은 케이스에 편향된 결과다. dataset-v1(DBP ME −0.95)와 방향이 반전됐다.

| 그래프               |                                             |
| -------------------- | ------------------------------------------- |
| Prediction vs Actual | ![](../data/images-v2/eval_plot/naive.png)  |
| Error Distribution   | ![](../data/images-v2/error_hist/naive.png) |
| 훈련 손실 곡선       | ![](../data/images-v2/loss_graph/naive.png) |
| 훈련 MAE 곡선        | ![](../data/images-v2/mae_graph/naive.png)  |

### 6.2 resnet1d_mini

best_epoch: 3 / 총 8 에폭

```
SBP — MAE: 16.75, ME: -4.51, SD: 22.17, RMSE: 22.62 | Grade D | AAMI: ❌
DBP — MAE:  8.22, ME: +1.01, SD: 10.35, RMSE: 10.40 | Grade D | AAMI: ❌
```

**v2 합산 최우수 모델(24.97)**. DBP MAE 8.22로 전 모델 중 최저. v1(21.86) 대비 합산 −3.11 mmHg
저하. best_epoch=3, 총 8에폭으로 매우 빠른 과적합을 보여준다. DBP ±15% 이내 비율 84.8%로 가장
높으며 Grade C 문턱(85%) 직전에 위치한다. SBP ME = −4.51 mmHg로 체계적 과소추정이 있으나
DBP에서는 편향이 낮다.

| 그래프               |                                                     |
| -------------------- | --------------------------------------------------- |
| Prediction vs Actual | ![](../data/images-v2/eval_plot/resnet1d_mini.png)  |
| Error Distribution   | ![](../data/images-v2/error_hist/resnet1d_mini.png) |
| 훈련 손실 곡선       | ![](../data/images-v2/loss_graph/resnet1d_mini.png) |
| 훈련 MAE 곡선        | ![](../data/images-v2/mae_graph/resnet1d_mini.png)  |

### 6.3 bpnet_cf (BPNet-CF Calibration-Free)

best_epoch: 2 / 총 7 에폭

```
SBP — MAE: 16.67, ME: -0.81, SD: 21.67, RMSE: 21.69 | Grade D | AAMI: ❌
DBP — MAE:  8.47, ME: +3.26, SD: 10.05, RMSE: 10.56 | Grade D | AAMI: ❌
```

**SBP ME = −0.81 mmHg로 전 모델 중 가장 낮은 SBP 편향**. SBP SD 21.67, DBP SD 10.05.
DBP SD 10.05는 전 모델 최저다. 편향이 작고 산포도 낮아 안정적인 예측을 보이나
DBP ME = +3.26 mmHg로 DBP 과추정 경향이 있다. best_epoch=2, 총 7에폭으로 매우 빠른 과적합.
v1에서는 평가되지 않은 신규 모델이다.

| 그래프               |                                                |
| -------------------- | ---------------------------------------------- |
| Prediction vs Actual | ![](../data/images-v2/eval_plot/bpnet_cf.png)  |
| Error Distribution   | ![](../data/images-v2/error_hist/bpnet_cf.png) |
| 훈련 손실 곡선       | ![](../data/images-v2/loss_graph/bpnet_cf.png) |
| 훈련 MAE 곡선        | ![](../data/images-v2/mae_graph/bpnet_cf.png)  |

### 6.4 cnn_bilstm_at (CNN + BiLSTM with Attention)

best_epoch: 20 / 총 25 에폭

```
SBP — MAE: 16.92, ME: -4.06, SD: 22.20, RMSE: 22.56 | Grade D | AAMI: ❌
DBP — MAE:  8.42, ME: +1.11, SD: 10.34, RMSE: 10.40 | Grade D | AAMI: ❌
```

**v2에서 상대적으로 안정적인 수렴을 보인 소수 모델 중 하나**. best_epoch=20, 총 25에폭으로
다른 모델에 비해 훈련이 늦게까지 지속됐다(v1에서는 best_epoch=3). 작은 데이터셋에서 BiLSTM의
순차적 학습이 더 많은 에폭을 요구하는 것으로 분석된다. v1(21.08) 대비 합산 −4.26 저하.
DBP ±10% 비율 63.3%로 Grade C 기준(65%)에 미달하나 상위권이다.

| 그래프               |                                                     |
| -------------------- | --------------------------------------------------- |
| Prediction vs Actual | ![](../data/images-v2/eval_plot/cnn_bilstm_at.png)  |
| Error Distribution   | ![](../data/images-v2/error_hist/cnn_bilstm_at.png) |
| 훈련 손실 곡선       | ![](../data/images-v2/loss_graph/cnn_bilstm_at.png) |
| 훈련 MAE 곡선        | ![](../data/images-v2/mae_graph/cnn_bilstm_at.png)  |

### 6.5 resnet1d_micro (초소형 ResNet1D)

best_epoch: 10 / 총 15 에폭

```
SBP — MAE: 16.60, ME: -3.96, SD: 21.65, RMSE: 22.01 | Grade D | AAMI: ❌
DBP — MAE:  8.77, ME: +3.22, SD: 10.46, RMSE: 10.94 | Grade D | AAMI: ❌
```

**SBP MAE 최우수(16.60)**. 15.1K 파라미터의 초소형 모델이 v2에서도 SBP 예측에서 강점을
보인다. best_epoch=10으로 소규모 데이터셋에서도 상대적으로 안정적인 수렴을 보인다.
v1(20.89) 대비 합산 −4.48 저하. DBP ME = +3.22 mmHg로 DBP 과추정 경향이 있다.
SBP SD 21.65로 전 모델 중 가장 낮은 SBP 오차 산포를 보인다.

| 그래프               |                                                      |
| -------------------- | ---------------------------------------------------- |
| Prediction vs Actual | ![](../data/images-v2/eval_plot/resnet1d_micro.png)  |
| Error Distribution   | ![](../data/images-v2/error_hist/resnet1d_micro.png) |
| 훈련 손실 곡선       | ![](../data/images-v2/loss_graph/resnet1d_micro.png) |
| 훈련 MAE 곡선        | ![](../data/images-v2/mae_graph/resnet1d_micro.png)  |

### 6.6 st_resnet (Spectro-Temporal ResNet)

best_epoch: 6 / 총 11 에폭

```
SBP — MAE: 17.10, ME: -3.19, SD: 22.58, RMSE: 22.81 | Grade D | AAMI: ❌
DBP — MAE:  8.66, ME: +1.26, SD: 10.69, RMSE: 10.77 | Grade D | AAMI: ❌
```

PPG·VPG·APG 3채널 입력 모델. v1(21.25) 대비 합산 −4.51 저하.
SBP ME = −3.19 mmHg로 다른 복합 모델보다 편향이 낮은 편이다.
best_epoch=6으로 acfa와 동일한 수렴 패턴을 보인다.

| 그래프               |                                                 |
| -------------------- | ----------------------------------------------- |
| Prediction vs Actual | ![](../data/images-v2/eval_plot/st_resnet.png)  |
| Error Distribution   | ![](../data/images-v2/error_hist/st_resnet.png) |
| 훈련 손실 곡선       | ![](../data/images-v2/loss_graph/st_resnet.png) |
| 훈련 MAE 곡선        | ![](../data/images-v2/mae_graph/st_resnet.png)  |

### 6.7 acfa (Attention CNN Feature Aggregation)

best_epoch: 6 / 총 11 에폭

```
SBP — MAE: 17.21, ME: -7.60, SD: 22.50, RMSE: 23.75 | Grade D | AAMI: ❌
DBP — MAE:  8.43, ME: -0.66, SD: 10.52, RMSE: 10.54 | Grade D | AAMI: ❌
```

SBP ME = −7.60 mmHg로 acfa가 SBP를 체계적으로 크게 과소추정함을 보여준다.
DBP에서는 ME = −0.66 mmHg로 편향이 낮아 SBP·DBP 간 편향 비대칭이 뚜렷하다.
v1(21.51) 대비 합산 −4.13 저하. SBP RMSE가 23.75로 이 그룹에서 높은 편이다.

| 그래프               |                                            |
| -------------------- | ------------------------------------------ |
| Prediction vs Actual | ![](../data/images-v2/eval_plot/acfa.png)  |
| Error Distribution   | ![](../data/images-v2/error_hist/acfa.png) |
| 훈련 손실 곡선       | ![](../data/images-v2/loss_graph/acfa.png) |
| 훈련 MAE 곡선        | ![](../data/images-v2/mae_graph/acfa.png)  |

### 6.8 conv_reg (Simple 1D CNN Regression)

best_epoch: 9 / 총 14 에폭

```
SBP — MAE: 17.19, ME: -3.47, SD: 23.59, RMSE: 23.84 | Grade D | AAMI: ❌
DBP — MAE:  8.55, ME: +1.00, SD: 11.10, RMSE: 11.15 | Grade D | AAMI: ❌
```

36.9K 파라미터의 단순 CNN. DBP ±5% 비율 37.6%로 단순 모델임에도 상위권.
v1(21.20) 대비 합산 −4.54 저하. SBP SD 23.59는 그룹 내에서 높은 편이다.

| 그래프               |                                                |
| -------------------- | ---------------------------------------------- |
| Prediction vs Actual | ![](../data/images-v2/eval_plot/conv_reg.png)  |
| Error Distribution   | ![](../data/images-v2/error_hist/conv_reg.png) |
| 훈련 손실 곡선       | ![](../data/images-v2/loss_graph/conv_reg.png) |
| 훈련 MAE 곡선        | ![](../data/images-v2/mae_graph/conv_reg.png)  |

### 6.9 minception (Multi-scale Inception 1D)

best_epoch: 5 / 총 10 에폭

```
SBP — MAE: 17.23, ME: -2.13, SD: 23.31, RMSE: 23.40 | Grade D | AAMI: ❌
DBP — MAE:  8.78, ME: +0.59, SD: 11.07, RMSE: 11.09 | Grade D | AAMI: ❌
```

SBP ME = −2.13 mmHg로 중간 모델 대비 낮은 편향을 보인다.
v1(21.85) 대비 합산 −4.16 저하. 440.7K 파라미터의 중형 모델이 7,283개 테스트 세그먼트에서
소형 모델과 큰 차이를 내지 못하는 것은 모델 복잡도보다 데이터 양이 더 큰 제약임을 시사한다.

| 그래프               |                                                  |
| -------------------- | ------------------------------------------------ |
| Prediction vs Actual | ![](../data/images-v2/eval_plot/minception.png)  |
| Error Distribution   | ![](../data/images-v2/error_hist/minception.png) |
| 훈련 손실 곡선       | ![](../data/images-v2/loss_graph/minception.png) |
| 훈련 MAE 곡선        | ![](../data/images-v2/mae_graph/minception.png)  |

### 6.10 pctn (Parallel CNN-Transformer)

best_epoch: 5 / 총 10 에폭

```
SBP — MAE: 17.29, ME: -1.42, SD: 23.39, RMSE: 23.43 | Grade D | AAMI: ❌
DBP — MAE:  8.64, ME: +1.77, SD: 10.88, RMSE: 11.03 | Grade D | AAMI: ❌
```

5.13M 파라미터의 대형 모델. SBP ME = −1.42 mmHg로 편향이 낮다. best_epoch=5, 총 10에폭으로
대형 모델임에도 매우 빠른 과적합을 보인다. 파라미터 5.13M이 32,594 훈련 세그먼트에 비해
과도하게 많아 심각한 과적합이 발생함을 시사한다.

| 그래프               |                                            |
| -------------------- | ------------------------------------------ |
| Prediction vs Actual | ![](../data/images-v2/eval_plot/pctn.png)  |
| Error Distribution   | ![](../data/images-v2/error_hist/pctn.png) |
| 훈련 손실 곡선       | ![](../data/images-v2/loss_graph/pctn.png) |
| 훈련 MAE 곡선        | ![](../data/images-v2/mae_graph/pctn.png)  |

### 6.11 resnet1d_tiny

best_epoch: 14 / 총 19 에폭

```
SBP — MAE: 17.31, ME: -0.99, SD: 23.50, RMSE: 23.52 | Grade D | AAMI: ❌
DBP — MAE:  8.63, ME: +0.90, SD: 10.97, RMSE: 11.01 | Grade D | AAMI: ❌
```

SBP ME = −0.99 mmHg로 SBP 편향이 작다. best_epoch=14, 총 19에폭으로 v2 모델 중에서
상대적으로 안정적인 수렴을 보인다. v1(21.17) 대비 합산 −4.77 저하.

| 그래프               |                                                     |
| -------------------- | --------------------------------------------------- |
| Prediction vs Actual | ![](../data/images-v2/eval_plot/resnet1d_tiny.png)  |
| Error Distribution   | ![](../data/images-v2/error_hist/resnet1d_tiny.png) |
| 훈련 손실 곡선       | ![](../data/images-v2/loss_graph/resnet1d_tiny.png) |
| 훈련 MAE 곡선        | ![](../data/images-v2/mae_graph/resnet1d_tiny.png)  |

### 6.12 xresnet1d (Deep XResNet)

best_epoch: 3 / 총 8 에폭

```
SBP — MAE: 17.34, ME: -0.93, SD: 22.07, RMSE: 22.09 | Grade D | AAMI: ❌
DBP — MAE:  9.41, ME: +2.71, SD: 11.34, RMSE: 11.66 | Grade D | AAMI: ❌
```

9.47M 파라미터의 대형 모델. SBP SD 22.07로 비교적 낮은 오차 산포를 보이나 DBP MAE 9.41은
전 모델 최저 수준이다. best_epoch=3, 총 8에폭으로 대형 모델의 빠른 과적합을 보인다.
v1(21.72) 대비 합산 −5.03 저하.

| 그래프               |                                                 |
| -------------------- | ----------------------------------------------- |
| Prediction vs Actual | ![](../data/images-v2/eval_plot/xresnet1d.png)  |
| Error Distribution   | ![](../data/images-v2/error_hist/xresnet1d.png) |
| 훈련 손실 곡선       | ![](../data/images-v2/loss_graph/xresnet1d.png) |
| 훈련 MAE 곡선        | ![](../data/images-v2/mae_graph/xresnet1d.png)  |

### 6.13 conv_reg_ds (Depthwise-Separable Conv Regression)

best_epoch: 4 / 총 9 에폭

```
SBP — MAE: 17.56, ME: -2.97, SD: 23.45, RMSE: 23.64 | Grade D | AAMI: ❌
DBP — MAE:  8.93, ME: +1.82, SD: 11.19, RMSE: 11.34 | Grade D | AAMI: ❌
```

14.1K 파라미터의 경량 모델. v1(21.19) 대비 합산 −5.30 저하(전 모델 중 mtae_tr·resnet1d 다음으로
큰 절대 낙폭). best_epoch=4, 총 9에폭.

| 그래프               |                                                   |
| -------------------- | ------------------------------------------------- |
| Prediction vs Actual | ![](../data/images-v2/eval_plot/conv_reg_ds.png)  |
| Error Distribution   | ![](../data/images-v2/error_hist/conv_reg_ds.png) |
| 훈련 손실 곡선       | ![](../data/images-v2/loss_graph/conv_reg_ds.png) |
| 훈련 MAE 곡선        | ![](../data/images-v2/mae_graph/conv_reg_ds.png)  |

### 6.14 resnet1d (기준 모델)

best_epoch: 12 / 총 17 에폭

```
SBP — MAE: 17.60, ME: -6.16, SD: 23.76, RMSE: 24.54 | Grade D | AAMI: ❌
DBP — MAE:  9.10, ME: -0.26, SD: 11.35, RMSE: 11.36 | Grade D | AAMI: ❌
```

2.18M 파라미터의 기준 ResNet. SBP ME = −6.16 mmHg로 상위권 모델 대비 편향이 크다.
SBP RMSE 24.54로 mtae를 제외하면 전 모델 중 최고다. v1(21.53) 대비 합산 −5.17 저하.
best_epoch=12로 중간 수준의 수렴을 보인다.

| 그래프               |                                                |
| -------------------- | ---------------------------------------------- |
| Prediction vs Actual | ![](../data/images-v2/eval_plot/resnet1d.png)  |
| Error Distribution   | ![](../data/images-v2/error_hist/resnet1d.png) |
| 훈련 손실 곡선       | ![](../data/images-v2/loss_graph/resnet1d.png) |
| 훈련 MAE 곡선        | ![](../data/images-v2/mae_graph/resnet1d.png)  |

### 6.15 mtae_tr (MTAE with Transformer)

best_epoch: 40 / 총 45 에폭

```
SBP — MAE: 17.67, ME: -9.61, SD: 21.63, RMSE: 23.67 | Grade D | AAMI: ❌
DBP — MAE:  8.78, ME: -1.52, SD: 10.94, RMSE: 11.04 | Grade D | AAMI: ❌
```

**v2에서 가장 늦게 수렴한 모델(best_epoch=40)**. Transformer 블록이 소규모 데이터셋에서도
더 많은 에폭이 필요함을 보여준다. SBP ME = −9.61 mmHg로 mtae 다음으로 큰 SBP 과소추정 편향을
보인다. SBP SD = 21.63으로 ME가 크지만 SD 자체는 낮아, 편향된 방향으로 일관성 있게 예측하는
패턴이다. v1(21.31) 대비 합산 −5.14 저하.

| 그래프               |                                               |
| -------------------- | --------------------------------------------- |
| Prediction vs Actual | ![](../data/images-v2/eval_plot/mtae_tr.png)  |
| Error Distribution   | ![](../data/images-v2/error_hist/mtae_tr.png) |
| 훈련 손실 곡선       | ![](../data/images-v2/loss_graph/mtae_tr.png) |
| 훈련 MAE 곡선        | ![](../data/images-v2/mae_graph/mtae_tr.png)  |

### 6.16 ae_lstm (Autoencoder + LSTM)

best_epoch: 18 / 총 23 에폭

```
SBP — MAE: 17.84, ME: -8.32, SD: 21.83, RMSE: 23.36 | Grade D | AAMI: ❌
DBP — MAE:  8.98, ME: -1.02, SD: 11.17, RMSE: 11.22 | Grade D | AAMI: ❌
```

**v1 최우수 모델(20.74)이 v2에서 15위(26.82)로 급락한 대표 사례**. SBP ME = −8.32 mmHg로
큰 과소추정 편향을 보인다. v1 대비 합산 −6.08 mmHg 저하는 전 모델 중 최대 낙폭이다.
오토인코더 기반 재구성 손실이 소규모 데이터셋에서 특히 취약함을 시사한다.
best_epoch=18로 수렴은 안정적이나 절대 성능이 크게 저하됐다.

| 그래프               |                                               |
| -------------------- | --------------------------------------------- |
| Prediction vs Actual | ![](../data/images-v2/eval_plot/ae_lstm.png)  |
| Error Distribution   | ![](../data/images-v2/error_hist/ae_lstm.png) |
| 훈련 손실 곡선       | ![](../data/images-v2/loss_graph/ae_lstm.png) |
| 훈련 MAE 곡선        | ![](../data/images-v2/mae_graph/ae_lstm.png)  |

### 6.17 mtae (Multi-Task AutoEncoder)

best_epoch: 99 / 총 100 에폭

```
SBP — MAE: 20.62, ME: -16.86, SD: 21.65, RMSE: 27.44 | Grade D | AAMI: ❌
DBP — MAE:  8.66, ME: -2.15, SD: 10.70, RMSE: 10.91 | Grade D | AAMI: ❌
```

**SBP 수렴 실패 모델**. SBP ME = −16.86 mmHg은 모델이 SBP를 실제 값보다 평균 17 mmHg 낮게
예측했음을 의미한다. best_epoch=99 (전체 100 에폭 거의 소진)는 val_loss가 에폭 84 이후에도
개선되지 않았음을 나타내며, 훈련 자체가 안정적으로 수렴하지 못한 것으로 분석된다.

**DBP는 상대적으로 정상**: DBP MAE = 8.66, ME = −2.15로 다른 모델과 비슷한 수준이다.
BP 헤드의 SBP 출력만 수렴 실패한 것으로, 재구성 손실과 BP 회귀 손실의 불균형이 원인으로
추정된다. dataset/v1에서는 best_epoch=14 (정상 수렴)으로 문제가 없었지만,
32,594 훈련 세그먼트라는 극단적 소규모에서 다중 태스크 손실의 균형이 무너진 것으로 판단된다.
합산 MAE 29.28로 naive(28.30)보다도 낮다.

| 그래프               |                                            |
| -------------------- | ------------------------------------------ |
| Prediction vs Actual | ![](../data/images-v2/eval_plot/mtae.png)  |
| Error Distribution   | ![](../data/images-v2/error_hist/mtae.png) |
| 훈련 손실 곡선       | ![](../data/images-v2/loss_graph/mtae.png) |
| 훈련 MAE 곡선        | ![](../data/images-v2/mae_graph/mtae.png)  |

### 6.18 conv_reg_at (Conv Regression with Attention)

best_epoch: 38 / 총 43 에폭 (조기 종료)

```
SBP — MAE: 112.29, ME: -112.29, SD: 22.63, RMSE: 114.55 | Grade D | AAMI: ❌
DBP — MAE:  57.87, ME: -57.87,  SD: 11.39, RMSE:  58.98  | Grade D | AAMI: ❌
```

**학습 완전 실패 모델**. SBP ME = −112.29 mmHg, DBP ME = −57.87 mmHg은 모델이
SBP ~5.8 mmHg, DBP ~3.7 mmHg의 극단적으로 낮은 고정값을 출력하고 있음을 의미한다.

훈련 과정을 보면 **훈련 손실은 정상 수렴** (epoch 1: 284 → epoch 43: 52, train SBP MAE 83→16)하였으나
**검증 손실은 epoch 1(448.9)부터 epoch 43(429.3)까지 고착** (val SBP MAE ~115–120, val DBP MAE ~60–63)됐다.
훈련 데이터에 과도하게 특화된 어텐션 가중치가 검증 입력에서 near-zero 활성화를 생성한 것으로 분석된다.

dataset-v1에서도 conv_reg_at는 훈련이 발산(기록 없음)했으나, v2에서는 훈련 자체는 진행됐다는 점에서
메커니즘이 다르다. v2의 극단적 소규모(32,594 세그먼트)에서 어텐션 모듈이 훈련셋에만 특화된
과적합 표현을 학습한 결과이며, eval JSON도 생성됐지만 수치는 임상적으로 무의미하다.

| 그래프               |                                                   |
| -------------------- | ------------------------------------------------- |
| Prediction vs Actual | ![](../data/images-v2/eval_plot/conv_reg_at.png)  |
| Error Distribution   | ![](../data/images-v2/error_hist/conv_reg_at.png) |
| 훈련 손실 곡선       | ![](../data/images-v2/loss_graph/conv_reg_at.png) |
| 훈련 MAE 곡선        | ![](../data/images-v2/mae_graph/conv_reg_at.png)  |

## 7. 훈련 과정 분석

### 7.1 Early Stopping 동작 요약

| 모델             | Best Epoch | 총 에폭 | 과적합 패턴                                             | v1 Best Epoch |
| ---------------- | ---------- | ------- | ------------------------------------------------------- | ------------- |
| `mtae_tr`        | **40**     | 45      | 완만한 감소 후 수렴 (가장 늦은 수렴)                    | 3             |
| `ae_lstm`        | **18**     | 23      | 완만한 감소 후 수렴                                     | 4             |
| `cnn_bilstm_at`  | **20**     | 25      | 완만한 수렴                                             | 3             |
| `resnet1d`       | 12         | 17      | 중간 수렴                                               | —             |
| `resnet1d_tiny`  | 14         | 19      | 중간 수렴                                               | —             |
| `resnet1d_micro` | 10         | 15      | 중간 수렴                                               | 14            |
| `conv_reg`       | 9          | 14      | 빠른 수렴                                               | —             |
| `st_resnet`      | 6          | 11      | 빠른 수렴                                               | —             |
| `acfa`           | 6          | 11      | 빠른 수렴                                               | —             |
| `minception`     | 5          | 10      | 매우 빠른 과적합                                        | —             |
| `pctn`           | 5          | 10      | 매우 빠른 과적합 (5.13M params)                         | —             |
| `bpnet_cf`       | 2          | 7       | 즉각 과적합                                             | —             |
| `resnet1d_mini`  | 3          | 8       | 즉각 과적합                                             | —             |
| `xresnet1d`      | 3          | 8       | 즉각 과적합 (9.47M params)                              | —             |
| `conv_reg_ds`    | 4          | 9       | 즉각 과적합                                             | —             |
| `naive`          | **100**    | 100     | 수렴 불안정 (88~100 에폭 개선 없음)                     | 7             |
| `mtae`           | **99**     | 100     | **SBP 수렴 실패** (다중 태스크 불균형)                  | 3             |
| `conv_reg_at`    | 38         | 43      | **학습 완전 실패** (val_loss ~430 고착, near-zero 출력) | —             |

### 7.2 과적합 특성 분석

dataset-v2의 훈련 세그먼트 수(32,594개)는 dataset-v1(6,769,507개)의 0.48%에 불과하다.
이로 인해 대부분의 모델이 1~6 에폭 내에 과적합이 발생하며, 총 학습 에폭도 7~25에폭 수준에서
종료된다. dataset-v1에서 best_epoch=3~15였던 모델들이 v2에서도 유사한 패턴을 보인다.

**주목할 v1 대비 변화**:

- `mtae_tr`: v1 best_epoch=3 → v2 best_epoch=**40** (13배 증가). Transformer 구조가 소규모
  데이터셋에서도 더 많은 에폭을 요구하는 특성을 보인다. 단, 최종 성능은 크게 저하됐다.
- `ae_lstm`: v1 best_epoch=4 → v2 best_epoch=18 (4.5배 증가). 오토인코더가 소규모 데이터에서
  더 오래 학습하나 성능은 크게 저하됐다.
- `mtae`: v1 best_epoch=3 → v2 best_epoch=**99** (수렴 실패).

**수렴 패턴 유형**:

- **즉각 과적합** (best_epoch 1~4): bpnet_cf, resnet1d_mini, xresnet1d, conv_reg_ds — 모델이 첫 몇 에폭에 최적점 도달
- **중간 수렴** (best_epoch 5~20): cnn_bilstm_at, resnet1d, resnet1d_tiny, resnet1d_micro, st_resnet, acfa, minception, pctn — 적절한 학습 곡선
- **불안정 수렴** (best_epoch 40+): mtae_tr, ae_lstm — 훈련은 진행되나 효과 제한
- **수렴 실패** (best_epoch 99~100): mtae, naive — 유의미한 수렴 없음
- **학습 완전 실패**: conv_reg_at — 훈련 손실 수렴에도 불구하고 검증 손실이 epoch 1부터 고착, near-zero 출력

## 8. 국제 표준 기준 달성 현황

### 8.1 AAMI 기준 분석

| 기준         | SBP                                                                                                                                                | DBP                                                   |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| ME ≤ ±5 mmHg | **부분 충족**: bpnet_cf(0.81), resnet1d_tiny(0.99), xresnet1d(0.93) 등 일부 충족. acfa(-7.60), mtae_tr(-9.61), ae_lstm(-8.32), mtae(-16.86)는 미달 | **충족**: 전 모델 ME ≤ 4.0 mmHg (단 naive +6.83 미달) |
| SD ≤ 8 mmHg  | ❌ **전 모델 미달** (최소 21.63 / `mtae_tr`)                                                                                                       | ❌ **전 모델 미달** (최소 10.05 / `bpnet_cf`)         |

SBP ME 기준: v1에서는 전 모델이 ME ≤ ±5를 충족했으나, v2에서는 SBP 체계적 과소추정(음의 편향)으로
인해 acfa, mtae_tr, ae_lstm, mtae, resnet1d(ME −6.16)가 ME ≤ ±5 기준을 초과한다.
DBP ME: naive(+6.83)를 제외한 모든 모델이 ±5 mmHg 이내를 충족한다.
`conv_reg_at`는 SBP ME −112.29, DBP ME −57.87로 AAMI 기준과 비교가 무의미한 학습 완전 실패 모델이며 위 분석에서 제외했다.

### 8.2 BHS 등급 달성 현황

| 등급             | SBP         | DBP                |
| ---------------- | ----------- | ------------------ |
| A (≥60%/85%/95%) | 전무        | 전무               |
| B (≥50%/75%/90%) | 전무        | 전무               |
| C (≥40%/65%/85%) | 전무        | **전무** (v1: 9종) |
| D                | **전 모델** | **전 모델**        |

dataset-v1에서 DBP Grade C를 달성한 9종이 **v2에서 전부 Grade D로 하락**했다.
DBP Grade C 기준은 ±5 mmHg 이내 ≥ 40%, ±10 mmHg 이내 ≥ 65%, ±15 mmHg 이내 ≥ 85%이다.
v2에서 최우수 모델(resnet1d_mini)의 DBP ±15% = 84.8%로 Grade C 기준(85%) 직전에 위치한다.

## 9. v1 대비 성능 변화 분석

### 9.1 전 모델 성능 저하 요약

| 지표           | v1 최우수       | v2 최우수              | 저하량             |
| -------------- | --------------- | ---------------------- | ------------------ |
| SBP MAE        | 12.89 (ae_lstm) | 16.60 (resnet1d_micro) | **+3.71 (+28.8%)** |
| DBP MAE        | 7.84 (mtae)     | 8.22 (resnet1d_mini)   | **+0.38 (+4.8%)**  |
| 합산 MAE       | 20.74 (ae_lstm) | 24.97 (resnet1d_mini)  | **+4.23 (+20.4%)** |
| DBP Grade C 수 | 9종             | **0종**                | **−9종**           |

**DBP 저하 < SBP 저하**: DBP 최우수 성능은 +0.38 mmHg(4.8%) 저하인 반면, SBP는 +3.71 mmHg(28.8%)
저하로 SBP 성능이 훨씬 더 크게 영향을 받았다.

### 9.2 SBP 체계적 과소추정의 원인

v2에서 대부분 모델의 SBP ME가 음수(과소추정)이고, 특히 일부 모델에서 −7 ~ −17 mmHg로 크게 나타난다.

주요 원인 후보:

1. **훈련셋 SBP 분포 편향**: v2 SBP 분포는 ~100 mmHg와 ~150 mmHg 부근에서 이중봉(bimodal)을
   보이며, 상위 10% 케이스가 세그먼트의 62%를 점유한다. 케이스 불균형이 모델의 SBP 예측을
   특정 값으로 끌어당긴다.
2. **레이블 범위 변화**: v2 SBP 범위는 63–180 mmHg로, v1(50–249 mmHg)보다 상단이 절사됐다.
   훈련셋에 없는 고혈압 구간을 예측할 때 모델이 과소추정할 수 있다.
3. **소규모 데이터 과적합**: 적은 에폭만에 best_epoch에 도달하므로, 모델이 특정 SBP 영역에
   편향된 채로 학습이 종료된다.

### 9.3 상대적 순위 변화

| 모델            | v1 순위 | v2 순위  | 변화    |
| --------------- | ------- | -------- | ------- |
| `ae_lstm`       | 1       | 15       | **−14** |
| `mtae`          | 2       | 17(최하) | **−15** |
| `resnet1d_mini` | 16      | 1        | **+15** |
| `bpnet_cf`      | (신규)  | 2        | —       |
| `xresnet1d`     | 14      | 14       | ≈       |

v1 최우수 모델들(ae_lstm, mtae)이 v2에서 최하위권으로 이동한 반면, v1에서 하위권이었던
resnet1d_mini가 v2에서 최우수를 달성했다. 이는 데이터 규모에 따라 최적 아키텍처가 다를 수 있음을
보여준다.

## 10. 주요 발견 및 시사점

### 10.1 소규모 데이터셋의 전반적 성능 저하

9단계 정제로 얻은 고품질 소규모 데이터셋(44,664 세그먼트)은 전 모델에서 성능이 저하됐다.
데이터 품질이 개선됐음에도 데이터 양의 감소(−99.5%)가 더 큰 부정적 영향을 미친 결과다.

### 10.2 소형 모델의 상대적 강세

v2에서 상위 4개 모델(resnet1d_mini, bpnet_cf, cnn_bilstm_at, resnet1d_micro)이 모두 중·소형
파라미터를 가진다. 대형 모델(xresnet1d 9.47M, pctn 5.13M)은 과적합으로 인해 상대적으로
성능이 나빴다. 소규모 데이터셋에서는 모델 복잡도를 데이터 양에 맞게 제한해야 한다는 것을
보여준다.

### 10.3 다중 태스크 학습의 취약성

`mtae`(재구성+BP 다중 손실)는 v1에서 2위였으나 v2에서 SBP 수렴 실패로 최하위로 떨어졌다.
소규모 데이터에서 다중 태스크 손실의 균형이 무너지기 쉬우며, 특히 재구성 손실과 BP 회귀 손실의
스케일 차이가 문제가 될 수 있다.

### 10.4 Transformer 구조의 비선형 수렴 패턴

`mtae_tr`은 v2에서 best_epoch=40으로 전 모델 중 가장 늦게 수렴했다. Transformer 구조가 소규모
데이터에서 수렴에 더 많은 에폭이 필요하지만, 최종 성능은 v1 대비 크게 저하됐다.

### 10.5 케이스 불균형의 영향

`naive` 모델의 DBP ME = +6.83 mmHg는 케이스 불균형(상위 10% 케이스가 세그먼트의 62% 점유)이
훈련 분포를 편향시켜 예측 편향으로 이어질 수 있음을 보여준다. 학습 전 케이스별 샘플링 균등화가
필요하다.

### 10.6 SBP vs DBP 성능 격차 확대

v1에서 SBP MAE - DBP MAE ≈ 5 mmHg 차이였으나, v2에서는 최우수 모델 기준 SBP 16.60 - DBP 8.22
= 8.38 mmHg로 격차가 확대됐다. SBP가 DBP보다 소규모 데이터에 훨씬 더 취약하다.

## 11. 미완료 실험 및 향후 과제

### 11.1 미완료 실험

| 항목          | 상태                     | 비고                                                                 |
| ------------- | ------------------------ | -------------------------------------------------------------------- |
| `conv_reg_at` | ✅ 평가 완료 (학습 실패) | 훈련 손실은 수렴했으나 검증 손실 고착 → near-zero 출력 → 6.18절 참조 |

### 11.2 주요 향후 과제

1. **케이스 불균형 대처**: 케이스별 weighted sampling 또는 균등 샘플링을 적용하여 상위 케이스
   편향을 제거하고 DBP Grade C 달성을 시도한다.

2. **SBP 체계적 편향 해소**: SBP 분포 범위(63–180 mmHg)에 맞는 custom loss(quantile loss,
   weighted MSE) 또는 레이블 정규화 적용을 검토한다.

3. **데이터 보강**: dataset-v2에 추가 정제 없이 dataset-v1 세그먼트 일부를 혼합 학습하거나,
   데이터 증강(augmentation)을 적용하여 소규모 데이터 문제를 완화한다.

4. **mtae 다중 태스크 손실 재조정**: 소규모 데이터에서 재구성 손실 가중치를 낮추거나(현재 0.5:0.5),
   BP 회귀 손실에 우선순위를 두는 방식으로 SBP 수렴 실패를 해소한다.

5. **dataset-v2 정제 파라미터 재검토**: 9단계 룰 중 특히 룰 8(SBP 변동 ≤ 40 mmHg)과 룰 7
   (`--min-peaks=4`) 완화를 통해 생존 세그먼트 수를 늘리고 분포 균형을 회복한다.

6. **교차 데이터셋 학습**: dataset-v1으로 사전학습 후 dataset-v2로 파인튜닝하여, 레이블 품질과
   데이터 양을 모두 활용하는 혼합 전략을 실험한다.

## 부록: 모델별 그래프 인덱스

| 모델           | eval_plot                                           | error_hist                                           | loss_graph                                           | mae_graph                                           |
| -------------- | --------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- | --------------------------------------------------- |
| naive          | ![](../data/images-v2/eval_plot/naive.png)          | ![](../data/images-v2/error_hist/naive.png)          | ![](../data/images-v2/loss_graph/naive.png)          | ![](../data/images-v2/mae_graph/naive.png)          |
| resnet1d       | ![](../data/images-v2/eval_plot/resnet1d.png)       | ![](../data/images-v2/error_hist/resnet1d.png)       | ![](../data/images-v2/loss_graph/resnet1d.png)       | ![](../data/images-v2/mae_graph/resnet1d.png)       |
| resnet1d_mini  | ![](../data/images-v2/eval_plot/resnet1d_mini.png)  | ![](../data/images-v2/error_hist/resnet1d_mini.png)  | ![](../data/images-v2/loss_graph/resnet1d_mini.png)  | ![](../data/images-v2/mae_graph/resnet1d_mini.png)  |
| resnet1d_tiny  | ![](../data/images-v2/eval_plot/resnet1d_tiny.png)  | ![](../data/images-v2/error_hist/resnet1d_tiny.png)  | ![](../data/images-v2/loss_graph/resnet1d_tiny.png)  | ![](../data/images-v2/mae_graph/resnet1d_tiny.png)  |
| resnet1d_micro | ![](../data/images-v2/eval_plot/resnet1d_micro.png) | ![](../data/images-v2/error_hist/resnet1d_micro.png) | ![](../data/images-v2/loss_graph/resnet1d_micro.png) | ![](../data/images-v2/mae_graph/resnet1d_micro.png) |
| st_resnet      | ![](../data/images-v2/eval_plot/st_resnet.png)      | ![](../data/images-v2/error_hist/st_resnet.png)      | ![](../data/images-v2/loss_graph/st_resnet.png)      | ![](../data/images-v2/mae_graph/st_resnet.png)      |
| minception     | ![](../data/images-v2/eval_plot/minception.png)     | ![](../data/images-v2/error_hist/minception.png)     | ![](../data/images-v2/loss_graph/minception.png)     | ![](../data/images-v2/mae_graph/minception.png)     |
| xresnet1d      | ![](../data/images-v2/eval_plot/xresnet1d.png)      | ![](../data/images-v2/error_hist/xresnet1d.png)      | ![](../data/images-v2/loss_graph/xresnet1d.png)      | ![](../data/images-v2/mae_graph/xresnet1d.png)      |
| acfa           | ![](../data/images-v2/eval_plot/acfa.png)           | ![](../data/images-v2/error_hist/acfa.png)           | ![](../data/images-v2/loss_graph/acfa.png)           | ![](../data/images-v2/mae_graph/acfa.png)           |
| ae_lstm        | ![](../data/images-v2/eval_plot/ae_lstm.png)        | ![](../data/images-v2/error_hist/ae_lstm.png)        | ![](../data/images-v2/loss_graph/ae_lstm.png)        | ![](../data/images-v2/mae_graph/ae_lstm.png)        |
| bpnet_cf       | ![](../data/images-v2/eval_plot/bpnet_cf.png)       | ![](../data/images-v2/error_hist/bpnet_cf.png)       | ![](../data/images-v2/loss_graph/bpnet_cf.png)       | ![](../data/images-v2/mae_graph/bpnet_cf.png)       |
| cnn_bilstm_at  | ![](../data/images-v2/eval_plot/cnn_bilstm_at.png)  | ![](../data/images-v2/error_hist/cnn_bilstm_at.png)  | ![](../data/images-v2/loss_graph/cnn_bilstm_at.png)  | ![](../data/images-v2/mae_graph/cnn_bilstm_at.png)  |
| conv_reg       | ![](../data/images-v2/eval_plot/conv_reg.png)       | ![](../data/images-v2/error_hist/conv_reg.png)       | ![](../data/images-v2/loss_graph/conv_reg.png)       | ![](../data/images-v2/mae_graph/conv_reg.png)       |
| conv_reg_ds    | ![](../data/images-v2/eval_plot/conv_reg_ds.png)    | ![](../data/images-v2/error_hist/conv_reg_ds.png)    | ![](../data/images-v2/loss_graph/conv_reg_ds.png)    | ![](../data/images-v2/mae_graph/conv_reg_ds.png)    |
| mtae           | ![](../data/images-v2/eval_plot/mtae.png)           | ![](../data/images-v2/error_hist/mtae.png)           | ![](../data/images-v2/loss_graph/mtae.png)           | ![](../data/images-v2/mae_graph/mtae.png)           |
| mtae_tr        | ![](../data/images-v2/eval_plot/mtae_tr.png)        | ![](../data/images-v2/error_hist/mtae_tr.png)        | ![](../data/images-v2/loss_graph/mtae_tr.png)        | ![](../data/images-v2/mae_graph/mtae_tr.png)        |
| pctn           | ![](../data/images-v2/eval_plot/pctn.png)           | ![](../data/images-v2/error_hist/pctn.png)           | ![](../data/images-v2/loss_graph/pctn.png)           | ![](../data/images-v2/mae_graph/pctn.png)           |
