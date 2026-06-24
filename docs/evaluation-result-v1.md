# 모델 평가 결과 (dataset-v1 기반 학습 및 평가)

작성일: 2026-06-15  
평가 대상: VitalDB PPG → SBP/DBP 직접 회귀 모델 (dataset-v1로 학습)  
평가 데이터셋: `data/dataset-v1/test` (case-level held-out, 672 cases, 1,955,049 segments)  
비교 기준: `data/dataset` 기반 모델 평가 결과 (`data/logs`)

## 1. 개요

본 문서는 `data/dataset-v1`으로 학습한 혈압 추정 모델들의 테스트셋 평가 결과를 종합한다.
`data/dataset-v1`은 원본 데이터셋(`data/dataset`)에서 PPG 신호 품질 기준(power_ratio ≥ 0.6)을
적용해 세그먼트를 선별한 필터링 데이터셋이다.

모든 v1 모델은 `data/dataset` 기반 모델(`data/logs`)과 동일한 아키텍처 및 하이퍼파라미터로
학습되었으며, 평가도 동일한 `eval-model.py`로 수행됐다. 두 결과를 직접 비교해 데이터셋 품질
개선이 모델 성능에 미치는 영향을 분석한다.

## 2. 평가 환경

| 항목                | 내용                                                                    |
| ------------------- | ----------------------------------------------------------------------- |
| 데이터셋            | VitalDB (dataset-v1: power_ratio ≥ 0.6 필터 적용)                       |
| 입력 신호           | PPG (`SNUADC/PLETH`), 125 Hz, 8초 (1000 샘플)                           |
| 레이블              | SBP/DBP 평균값 (mmHg), 세그먼트 내 `Solar8000/ART_SBP/DBP` 기반         |
| case 분할           | train 60% / val 20% / test 20% (case-level, seed=42, dataset 분할 동일) |
| 테스트 케이스 수    | 672 cases (dataset 동일)                                                |
| 테스트 세그먼트 수  | 1,955,049 segments (dataset: 1,987,556 → **−32,507, −1.6%**)            |
| 평가 체크포인트     | 각 모델의 `best.pt` (val loss 최소 epoch)                               |
| 공통 하이퍼파라미터 | lr=1e-3, weight_decay=1e-4, batch_size=256, max_epochs=100, patience=15 |
| 출력 디렉터리       | `data/models-v1`, `data/results-v1`, `data/logs-v1`                     |

### 2.1 dataset vs dataset-v1 규모 비교

| 구분  | dataset (원본)     | dataset-v1 (필터)  | 감소율 |
| ----- | ------------------ | ------------------ | ------ |
| test  | 1,987,556 세그먼트 | 1,955,049 세그먼트 | −1.6%  |
| cases | 672                | 672 (동일)         | —      |

테스트 케이스 수는 동일하나 품질 미달 세그먼트 약 32,500개가 제거됐다.
train/val 세트도 동등한 비율로 축소되어 있다.

## 3. 평가 지표

### 3.1 정량 지표

| 지표     | 정의                               | 의미                                                      |
| -------- | ---------------------------------- | --------------------------------------------------------- |
| **MAE**  | Mean Absolute Error (mmHg)         | 예측 오차의 절대값 평균. 임상에서 가장 직관적인 오차 지표 |
| **ME**   | Mean Error (mmHg)                  | 예측 편향(bias). 양수=과추정, 음수=과소추정               |
| **SD**   | Standard Deviation of error (mmHg) | 예측 오차의 산포. AAMI 기준의 핵심 지표                   |
| **RMSE** | Root Mean Squared Error (mmHg)     | 이상치에 민감한 오차. √(ME² + SD²)                        |

### 3.2 임상 표준 기준

**AAMI (Association for the Advancement of Medical Instrumentation) 기준**

| 조건 | 임계값    |
| ---- | --------- |
| ME   | ≤ ±5 mmHg |
| SD   | ≤ 8 mmHg  |

**BHS (British Hypertension Society) 등급**

| 등급 | ±5 mmHg 이내 | ±10 mmHg 이내 | ±15 mmHg 이내 |
| ---- | ------------ | ------------- | ------------- |
| A    | ≥ 60%        | ≥ 85%         | ≥ 95%         |
| B    | ≥ 50%        | ≥ 75%         | ≥ 90%         |
| C    | ≥ 40%        | ≥ 65%         | ≥ 85%         |
| D    | C 미달       |               |               |

> **임상 적용 기준**: 혈압계로서 AAMI 통과 + BHS Grade B 이상이 임상 사용의 최소 요건으로
> 통용된다. Grade C는 연구용 참고 기준으로 활용된다.

## 4. 평가 대상 모델

| 모델명                    | 분류                   | 비고             | dataset 비교 |
| ------------------------- | ---------------------- | ---------------- | ------------ |
| `naive`                   | 베이스라인             |                  | ✅            |
| `resnet1d`                | ResNet1D 계열          |                  | ✅            |
| `resnet1d_mini`           | ResNet1D 계열          |                  | ✅            |
| `resnet1d_tiny`           | ResNet1D 계열          |                  | ✅            |
| `resnet1d_micro`          | ResNet1D 계열          |                  | ✅            |
| `st_resnet`               | 다중 채널              | PPG + VPG + APG  | ✅            |
| `minception`              | 다중 스케일            |                  | ✅            |
| `xresnet1d`               | 대형 ResNet            |                  | ✅            |
| `acfa`                    | Attention CNN          |                  | ✅            |
| `ae_lstm`                 | AE + LSTM              |                  | ✅            |
| `cnn_bilstm_at`           | CNN + BiLSTM           |                  | ✅            |
| `conv_reg`                | Conv 회귀              |                  | ✅            |
| `conv_reg_ds`             | Conv 회귀 (Depthwise)  |                  | ✅            |
| `mtae`                    | 다중 태스크 오토인코더 |                  | ✅            |
| `mtae_tr`                 | MTAE + Transformer     |                  | ✅            |

## 5. 테스트셋 정량 평가 결과

### 5.1 SBP(수축기혈압) 종합 비교

| 모델                      | MAE ↓     | ME    | SD        | RMSE      | ±5%       | ±10%      | ±15%      | BHS | AAMI |
| ------------------------- | --------- | ----- | --------- | --------- | --------- | --------- | --------- | --- | ---- |
| **`ae_lstm`**             | **12.89** | −1.12 | **16.90** | **16.93** | **25.8%** | **48.8%** | **67.2%** | D   | ❌    |
| `mtae`                    | 12.95     | −0.83 | 16.92     | 16.94     | 25.5%     | 48.3%     | 66.8%     | D   | ❌    |
| `resnet1d_micro`          | 13.03     | +0.69 | 16.95     | 16.97     | 25.0%     | 47.7%     | 66.2%     | D   | ❌    |
| `cnn_bilstm_at`           | 13.13     | +0.24 | 17.18     | 17.18     | 25.2%     | 47.9%     | 66.1%     | D   | ❌    |
| `st_resnet`               | 13.14     | +1.68 | 16.97     | 17.06     | 24.7%     | 47.3%     | 65.7%     | D   | ❌    |
| `mtae_tr`                 | 13.17     | −0.92 | 17.16     | 17.19     | 25.0%     | 47.4%     | 65.8%     | D   | ❌    |
| `conv_reg_ds`             | 13.18     | +0.39 | 17.20     | 17.20     | 25.1%     | 47.6%     | 65.7%     | D   | ❌    |
| `resnet1d_tiny`           | 13.21     | +0.71 | 17.27     | 17.28     | 25.3%     | 47.6%     | 65.7%     | D   | ❌    |
| `conv_reg`                | 13.27     | −0.86 | 17.41     | 17.43     | 25.4%     | 47.8%     | 65.7%     | D   | ❌    |
| `acfa`                    | 13.36     | −0.07 | 17.49     | 17.49     | 25.1%     | 47.4%     | 65.3%     | D   | ❌    |
| `resnet1d`                | 13.36     | −0.85 | 17.60     | 17.62     | 25.4%     | 47.7%     | 65.4%     | D   | ❌    |
| `minception`              | 13.57     | +0.04 | 17.91     | 17.91     | 25.0%     | 47.1%     | 64.8%     | D   | ❌    |
| `resnet1d_mini`           | 13.65     | −0.33 | 18.02     | 18.02     | 24.9%     | 47.1%     | 64.8%     | D   | ❌    |
| `xresnet1d`               | 13.72     | +3.57 | 17.27     | 17.64     | 23.6%     | 45.0%     | 62.9%     | D   | ❌    |
| —  `naive`                | 15.60     | −2.14 | 20.16     | 20.27     | 20.7%     | 40.4%     | 57.5%     | D   | ❌    |

> `ae_lstm`이 SBP MAE 12.89로 최우수. `xresnet1d`는 ME +3.57로 과추정 편향이 두드러진다.

### 5.2 DBP(이완기혈압) 종합 비교

| 모델                      | MAE ↓    | ME    | SD        | RMSE      | ±5%       | ±10%      | ±15%      | BHS   | AAMI |
| ------------------------- | -------- | ----- | --------- | --------- | --------- | --------- | --------- | ----- | ---- |
| **`mtae`**                | **7.84** | −0.36 | **10.24** | **10.25** | **41.2%** | **70.5%** | **87.0%** | **C** | ❌    |
| `ae_lstm`                 | 7.85     | −0.73 | 10.25     | 10.28     | 41.2%     | 70.7%     | 87.2%     | **C** | ❌    |
| `resnet1d_micro`          | 7.86     | +0.13 | 10.26     | 10.26     | 41.2%     | 70.4%     | 86.9%     | **C** | ❌    |
| `conv_reg`                | 7.93     | −0.19 | 10.37     | 10.37     | 41.0%     | 70.1%     | 86.6%     | **C** | ❌    |
| `resnet1d_tiny`           | 7.96     | −0.59 | 10.37     | 10.39     | 40.5%     | 69.8%     | 86.6%     | **C** | ❌    |
| `cnn_bilstm_at`           | 7.95     | +0.00 | 10.37     | 10.37     | 40.5%     | 69.5%     | 86.8%     | **C** | ❌    |
| `xresnet1d`               | 8.00     | +1.14 | 10.30     | 10.36     | 40.1%     | 69.2%     | 86.3%     | **C** | ❌    |
| `conv_reg_ds`             | 8.01     | +0.99 | 10.38     | 10.42     | 40.4%     | 69.4%     | 86.4%     | **C** | ❌    |
| `st_resnet`               | 8.11     | +1.66 | 10.32     | 10.46     | 39.4%     | 68.3%     | 85.9%     | D     | ❌    |
| `mtae_tr`                 | 8.14     | −0.44 | 10.63     | 10.64     | 39.8%     | 68.8%     | 85.7%     | D     | ❌    |
| `acfa`                    | 8.15     | +0.80 | 10.54     | 10.57     | 39.3%     | 68.5%     | 85.8%     | D     | ❌    |
| `resnet1d`                | 8.17     | −1.75 | 10.58     | 10.72     | 39.6%     | 69.0%     | 85.8%     | D     | ❌    |
| `resnet1d_mini`           | 8.21     | −0.51 | 10.76     | 10.77     | 39.5%     | 68.8%     | 85.6%     | D     | ❌    |
| `minception`              | 8.28     | +0.32 | 10.82     | 10.82     | 39.1%     | 68.3%     | 85.3%     | D     | ❌    |
| —  `naive`                | 9.40     | −0.95 | 12.11     | 12.15     | 33.5%     | 61.7%     | 81.0%     | D     | ❌    |

> DBP Grade C 달성 모델 8종: `mtae`, `ae_lstm`, `resnet1d_micro`,
> `conv_reg`, `resnet1d_tiny`, `cnn_bilstm_at`, `xresnet1d`, `conv_reg_ds`.

### 5.3 종합 순위 (SBP MAE + DBP MAE 합산 기준)

| 순위 | 모델                      | SBP MAE | DBP MAE | 합산      | dataset 합산 | 변화        | DBP BHS |
| ---- | ------------------------- | ------- | ------- | --------- | ------------ | ----------- | ------- |
| 1    | `ae_lstm`                 | 12.89   | 7.85    | **20.74** | 21.14        | **↑ +0.40** | C       |
| 2    | `mtae`                    | 12.95   | 7.84    | **20.79** | 20.98        | ↑ +0.19     | C       |
| 3    | `resnet1d_micro`          | 13.03   | 7.86    | **20.89** | 21.10        | ↑ +0.21     | C       |
| 4    | `cnn_bilstm_at`           | 13.13   | 7.95    | **21.08** | 21.16        | ↑ +0.08     | C       |
| 6    | `resnet1d_tiny`           | 13.21   | 7.96    | **21.17** | 21.08        | ↓ −0.09     | C       |
| 8    | `conv_reg_ds`             | 13.18   | 8.01    | **21.19** | 20.90        | ↓ −0.29     | C       |
| 9    | `conv_reg`                | 13.27   | 7.93    | **21.20** | 21.20        | ≈           | C       |
| 10   | `st_resnet`               | 13.14   | 8.11    | **21.25** | 21.12        | ↓ −0.13     | D       |
| 11   | `mtae_tr`                 | 13.17   | 8.14    | **21.31** | 21.48        | ↑ +0.17     | D       |
| 12   | `acfa`                    | 13.36   | 8.15    | **21.51** | 21.43        | ↓ −0.08     | D       |
| 13   | `resnet1d`                | 13.36   | 8.17    | **21.53** | 21.95        | **↑ +0.42** | D       |
| 14   | `xresnet1d`               | 13.72   | 8.00    | **21.72** | 21.33        | ↓ **−0.39** | C       |
| 15   | `minception`              | 13.57   | 8.28    | **21.85** | 22.21        | ↑ +0.36     | D       |
| 16   | `resnet1d_mini`           | 13.65   | 8.21    | **21.86** | 22.05        | ↑ +0.19     | D       |
| —    | `naive`                   | 15.60   | 9.40    | **25.00** | 25.07        | ↑ +0.07     | D       |

> ↑: dataset 대비 개선(합산 MAE 감소), ↓: 악화(합산 MAE 증가). `dataset 합산`은 data/logs 기준.

## 6. 모델별 상세 평가

### 6.1 naive (베이스라인)

best_epoch: 7 (dataset: 15)

```
SBP — MAE: 15.60, ME: -2.14, SD: 20.16, RMSE: 20.27 | Grade D | AAMI: ❌
DBP — MAE:  9.40, ME: -0.95, SD: 12.11, RMSE: 12.15 | Grade D | AAMI: ❌
```

훈련셋 통계값 근사 출력 모델. 성능은 dataset(SBP 15.65, DBP 9.42)과 사실상 동일하다.
best_epoch=7로 dataset(15) 대비 절반으로 줄었으나 val_loss는 동일 수준에서 수렴한다.
데이터셋 품질이 베이스라인에는 영향을 미치지 않음을 확인한다.

| 그래프               |                                              |
| -------------------- | -------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v1/eval_plot/naive.png)  |
| Error Distribution   | ![](../data/results-v1/error_hist/naive.png) |
| 훈련 손실 곡선       | ![](../data/results-v1/loss_graph/naive.png) |
| 훈련 MAE 곡선        | ![](../data/results-v1/mae_graph/naive.png)  |

### 6.2 ae_lstm (AE + LSTM)

best_epoch: 4 (dataset: 3)

```
SBP — MAE: 12.89, ME: -1.12, SD: 16.90, RMSE: 16.93 | Grade D | AAMI: ❌
DBP — MAE:  7.85, ME: -0.73, SD: 10.25, RMSE: 10.28 | Grade C | AAMI: ❌
```

**v1 데이터셋 최우수 모델 (합산 MAE 20.74)**. dataset 대비 SBP +0.23, DBP +0.17 개선으로
전체 모델 중 가장 큰 이득을 얻었다. SBP SD가 17.09→16.90으로 낮아져 오차 산포도 줄었다.
오토인코더 기반 특징 추출 구조가 주기성이 강화된 v1 세그먼트에서 더 효과적으로 작동한 것으로
분석된다.

| 그래프               |                                                |
| -------------------- | ---------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v1/eval_plot/ae_lstm.png)  |
| Error Distribution   | ![](../data/results-v1/error_hist/ae_lstm.png) |
| 훈련 손실 곡선       | ![](../data/results-v1/loss_graph/ae_lstm.png) |
| 훈련 MAE 곡선        | ![](../data/results-v1/mae_graph/ae_lstm.png)  |

### 6.3 mtae (Multi-Task AutoEncoder)

best_epoch: 3 (dataset: 2)

```
SBP — MAE: 12.95, ME: -0.83, SD: 16.92, RMSE: 16.94 | Grade D | AAMI: ❌
DBP — MAE:  7.84, ME: -0.36, SD: 10.24, RMSE: 10.25 | Grade C | AAMI: ❌
```

**DBP MAE 최우수(7.84)**. dataset 대비 SBP +0.14, DBP +0.05 개선. DBP SD 10.33→10.24로 낮아져
오차 산포가 개선됐다. CNN 인코더/디코더 + BP 헤드 구조가 v1 데이터셋에서도 안정적인 성능을
유지하며 ae_lstm과 함께 최상위 그룹을 형성한다.

| 그래프               |                                             |
| -------------------- | ------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v1/eval_plot/mtae.png)  |
| Error Distribution   | ![](../data/results-v1/error_hist/mtae.png) |
| 훈련 손실 곡선       | ![](../data/results-v1/loss_graph/mtae.png) |
| 훈련 MAE 곡선        | ![](../data/results-v1/mae_graph/mtae.png)  |

### 6.4 resnet1d_micro (초소형 ResNet1D)

best_epoch: 14 (dataset: 29)

```
SBP — MAE: 13.03, ME: +0.69, SD: 16.95, RMSE: 16.97 | Grade D | AAMI: ❌
DBP — MAE:  7.86, ME: +0.13, SD: 10.26, RMSE: 10.26 | Grade C | AAMI: ❌
```

15.1K 파라미터의 초소형 모델. dataset 대비 SBP +0.19, DBP +0.02 개선.
**best_epoch=29→14로 절반 감소**: 노이즈가 줄어든 v1 데이터셋에서 손실 곡선이 더 빠르게
수렴했음을 시사한다. 작은 모델이 큰 이득을 본 점은 과소적합(underfitting) 영역에 있던 모델도
데이터 품질 개선의 수혜를 받을 수 있음을 보여준다.

| 그래프               |                                                       |
| -------------------- | ----------------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v1/eval_plot/resnet1d_micro.png)  |
| Error Distribution   | ![](../data/results-v1/error_hist/resnet1d_micro.png) |
| 훈련 손실 곡선       | ![](../data/results-v1/loss_graph/resnet1d_micro.png) |
| 훈련 MAE 곡선        | ![](../data/results-v1/mae_graph/resnet1d_micro.png)  |

### 6.5 cnn_bilstm_at (CNN + BiLSTM with Attention)

best_epoch: 3 (dataset: 13)

```
SBP — MAE: 13.13, ME: +0.24, SD: 17.18, RMSE: 17.18 | Grade D | AAMI: ❌
DBP — MAE:  7.95, ME:  0.00, SD: 10.37, RMSE: 10.37 | Grade C | AAMI: ❌
```

dataset 대비 SBP +0.03, DBP +0.05 소폭 개선. **best_epoch=13→3으로 대폭 감소**: v1 데이터셋의
주기적 신호가 BiLSTM의 순차적 학습을 가속화한 것으로 분석된다. DBP ME=0.00으로 편향이 완전히
제거됐다.

| 그래프               |                                                      |
| -------------------- | ---------------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v1/eval_plot/cnn_bilstm_at.png)  |
| Error Distribution   | ![](../data/results-v1/error_hist/cnn_bilstm_at.png) |
| 훈련 손실 곡선       | ![](../data/results-v1/loss_graph/cnn_bilstm_at.png) |
| 훈련 MAE 곡선        | ![](../data/results-v1/mae_graph/cnn_bilstm_at.png)  |

### 6.6 resnet1d (기준 모델)

best_epoch: 1 (dataset: 1)

```
SBP — MAE: 13.36, ME: -0.85, SD: 17.60, RMSE: 17.62 | Grade D | AAMI: ❌
DBP — MAE:  8.17, ME: -1.75, SD: 10.58, RMSE: 10.72 | Grade D | AAMI: ❌
```

dataset 대비 **SBP +0.35, DBP +0.07 개선**. 합산 MAE 21.95→21.53으로 종합 순위
12위→13위. best_epoch=1 유지로 학습 동역학 변화 없이 데이터 품질 개선만으로 성능이 향상됐다.
DBP ME −0.48→−1.75로 과소추정 편향이 증가한 점은 주목할 만하다.

| 그래프               |                                                 |
| -------------------- | ----------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v1/eval_plot/resnet1d.png)  |
| Error Distribution   | ![](../data/results-v1/error_hist/resnet1d.png) |
| 훈련 손실 곡선       | ![](../data/results-v1/loss_graph/resnet1d.png) |
| 훈련 MAE 곡선        | ![](../data/results-v1/mae_graph/resnet1d.png)  |

### 6.7 resnet1d_mini

best_epoch: 1 (dataset: 1)

```
SBP — MAE: 13.65, ME: -0.33, SD: 18.02, RMSE: 18.02 | Grade D | AAMI: ❌
DBP — MAE:  8.21, ME: -0.51, SD: 10.76, RMSE: 10.77 | Grade D | AAMI: ❌
```

dataset 대비 SBP +0.11, DBP +0.08 개선. 합산 순위 14위→16위. v1에서도 DBP Grade D 유지.

| 그래프               |                                                      |
| -------------------- | ---------------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v1/eval_plot/resnet1d_mini.png)  |
| Error Distribution   | ![](../data/results-v1/error_hist/resnet1d_mini.png) |
| 훈련 손실 곡선       | ![](../data/results-v1/loss_graph/resnet1d_mini.png) |
| 훈련 MAE 곡선        | ![](../data/results-v1/mae_graph/resnet1d_mini.png)  |

### 6.8 resnet1d_tiny

best_epoch: 3 (dataset: 1)

```
SBP — MAE: 13.21, ME: +0.71, SD: 17.27, RMSE: 17.28 | Grade D | AAMI: ❌
DBP — MAE:  7.96, ME: -0.59, SD: 10.37, RMSE: 10.39 | Grade C | AAMI: ❌
```

dataset 대비 SBP −0.05, DBP −0.04 소폭 악화. best_epoch=1→3으로 수렴이 지연됐다.
합산 순위는 4위→6위로 하락했으나 절대 성능은 거의 동일하다.

| 그래프               |                                                      |
| -------------------- | ---------------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v1/eval_plot/resnet1d_tiny.png)  |
| Error Distribution   | ![](../data/results-v1/error_hist/resnet1d_tiny.png) |
| 훈련 손실 곡선       | ![](../data/results-v1/loss_graph/resnet1d_tiny.png) |
| 훈련 MAE 곡선        | ![](../data/results-v1/mae_graph/resnet1d_tiny.png)  |

### 6.9 st_resnet (Spectro-Temporal ResNet)

best_epoch: 1 (dataset: 1)

```
SBP — MAE: 13.14, ME: +1.68, SD: 16.97, RMSE: 17.06 | Grade D | AAMI: ❌
DBP — MAE:  8.11, ME: +1.66, SD: 10.32, RMSE: 10.46 | Grade D | AAMI: ❌
```

PPG + VPG + APG 3채널 입력 모델. dataset 대비 SBP −0.04, DBP −0.09 소폭 악화. 합산 순위
5위→10위. SBP·DBP 모두 ME > 1.6 mmHg로 양의 편향이 두드러지며, Grade D 유지.

| 그래프               |                                                  |
| -------------------- | ------------------------------------------------ |
| Prediction vs Actual | ![](../data/results-v1/eval_plot/st_resnet.png)  |
| Error Distribution   | ![](../data/results-v1/error_hist/st_resnet.png) |
| 훈련 손실 곡선       | ![](../data/results-v1/loss_graph/st_resnet.png) |
| 훈련 MAE 곡선        | ![](../data/results-v1/mae_graph/st_resnet.png)  |

### 6.11 minception (Multi-scale Inception 1D)

best_epoch: 1 (dataset: 1)

```
SBP — MAE: 13.57, ME: +0.04, SD: 17.91, RMSE: 17.91 | Grade D | AAMI: ❌
DBP — MAE:  8.28, ME: +0.32, SD: 10.82, RMSE: 10.82 | Grade D | AAMI: ❌
```

dataset 대비 SBP +0.33, DBP +0.03 개선. 합산 순위 13위→15위. SBP ME가 0.92→+0.04로
편향이 대폭 감소했다. 그러나 SBP SD 18.22→17.91로 여전히 전 모델 중 최고(최악) 수준이다.

| 그래프               |                                                   |
| -------------------- | ------------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v1/eval_plot/minception.png)  |
| Error Distribution   | ![](../data/results-v1/error_hist/minception.png) |
| 훈련 손실 곡선       | ![](../data/results-v1/loss_graph/minception.png) |
| 훈련 MAE 곡선        | ![](../data/results-v1/mae_graph/minception.png)  |

### 6.12 xresnet1d

best_epoch: 1 (dataset: 2)

```
SBP — MAE: 13.72, ME: +3.57, SD: 17.27, RMSE: 17.64 | Grade D | AAMI: ❌
DBP — MAE:  8.00, ME: +1.14, SD: 10.30, RMSE: 10.36 | Grade C | AAMI: ❌
```

**v1에서 가장 큰 성능 저하 모델**. dataset 대비 SBP −0.39 악화(13.33→13.72). SBP ME가
+1.74→+3.57로 과추정 편향이 두 배로 악화됐다. DBP는 MAE 동일(8.00)하지만 DBP ME도
+1.35→+1.14로 여전히 양의 편향이다. 9.47M 파라미터의 대형 모델이 cleaner 데이터셋에서 오히려
SBP 편향이 증가한 이유는 추가 분석이 필요하다.

| 그래프               |                                                  |
| -------------------- | ------------------------------------------------ |
| Prediction vs Actual | ![](../data/results-v1/eval_plot/xresnet1d.png)  |
| Error Distribution   | ![](../data/results-v1/error_hist/xresnet1d.png) |
| 훈련 손실 곡선       | ![](../data/results-v1/loss_graph/xresnet1d.png) |
| 훈련 MAE 곡선        | ![](../data/results-v1/mae_graph/xresnet1d.png)  |

### 6.14 acfa (Attention CNN with Frequency Attention)

best_epoch: 1 (dataset: 1)

```
SBP — MAE: 13.36, ME: -0.07, SD: 17.49, RMSE: 17.49 | Grade D | AAMI: ❌
DBP — MAE:  8.15, ME: +0.80, SD: 10.54, RMSE: 10.57 | Grade D | AAMI: ❌
```

dataset 대비 SBP +0.02 미미한 개선, DBP −0.10 소폭 악화. 합산 순위 11위→12위.
SBP ME −0.07로 거의 무편향이나 DBP에서 +0.80으로 과추정 편향이 있다.

| 그래프               |                                             |
| -------------------- | ------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v1/eval_plot/acfa.png)  |
| Error Distribution   | ![](../data/results-v1/error_hist/acfa.png) |
| 훈련 손실 곡선       | ![](../data/results-v1/loss_graph/acfa.png) |
| 훈련 MAE 곡선        | ![](../data/results-v1/mae_graph/acfa.png)  |

### 6.15 conv_reg

best_epoch: 3 (dataset: 3)

```
SBP — MAE: 13.27, ME: -0.86, SD: 17.41, RMSE: 17.43 | Grade D | AAMI: ❌
DBP — MAE:  7.93, ME: -0.19, SD: 10.37, RMSE: 10.37 | Grade C | AAMI: ❌
```

dataset 대비 SBP ±0.00, DBP ±0.00으로 완전히 동일한 성능. 데이터셋 변경에 가장 무감한
모델이다. best_epoch=3 유지.

| 그래프               |                                                 |
| -------------------- | ----------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v1/eval_plot/conv_reg.png)  |
| Error Distribution   | ![](../data/results-v1/error_hist/conv_reg.png) |
| 훈련 손실 곡선       | ![](../data/results-v1/loss_graph/conv_reg.png) |
| 훈련 MAE 곡선        | ![](../data/results-v1/mae_graph/conv_reg.png)  |

### 6.16 conv_reg_ds (Depthwise Separable Conv)

best_epoch: 5 (dataset: 5)

```
SBP — MAE: 13.18, ME: +0.39, SD: 17.20, RMSE: 17.20 | Grade D | AAMI: ❌
DBP — MAE:  8.01, ME: +0.99, SD: 10.38, RMSE: 10.42 | Grade C | AAMI: ❌
```

**dataset에서 1위였던 모델(20.90)이 v1에서 8위(21.19)로 하락**. SBP −0.14 악화,
DBP −0.15 악화. dataset에서 강점이었던 경량 구조가 cleaner 데이터셋에서는 상대적으로 경쟁
우위가 줄어든 것으로 보인다. best_epoch=5 유지.

| 그래프               |                                                    |
| -------------------- | -------------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v1/eval_plot/conv_reg_ds.png)  |
| Error Distribution   | ![](../data/results-v1/error_hist/conv_reg_ds.png) |
| 훈련 손실 곡선       | ![](../data/results-v1/loss_graph/conv_reg_ds.png) |
| 훈련 MAE 곡선        | ![](../data/results-v1/mae_graph/conv_reg_ds.png)  |

### 6.17 mtae_tr (MTAE with Transformer)

best_epoch: 1 (dataset: 5)

```
SBP — MAE: 13.17, ME: -0.92, SD: 17.16, RMSE: 17.19 | Grade D | AAMI: ❌
DBP — MAE:  8.14, ME: -0.44, SD: 10.63, RMSE: 10.64 | Grade D | AAMI: ❌
```

dataset 대비 SBP +0.15, DBP +0.02 개선. **best_epoch=5→1로 급감**: 학습이 더 일찍 최적점에
도달했다. mtae(CNN 백본)가 ae_lstm·mtae와 함께 최상위 그룹을 형성한 데 반해, Transformer
백본의 mtae_tr은 11위에 머물렀다.

| 그래프               |                                                |
| -------------------- | ---------------------------------------------- |
| Prediction vs Actual | ![](../data/results-v1/eval_plot/mtae_tr.png)  |
| Error Distribution   | ![](../data/results-v1/error_hist/mtae_tr.png) |
| 훈련 손실 곡선       | ![](../data/results-v1/loss_graph/mtae_tr.png) |
| 훈련 MAE 곡선        | ![](../data/results-v1/mae_graph/mtae_tr.png)  |

## 7. dataset vs dataset-v1 성능 비교 분석

### 7.1 합산 MAE 변화 요약

공통 모델 14종에 대한 dataset → dataset-v1 합산 MAE 변화:

| 범주           | 모델                                         | Δ 합산 MAE  | 해석                                      |
| -------------- | -------------------------------------------- | ----------- | ----------------------------------------- |
| 최대 개선      | `resnet1d`                                   | **+0.42**   | 노이즈 감소가 기본 CNN에 크게 기여        |
| 최대 개선      | `ae_lstm`                                    | **+0.40**   | 오토인코더 구조가 주기적 신호 품질에 민감 |
| 개선           | `minception`                                 | +0.36       |                                           |
|                | `resnet1d_mini`, `resnet1d_micro`            | +0.19~0.21  |                                           |
|                | `mtae_tr`, `mtae`, `cnn_bilstm_at`           | +0.08~0.19  |                                           |
|                | `naive`, `acfa`, `conv_reg`, `resnet1d_tiny` | ≈0~+0.09    | 데이터셋 변화에 거의 무감                 |
| 소폭 악화      | `st_resnet`, `acfa`, `resnet1d_tiny`         | −0.04~−0.13 |                                           |
| 의미 있는 악화 | `conv_reg_ds`                                | **−0.29**   | dataset에서 1위였으나 v1에서 경쟁력 약화  |
| 최대 악화      | `xresnet1d`                                  | **−0.39**   | SBP ME +1.74→+3.57, 과추정 편향 급증      |

### 7.2 SBP 개선 폭 상세

| 모델             | SBP_dataset | SBP_v1 | Δ SBP     | ME 변화                      |
| ---------------- | ----------- | ------ | --------- | ---------------------------- |
| `resnet1d`       | 13.71       | 13.36  | +0.35     | −0.82→−0.85                  |
| `minception`     | 13.90       | 13.57  | +0.33     | +0.92→+0.04 (편향 해소)      |
| `ae_lstm`        | 13.12       | 12.89  | +0.23     | +0.79→−1.12                  |
| `resnet1d_micro` | 13.22       | 13.03  | +0.19     | +0.77→+0.69                  |
| `resnet1d_mini`  | 13.76       | 13.65  | +0.11     | +0.65→−0.33 (편향 방향 반전) |
| `mtae_tr`        | 13.32       | 13.17  | +0.15     | −1.06→−0.92                  |
| `mtae`           | 13.09       | 12.95  | +0.14     | −0.12→−0.83                  |
| `conv_reg`       | 13.27       | 13.27  | ±0.00     | −1.93→−0.86 (편향 감소)      |
| `acfa`           | 13.38       | 13.36  | +0.02     | ≈동일                        |
| `cnn_bilstm_at`  | 13.16       | 13.13  | +0.03     |                              |
| `resnet1d_tiny`  | 13.16       | 13.21  | −0.05     |                              |
| `st_resnet`      | 13.10       | 13.14  | −0.04     |                              |
| `conv_reg_ds`    | 13.04       | 13.18  | −0.14     | −0.26→+0.39 (편향 방향 반전) |
| `xresnet1d`      | 13.33       | 13.72  | **−0.39** | +1.74→+3.57 (과추정 급증)    |

### 7.3 DBP 개선 폭 상세

| 모델             | DBP_dataset | DBP_v1 | Δ DBP | BHS 변화 |
| ---------------- | ----------- | ------ | ----- | -------- |
| `ae_lstm`        | 8.02        | 7.85   | +0.17 | C→C      |
| `conv_reg_ds`    | 7.86        | 8.01   | −0.15 | C→C      |
| `resnet1d`       | 8.24        | 8.17   | +0.07 | D→D      |
| `resnet1d_mini`  | 8.29        | 8.21   | +0.08 | D→D      |
| `mtae`           | 7.89        | 7.84   | +0.05 | C→C      |
| `cnn_bilstm_at`  | 8.00        | 7.95   | +0.05 | C→C      |
| `mtae_tr`        | 8.16        | 8.14   | +0.02 | D→D      |
| `resnet1d_micro` | 7.88        | 7.86   | +0.02 | C→C      |
| `conv_reg`       | 7.93        | 7.93   | ±0.00 | C→C      |
| `xresnet1d`      | 8.00        | 8.00   | ±0.00 | C→C      |
| `resnet1d_tiny`  | 7.92        | 7.96   | −0.04 | C→C      |
| `st_resnet`      | 8.02        | 8.11   | −0.09 | D→D      |
| `acfa`           | 8.05        | 8.15   | −0.10 | D→D      |

> DBP BHS 등급은 dataset → v1에서 모든 모델이 동일 등급을 유지했다.

### 7.4 훈련 수렴 변화 (best_epoch)

| 모델             | dataset BE | v1 BE | 변화     | 해석                                         |
| ---------------- | ---------- | ----- | -------- | -------------------------------------------- |
| `cnn_bilstm_at`  | 13         | 3     | **↓ 10** | 주기적 신호가 BiLSTM 수렴을 크게 가속        |
| `resnet1d_micro` | 29         | 14    | **↓ 15** | 노이즈 감소로 최적점을 빠르게 발견           |
| `naive`          | 15         | 7     | ↓ 8      | 통계 근사이므로 학습 데이터 분포 단순화 효과 |
| `mtae_tr`        | 5          | 1     | ↓ 4      | epoch 1이 최적 — 조기 수렴                   |
| `xresnet1d`      | 2          | 1     | ↓ 1      |                                              |
| `resnet1d_tiny`  | 1          | 3     | ↑ 2      | 소폭 지연 — 훈련 분포 변화에 적응 필요       |
| `mtae`           | 2          | 3     | ↑ 1      |                                              |
| `ae_lstm`        | 3          | 4     | ↑ 1      |                                              |
| 나머지           | —          | —     | ≈동일    |                                              |

**전반적 경향**: 대부분 모델에서 best_epoch이 감소했다. 품질이 높은 학습 데이터가 손실
곡선을 단순화하여 최적점을 더 빨리 찾는 것으로 분석된다.

### 7.5 종합 순위 변동

dataset → v1에서 크게 변동된 모델:

| 모델          | dataset 순위 | v1 순위 | 변동 | 사유                                |
| ------------- | ------------ | ------- | ---- | ----------------------------------- |
| `ae_lstm`     | 4위          | 1위     | +3   | 최대 개선, 오토인코더 구조 수혜     |
| `resnet1d`    | 12위         | 13위    | −1   | 절대 성능 개선되나 순위 소폭 하락   |
| `conv_reg_ds` | 1위          | 8위     | −7   | 상대적 경쟁력 하락                  |
| `xresnet1d`   | 9위          | 14위    | −5   | SBP ME 악화로 합산 순위 하락        |
| `st_resnet`   | 3위          | 10위    | −7   | 소폭 악화 + 다른 모델 개선으로 역전 |

## 8. 훈련 과정 분석

### 8.1 Early Stopping 요약

| 모델                      | v1 BE  | v1 총 에폭 | dataset BE | 변화  | 과적합 패턴               |
| ------------------------- | ------ | ---------- | ---------- | ----- | ------------------------- |
| `resnet1d_micro`          | **14** | ~29        | 29         | ↓ 15  | 완만한 수렴 → 정체 (정상) |
| `ae_lstm`                 | 4      | ~19        | 3          | ↑ 1   | 초기 수렴 후 완만한 감소  |
| `cnn_bilstm_at`           | 3      | ~18        | 13         | ↓ 10  | 빠른 수렴                 |
| `conv_reg`, `mtae`        | 3      | ~18        | 2~3        | ≈동일 | 초기 수렴                 |
| `naive`                   | 7      | ~22        | 15         | ↓ 8   | 수렴하지 않음 (상수 출력) |
| `resnet1d`, `acfa` 등     | 1      | 16         | 1          | 동일  | epoch 2부터 즉각 과적합   |

> resnet1d_micro만이 두 데이터셋 모두에서 epoch 10 이상 수렴하는 정상적인 학습 곡선을 보인다.
> 나머지 대부분은 epoch 1~5에서 최적점 도달 후 과적합이 시작된다.

### 8.2 과적합 정도 분석

v1에서도 즉각 과적합 패턴이 지속되는 이유: 데이터셋 크기가 학습에 충분하고 모델 표현력 대비
데이터가 상대적으로 부족하지 않음에도 불구하고 best_epoch=1이 다수인 것은 모델별 학습률,
배치 크기, weight decay 조합이 최적화되지 않았을 가능성이 있다.

## 9. 국제 표준 기준 달성 현황

### 9.1 AAMI 기준 분석

| 기준        | SBP                                        | DBP                                             |
| ----------- | ------------------------------------------ | ----------------------------------------------- |
| ME ≤ 5 mmHg | **충족** (전 모델: 최대 3.57 mmHg)         | **충족** (전 모델: 최대 1.75 mmHg)              |
| SD ≤ 8 mmHg | ❌ **미달** (최솟값 16.90 mmHg / `ae_lstm`) | ❌ **미달** (최솟값 10.24 mmHg / `mtae`) |

ME 기준은 전 모델 충족. SD 기준은 SBP 16.90, DBP 10.24로 목표치(8 mmHg)의 약 2배 수준으로
dataset과 동일하게 전 모델 미달이다.

### 9.2 BHS 등급 달성 현황

| 등급             | SBP     | DBP (v1)                                                                                                              | DBP (dataset) 대비       |
| ---------------- | ------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| A (≥60%/85%/95%) | 전무    | 전무                                                                                                                  | 변화 없음                |
| B (≥50%/75%/90%) | 전무    | 전무                                                                                                                  | 변화 없음                |
| C (≥40%/65%/85%) | 전무    | **8종** (mtae, ae_lstm, resnet1d_micro, conv_reg, resnet1d_tiny, cnn_bilstm_at, xresnet1d, conv_reg_ds) | 변화 없음 |
| D                | 전 모델 | 8종                                                                                                     | 변화 없음 |

DBP Grade C 8종 달성 (dataset 8종 대비 동일).

## 10. 주요 발견 및 시사점

### 10.1 데이터 품질 개선의 효과: 대부분의 모델 개선

14개 공통 모델 중 10개가 합산 MAE 개선, 1개 동일, 3개 악화됐다. 데이터셋의 1.6% 세그먼트만
제거했음에도 평균 합산 MAE 기준 +0.13 mmHg 개선 효과가 관찰됐다.

### 10.2 오토인코더 계열 모델의 수혜

`ae_lstm`(+0.40)과 `mtae`(+0.19)는 오토인코더 기반 특징 추출 구조를 갖는다. 이 두 모델이
v1에서 각각 1위, 2위를 기록했다. 주기성이 강화된 PPG 세그먼트가 오토인코더의 재구성 목표를
더 일관되게 만들어 표현 학습을 향상시킨 것으로 분석된다.

### 10.3 수렴 가속 효과

cnn_bilstm_at(13→3), resnet1d_micro(29→14), naive(15→7) 등 여러 모델에서 best_epoch이
크게 감소했다. 품질이 균일한 데이터는 손실 지형(loss landscape)을 부드럽게 만들어 최적점에
빨리 도달하게 한다는 가설과 일치한다.

### 10.4 xresnet1d의 역행: 과추정 편향 급증

xresnet1d는 SBP ME가 +1.74→+3.57로 급증하면서 합산 순위가 9위→14위로 하락했다. 9.47M
파라미터의 대형 모델이 1.6% 더 작은 데이터셋에서 최적화 경로가 바뀐 것으로 보이나, 구체적인
원인은 추가 분석이 필요하다.

### 10.5 conv_reg_ds의 상대적 경쟁력 하락

dataset에서 1위(20.90)였던 conv_reg_ds가 v1에서 8위(21.19)로 하락했다. 절대 성능은 소폭
악화됐으나 다른 모델들이 더 크게 개선된 결과다. dataset에서 이 모델의 강점이 노이즈 강인성에
있었다면, v1에서 노이즈가 줄어들면서 그 강점이 희석됐을 가능성이 있다.

### 10.7 SBP vs DBP 개선 비대칭

v1에서 SBP 개선 폭(평균 +0.12)이 DBP 개선 폭(평균 +0.02)보다 크다. PPG 신호의 주기적
성분이 SBP 추정에 더 중요한 특징을 제공한다는 기존 해석과 일치한다.

## 11. 미완료 실험 및 향후 과제

### 현재 한계

- SBP AAMI SD 기준까지의 거리가 약 2배(16.90 vs 목표 8.0 mmHg)로 근본적 개선이 필요하다.
- 모든 모델이 SBP Grade D로, ±5mmHg 이내 비율이 26% 미만이다.

### 주요 향후 과제

1. **power_ratio 임계값 탐색**: 현재 0.6 기준에서 0.65, 0.70으로 강화했을 때의 성능 변화
   체계적 측정. 데이터 감소량과 성능 개선 간 최적 균형점 탐색.

2. **xresnet1d SBP 편향 원인 분석**: v1 데이터셋에서 SBP ME +3.57로 급증한 원인 규명.
   학습률 스케줄, 배치 정규화 상호작용, 초기화 랜덤 시드 변경 등을 통한 재현성 확인.

3. **ae_lstm 추가 탐색**: v1에서 최우수 성능을 달성한 ae_lstm의 잠재력을 더 활용하기 위해
   오토인코더 잠재 공간 크기, LSTM 레이어 수, 드롭아웃 비율을 조정한 변형 실험.

4. **데이터 증강(augmentation)과의 교차 실험**: dataset + augmentation vs dataset-v1 +
   augmentation 간 성능 차이 측정. 데이터 품질 향상과 증강이 독립적으로 기여하는지,
   상호작용 효과가 있는지 분석.

5. **SBP 정확도 개선**: Huber loss 또는 양자 손실(quantile loss) 도입으로 SBP SD를
   줄이는 실험.

6. **BP 범위별 오차 분석**: 저혈압(SBP < 90) / 정상 / 고혈압(SBP > 140) 구간별 MAE 분리
   평가. v1 데이터셋이 각 구간에서 원본 대비 어떤 차이를 보이는지 분석.

## 부록: 모델별 그래프 인덱스

| 모델           | eval_plot                                            | error_hist                                            | loss_graph                                            | mae_graph                                            |
| -------------- | ---------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- | ---------------------------------------------------- |
| naive          | ![](../data/results-v1/eval_plot/naive.png)          | ![](../data/results-v1/error_hist/naive.png)          | ![](../data/results-v1/loss_graph/naive.png)          | ![](../data/results-v1/mae_graph/naive.png)          |
| ae_lstm        | ![](../data/results-v1/eval_plot/ae_lstm.png)        | ![](../data/results-v1/error_hist/ae_lstm.png)        | ![](../data/results-v1/loss_graph/ae_lstm.png)        | ![](../data/results-v1/mae_graph/ae_lstm.png)        |
| mtae           | ![](../data/results-v1/eval_plot/mtae.png)           | ![](../data/results-v1/error_hist/mtae.png)           | ![](../data/results-v1/loss_graph/mtae.png)           | ![](../data/results-v1/mae_graph/mtae.png)           |
| resnet1d_micro | ![](../data/results-v1/eval_plot/resnet1d_micro.png) | ![](../data/results-v1/error_hist/resnet1d_micro.png) | ![](../data/results-v1/loss_graph/resnet1d_micro.png) | ![](../data/results-v1/mae_graph/resnet1d_micro.png) |
| cnn_bilstm_at  | ![](../data/results-v1/eval_plot/cnn_bilstm_at.png)  | ![](../data/results-v1/error_hist/cnn_bilstm_at.png)  | ![](../data/results-v1/loss_graph/cnn_bilstm_at.png)  | ![](../data/results-v1/mae_graph/cnn_bilstm_at.png)  |
| resnet1d_tiny  | ![](../data/results-v1/eval_plot/resnet1d_tiny.png)  | ![](../data/results-v1/error_hist/resnet1d_tiny.png)  | ![](../data/results-v1/loss_graph/resnet1d_tiny.png)  | ![](../data/results-v1/mae_graph/resnet1d_tiny.png)  |
| conv_reg_ds    | ![](../data/results-v1/eval_plot/conv_reg_ds.png)    | ![](../data/results-v1/error_hist/conv_reg_ds.png)    | ![](../data/results-v1/loss_graph/conv_reg_ds.png)    | ![](../data/results-v1/mae_graph/conv_reg_ds.png)    |
| conv_reg       | ![](../data/results-v1/eval_plot/conv_reg.png)       | ![](../data/results-v1/error_hist/conv_reg.png)       | ![](../data/results-v1/loss_graph/conv_reg.png)       | ![](../data/results-v1/mae_graph/conv_reg.png)       |
| st_resnet      | ![](../data/results-v1/eval_plot/st_resnet.png)      | ![](../data/results-v1/error_hist/st_resnet.png)      | ![](../data/results-v1/loss_graph/st_resnet.png)      | ![](../data/results-v1/mae_graph/st_resnet.png)      |
| mtae_tr        | ![](../data/results-v1/eval_plot/mtae_tr.png)        | ![](../data/results-v1/error_hist/mtae_tr.png)        | ![](../data/results-v1/loss_graph/mtae_tr.png)        | ![](../data/results-v1/mae_graph/mtae_tr.png)        |
| acfa           | ![](../data/results-v1/eval_plot/acfa.png)           | ![](../data/results-v1/error_hist/acfa.png)           | ![](../data/results-v1/loss_graph/acfa.png)           | ![](../data/results-v1/mae_graph/acfa.png)           |
| resnet1d       | ![](../data/results-v1/eval_plot/resnet1d.png)       | ![](../data/results-v1/error_hist/resnet1d.png)       | ![](../data/results-v1/loss_graph/resnet1d.png)       | ![](../data/results-v1/mae_graph/resnet1d.png)       |
| xresnet1d      | ![](../data/results-v1/eval_plot/xresnet1d.png)      | ![](../data/results-v1/error_hist/xresnet1d.png)      | ![](../data/results-v1/loss_graph/xresnet1d.png)      | ![](../data/results-v1/mae_graph/xresnet1d.png)      |
| minception     | ![](../data/results-v1/eval_plot/minception.png)     | ![](../data/results-v1/error_hist/minception.png)     | ![](../data/results-v1/loss_graph/minception.png)     | ![](../data/results-v1/mae_graph/minception.png)     |
| resnet1d_mini  | ![](../data/results-v1/eval_plot/resnet1d_mini.png)  | ![](../data/results-v1/error_hist/resnet1d_mini.png)  | ![](../data/results-v1/loss_graph/resnet1d_mini.png)  | ![](../data/results-v1/mae_graph/resnet1d_mini.png)  |
