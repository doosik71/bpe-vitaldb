# 모델 평가 결과 (Evaluation Results — Data Augmentation 적용)

작성일: 2026-06-08  
평가 대상: VitalDB PPG → SBP/DBP 직접 회귀 모델 (Data Augmentation 적용 재학습)  
평가 데이터셋: `data/dataset/test` (case-level held-out, 672 cases, 1,987,561 segments)

## 1. 개요

본 문서는 `bpe-vitaldb` 프로젝트에서 Data Augmentation을 적용하여 재학습한 혈압 추정
모델들의 테스트셋 평가 결과를 종합적으로 정리한다. 모든 모델은 augmentation 미적용
결과(`evaluation_result_no_aug.md`)와 동일한 테스트셋·평가 방식으로 비교된다.

평가는 `scripts/eval-model.py`를 사용하며 `best.pt` 체크포인트(validation loss 기준)를
테스트셋에 단 1회 적용하는 방식으로 수행된다.

## 2. 평가 환경

| 항목                | 내용                                                                    |
| ------------------- | ----------------------------------------------------------------------- |
| 데이터셋            | VitalDB (수술 환자 6,388 케이스 중 PPG+ABP 보유 케이스)                 |
| 입력 신호           | PPG (`SNUADC/PLETH`), 125 Hz, 8초 (1000 샘플)                           |
| 레이블              | SBP/DBP 평균값 (mmHg), 세그먼트 내 `Solar8000/ART_SBP/DBP` 기반         |
| case 분할           | train 60% / val 20% / test 20% (case-level, seed=42)                    |
| 테스트 케이스 수    | 672 cases                                                               |
| 테스트 세그먼트 수  | 1,987,561 segments                                                      |
| 평가 체크포인트     | 각 모델의 `best.pt` (val loss 최소 epoch)                               |
| 공통 하이퍼파라미터 | lr=1e-3, weight_decay=1e-4, batch_size=256, max_epochs=100, patience=15 |

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

| 모델명             | 분류                   | 파라미터 수 | 레이어 수 | 입력 방식              | 평가 여부 |
| ------------------ | ---------------------- | ----------- | --------- | ---------------------- | --------- |
| `naive`            | 베이스라인             | —           | —         | 8초 전체 (1000 샘플)   | ✅         |
| `resnet1d`         | ResNet1D 계열          | 2.18 M      | 100       | 8초 전체               | ✅         |
| `resnet1d_mini`    | ResNet1D 계열          | 964.4 K     | 60        | 8초 전체               | ✅         |
| `resnet1d_tiny`    | ResNet1D 계열          | 60.6 K      | 34        | 8초 전체               | ✅         |
| `resnet1d_micro`   | ResNet1D 계열          | 15.1 K      | 21        | 8초 전체               | ✅         |
| `st_resnet`        | 다중 채널              | 478.9 K     | 140       | PPG + VPG + APG        | ✅         |
| `minception`       | 다중 스케일            | 440.7 K     | 134       | 8초 전체               | ✅         |
| `xresnet1d`        | 대형 ResNet            | 9.47 M      | 484       | 8초 전체               | ✅         |
| `xresnet1d101`     | 대형 ResNet            | —           | —         | 8초 전체               | ✅         |
| `pulse_resnet1d`   | 맥박 분할              | ~58 K       | —         | 8×125 샘플 (맥박 단위) | ✅         |
| `pulsewo_resnet1d` | 맥박 분할              | —           | —         | 8×125 샘플             | ✅         |
| `pulsew_resnet1d`  | 맥박 분할              | —           | —         | 8×125 샘플             | ✅         |
| `mtae`             | 다중 태스크 오토인코더 | 119.5 K     | 37        | 8초 전체               | ✅         |
| `mtae_tr`          | MTAE + Transformer     | 109.4 K     | 93        | 8초 전체               | ✅         |

## 5. 테스트셋 정량 평가 결과

### 5.1 SBP(수축기혈압) 종합 비교

| 모델                   | MAE ↓     | ME    | SD        | RMSE      | ±5%       | ±10%      | ±15%      | BHS   | AAMI |
| ---------------------- | --------- | ----- | --------- | --------- | --------- | --------- | --------- | ----- | ---- |
| `naive`                | 15.65     | -2.19 | 20.22     | 20.34     | 20.6%     | 40.3%     | 57.4%     | D     | ❌    |
| `mtae_tr`              | 13.53     | -1.51 | 17.81     | 17.87     | 25.0%     | 47.3%     | 65.1%     | D     | ❌    |
| `minception`           | 13.49     | -0.82 | 17.76     | 17.78     | 25.1%     | 47.2%     | 65.0%     | D     | ❌    |
| `resnet1d`             | 13.42     | +0.45 | 17.65     | 17.66     | 25.1%     | 47.4%     | 65.2%     | D     | ❌    |
| `xresnet1d`            | 13.33     | -0.51 | 17.51     | 17.52     | 25.2%     | 47.7%     | 65.4%     | D     | ❌    |
| `resnet1d_mini`        | 13.27     | -0.69 | 17.41     | 17.42     | 25.3%     | 47.8%     | 65.6%     | D     | ❌    |
| `pulsew_resnet1d`      | 13.14     | -1.58 | 17.22     | 17.29     | 25.5%     | 48.2%     | 66.3%     | D     | ❌    |
| `pulse_resnet1d`       | 13.13     | -1.27 | 17.23     | 17.28     | 25.6%     | 48.2%     | 66.2%     | D     | ❌    |
| `xresnet1d101`         | 13.11     | -2.71 | 17.10     | 17.31     | 25.9%     | 48.6%     | 66.3%     | D     | ❌    |
| `mtae`                 | 13.11     | -1.98 | 17.14     | 17.25     | 25.5%     | 48.1%     | 66.4%     | D     | ❌    |
| `resnet1d_tiny`        | 13.09     | -0.41 | 17.14     | 17.15     | 25.5%     | 48.1%     | 66.1%     | D     | ❌    |
| `st_resnet`            | 13.02     | -1.99 | 17.08     | 17.19     | **26.0%** | **48.8%** | **66.6%** | D     | ❌    |
| `resnet1d_micro`       | 13.01     | -1.33 | 17.04     | 17.10     | 25.6%     | 48.4%     | 66.6%     | D     | ❌    |
| **`pulsewo_resnet1d`** | **12.97** | -1.05 | **16.97** | **17.01** | 25.6%     | 48.5%     | **66.7%** | **D** | ❌    |

> ↓: 낮을수록 좋음. ME 부호: 양수=과추정, 음수=과소추정.  
> BHS 등급 기준으로 SBP ±5mmHg 이내 비율이 40%에 크게 미달하여 전 모델이 Grade D.  
> `pulsewo_resnet1d`가 augmentation 적용 후 SBP MAE 최고(12.97) 달성 — no_aug 대비 0.41 개선.

### 5.2 DBP(이완기혈압) 종합 비교

| 모델                 | MAE ↓    | ME    | SD        | RMSE      | ±5%       | ±10%      | ±15%      | BHS   | AAMI |
| -------------------- | -------- | ----- | --------- | --------- | --------- | --------- | --------- | ----- | ---- |
| `naive`              | 9.41     | -1.01 | 12.14     | 12.18     | 33.5%     | 61.7%     | 80.9%     | D     | ❌    |
| `mtae_tr`            | 8.28     | -0.89 | 10.82     | 10.86     | 39.3%     | 68.5%     | 85.3%     | D     | ❌    |
| `resnet1d_mini`      | 8.18     | +0.44 | 10.62     | 10.63     | 39.4%     | 68.2%     | 85.6%     | D     | ❌    |
| `minception`         | 8.16     | -0.26 | 10.67     | 10.67     | 39.8%     | 68.9%     | 85.8%     | D     | ❌    |
| `xresnet1d`          | 8.11     | -0.41 | 10.61     | 10.61     | 40.0%     | 69.1%     | 85.9%     | D     | ❌    |
| `resnet1d`           | 8.09     | -0.85 | 10.57     | 10.60     | 40.0%     | 69.4%     | 86.2%     | **C** | ❌    |
| `pulse_resnet1d`     | 8.05     | -1.40 | 10.46     | 10.55     | 40.4%     | 69.5%     | 86.2%     | **C** | ❌    |
| `pulsew_resnet1d`    | 7.96     | -0.64 | 10.40     | 10.42     | 40.7%     | 69.9%     | 86.6%     | **C** | ❌    |
| `mtae`               | 7.96     | -0.91 | 10.41     | 10.45     | 40.9%     | 69.9%     | 86.5%     | **C** | ❌    |
| `xresnet1d101`       | 7.97     | -1.72 | 10.29     | 10.43     | 40.4%     | 69.9%     | 86.8%     | **C** | ❌    |
| `resnet1d_tiny`      | 7.97     | -0.17 | 10.40     | 10.40     | 40.5%     | 69.7%     | 86.5%     | **C** | ❌    |
| `st_resnet`          | 7.93     | -0.36 | 10.36     | 10.37     | 40.8%     | 70.1%     | 86.7%     | **C** | ❌    |
| `pulsewo_resnet1d`   | 7.91     | -1.07 | 10.31     | 10.36     | 41.1%     | 70.4%     | 86.8%     | **C** | ❌    |
| **`resnet1d_micro`** | **7.89** | -0.92 | **10.32** | **10.36** | **41.2%** | **70.4%** | **86.9%** | **C** | ❌    |

> DBP에서 Grade C 달성 모델: `resnet1d`, `pulse_resnet1d`, `pulsew_resnet1d`, `mtae`,
> `xresnet1d101`, `resnet1d_tiny`, `st_resnet`, `pulsewo_resnet1d`, `resnet1d_micro` (9종).  
> AAMI 기준: SD가 모든 모델에서 10 mmHg를 초과하여 전 모델 불통과 (기준: SD ≤ 8 mmHg).  
> no_aug 대비 `resnet1d` DBP Grade D→C 개선, `mtae_tr` · `xresnet1d` C→D 하락.

### 5.3 종합 순위 (SBP MAE + DBP MAE 합산 기준)

| 순위 | 모델               | SBP MAE | DBP MAE | 합산      | no_aug 합산 | 변화        | DBP BHS |
| ---- | ------------------ | ------- | ------- | --------- | ----------- | ----------- | ------- |
| 1    | `pulsewo_resnet1d` | 12.97   | 7.91    | **20.87** | 21.49       | ↑ +0.62     | C       |
| 2    | `resnet1d_micro`   | 13.01   | 7.89    | **20.90** | 20.79       | ↓ −0.11     | C       |
| 3    | `st_resnet`        | 13.02   | 7.93    | **20.95** | 20.93       | ↓ −0.02     | C       |
| 4    | `resnet1d_tiny`    | 13.09   | 7.97    | **21.06** | 20.81       | ↓ −0.25     | C       |
| 5    | `mtae`             | 13.11   | 7.96    | **21.08** | 21.07       | ≈           | C       |
| 6    | `xresnet1d101`     | 13.11   | 7.97    | **21.08** | 21.01       | ↓ −0.07     | C       |
| 7    | `pulsew_resnet1d`  | 13.14   | 7.96    | **21.10** | 21.02       | ↓ −0.08     | C       |
| 8    | `pulse_resnet1d`   | 13.13   | 8.05    | **21.18** | 21.20       | ↑ +0.02     | C       |
| 9    | `xresnet1d`        | 13.33   | 8.11    | **21.44** | 21.20       | ↓ −0.24     | D       |
| 10   | `resnet1d_mini`    | 13.27   | 8.18    | **21.45** | 21.66       | ↑ +0.21     | D       |
| 11   | `resnet1d`         | 13.42   | 8.09    | **21.51** | 21.49       | ↓ −0.02     | C       |
| 12   | `minception`       | 13.49   | 8.16    | **21.65** | 21.59       | ↓ −0.06     | D       |
| 13   | `mtae_tr`          | 13.53   | 8.28    | **21.82** | 21.38       | ↓ **−0.44** | D       |
| —    | `naive`            | 15.65   | 9.41    | **25.06** | 25.07       | ≈           | D       |

> ↑: no_aug 대비 개선, ↓: 성능 하락. 합산 MAE 기준.

## 6. 모델별 상세 평가

### 6.1 naive (베이스라인)

훈련 실행: `20260605_151217` | best_epoch: 9 (no_aug: 17)

```
SBP — MAE: 15.65, ME: -2.19, SD: 20.22, RMSE: 20.34 | Grade D | AAMI: ❌
DBP — MAE:  9.41, ME: -1.01, SD: 12.14, RMSE: 12.18 | Grade D | AAMI: ❌
```

훈련셋 통계값 근사 출력 모델. 수치 결과는 no_aug(SBP 15.65, DBP 9.42)와 사실상 동일하여
augmentation이 베이스라인에 영향을 미치지 않음을 확인한다. best_epoch=9로 no_aug(17)보다
빠르게 결정되었으나 val_loss=51.26 수준에서 변화 없이 early stopping 발동.

| 그래프               |                                                     |
| -------------------- | --------------------------------------------------- |
| Prediction vs Actual | ![no caption](../data/results/eval_plot/naive.png)  |
| Error Distribution   | ![no caption](../data/results/error_hist/naive.png) |
| 훈련 손실 곡선       | ![no caption](../data/results/loss_graph/naive.png) |
| 훈련 MAE 곡선        | ![no caption](../data/results/mae_graph/naive.png)  |

### 6.2 resnet1d (기준 모델)

훈련 실행: `20260605_233256` | best_epoch: 1 (no_aug: 1)

```
SBP — MAE: 13.42, ME: +0.45, SD: 17.65, RMSE: 17.66 | Grade D | AAMI: ❌
DBP — MAE:  8.09, ME: -0.85, SD: 10.57, RMSE: 10.60 | Grade C | AAMI: ❌
```

2.18M 파라미터의 100-layer 1D ResNet. **DBP BHS Grade D → C로 개선** (±5mmHg 이내 40.0%,
no_aug: 39.9%). SBP MAE는 13.39→13.42로 소폭 증가. ME가 SBP에서 -3.06→+0.45로 편향 방향이
반전되어 no_aug의 과소추정 경향이 해소됐다. best_epoch=1로 즉각 과적합 패턴은 동일.

| 그래프               |                                                        |
| -------------------- | ------------------------------------------------------ |
| Prediction vs Actual | ![no caption](../data/results/eval_plot/resnet1d.png)  |
| Error Distribution   | ![no caption](../data/results/error_hist/resnet1d.png) |
| 훈련 손실 곡선       | ![no caption](../data/results/loss_graph/resnet1d.png) |
| 훈련 MAE 곡선        | ![no caption](../data/results/mae_graph/resnet1d.png)  |

### 6.3 resnet1d_mini

훈련 실행: `20260605_190738` | best_epoch: 1 (no_aug: 1)

```
SBP — MAE: 13.27, ME: -0.69, SD: 17.41, RMSE: 17.42 | Grade D | AAMI: ❌
DBP — MAE:  8.18, ME: +0.44, SD: 10.62, RMSE: 10.63 | Grade D | AAMI: ❌
```

no_aug 대비 SBP 13.51→13.27 (+0.24 개선), DBP 8.15→8.18 (소폭 악화). 종합 순위는
13위→10위로 상승. DBP BHS Grade D 유지(±5mmHg 39.4%로 C 문턱 미달). ME가 DBP에서
-0.56→+0.44로 반전되는 등 augmentation이 편향 방향에 영향을 미쳤다.

| 그래프               |                                                             |
| -------------------- | ----------------------------------------------------------- |
| Prediction vs Actual | ![no caption](../data/results/eval_plot/resnet1d_mini.png)  |
| Error Distribution   | ![no caption](../data/results/error_hist/resnet1d_mini.png) |
| 훈련 손실 곡선       | ![no caption](../data/results/loss_graph/resnet1d_mini.png) |
| 훈련 MAE 곡선        | ![no caption](../data/results/mae_graph/resnet1d_mini.png)  |

### 6.4 resnet1d_tiny

훈련 실행: `20260605_191547` | best_epoch: 2 (no_aug: 1)

```
SBP — MAE: 13.09, ME: -0.41, SD: 17.14, RMSE: 17.15 | Grade D | AAMI: ❌
DBP — MAE:  7.97, ME: -0.17, SD: 10.40, RMSE: 10.40 | Grade C | AAMI: ❌
```

no_aug 대비 SBP 12.95→13.09 (소폭 악화), DBP 7.86→7.97 (소폭 악화). 종합 순위 2위→4위.
best_epoch=2로 no_aug(1)보다 1 에폭 늦어져 augmentation이 약간의 정규화 효과를 발휘했다.
SBP ME가 -2.21→-0.41로 대폭 감소하여 편향이 크게 개선됐다.

| 그래프               |                                                             |
| -------------------- | ----------------------------------------------------------- |
| Prediction vs Actual | ![no caption](../data/results/eval_plot/resnet1d_tiny.png)  |
| Error Distribution   | ![no caption](../data/results/error_hist/resnet1d_tiny.png) |
| 훈련 손실 곡선       | ![no caption](../data/results/loss_graph/resnet1d_tiny.png) |
| 훈련 MAE 곡선        | ![no caption](../data/results/mae_graph/resnet1d_tiny.png)  |

### 6.5 resnet1d_micro

훈련 실행: `20260606_215613` | best_epoch: **20** ← 정상 수렴 (no_aug와 동일)

```
SBP — MAE: 13.01, ME: -1.33, SD: 17.04, RMSE: 17.10 | Grade D | AAMI: ❌
DBP — MAE:  7.89, ME: -0.92, SD: 10.32, RMSE: 10.36 | Grade C | AAMI: ❌
```

15.1K 파라미터의 초소형 ResNet. **DBP MAE 최고(7.89)** 유지. SBP 12.96→13.01로 미미하게
증가했으나 종합 순위는 1위→2위. best_epoch=20, 총 35 에폭으로 no_aug와 학습 동역학이
동일하다. 파라미터가 너무 적어 augmentation이 추가적인 정규화 효과를 발휘하지 못하는
underfitting 영역에 있음을 시사한다.

| 그래프               |                                                              |
| -------------------- | ------------------------------------------------------------ |
| Prediction vs Actual | ![no caption](../data/results/eval_plot/resnet1d_micro.png)  |
| Error Distribution   | ![no caption](../data/results/error_hist/resnet1d_micro.png) |
| 훈련 손실 곡선       | ![no caption](../data/results/loss_graph/resnet1d_micro.png) |
| 훈련 MAE 곡선        | ![no caption](../data/results/mae_graph/resnet1d_micro.png)  |

### 6.6 st_resnet (Spectro-Temporal ResNet)

훈련 실행: `20260605_203424` | best_epoch: 2 (no_aug: 1)

```
SBP — MAE: 13.02, ME: -1.99, SD: 17.08, RMSE: 17.19 | Grade D | AAMI: ❌
DBP — MAE:  7.93, ME: -0.36, SD: 10.36, RMSE: 10.37 | Grade C | AAMI: ❌
```

PPG와 그 1차(VPG)·2차(APG) 미분을 3채널 입력으로 사용하는 모델. **SBP ±5mmHg 이내
비율 26.0%로 전 모델 중 최고**. SBP MAE 13.00→13.02, DBP MAE 7.93으로 no_aug 대비 거의
동일한 성능을 유지한다. best_epoch=2로 no_aug(1) 대비 1 에폭 늦어졌다. DBP ME -0.36으로
편향이 작다.

| 그래프               |                                                         |
| -------------------- | ------------------------------------------------------- |
| Prediction vs Actual | ![no caption](../data/results/eval_plot/st_resnet.png)  |
| Error Distribution   | ![no caption](../data/results/error_hist/st_resnet.png) |
| 훈련 손실 곡선       | ![no caption](../data/results/loss_graph/st_resnet.png) |
| 훈련 MAE 곡선        | ![no caption](../data/results/mae_graph/st_resnet.png)  |

### 6.7 minception (Multi-scale Inception 1D)

훈련 실행: `20260605_151212` | best_epoch: 1 (no_aug: 2)

```
SBP — MAE: 13.49, ME: -0.82, SD: 17.76, RMSE: 17.78 | Grade D | AAMI: ❌
DBP — MAE:  8.16, ME: -0.26, SD: 10.67, RMSE: 10.67 | Grade D | AAMI: ❌
```

no_aug 대비 SBP 13.40→13.49 (소폭 악화), DBP 8.19→8.16 (소폭 개선). ME 방향이 SBP
+0.24→-0.82, DBP +0.48→-0.26으로 양 편향에서 음 편향으로 전환됐다. 종합 순위는 12위 유지.
best_epoch=1로 no_aug(2) 대비 더 빠른 과적합이 발생했다.

| 그래프               |                                                          |
| -------------------- | -------------------------------------------------------- |
| Prediction vs Actual | ![no caption](../data/results/eval_plot/minception.png)  |
| Error Distribution   | ![no caption](../data/results/error_hist/minception.png) |
| 훈련 손실 곡선       | ![no caption](../data/results/loss_graph/minception.png) |
| 훈련 MAE 곡선        | ![no caption](../data/results/mae_graph/minception.png)  |

### 6.8 xresnet1d (Deep XResNet-101-style)

훈련 실행: `20260606_092522` | best_epoch: 2 (no_aug: 1)

```
SBP — MAE: 13.33, ME: -0.51, SD: 17.51, RMSE: 17.52 | Grade D | AAMI: ❌
DBP — MAE:  8.11, ME: -0.41, SD: 10.61, RMSE: 10.61 | Grade D | AAMI: ❌
```

9.47M 파라미터의 대형 모델. **DBP BHS Grade C → D 하락**: ±5mmHg 이내 40.0%로 40% 문턱에
0.001pp 미달. no_aug 대비 SBP 13.25→13.33 (소폭 악화), DBP 7.95→8.11 (악화). SBP ME가
+1.65→-0.51로 과추정 편향이 해소됐으나 전체 정확도는 하락했다. best_epoch=2로 no_aug(1)
대비 소폭 개선됐으나 대형 모델의 과적합 문제는 지속된다.

| 그래프               |                                                         |
| -------------------- | ------------------------------------------------------- |
| Prediction vs Actual | ![no caption](../data/results/eval_plot/xresnet1d.png)  |
| Error Distribution   | ![no caption](../data/results/error_hist/xresnet1d.png) |
| 훈련 손실 곡선       | ![no caption](../data/results/loss_graph/xresnet1d.png) |
| 훈련 MAE 곡선        | ![no caption](../data/results/mae_graph/xresnet1d.png)  |

### 6.13 mtae (Multi-Task AutoEncoder)

훈련 실행: `20260605_151222` | best_epoch: **5** (no_aug: 14)

```
SBP — MAE: 13.11, ME: -1.98, SD: 17.14, RMSE: 17.25 | Grade D | AAMI: ❌
DBP — MAE:  7.96, ME: -0.91, SD: 10.41, RMSE: 10.45 | Grade C | AAMI: ❌
```

CNN 인코더/디코더 + BP 헤드. no_aug 대비 SBP 13.09→13.11 (≈동일), DBP 7.98→7.96 (미미한
개선). 종합 순위 6위→5위로 소폭 상승.

**주목할 변화**: best_epoch=14→**5**로 대폭 감소(총 에폭 29→20). augmentation으로 입력
데이터 분포가 다양해지면서 재구성 손실 기반 정규화 효과와 상호작용이 변화한 것으로 분석된다.
초기 수렴이 빨라졌으나 절대 성능은 거의 동일하다.

| 그래프               |                                                    |
| -------------------- | -------------------------------------------------- |
| Prediction vs Actual | ![no caption](../data/results/eval_plot/mtae.png)  |
| Error Distribution   | ![no caption](../data/results/error_hist/mtae.png) |
| 훈련 손실 곡선       | ![no caption](../data/results/loss_graph/mtae.png) |
| 훈련 MAE 곡선        | ![no caption](../data/results/mae_graph/mtae.png)  |

### 6.14 mtae_tr (MTAE with Transformer)

훈련 실행: `20260605_151226` | best_epoch: **14** (no_aug: 3)

```
SBP — MAE: 13.53, ME: -1.51, SD: 17.81, RMSE: 17.87 | Grade D | AAMI: ❌
DBP — MAE:  8.28, ME: -0.89, SD: 10.82, RMSE: 10.86 | Grade D | AAMI: ❌
```

**augmentation 최대 피해 모델**. no_aug 대비 SBP 13.25→13.53 (**+0.28 악화**), DBP 8.13→8.28
(**+0.15 악화**). 종합 순위 **9위→13위**. **DBP BHS Grade C → D** 하락(±5mmHg 40.2%→39.3%).

**역설적 best_epoch 변화**: best_epoch=3→**14**로 대폭 증가(총 에폭 18→29). mtae(14→5)와
정반대 방향. Transformer 구조가 augmented 데이터에서 수렴을 늦게 찾지만 최종 성능은 더
나빠진다. 현재 d_model=32, 4-head, 4-layer 구성의 Transformer가 augmented 데이터의 복잡한
분포를 처리하기에 표현력이 부족하거나 학습률 일정이 맞지 않는 것으로 분석된다.

| 그래프               |                                                       |
| -------------------- | ----------------------------------------------------- |
| Prediction vs Actual | ![no caption](../data/results/eval_plot/mtae_tr.png)  |
| Error Distribution   | ![no caption](../data/results/error_hist/mtae_tr.png) |
| 훈련 손실 곡선       | ![no caption](../data/results/loss_graph/mtae_tr.png) |
| 훈련 MAE 곡선        | ![no caption](../data/results/mae_graph/mtae_tr.png)  |

## 7. 훈련 과정 분석

### 7.1 Early Stopping 동작 요약

| 모델               | Best Epoch | 총 에폭 | Val Loss 최소 | 과적합 패턴                            | no_aug BE |
| ------------------ | ---------- | ------- | ------------- | -------------------------------------- | --------- |
| `resnet1d_micro`   | **20**     | 35      | 41.07         | 정상 수렴 (완만한 감소)                | 20        |
| `mtae_tr`          | **14**     | 29      | 21.76 †       | 완만한 감소 후 수렴 (no_aug 3→14 급증) | 3         |
| `mtae`             | **5**      | 20      | 20.99 †       | 빠른 수렴 후 정체 (no_aug 14→5 급감)   | 14        |
| `resnet1d_tiny`    | 2          | 17      | 41.42         | 초기 수렴, 이후 증가                   | 1         |
| `st_resnet`        | 2          | 17      | 41.58         | 초기 수렴, 이후 증가                   | 1         |
| `xresnet1d`        | 2          | 17      | 42.37         | 초기 수렴, 이후 증가                   | 1         |
| `pulsew_resnet1d`  | 2          | 17      | 41.54         | 초기 수렴, 이후 증가                   | 2         |
| `naive`            | 9          | 24      | 51.26         | 수렴하지 않음 (상수 출력)              | 17        |
| `resnet1d`         | 1          | 16      | 42.54         | **epoch 2부터 즉각 과적합**            | 1         |
| `resnet1d_mini`    | 1          | 16      | 42.10         | 즉각 과적합                            | 1         |
| `minception`       | 1          | 16      | 43.05         | 즉각 과적합                            | 2         |
| `xresnet1d101`     | 1          | 16      | 41.87         | epoch 2부터 급격한 과적합              | 1         |
| `pulse_resnet1d`   | 1          | 16      | 41.84         | 즉각 과적합                            | 1         |
| `pulsewo_resnet1d` | 1          | 16      | 41.21         | 즉각 과적합 (no_aug 2→1)               | 2         |

> † `mtae` / `mtae_tr`의 val_loss 스케일은 다중 태스크 손실(BP×0.5 + 재구성×0.5)로
> 계산되어 단순 BP 회귀 모델의 ~42 대비 ~21 수준. 직접 비교 불가.

### 7.2 과적합 분석

**augmentation 적용 전후 best_epoch 변화 요약**:

| 변화 방향              | 모델                                                                                               | 해석                                      |
| ---------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| best_epoch 증가 (지연) | `resnet1d_tiny` 1→2, `st_resnet` 1→2, `xresnet1d` 1→2, `mtae_tr` 3→14                              | augmentation이 정규화 역할, 수렴 지연     |
| best_epoch 감소 (가속) | `mtae` 14→5, `minception` 2→1, `pulsewo_resnet1d` 2→1                                              | augmented 데이터가 기존 최적점을 변화시킴 |
| 변화 없음              | `resnet1d`, `resnet1d_mini`, `xresnet1d101`, `pulse_resnet1d`, `pulsew_resnet1d`, `resnet1d_micro` | 구조적으로 augmentation 영향 제한적       |

### 7.3 훈련셋 vs 검증셋 vs 테스트셋 성능 비교

#### SBP 비교 (단위: mmHg)

| 모델               | BE  | Train(BE) | Val(BE) | Test  | Train(fin) | 과적합 지수¹ |
| ------------------ | --- | --------- | ------- | ----- | ---------- | ------------ |
| `naive`            | 9   | 15.996    | 15.844  | 15.65 | 15.995     | **−0.35**    |
| `resnet1d_micro`   | 20  | 12.951    | 13.129  | 13.01 | 12.782     | **0.22**     |
| `mtae_tr`          | 14  | 12.838    | 13.605  | 13.53 | 12.597     | 0.94         |
| `mtae`             | 5   | 12.766    | 13.210  | 13.11 | 12.290     | 0.82         |
| `pulsew_resnet1d`  | 2   | 12.681    | 13.289  | 13.14 | 12.035     | 1.10         |
| `pulsewo_resnet1d` | 1   | 14.109†   | 13.161  | 12.97 | 11.728     | 1.24         |
| `pulse_resnet1d`   | 1   | 14.922†   | 13.283  | 13.13 | 12.050     | 1.08         |
| `resnet1d_tiny`    | 2   | 13.097    | 13.256  | 13.09 | 11.925     | 1.17         |
| `st_resnet`        | 2   | 13.271    | 13.273  | 13.02 | 10.246     | 2.78         |
| `xresnet1d`        | 2   | 12.031    | 13.451  | 13.33 | 8.139      | 5.19         |
| `minception`       | 1   | 13.315†   | 13.689  | 13.49 | 9.654      | 3.84         |
| `resnet1d_mini`    | 1   | 13.346    | 13.336  | 13.27 | 9.564      | 3.71         |
| `resnet1d`         | 1   | 13.144    | 13.507  | 13.42 | 8.557      | 4.86         |
| `xresnet1d101`     | 1   | 13.428†   | 13.352  | 13.11 | 8.427      | 4.69         |

#### DBP 비교 (단위: mmHg)

| 모델               | BE  | Train(BE) | Val(BE) | Test | Train(fin) | 과적합 지수¹ |
| ------------------ | --- | --------- | ------- | ---- | ---------- | ------------ |
| `naive`            | 9   | 9.416     | 9.200   | 9.41 | 9.411      | **0.00**     |
| `resnet1d_micro`   | 20  | 7.869     | 7.733   | 7.89 | 7.783      | **0.11**     |
| `mtae`             | 5   | 7.824     | 7.821   | 7.96 | 7.566      | 0.40         |
| `mtae_tr`          | 14  | 7.896     | 8.050   | 8.28 | 7.745      | 0.54         |
| `pulsew_resnet1d`  | 2   | 7.744     | 7.760   | 7.96 | 7.383      | 0.58         |
| `resnet1d_tiny`    | 2   | 7.926     | 7.743   | 7.97 | 7.309      | 0.66         |
| `pulse_resnet1d`   | 1   | 8.614†    | 7.892   | 8.05 | 7.381      | 0.67         |
| `pulsewo_resnet1d` | 1   | 8.345†    | 7.753   | 7.91 | 7.210      | 0.70         |
| `st_resnet`        | 2   | 7.981     | 7.796   | 7.93 | 6.397      | 1.53         |
| `resnet1d`         | 1   | 7.895     | 7.959   | 8.09 | 5.325      | 2.77         |
| `minception`       | 1   | 7.941†    | 7.983   | 8.16 | 5.995      | 2.16         |
| `resnet1d_mini`    | 1   | 7.943     | 7.950   | 8.18 | 5.913      | 2.27         |
| `xresnet1d101`     | 1   | 8.142†    | 7.834   | 7.97 | 5.285      | 2.68         |
| `xresnet1d`        | 2   | 7.399     | 7.944   | 8.11 | 5.105      | 3.00         |

> ¹ **과적합 지수** = Test MAE − Train(final epoch) MAE. 값이 클수록 과적합이 심함.  
> † best_epoch=1에서 초기 무작위 가중치 배치가 포함되어 Train(BE) > Val(BE)인 "워밍업
> 효과" 발생.

#### 주요 관찰

**1. 검증셋 ↔ 테스트셋 일관성 유지**

Val(BE)와 Test의 차이가 SBP/DBP 모두 ±0.3 mmHg 이내로 case-level split의 신뢰성이
augmentation 적용 후에도 유지된다. Data leakage 없음이 재확인됐다.

**2. augmentation의 과적합 억제 효과 — 모델별 차이**

과적합 지수가 no_aug 대비 크게 개선된 모델과 악화된 모델이 공존한다:

| 모델             | no_aug SBP 과적합 지수 | aug SBP 과적합 지수 | 변화   |
| ---------------- | ---------------------- | ------------------- | ------ |
| `resnet1d_micro` | 0.68                   | 0.22                | ↑ 개선 |
| `mtae`           | 1.57                   | 0.82                | ↑ 개선 |
| `xresnet1d`      | 4.80                   | 5.19                | ↓ 악화 |
| `resnet1d`       | 4.94                   | 4.86                | ≈ 동일 |

대형 모델(xresnet1d, resnet1d)의 과적합 지수는 augmentation으로도 크게 개선되지 않았다.

## 8. 국제 표준 기준 달성 현황

### 8.1 AAMI 기준 분석

| 기준        | SBP (최우수 모델)                                 | DBP (최우수 모델)                             |
| ----------- | ------------------------------------------------- | --------------------------------------------- |
| ME ≤ 5 mmHg | **충족** (전 모델: 최대 2.71 mmHg)                | **충족** (전 모델: 최대 1.72 mmHg)            |
| SD ≤ 8 mmHg | ❌ **미달** (최소 16.97 mmHg / `pulsewo_resnet1d`) | ❌ **미달** (최소 10.29 mmHg / `xresnet1d101`) |

ME 기준은 전 모델 충족. SD 기준은 no_aug와 동일하게 전 모델 미달.

### 8.2 BHS 등급 달성 현황

| 등급             | SBP         | DBP (no_aug 대비)            |
| ---------------- | ----------- | ---------------------------- |
| A (≥60%/85%/95%) | 전무        | 전무                         |
| B (≥50%/75%/90%) | 전무        | 전무                         |
| C (≥40%/65%/85%) | 전무        | **9종** (no_aug: 10종, −1종) |
| D                | **전 모델** | 5종 (no_aug: 4종, +1종)      |

DBP Grade C: 9종 달성 (no_aug 10종 대비 1종 감소). `mtae_tr`·`xresnet1d` 탈락, `resnet1d` 신규 진입.

## 9. 주요 발견 및 시사점

### 9.1 Augmentation 효과: 모델별 비균일 반응

| 범주      | 모델                                               | aug 전후 합산 MAE 변화 | 해석                                             |
| --------- | -------------------------------------------------- | ---------------------- | ------------------------------------------------ |
| 최대 이득 | `pulsewo_resnet1d`                                 | 21.49 → 20.87 (+0.62)  | 맥박 분할 구조가 다양한 augmented 파형에 잘 반응 |
| 소폭 이득 | `pulse_resnet1d`                                   | 21.20 → 21.18 (+0.02)  |                                                  |
|           | `resnet1d_mini`                                    | 21.66 → 21.45 (+0.21)  |                                                  |
| 중립      | `naive`, `resnet1d`, `st_resnet`, `mtae`           | ≈동일                  | 구조상 augmentation 영향 제한적                  |
| 소폭 손실 | `resnet1d_tiny`, `xresnet1d101`, `pulsew_resnet1d` | −0.07~−0.25            | 기존 최적점에서 벗어남                           |
| 최대 손실 | `mtae_tr`                                          | 21.38 → 21.82 (−0.44)  | Transformer가 augmented 분포 변화에 취약         |

### 9.2 mtae vs mtae_tr의 상반된 반응

| 항목            | mtae (CNN)             | mtae_tr (Transformer)  |
| --------------- | ---------------------- | ---------------------- |
| best_epoch 변화 | 14 → **5** (빠른 수렴) | 3 → **14** (느린 수렴) |
| 성능 변화       | ≈동일 (−0.01 / +0.02)  | 악화 (+0.28 / +0.15)   |
| DBP BHS         | C 유지                 | C → **D** 하락         |

CNN 기반 mtae는 augmentation에 안정적으로 반응한 반면, Transformer 기반 mtae_tr은
수렴 dynamics가 크게 바뀌면서 성능이 저하됐다. 동일한 다중 태스크 접근법이라도 백본
구조에 따라 augmentation과의 상호작용이 상이함을 보여준다.

### 9.3 pulsewo_resnet1d의 도약

no_aug에서 10위(21.49)였던 `pulsewo_resnet1d`가 augmentation 적용 후 1위(20.87)로 도약한
것은 이 모델의 맥박 분할 구조가 증강된 파형 다양성에 특히 잘 맞는다는 것을 시사한다.
SBP SD가 17.32→16.97로 전 모델 최저 SD를 달성하여 오차 산포가 크게 개선됐다.

### 9.4 편향(ME) 패턴 변화

augmentation 적용으로 대부분 모델의 SBP ME 절대값이 감소했다:

| 모델               | no_aug SBP ME | aug SBP ME | 변화             |
| ------------------ | ------------- | ---------- | ---------------- |
| `resnet1d`         | −3.06         | +0.45      | 과소→과추정 반전 |
| `resnet1d_tiny`    | −2.21         | −0.41      | 편향 대폭 감소   |
| `pulsewo_resnet1d` | −3.37         | −1.05      | 편향 대폭 감소   |
| `pulse_resnet1d`   | −2.53         | −1.27      | 편향 감소        |
| `xresnet1d`        | +1.65         | −0.51      | 과추정→과소 반전 |

augmentation이 전반적으로 SBP 과소추정 편향을 완화하는 방향으로 작용했다.

### 9.5 SBP vs DBP 예측 난이도 차이

모든 모델에서 DBP MAE < SBP MAE (약 5 mmHg 차)가 유지됐다. augmentation이 DBP보다
SBP 개선에 더 이득을 준 모델(pulsewo_resnet1d: SBP −0.41, DBP −0.20)이 있는 반면,
mtae_tr은 DBP 악화가 더 큰(SBP +0.28, DBP +0.15) 비대칭 패턴도 관찰된다.

## 10. 미완료 실험 및 향후 과제

### 미완료 실험

현재 모든 등록 모델의 테스트셋 평가가 완료되어 미완료 실험 없음.

### 주요 향후 과제

1. **Augmentation 전략 최적화**: pulsewo_resnet1d에서 효과적이었던 augmentation이
   mtae_tr에서는 역효과를 낸 원인 분석이 필요하다. 모델별 최적 augmentation 유형·강도
   탐색(noise 수준, stretch 비율, flip 적용 여부 분리 실험)이 요구된다.

2. **mtae_tr 개선**: best_epoch=14로 수렴이 안정화됐으나 절대 성능이 하락했다. 학습률
   warmup + cosine annealing 적용, d_model 확대, augmentation 강도 조정이 필요하다.

3. **SBP 정확도 개선**: AAMI 기준 SD ≤ 8 mmHg까지 현재 최솟값(16.97 mmHg)의 절반
   이하로 줄여야 한다. Huber loss, quantile loss, 또는 더 강한 augmentation 탐색.

4. **pulsewo_resnet1d 심층 분석**: augmentation 후 SBP SD 16.97로 최저를 달성한 원인
   분석. 유사한 구조 변형(pulsewo_v2 등)으로 추가 개선 가능성 탐색.

5. **waveform reconstruction 접근법**: PPG2ABP, ABP-Net 등 ABP 파형을 먼저 복원 후
   SBP/DBP를 추출하는 접근법.

6. **BP 범위별 오차 분석**: 정상혈압/고혈압/저혈압 구간별 MAE 분리 평가.

7. **case-level 오차 분포**: 세그먼트 평균이 아닌 케이스별 평균 오차 분석.

## 부록: 모델별 그래프 인덱스

| 모델           | eval_plot                                                   | error_hist                                                   | loss_graph                                                   | mae_graph                                                   |
| -------------- | ----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------------------- |
| naive          | ![no caption](../data/results/eval_plot/naive.png)          | ![no caption](../data/results/error_hist/naive.png)          | ![no caption](../data/results/loss_graph/naive.png)          | ![no caption](../data/results/mae_graph/naive.png)          |
| resnet1d       | ![no caption](../data/results/eval_plot/resnet1d.png)       | ![no caption](../data/results/error_hist/resnet1d.png)       | ![no caption](../data/results/loss_graph/resnet1d.png)       | ![no caption](../data/results/mae_graph/resnet1d.png)       |
| resnet1d_mini  | ![no caption](../data/results/eval_plot/resnet1d_mini.png)  | ![no caption](../data/results/error_hist/resnet1d_mini.png)  | ![no caption](../data/results/loss_graph/resnet1d_mini.png)  | ![no caption](../data/results/mae_graph/resnet1d_mini.png)  |
| resnet1d_tiny  | ![no caption](../data/results/eval_plot/resnet1d_tiny.png)  | ![no caption](../data/results/error_hist/resnet1d_tiny.png)  | ![no caption](../data/results/loss_graph/resnet1d_tiny.png)  | ![no caption](../data/results/mae_graph/resnet1d_tiny.png)  |
| resnet1d_micro | ![no caption](../data/results/eval_plot/resnet1d_micro.png) | ![no caption](../data/results/error_hist/resnet1d_micro.png) | ![no caption](../data/results/loss_graph/resnet1d_micro.png) | ![no caption](../data/results/mae_graph/resnet1d_micro.png) |
| st_resnet      | ![no caption](../data/results/eval_plot/st_resnet.png)      | ![no caption](../data/results/error_hist/st_resnet.png)      | ![no caption](../data/results/loss_graph/st_resnet.png)      | ![no caption](../data/results/mae_graph/st_resnet.png)      |
| minception     | ![no caption](../data/results/eval_plot/minception.png)     | ![no caption](../data/results/error_hist/minception.png)     | ![no caption](../data/results/loss_graph/minception.png)     | ![no caption](../data/results/mae_graph/minception.png)     |
| xresnet1d      | ![no caption](../data/results/eval_plot/xresnet1d.png)      | ![no caption](../data/results/error_hist/xresnet1d.png)      | ![no caption](../data/results/loss_graph/xresnet1d.png)      | ![no caption](../data/results/mae_graph/xresnet1d.png)      |
| mtae           | ![no caption](../data/results/eval_plot/mtae.png)           | ![no caption](../data/results/error_hist/mtae.png)           | ![no caption](../data/results/loss_graph/mtae.png)           | ![no caption](../data/results/mae_graph/mtae.png)           |
| mtae_tr        | ![no caption](../data/results/eval_plot/mtae_tr.png)        | ![no caption](../data/results/error_hist/mtae_tr.png)        | ![no caption](../data/results/loss_graph/mtae_tr.png)        | ![no caption](../data/results/mae_graph/mtae_tr.png)        |
