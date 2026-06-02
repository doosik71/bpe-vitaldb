# Phase 3 Model Development Plan

## 목적

이 문서는 `bpe-vitaldb` 프로젝트의 Phase 3, 즉 PPG waveform 기반 혈압 추정
모델 개발 계획을 정의한다. 프로젝트 목표는 VitalDB의 PPG segment를 입력으로
받아 SBP, DBP, MBP를 예측하는 PyTorch 모델을 개발하는 것이다.

현재 Phase 2 데이터셋은 다음 형태를 가진다.

```text
x  float32  (N, segment_samples)   PPG segments
y  float32  (N, 2)                 [SBP_mean, DBP_mean] in mmHg
```

따라서 Phase 3의 첫 학습 목표는 SBP/DBP 예측이다.

## 개발 원칙

- case-level train/val/test split을 유지하여 segment leakage를 방지한다.
- 모델 비교는 같은 VitalDB split, 같은 preprocessing, 같은 metric으로 수행한다.
- 논문 성능 수치는 데이터셋, subject split, calibration 방식이 달라 직접 비교하지
  않는다. 모델 채택은 구조적 적합성과 프로젝트 내 재현 성능을 기준으로 한다.
- PPG-only 모델을 우선한다. ECG/PAT 기반 모델은 현재 목표와 입력 조건이 달라
  참고 모델로만 둔다.
- Python 의존성 관리는 `uv`로만 한다. `pip`는 사용하지 않는다.

## 기존 모델 조사와 채택 후보

### 성능 해석 주의

PulseDB 논문은 기존 cuff-less BP 연구들이 공개 데이터셋을 쓰더라도 subject 수,
전처리, split, BP 분포가 서로 달라 모델 간 성능 비교가 불공정해질 수 있다고
지적한다. 또한 MIMIC의 ECG/PPG/ABP alignment에는 최대 500 ms 수준의 불확실성이
있을 수 있고, VitalDB는 signal alignment가 확보되어 alignment 기반 분석에 더
적합하다고 설명한다. 이 프로젝트는 VitalDB의 case-level split과 동일 metric을
사용하여 후보 모델을 다시 평가한다.

| 채택 후보                 | 유형                                  | 입력                  | 주요 특징                                                                                                                                 | 논문 보고 성능                                                                                                                                                        | 프로젝트 채택 이유                                                                            |
| ------------------------- | ------------------------------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Spectro-Temporal ResNet   | 직접 SBP/DBP 회귀                     | PPG, VPG, APG         | PPG 원신호와 1차/2차 derivative를 함께 쓰는 residual CNN. 공개 MIMIC III subset과 code를 제공해 재현 baseline으로 좋다.                   | MIMIC III 510 subjects, leave-one-subject-out. MAE SBP 9.43 mmHg, DBP 6.88 mmHg.                                                                                      | 현재 `x -> y` 데이터셋에 바로 적용 가능하다. derivative channel만 on-the-fly로 추가하면 된다. |
| XResNet1d101              | 직접 SBP/DBP 회귀                     | PPG                   | 1D residual backbone. 2025 benchmarking study에서 PulseDB 기반 generalization 평가의 최상위 모델로 보고됨.                                | PulseDB in-distribution MAE SBP/DBP 9.4/6.0 mmHg with calibration, 14.0/8.5 mmHg without calibration. External no-calibration MAE는 SBP 15.0-25.1, DBP 7.0-10.4 mmHg. | 낮은 leakage 위험의 현실적인 generalization baseline으로 채택한다.                            |
| MInception + demographics | 직접 SBP/DBP 회귀                     | PPG, age, sex, BMI 등 | multi-scale Inception 계열. demographic side-channel을 붙이면 성능이 크게 개선된다고 보고됨.                                              | 2026 NBPDB benchmark에서 demographic 추가 후 MAE SBP 4.75 mmHg, DBP 2.90 mmHg.                                                                                        | VitalDB clinical metadata를 활용할 수 있게 되면 가장 강한 직접 회귀 후보가 될 수 있다.        |
| PPG2ABP                   | ABP waveform 복원 후 SBP/DBP/MBP 계산 | PPG                   | 1D U-Net approximation network와 MultiResUNet refinement network로 ABP waveform을 생성한다. BP 값은 예측 ABP의 max/min/mean에서 계산한다. | ABP waveform MAE 4.604 mmHg. BP parameter MAE DBP 3.449 +/- 6.147, MAP 2.310 +/- 4.437, SBP 5.727 +/- 9.162 mmHg.                                                     | 프로젝트가 ABP waveform ground truth를 보유하므로 MBP까지 자연스럽게 예측할 수 있다.          |
| ABP-Net                   | ABP waveform 복원 후 BP 계산          | PPG, VPG, APG         | Wave-U-Net 기반 fully convolutional model. MSE와 maximal absolute loss를 결합해 waveform peak/trough 보존을 강화한다.                     | waveform MAE/RMSE: subject-dependent 3.20/4.38 mmHg, subject-independent 5.57/7.15 mmHg.                                                                              | PPG2ABP보다 구현 단위가 단순한 waveform branch 후보로 채택한다.                               |
| PPG-to-ABP Transformer    | ABP waveform 복원 후 BP 계산          | PPG                   | encoder-decoder transformer와 frequency-domain learning을 비교한다.                                                                       | Transformer waveform MAE 3.01 mmHg. SBP MAE 3.77 mmHg, DBP MAE 2.69 mmHg. AAMI criterion 충족, BHS Grade A 보고.                                                      | U-Net 계열 이후 장기 dependency와 frequency structure를 비교할 고성능 연구 후보로 채택한다.   |

## 제외 또는 후순위 모델

- Kachuee et al.의 PAT 기반 모델과 waveform-based ANN-LSTM은 ECG+PPG 또는 PAT
  feature를 주로 사용한다. 현재 프로젝트의 1차 목표는 PPG-only 추론이므로
  Phase 3 주력 후보에서는 제외한다.
- 개인별 calibration을 요구하는 모델은 별도 실험군으로 둔다. 프로젝트 기본 목표는
  case-unseen, calibration-free 성능이다.

## 모델 구현 계획

### 공통 데이터 인터페이스

`scripts/train.py`와 `scripts/evaluate.py`는 공통 dataset loader를 사용한다.

```text
direct regression:
  input:  x_ppg                   (B, 1, T)
          or [ppg, vpg, apg]      (B, 3, T)
  target: [SBP, DBP]              (B, 2)
```

Derivative channels are computed in the dataset layer so that raw NPZ files do
not need to store duplicate arrays.

### 코드 레이아웃

```text
scripts/train.py          # train loop, checkpoint, CLI config
scripts/evaluate.py       # test-set evaluation and report export
bpe/models/<model-id>.py  # model registry and architecture definitions
bpe/eval/metrics.py       # MAE, RMSE, ME, SDE, R2, AAMI/BHS helpers
```

모델 id 예시는 다음과 같다.

```text
resnet1d
st_resnet
minception
ppg2abp_unet
abp_net
abp_transformer
```

## 단계별 실행 계획

### Milestone 1: Direct Regression Baseline

- `bpe/eval/metrics.py` 작성
- `scripts/train.py` 최소 구현
- constant median baseline과 small CNN baseline 추가
- `resnet1d` 또는 `st_resnet`을 첫 실제 모델로 학습
- 산출물: validation/test MAE, RMSE, ME, SDE, R2

성공 기준:

- constant median baseline보다 SBP/DBP MAE가 모두 개선된다.
- test case별 error 분포를 확인할 수 있다.
- 재실행 가능한 seed와 checkpoint 저장이 가능하다.

### Milestone 2: Multi-Scale Direct Models

- `st_resnet`: PPG/VPG/APG 3-channel 입력
- `minception`: multi-scale convolution branches
- metadata가 준비되면 `minception_demographic` variant 추가

성공 기준:

- `resnet1d` 대비 validation MAE가 10% 이상 개선되거나,
  SBP/DBP 중 하나가 크게 개선되고 다른 하나가 악화되지 않는다.
- BP range bin별 error를 보고해 normotension/hypotension/hypertension 구간
  편향을 확인한다.

### Milestone 3: 최종 모델 선택

최종 모델은 하나의 수치만으로 고르지 않는다. 다음 조건을 함께 본다.

- test MAE/RMSE
- ME와 SDE, AAMI/ISO numerical limits 근접 여부
- case-level error distribution
- BP range별 bias
- calibration-free 성능
- 학습/추론 비용
- 구현 복잡도와 재현 가능성

## 평가 프로토콜

- split: case-level train/val/test 고정
- model selection: validation only
- final report: test set 1회 평가
- metrics:
  - MAE, RMSE for SBP/DBP
  - ME, SDE for AAMI-style numerical review
  - R2
  - BHS-style cumulative error bands
  - case-level mean and worst-case error
- plots:
  - prediction vs reference scatter
  - Bland-Altman plot
  - BP bin별 error histogram
  - waveform model의 ABP overlay examples

## 리스크와 대응

- 데이터 누수: segment-level split 금지, case-level split만 허용한다.
- 과도하게 좋은 논문 수치: small subject, subject-dependent split, calibration 포함 여부를
  따로 기록한다.
- SBP peak 오차: waveform model은 `L_bp` 또는 peak-sensitive loss를 병행한다.
- metadata 누락: demographic model은 metadata가 없는 case를 처리할 fallback을 둔다.

## 우선순위

1. `st_resnet`
2. `resnet1d` or `xresnet1d`
3. `minception`
4. `ppg2abp_unet`
5. `abp_net`
6. `abp_transformer`

직접 회귀 모델을 먼저 구현해 빠른 baseline을 만든다.

## 참고 문헌

- Slapničar, G., Mlakar, N., and Luštrek, M. (2019).
  Blood Pressure Estimation from Photoplethysmogram Using a Spectro-Temporal
  Deep Neural Network. Sensors.
  <https://mdpi-res.com/d_attachment/sensors/sensors-19-03420/article_deploy/sensors-19-03420.pdf>
- Moulaeifard, M., Charlton, P. H., and Strodthoff, N. (2025).
  Generalizable deep learning for photoplethysmography-based blood pressure
  estimation: A benchmarking study.
  <https://arxiv.org/abs/2502.19167>
- Mathew, N., Shen, Y., Hu, R., Rahimi, M., and Zouridakis, G. (2026).
  Benchmarking and Enhancing PPG-Based Cuffless Blood Pressure Estimation
  Methods.
  <https://arxiv.org/abs/2602.04725>
- Ibtehaz, N. et al. (2022).
  PPG2ABP: Translating Photoplethysmogram Signals to Arterial Blood Pressure
  Waveforms.
  <https://arxiv.org/abs/2005.01669>
- PPG2ABP journal/PMC summary.
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC9687508/>
- Wang, D. et al. (2021).
  Prediction of arterial blood pressure waveforms from photoplethysmogram
  signals via fully convolutional neural networks.
  <https://www.sciencedirect.com/science/article/pii/S0010482521006715>
- Nawaz, M. W. et al. (2024).
  Cuff-less Arterial Blood Pressure Waveform Synthesis from Single-site PPG
  using Transformer and Frequency-domain Learning.
  <https://arxiv.org/abs/2401.05452>
- Wang, W. et al. (2023).
  PulseDB: A large, cleaned dataset based on MIMIC-III and VitalDB for
  benchmarking cuff-less blood pressure estimation methods.
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC9944565/>
