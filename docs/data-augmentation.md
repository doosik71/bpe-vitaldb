# PPG 신호 Data Augmentation

구현 위치: `bpe/train/augment.py`  

## 1. 개요

본 프로젝트에서는 PPG 신호에 대한 온라인(on-the-fly) 데이터 증강을 훈련 시 적용한다.
증강은 DataLoader 워커 내부에서 샘플마다 독립적으로 적용되며, 검증·테스트셋에는 일절
적용하지 않는다.

**증강 적용 순서** (훈련 시 모든 기법이 순차 적용):

```text
원본 PPG 세그먼트
   ↓  z-score 정규화  (mean=0, std=1)
   ↓  GaussianNoise
   ↓  AmplitudeScaling
   ↓  TimeShift
   ↓  RandomMasking
   → 모델 입력
```

> z-score 정규화 **이후**에 증강이 적용되므로, 모든 증강 파라미터는 정규화된
> 신호 단위(무차원)를 기준으로 설계되었다.

## 2. 증강 기법 상세

### 2.1 Gaussian Noise (가우시안 잡음 추가)

**클래스:** `GaussianNoise(std=0.01)`  
**구현:** `bpe/train/augment.py` — `GaussianNoise`

#### 동작 원리

세그먼트 전체에 평균 0, 표준편차 `std`의 가우시안 잡음을 독립적으로 추가한다.

```text
x_aug[i] = x[i] + ε[i],    ε[i] ~ N(0, std^2)
```

#### 파라미터

| 파라미터 | 기본값 | 의미                             |
| -------- | ------ | -------------------------------- |
| `std`    | 0.01   | 잡음 표준편차 (정규화 신호 단위) |

#### 설계 근거

z-score 정규화 후 신호의 표준편차가 약 1이므로, std=0.01은 신호 대비 약 **1% 수준의
잡음**에 해당한다. 측정 센서의 양자화 잡음 및 전기적 노이즈를 모사하여 모델의 노이즈
강건성을 향상시킨다.

### 2.2 Amplitude Scaling (진폭 스케일링)

**클래스:** `AmplitudeScaling(lo=0.8, hi=1.2)`  
**구현:** `bpe/train/augment.py` — `AmplitudeScaling`

#### 동작 원리

세그먼트 전체를 균일 분포에서 샘플링한 스칼라 `α`로 곱한다.

```text
α ~ Uniform(lo, hi)
x_aug = α · x
```

#### 파라미터

| 파라미터 | 기본값 | 의미                      |
| -------- | ------ | ------------------------- |
| `lo`     | 0.8    | 스케일 하한 (원본의 80%)  |
| `hi`     | 1.2    | 스케일 상한 (원본의 120%) |

#### 설계 근거

z-score 정규화 후에 적용하므로 평균은 ~0으로 유지되고 분산은 대략 -20%~+20% 범위에서
섭동된다. PPG 센서의 접촉 압력 차이나 피부 색소 농도에 따른 신호 이득 변화를 모사한다.
하한을 0.8까지 열어 둠으로써 신호 감쇠 상황을 조금 더 넓게 포함하면서도, 상한은 1.2로
유지해 정규화 신호의 형태 정보를 크게 훼손하지 않도록 설정했다.

### 2.3 Time Shift (순환 시간 이동)

**클래스:** `TimeShift(max_shift=50)`  
**구현:** `bpe/train/augment.py` — `TimeShift`

#### 동작 원리

세그먼트를 무작위 오프셋만큼 순환(circular) 이동한다. `torch.roll`을 사용하므로 경계에서
제로 패딩 없이 반대쪽 끝에서 신호가 순환된다.

```text
shift ~ Uniform{-max_shift, ..., +max_shift}   (정수)
x_aug[i] = x[(i - shift) mod N]
```

#### 파라미터

| 파라미터    | 기본값 | 의미                                      |
| ----------- | ------ | ----------------------------------------- |
| `max_shift` | 50     | 최대 이동량 (샘플 수), 125 Hz 기준 ±0.4초 |

#### 설계 근거

8초 세그먼트(1000 샘플) 내에서 최대 ±50샘플(±0.4초) 이동은 세그먼트 길이의 ±5%에
해당한다. 맥박 위상의 미세한 시작점 차이나 세그먼트 추출 타이밍 오차에 대한 불변성을
학습시킨다. 순환 이동(circular shift)을 사용하므로 신호 에너지 손실 없이 경계 아티팩트를
방지한다.

### 2.4 Random Masking (연속 구간 마스킹)

**클래스:** `RandomMasking(lo_frac=0.05, hi_frac=0.10)`  
**구현:** `bpe/train/augment.py` — `RandomMasking`

#### 동작 원리

세그먼트에서 무작위로 선택한 **하나의 연속된 구간(span)** 을 0으로 대체한다. 마스킹 비율
`p`는 매 호출 시 균일 분포에서 샘플링되며, 이 값으로 계산한 길이를 바탕으로 span 길이를
정한다. 단, span 길이는 **125 Hz 기준 최대 1초(125 샘플)** 를 넘지 않도록 제한한다.

```text
p ~ Uniform(lo_frac, hi_frac)
n_mask = min(max(1, floor(N · p)), min(N, 125))
start = randint(0, N - n_mask)
x_aug[start : start + n_mask] = 0.0   (z-score 정규화 후 평균값)
```

#### 파라미터

| 파라미터  | 기본값 | 의미                      |
| --------- | ------ | ------------------------- |
| `lo_frac` | 0.05   | span 길이 비율 하한 (5%)  |
| `hi_frac` | 0.10   | span 길이 비율 상한 (10%) |

#### 설계 근거

마스킹값 0.0은 z-score 정규화된 신호의 평균값과 동일하여 마스킹이 평균 보간과 동등하게
처리된다. 연속 span 마스킹은 센서 접촉 불안정, 짧은 동작 아티팩트, 순간적인 포화처럼
**일정 시간 구간이 통째로 손상되는 패턴**을 더 직접적으로 모사한다. 기본 8초 세그먼트
기준 5~10% 길이는 50~100 샘플(약 0.4~0.8초)에 해당하며, 구현은 최대 125 샘플(1초)
이내로 제한해 과도한 정보 손실을 방지한다. MTAE 계열 모델에서 특히 재구성 손실과
시너지 효과가 예상된다.

## 3. 적용 방식

### 3.1 파이프라인 구성

`scripts/train.py`에서 증강 파이프라인을 조립하여 훈련 데이터셋에 전달한다.

```python
# scripts/train.py (기본값: 전체 증강 활성화)
aug_transforms = []
if args.aug_noise:  aug_transforms.append(GaussianNoise(std=0.01))
if args.aug_scale:  aug_transforms.append(AmplitudeScaling(lo=0.8, hi=1.2))
if args.aug_shift:  aug_transforms.append(TimeShift(max_shift=50))
if args.aug_mask:   aug_transforms.append(RandomMasking(lo_frac=0.05, hi_frac=0.10))
augment = PPGAugment(aug_transforms) if aug_transforms else None

train_ds = PPGDataset(..., augment=augment)   # 훈련셋: 증강 적용
val_ds   = PPGDataset(..., augment=None)      # 검증셋: 증강 없음
```

### 3.2 적용 위치

`PPGDataset.__getitem__()` 내부에서 z-score 정규화 직후에 증강이 실행된다.

```text
PPGDataset.__getitem__()
   1. NPZ 파일에서 세그먼트 로드 → Tensor 변환
   2. z-score 정규화: x = (x - mean) / std
   3. augment(x)  ← 훈련셋에만 적용
   4. (x, y) 반환
```

### 3.3 CLI 옵션

기본값은 전체 활성화이며, 개별 기법을 비활성화할 수 있다.

| 옵션             | 효과                         |
| ---------------- | ---------------------------- |
| `--no-aug-noise` | Gaussian Noise 비활성화      |
| `--no-aug-scale` | Amplitude Scaling 비활성화   |
| `--no-aug-shift` | Time Shift 비활성화          |
| `--no-aug-mask`  | Random span masking 비활성화 |

사용 예:

```bash
# 전체 증강 적용 (기본)
bin/train-model resnet1d

# 잡음만 비활성화
bin/train-model resnet1d --no-aug-noise

# 증강 완전 비활성화 (no-aug 실험 재현)
bin/train-model resnet1d --no-aug-noise --no-aug-scale --no-aug-shift --no-aug-mask
```

## 4. 실험 결과 요약

증강 적용 전후 성능 변화 (SBP+DBP MAE 합산, 단위: mmHg):

| 모델               | no_aug | aug   | 변화        |
| ------------------ | ------ | ----- | ----------- |
| `pulsewo_resnet1d` | 21.49  | 20.87 | **+0.62 ↑** |
| `resnet1d_mini`    | 21.66  | 21.45 | +0.21 ↑     |
| `pulse_resnet1d`   | 21.20  | 21.18 | +0.02 ↑     |
| `naive`            | 25.07  | 25.06 | ≈           |
| `resnet1d`         | 21.49  | 21.51 | −0.02       |
| `st_resnet`        | 20.93  | 20.95 | −0.02       |
| `mtae`             | 21.07  | 21.08 | ≈           |
| `xresnet1d101`     | 21.01  | 21.08 | −0.07       |
| `pulsew_resnet1d`  | 21.02  | 21.10 | −0.08       |
| `resnet1d_tiny`    | 20.81  | 21.06 | −0.25       |
| `xresnet1d`        | 21.20  | 21.44 | −0.24       |
| `resnet1d_micro`   | 20.79  | 20.90 | −0.11       |
| `minception`       | 21.59  | 21.65 | −0.06       |
| `mtae_tr`          | 21.38  | 21.82 | **−0.44 ↓** |

**주요 관찰:**

- **최대 수혜**: `pulsewo_resnet1d` — 10위(21.49) → 1위(20.87). SBP SD 17.32→16.97로
  전 모델 최저 달성. SBP 과소추정 편향(ME −3.37→−1.05) 대폭 개선.
- **최대 손실**: `mtae_tr` — Transformer 백본이 증강된 데이터 분포 변화에 취약하게 반응.
  best_epoch 3→14로 지연되었으나 최종 성능 하락. DBP BHS Grade C→D.
- **중립**: `mtae`(CNN) — best_epoch 14→5로 수렴이 빨라졌으나 절대 성능은 거의 동일.
  CNN과 Transformer의 상반된 반응은 백본 구조에 따른 증강 민감도 차이를 시사한다.
- **소형 모델 한계**: `resnet1d_micro`(15K 파라미터)는 증강 전후 동일한 best_epoch=20을
  유지하며 성능 변화도 미미. 이미 underfitting 영역에 있어 증강의 정규화 효과가 작용하지
  않는다.

상세 결과는 [evaluation_result.md](evaluation-result.md) 참조.

## 5. 구현 참고

### 소스 파일

| 파일                   | 역할                                                                                               |
| ---------------------- | -------------------------------------------------------------------------------------------------- |
| `bpe/train/augment.py` | 증강 클래스 정의 (`GaussianNoise`, `AmplitudeScaling`, `TimeShift`, `RandomMasking`, `PPGAugment`) |
| `bpe/train/dataset.py` | `PPGDataset.__getitem__()` — 정규화 후 증강 적용                                                   |
| `scripts/train.py`     | CLI 옵션 파싱, 파이프라인 조립, 훈련셋에 전달                                                      |

### 확장 방법

새 증강 기법을 추가하려면 `bpe/train/augment.py`에 `__call__(self, x) -> Tensor` 인터페이스를 구현한 클래스를 작성하고 `scripts/train.py`의 파이프라인 조립 부분에 추가하면 된다.
