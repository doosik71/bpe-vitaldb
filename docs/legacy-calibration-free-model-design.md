# 비보정(Calibration-Free) 혈압 추정 모델 설계 문서

## 1. 개요

BPNet-CF는 PPG 신호 하나만으로 SBP와 DBP를 동시에 추정하는 단일 end-to-end 모델입니다. 개인 보정 측정(초기 PPG·혈압)과 수작업 특징 추출 없이 동작합니다.

### 1.1 설계 원칙

| 원칙                                                   | 내용                                                                                    |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| 이중 스케일 PPG 인코더 (단기 k=5/5/3 + 장기 k=15/11/7) | 펄스 단위 형태(수 ms)부터 PPI·수축-이완 타이밍(수 백 ms)까지 다중 수용 범위로 자동 학습 |
| 공유 백본 + 비대칭 이중 헤드                           | 저차원 특징 추출은 공유하고, 비대칭 채널 사영으로 분리 후 SBP/DBP 독립 헤드 적용        |
| SBP 헤드 우선 용량 배분                                | SBP 추정이 DBP보다 어려우므로 SBP 헤드에 더 많은 파라미터를 할당 (약 2.9:1)             |
| 시간적 자기 주의 모듈 (분리 적용)                      | 각 헤드에서 독립적으로 적용하여 SBP/DBP 고차원 표현을 분리                              |

### 1.2 모델 파일 명칭 규칙

단일 모델, 두 출력: `BPNetCF-{epoch:02d}-{val_loss:.4f}.h5`

## 2. 입력 및 출력 사양

모델은 단일 텐서를 입력으로 받고 두 스칼라를 출력합니다.

| 항목   | 이름       | 형태          | dtype   | 설명                     |
| ------ | ---------- | ------------- | ------- | ------------------------ |
| 입력   | `curr_ppg` | `(N, 800, 1)` | float32 | 현재(추정 시점) PPG 파형 |
| 출력 1 | `sbp`      | `(N, 1)`      | float32 | 추정 SBP (mmHg)          |
| 출력 2 | `dbp`      | `(N, 1)`      | float32 | 추정 DBP (mmHg)          |

- `N`: 배치 크기 (추론 시 1)
- PPG 800 샘플 = 100 Hz × 8초

## 3. 전처리 파이프라인

```text
원시 PPG (100 Hz, 8 sec = 800 samples)
        │
        ▼
Band-pass 필터 (Butterworth 4차, 0.5–7.0 Hz, filtfilt)
        │
        ▼
Min-max 정규화 ([0, 1])
        │
        ▼
입력 텐서 (800, 1)
```

## 4. 모델 아키텍처

```text
curr_ppg (N, 800, 1)
        │
        ├───────────────────────────────────────────────────────────┐
        │                                                           │
[Branch S: 단기 스케일 인코더]                         [Branch L: 장기 스케일 인코더]
 (커널 크기 5 / 5 / 3)                                  (커널 크기 15 / 11 / 7)
        │                                                           │
        └─────────── Concatenate (axis=-1) → (132, 2) ─────────────┘
                                │
                    Conv1D(32, kernel=1) → (132, 32)  ← 공유 표현
                                │
                    Conv1D(48, kernel=1) → (132, 48)  ← 비대칭 사영 [분기점]
                                │
              ┌─────────────────┴──────────────────┐
    ch 0:32 ──┤                                    ├── ch 32:48
  (132, 32)   │                                    │   (132, 16)
              ▼                                    ▼
  [SBP 어텐션 헤드]                       [DBP 어텐션 헤드]
   4-head MHA (d_k=8)                      2-head MHA (d_k=8)
   FFN(64→32), Add & LayerNorm             FFN(32→16), Add & LayerNorm
   → (132, 32)                             → (132, 16)
              │                                    │
  [SBP 회귀 헤드]                         [DBP 회귀 헤드]
   Spatial + Channel 요약                  Spatial + Channel 요약
   Flatten → (164,), Dropout(0.2)          Flatten → (148,), Dropout(0.2)
   Dense(128)→Dense(64)                    Dense(64)→Dense(32)
   →Dense(32)→Dense(1)                     →Dense(1)
              │                                    │
           SBP (mmHg)                          DBP (mmHg)
```

## 5. 이중 스케일 PPG 인코더

`curr_ppg`를 단기(S)와 장기(L) 두 경로로 병렬 처리합니다. 두 경로는 동일한 블록 구조를 공유하되 커널 크기가 다릅니다.

### 5.1 블록 구조

| 블록    | Branch S 커널 크기 | Branch L 커널 크기 | 설명                 |
| ------- | ------------------ | ------------------ | -------------------- |
| Block 1 | 5                  | 15                 | (800,1) → (400,32)   |
| Block 2 | 5                  | 11                 | (400,32) → (200,64)  |
| Block 3 | 3                  | 7                  | (200,64) → (200,128) |

Branch L의 큰 커널은 수 백 ms 단위의 PPI 및 수축-이완 타이밍을 직접 수용 범위에 포함합니다.

### 5.2 각 브랜치 상세 레이어 (Branch S / Branch L 공통 구조)

#### Block 1 — (800, 1) → (400, 32)

| 레이어             | Branch S 설명                        | Branch L 설명                         | 출력 형태 |
| ------------------ | ------------------------------------ | ------------------------------------- | --------- |
| DepthwiseConv1D    | kernel=5, depth_mult=1, padding=same | kernel=15, depth_mult=1, padding=same | (800, 1)  |
| Conv1D             | filters=32, kernel=1, padding=same   | filters=32, kernel=1, padding=same    | (800, 32) |
| BatchNormalization | —                                    | —                                     | (800, 32) |
| Activation         | ReLU                                 | ReLU                                  | (800, 32) |
| MaxPooling1D       | pool_size=2                          | pool_size=2                           | (400, 32) |

#### Block 2 — (400, 32) → (200, 64)

| 레이어             | Branch S 설명                        | Branch L 설명                         | 출력 형태 |
| ------------------ | ------------------------------------ | ------------------------------------- | --------- |
| DepthwiseConv1D    | kernel=5, depth_mult=1, padding=same | kernel=11, depth_mult=1, padding=same | (400, 32) |
| Conv1D             | filters=64, kernel=1, padding=same   | filters=64, kernel=1, padding=same    | (400, 64) |
| BatchNormalization | —                                    | —                                     | (400, 64) |
| Activation         | ReLU                                 | ReLU                                  | (400, 64) |
| MaxPooling1D       | pool_size=2                          | pool_size=2                           | (200, 64) |

#### Block 3 — (200, 64) → (200, 128)

| 레이어             | Branch S 설명                        | Branch L 설명                        | 출력 형태  |
| ------------------ | ------------------------------------ | ------------------------------------ | ---------- |
| DepthwiseConv1D    | kernel=3, depth_mult=1, padding=same | kernel=7, depth_mult=1, padding=same | (200, 64)  |
| Conv1D             | filters=128, kernel=1, padding=same  | filters=128, kernel=1, padding=same  | (200, 128) |
| BatchNormalization | —                                    | —                                    | (200, 128) |
| Activation         | ReLU                                 | ReLU                                 | (200, 128) |

#### SE 블록 — 채널 어텐션 (양 브랜치 동일)

(200, 128)에 채널 재가중을 적용합니다.

```text
GlobalAveragePooling1D → (128,) → Reshape(1,128)
→ Dense(32, relu) → Dense(128, sigmoid)
→ Multiply[(200,128) × (1,128)] → (200,128)
```

채널 압축비(ratio) = 4 (128 → 32 → 128).

#### Post-SE 처리 — (200, 128) → (132, 1)

| 레이어                                                         | 출력 형태 |
| -------------------------------------------------------------- | --------- |
| Conv1D(32, 1) + AveragePooling1D(2) + BN + ReLU + Dropout(0.2) | (100, 32) |
| Conv1D(1, 1) 공간 요약                                         | (100, 1)  |
| GlobalAveragePooling1D 채널 요약                               | (32,)     |
| Reshape (32,) → (32, 1)                                        | (32, 1)   |
| Concatenate [(100,1), (32,1)]                                  | (132, 1)  |

## 6. 공유 사영 및 채널 분기

두 브랜치 출력을 통합하고 SBP/DBP 경로로 분리합니다.

```text
Branch S (132,1) ┐
                 ├── Concatenate(axis=-1) → (132, 2)
Branch L (132,1) ┘
                 │
       Conv1D(32, kernel=1) → (132, 32)   ← 공유 표현
                 │
       Conv1D(48, kernel=1) → (132, 48)   ← 비대칭 사영
                 │
     ┌───────────┴───────────┐
  ch[:32]                 ch[32:]
  (132, 32)               (132, 16)
  SBP 경로                DBP 경로
```

비대칭 채널 사영에서 `groups`를 사용하지 않는 이유: Keras `Conv1D`의 `groups` 파라미터는 입력 채널과 출력 필터를 균등 분할해야 하므로 32:16의 비대칭 분배에 사용할 수 없습니다. 대신 groups 없는 `Conv1D(48, 1)`로 사영 후 슬라이스합니다.

## 7. SBP/DBP 어텐션 헤드

채널 분기 이후 SBP와 DBP는 각각 독립적인 Temporal Self-Attention + 회귀 헤드를 통과합니다.

### 7.1 SBP 어텐션 헤드 — (132, 32)

```text
(132, 32)
    │
┌───▼────────────────────┐
│  Multi-Head Attention   │  num_heads=4, key_dim=8, value_dim=8
│  Q = K = V = (132, 32)  │
│  → (132, 32)            │
└───┬────────────────────┘
    │ Add & LayerNorm
    │
┌───▼────────────────────┐
│  Feed-Forward Network   │
│  Dense(64, relu)        │
│  Dense(32)              │
└───┬────────────────────┘
    │ Add & LayerNorm
    ▼
 (132, 32)
```

| 항목                        | 값  |
| --------------------------- | --- |
| 헤드 수 (`num_heads`)       | 4   |
| 헤드별 차원 (`key_dim`)     | 8   |
| FFN 내부 차원               | 64  |
| Dropout (attention weights) | 0.1 |

### 7.2 DBP 어텐션 헤드 — (132, 16)

```text
(132, 16)
    │
┌───▼────────────────────┐
│  Multi-Head Attention   │  num_heads=2, key_dim=8, value_dim=8
│  Q = K = V = (132, 16)  │
│  → (132, 16)            │
└───┬────────────────────┘
    │ Add & LayerNorm
    │
┌───▼────────────────────┐
│  Feed-Forward Network   │
│  Dense(32, relu)        │
│  Dense(16)              │
└───┬────────────────────┘
    │ Add & LayerNorm
    ▼
 (132, 16)
```

| 항목                        | 값  |
| --------------------------- | --- |
| 헤드 수 (`num_heads`)       | 2   |
| 헤드별 차원 (`key_dim`)     | 8   |
| FFN 내부 차원               | 32  |
| Dropout (attention weights) | 0.1 |

## 8. 회귀 헤드

### 8.1 SBP 회귀 헤드

```text
(132, 32)
    │
    ├── Conv1D(1, 1) → (132, 1)          공간 요약
    └── GlobalAveragePooling1D → (32,)
        Reshape → (32, 1)                 채널 요약
    │
    Concatenate [(132,1), (32,1)] → (164, 1)
    Flatten → (164,)
    Dropout(0.2)
    Dense(128, relu)
    Dense(64, relu)
    Dense(32, relu)
    Dense(1, linear) → SBP (mmHg)
```

### 8.2 DBP 회귀 헤드

```text
(132, 16)
    │
    ├── Conv1D(1, 1) → (132, 1)          공간 요약
    └── GlobalAveragePooling1D → (16,)
        Reshape → (16, 1)                 채널 요약
    │
    Concatenate [(132,1), (16,1)] → (148, 1)
    Flatten → (148,)
    Dropout(0.2)
    Dense(64, relu)
    Dense(32, relu)
    Dense(1, linear) → DBP (mmHg)
```

## 9. 파라미터 요약

### 9.1 공유 백본

| 모듈                               | 파라미터 수 | 비고                       |
| ---------------------------------- | ----------- | -------------------------- |
| Branch S (Block1~3 + SE + Post-SE) | 24,327      |                            |
| Branch L (Block1~3 + SE + Post-SE) | 24,688      | 장기 커널로 DW params 증가 |
| 브랜치 병합 Conv1D(32, 1)          | 96          | 2×32 + 32                  |
| 비대칭 사영 Conv1D(48, 1)          | 1,584       | 32×48 + 48                 |
| **공유 백본 합계**                 | **50,695**  |                            |

### 9.2 SBP 헤드

| 모듈                                       | 파라미터 수 | 비고 |
| ------------------------------------------ | ----------- | ---- |
| MHA (Q/K/V/O 사영, num_heads=4, key_dim=8) | 4,224       |      |
| FFN (Dense 64+32) + LayerNorm ×2           | 4,320       |      |
| 공간 요약 Conv1D(1,1)                      | 33          |      |
| 회귀 헤드 (164→128→64→32→1)                | 31,522      |      |
| **SBP 헤드 합계**                          | **40,099**  |      |

### 9.3 DBP 헤드

| 모듈                                       | 파라미터 수 | 비고 |
| ------------------------------------------ | ----------- | ---- |
| MHA (Q/K/V/O 사영, num_heads=2, key_dim=8) | 1,088       |      |
| FFN (Dense 32+16) + LayerNorm ×2           | 1,136       |      |
| 공간 요약 Conv1D(1,1)                      | 17          |      |
| 회귀 헤드 (148→64→32→1)                    | 11,649      |      |
| **DBP 헤드 합계**                          | **13,890**  |      |

### 9.4 전체 요약

| 구분          | 파라미터 수 | 비율  |
| ------------- | ----------- | ----- |
| 공유 백본     | 50,695      | 48.4% |
| SBP 헤드      | 40,099      | 38.3% |
| DBP 헤드      | 13,890      | 13.3% |
| **모델 합계** | **104,684** | 100%  |

SBP 헤드 : DBP 헤드 파라미터 비율 ≈ **2.9 : 1**

## 10. 학습 설정

### 10.1 기본 설정

| 항목            | 값                               |
| --------------- | -------------------------------- |
| 프레임워크      | TensorFlow 2.15.1 / Keras 2.15.0 |
| 옵티마이저      | Adam (lr=0.001)                  |
| 데이터셋        | VitalDB (PPG 열만 사용)          |
| 출력 범위 (SBP) | 60–180 mmHg                      |
| 출력 범위 (DBP) | 40–110 mmHg                      |

### 10.2 다중 출력 손실 설정

단일 모델에서 두 출력을 동시에 학습합니다. SBP 추정이 더 어렵기 때문에 SBP 손실에 더 높은 가중치를 부여합니다.

```python
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={'sbp': 'huber', 'dbp': 'huber'},
    loss_weights={'sbp': 1.5, 'dbp': 1.0}
)
```

| 항목            | 값                            |
| --------------- | ----------------------------- |
| 손실 함수       | Huber loss                    |
| SBP 손실 가중치 | 1.5                           |
| DBP 손실 가중치 | 1.0                           |
| 체크포인트      | val_loss 기준 best model 저장 |

## 11. 배포 형식

| 형식               | 경로                                     | 사용처                | 변환 도구                            |
| ------------------ | ---------------------------------------- | --------------------- | ------------------------------------ |
| `.h5` (Keras)      | `data/model/cf/`                         | Python 실험·학습 원본 | —                                    |
| `.tflite` (LiteRT) | `app-android/src/main/assets/models/cf/` | Android 앱            | `tools-python/h5_to_tflite.py`       |
| `.onnx`            | `result/model/cf/`                       | Desktop 워크벤치      | `tools-python/convert_h5_to_onnx.py` |

Temporal Self-Attention은 표준 Keras `MultiHeadAttention` 레이어를 사용하므로 TFLite/ONNX 변환 시 별도 처리가 필요 없습니다.

## 12. BPNet과의 비교 분석

### 12.1 아키텍처 비교

| 항목                 | BPNet (보정 기반)                                       | BPNet-CF (비보정)                      |
| -------------------- | ------------------------------------------------------- | -------------------------------------- |
| 입력                 | 4개 (`init_ppg`, `init_bp`, `curr_ppg`, `curr_feature`) | 1개 (`curr_ppg`)                       |
| 출력                 | 모델별 1개 (SBP 모델 / DBP 모델 분리)                   | 1개 모델에서 2개 (SBP + DBP 동시 출력) |
| PPG 인코더 브랜치    | 2개 (init/curr, 가중치 비공유)                          | 2개 (단기/장기 스케일, 가중치 비공유)  |
| 수작업 특징 네트워크 | O (29-dim → feat_s/d)                                   | X                                      |
| 관계 모듈            | R1, R2, R3 (합계 ~99K params)                           | X                                      |
| SBP/DBP 헤드 분리    | 완전 분리 (독립 모델)                                   | 공유 백본 + 비대칭 헤드 (단일 모델)    |
| 시간적 자기 주의     | X                                                       | O (SBP 4-head / DBP 2-head 비대칭)     |
| 파라미터 수          | 209,609 × 2모델 = 419,218                               | 104,684 (단일 모델, 약 75% 감소)       |
| 보정 측정 필요 여부  | 필요 (초기 PPG + 혈압)                                  | 불필요                                 |
| 개인화 Fine-tuning   | 지원 (레이어 136~ 재학습)                               | 해당 없음                              |

### 12.2 성능 비교 실험 설계

| 조건          | 값                                           |
| ------------- | -------------------------------------------- |
| 테스트 데이터 | VitalDB 동일 분할                            |
| 평가 지표     | MAE, RMSE (mmHg), SBP/DBP 각각               |
| 기준 표준     | AAMI/ISO 81060-2 (MAE <=5 mmHg, SD <=8 mmHg) |

### 12.3 예상 성능 차이 원인

BPNet-CF가 BPNet 대비 오차가 클 것으로 예상되는 주요 원인:

1. **개인 간 변동성 미반영**: 동일한 PPG 형태라도 개인마다 절대 혈압이 다를 수 있으며, 이를 보정할 기준점이 없어 집단 평균 통계에만 의존합니다.
2. **절대 혈압 기준점 부재**: 개인별 BP-PPG 대응 관계를 앵커링하는 정보(init_bp)가 없습니다.
3. **명시적 타이밍 특징 부재**: 수작업으로 계산된 PPI·맥파 폭 특징은 노이즈에 강인하지만, CNN/어텐션이 자동 학습한 특징은 이보다 민감할 수 있습니다.
