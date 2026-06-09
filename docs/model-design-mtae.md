# MTAE 모델 상세 설계서

## 1. 개요

MTAE(Multi-Task AutoEncoder)는 PPG 신호 **재구성(reconstruction)** 과
**혈압 회귀(regression)** 를 하나의 CNN 인코더에서 동시에 수행하는 멀티태스크
학습 모델이다.

- **구현 파일**: [`bpe/models/mtae.py`](../bpe/models/mtae.py)
- **모델 등록명**: `mtae`
- **Transformer 기반 변형**: [`docs/model-design-mtae_tr.md`](model-design-mtae_tr.md) 참조

멀티태스크 학습 인터페이스:

- **`forward(x)`**: 인코더 → BP 헤드만 실행. 추론 및 표준 평가용.
- **`compute_loss(x, y, criterion)`**: 인코더 → 디코더 + BP 헤드 동시 실행.
  Trainer가 `hasattr(model, "compute_loss")`로 자동 감지해 호출한다.

### 멀티태스크 손실 공식

```text
loss = (1 - recon_weight) × bp_loss + recon_weight × recon_loss
```

- `bp_loss`: `criterion(pred, y)` — BP 예측 오차
- `recon_loss`: `criterion(recon, x)` — PPG 재구성 오차
- 기본값 `recon_weight=0.5`: 두 손실의 동등 가중 합산

## 2. 전체 아키텍처

```text
입력: PPG 세그먼트
                (B, 1000)  또는  (B, 1, 1000)
                          │
                          ▼  ensure_3d
                     (B, 1, 1000)
                          │
┌─────────────────────────┴──────────────────────────┐
│  _Encoder                                          │
│  ConvBnAct1d(1→32,  k=7, s=2)   → (B, 32, 500)     │
│  ConvBnAct1d(32→64, k=7, s=2)   → (B, 64, 250)     │
│  ConvBnAct1d(64→128, k=5, s=2)  → (B, 128, 125)    │
│  AdaptiveAvgPool1d(1) + Flatten → (B, 128)         │
│  Linear(128 → latent_dim)                          │
│  Sigmoid                        → (B, latent_dim)  │
└─────────────────────────┬──────────────────────────┘
                  (B, latent_dim=16)
        ┌─────────────────┴──────────────┐
        │ BP Head                        │ Decoder (훈련 시)
 Linear(16→2)                  ┌─────────┴──────────────────┐
        │                      │  _Decoder                  │
     (B, 2)                    │  Linear(16→128)            │
  [SBP, DBP]                   │  unsqueeze(-1)             │
                               │  → (B, 128, 1)             │
                               │  Upsample(125)             │
                               │  ConvBnAct1d(128→64, k=5)  │
                               │  Upsample(250)             │
                               │  ConvBnAct1d(64→32, k=7)   │
                               │  Upsample(500)             │
                               │  ConvBnAct1d(32→16, k=7)   │
                               │  Upsample(1000)            │
                               │  Conv1d(16→1, k=7)         │
                               └─────────┬──────────────────┘
                                    (B, 1, 1000)
                                    [재구성 PPG]
```

## 3. 텐서 흐름 요약

### _Encoder (latent_dim=16 기준)

| 단계 | 처리                           | 입력 shape    | 출력 shape    |
| ---- | ------------------------------ | ------------- | ------------- |
| 0    | ensure_3d                      | (B, 1000)     | (B, 1, 1000)  |
| 1    | ConvBnAct1d(1→32, k=7, s=2)    | (B, 1, 1000)  | (B, 32, 500)  |
| 2    | ConvBnAct1d(32→64, k=7, s=2)   | (B, 32, 500)  | (B, 64, 250)  |
| 3    | ConvBnAct1d(64→128, k=5, s=2)  | (B, 64, 250)  | (B, 128, 125) |
| 4    | AdaptiveAvgPool1d(1) + flatten | (B, 128, 125) | (B, 128)      |
| 5    | Linear(128→16) + Sigmoid       | (B, 128)      | (B, 16)       |

### _Decoder (latent_dim=16 기준)

| 단계 | 처리                     | 입력 shape    | 출력 shape    |
| ---- | ------------------------ | ------------- | ------------- |
| 0    | Linear(16→128)           | (B, 16)       | (B, 128)      |
| 1    | unsqueeze(-1)            | (B, 128)      | (B, 128, 1)   |
| 2    | Upsample(125)            | (B, 128, 1)   | (B, 128, 125) |
| 3    | ConvBnAct1d(128→64, k=5) | (B, 128, 125) | (B, 64, 125)  |
| 4    | Upsample(250)            | (B, 64, 125)  | (B, 64, 250)  |
| 5    | ConvBnAct1d(64→32, k=7)  | (B, 64, 250)  | (B, 32, 250)  |
| 6    | Upsample(500)            | (B, 32, 250)  | (B, 32, 500)  |
| 7    | ConvBnAct1d(32→16, k=7)  | (B, 32, 500)  | (B, 16, 500)  |
| 8    | Upsample(1000)           | (B, 16, 500)  | (B, 16, 1000) |
| 9    | Conv1d(16→1, k=7, pad=3) | (B, 16, 1000) | (B, 1, 1000)  |

## 4. 디코더 설계: Upsample + Conv 패턴

디코더가 전치합성곱(ConvTranspose1d) 대신 **Upsample(nearest) + ConvBnAct1d**
조합을 사용하는 이유:

전치합성곱은 업샘플링 보폭에 따라 출력에 격자 무늬 아티팩트(checkerboard
artifact)가 발생하는 것으로 알려져 있다. nearest-neighbor 업샘플링 후
일반 합성곱을 적용하면 이 문제를 피하면서 더 부드러운 재구성 결과를 얻는다.

마지막 레이어(Conv1d(16→1, k=7))에는 BN·ReLU가 없다. 재구성 출력은 원본
PPG와 동일한 값 범위(정규화된 실수)를 가져야 하므로 비선형 활성화를 적용하지
않는다.

## 5. 하이퍼파라미터

| 파라미터       | 기본값 | 역할                                      |
| -------------- | ------ | ----------------------------------------- |
| `latent_dim`   | 16     | 시그모이드 병목 잠재 벡터 차원            |
| `recon_weight` | 0.5    | 재구성 손실 가중치 (0: BP만, 1: 재구성만) |

## 6. compute_loss 훈련 인터페이스

### 6.1 Trainer 연동 프로토콜

```python
# Trainer (bpe/train/trainer.py:165)
if hasattr(self.model, "compute_loss"):
    loss, pred = self.model.compute_loss(x, y, self.criterion)
else:
    pred = self.model(x)
    loss = self.criterion(pred, y)
```

`compute_loss`가 있으면 모델이 손실 계산을 직접 담당한다.
`forward()`는 추론 경로만 실행하므로 평가 속도에 영향을 주지 않는다.

### 6.2 compute_loss 흐름

```python
def compute_loss(self, x, y, criterion):
    x3d = ensure_3d(x)
    z    = self.encoder(x3d)          # (B, latent_dim)
    pred  = self.bp_head(z)           # (B, 2)
    recon = self.decoder(z)           # (B, 1, 1000)

    bp_loss    = criterion(pred, y)      # 스칼라
    recon_loss = criterion(recon, x3d)   # 스칼라

    loss = (1.0 - self.recon_weight) * bp_loss
         +         self.recon_weight  * recon_loss
    return loss, pred
```

### 6.3 forward()와 compute_loss()의 실행 경로 차이

| 상황                              | 실행 경로                   | 디코더 실행 |
| --------------------------------- | --------------------------- | ----------- |
| `model(x)` — 추론 / `print-model` | encoder → bp_head           | 없음        |
| `compute_loss` — 훈련             | encoder → bp_head + decoder | 있음        |
| `compute_loss` — 검증             | encoder → bp_head + decoder | 있음        |

디코더는 훈련과 검증 모두에서 실행된다. 검증 손실 계산에도 재구성 오차가
포함되어야 올바른 `recon_weight` 튜닝이 가능하기 때문이다.

### 6.4 시그모이드 병목의 역할

잠재 벡터를 `Sigmoid`로 활성화해 **[0, 1] 범위로 제한**한다.

일반 AE는 잠재 공간에 경계가 없어 훈련 초기에 폭발적 활성화가 발생할 수 있다.
Sigmoid 병목은:

- 잠재 공간을 하이퍼큐브 `[0, 1]^d` 안에 강제 → 안정적 학습
- BP 헤드(단순 Linear)가 균일한 범위의 입력을 받게 됨
- 재구성 손실과 BP 손실의 그래디언트가 비슷한 스케일로 유지됨

## 7. 훈련 방법

### 기본 훈련

```bash
bin\train-model.bat --model mtae
```

### recon_weight 조정

```bash
# 재구성보다 혈압 회귀에 집중 (recon_weight=0.2)
bin\train-model.bat --model mtae \
    --model-kwargs "recon_weight=0.2"

# 재구성 보조 없이 순수 회귀 (사실상 단일태스크)
bin\train-model.bat --model mtae \
    --model-kwargs "recon_weight=0.0"
# ※ recon_weight=0 이면 디코더는 그래디언트를 받지 못해 의미가 없음
```

### latent_dim 조정

```bash
# 더 넓은 병목
bin\train-model.bat --model mtae --model-kwargs "latent_dim=64"
```

## 8. 모델 검사

```bash
bin\print-model.bat --model mtae
```

출력 예시:

```text
MTAE
  (encoder): _Encoder
    (conv): Sequential
      (0): ConvBnAct1d(1→32,  k=7, s=2)
      (1): ConvBnAct1d(32→64, k=7, s=2)
      (2): ConvBnAct1d(64→128, k=5, s=2)
    (pool): AdaptiveAvgPool1d(1)
    (fc): Linear(128→16)
  (decoder): _Decoder
    (fc): Linear(16→128)
    (up): Sequential
      Upsample(125) → ConvBnAct1d(128→64) → Upsample(250) → ...
      Conv1d(16→1, k=7)
  (bp_head): Linear(16→2)

Total params    : <N>
Input shape     : (1, 1000)
```

## 9. 참고 문헌

- Odena, A., Dumoulin, V., and Olah, C. (2016). "Deconvolution and
  Checkerboard Artifacts." *Distill*.
  (Upsample + Conv 패턴의 근거)
