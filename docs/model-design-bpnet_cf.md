# BPNet-CF 모델 상세 설계서

## 1. 개요

BPNet-CF는 기존의 보정 기반 혈압 추정 모델을 프로젝트에 맞게 재정의한
설계 문서이다.

- **목표**: 단일 PPG 세그먼트만으로 `SBP`, `DBP`를 동시에 추정
- **프레임워크**: `TensorFlow/Keras` 대신 `PyTorch`
- **프로젝트 인터페이스 적합화**
  - 입력: `(B, 1000)` 또는 `(B, 1, 1000)`
  - 출력: `(B, 2)` 단일 텐서
  - 레지스트리 등록형 `nn.Module`
- **구현 대상 파일(권장)**: `bpe/models/bpnet_cf.py`
- **등록명(권장)**: `bpnet_cf`

이 프로젝트의 기본 데이터셋은 `125 Hz × 8초 = 1000 samples` 길이의 PPG 세그먼트를
사용하므로, 기존 모델의 `100 Hz × 8초 = 800 samples` 입력은 그대로 쓰지 않는다.

## 2. 기존 모델 설계에서 바뀌는 점

| 항목            | 기존 모델                | BPNet-CF                                                |
| --------------- | ------------------------ | ------------------------------------------------------- |
| 프레임워크      | TensorFlow 2.x / Keras   | PyTorch                                                 |
| 모델 이름       | BPNet-CF                 | BPNet-CF                                                |
| 입력 shape      | `(N, 800, 1)`            | `(B, 1000)` 또는 `(B, 1, 1000)`                         |
| 내부 텐서 축    | `(batch, time, channel)` | `(batch, channel, time)`                                |
| 출력 형식       | `sbp`, `dbp` 2개 텐서    | `[SBP, DBP]` 단일 텐서 `(B, 2)`                         |
| 데이터 샘플링   | 100 Hz                   | 125 Hz                                                  |
| 체크포인트 예시 | `.h5`                    | `best.pt`, `last.pt`                                    |
| 학습 진입점     | Keras `model.fit()`      | `uv run python scripts/train-model.py --model bpnet_cf` |

핵심 구조 자체는 유지하되, 구현 방식과 입출력 규약은 저장소의 기존 모델들과 동일한
관례를 따른다.

## 3. 입력 및 출력 사양

### 3.1 입력

| 항목 | 이름 | shape                           | dtype     | 설명                  |
| ---- | ---- | ------------------------------- | --------- | --------------------- |
| 입력 | `x`  | `(B, 1000)` 또는 `(B, 1, 1000)` | `float32` | 8초 길이 PPG 세그먼트 |

- `B`: 배치 크기
- 길이 `1000 = 125 Hz × 8 sec`
- `forward()` 시작 시 `ensure_3d()`와 동일한 방식으로 `(B, 1, 1000)`으로 정규화하는
  구현을 권장한다.

### 3.2 출력

| 항목 | 이름    | shape    | dtype     | 설명                           |
| ---- | ------- | -------- | --------- | ------------------------------ |
| 출력 | `y_hat` | `(B, 2)` | `float32` | `[:, 0] = SBP`, `[:, 1] = DBP` |

이 프로젝트의 `PPGDataset`은 타깃을 이미 `(N, 2)` 형태의 `[SBP_mean, DBP_mean]`
으로 제공하므로, 모델도 여기에 맞춰 단일 회귀 텐서를 반환해야 한다.

## 4. 전처리 가정

본 모델은 별도 수작업 특징 없이 기본 파이프라인에 따른 전처리된 **PPG 1채널**을 직접 입력으로 받는다.

## 5. 전체 아키텍처

```text
입력 PPG
   (B, 1000) or (B, 1, 1000)
            │
            ▼
        ensure_3d
            │
            ▼
       (B, 1, 1000)
            │
┌───────────┴──────────────┐
│                          │
│   Short-scale encoder    │
│   k = [5, 5, 3]          │
│                          │
│   Long-scale encoder     │
│   k = [15, 11, 7]        │
│                          │
└───────────┬──────────────┘
            │
            ▼
    branch features: (B, 1, 157) + (B, 1, 157)
            │
            ▼
    concat along channel
            │
            ▼
        (B, 2, 157)
            │
            ▼
    shared 1x1 projection      2 → 32
            │
            ▼
        (B, 32, 157)
            │
            ▼
    asymmetric 1x1 projection 32 → 48
            │
            ├── SBP path: first 32 ch  → (B, 32, 157)
            └── DBP path: last  16 ch  → (B, 16, 157)
                         │
                         ▼
          independent temporal self-attention
                         │
                         ▼
            independent regression heads
                         │
                         ├── SBP scalar: (B, 1)
                         └── DBP scalar: (B, 1)
                         │
                         ▼
               concat -> (B, 2) = [SBP, DBP]
```

## 6. 듀얼 스케일 PPG 인코더

두 브랜치는 같은 stage 구성을 공유하고 커널 크기만 다르다.

### 6.1 브랜치별 stage 구성

| Stage | Short branch | Long branch | 입력 shape     | 출력 shape      |
| ----- | ------------ | ----------- | -------------- | --------------- |
| 1     | DWConv k=5   | DWConv k=15 | `(B, 1, 1000)` | `(B, 32, 500)`  |
| 2     | DWConv k=5   | DWConv k=11 | `(B, 32, 500)` | `(B, 64, 250)`  |
| 3     | DWConv k=3   | DWConv k=7  | `(B, 64, 250)` | `(B, 128, 250)` |

각 stage의 공통 패턴:

```text
Depthwise Conv1d
→ Pointwise Conv1d(1x1)
→ BatchNorm1d
→ ReLU
→ (stage 1, 2 only) MaxPool1d(2)
```

- Stage 1: `1000 → 500`
- Stage 2: `500 → 250`
- Stage 3: 길이 유지 `250 → 250`

### 6.2 설계 의도

- **Short branch**는 개별 맥파의 국소 형태, 피크 주변 기울기, notch 같은 짧은 시간
  패턴에 민감하다.
- **Long branch**는 박동 간 간격, 수축-이완 리듬, 더 넓은 문맥을 직접 수용한다.

즉, 하나의 커널 크기로는 놓치기 쉬운 PPG의 다중 시간 스케일 정보를 병렬로 포착한다.

## 7. SE 기반 브랜치 후처리

각 브랜치의 Stage 3 출력 `(B, 128, 250)`에 채널 재가중을 적용한다.

### 7.1 SE 블록

```text
(B, 128, 250)
    │
    ▼  AdaptiveAvgPool1d(1)
(B, 128, 1)
    │
    ▼  Flatten
(B, 128)
    │
    ▼  Linear(128 → 32) + ReLU
    ▼  Linear(32 → 128) + Sigmoid
(B, 128)
    │
    ▼  reshape to (B, 128, 1)
    ▼  channel-wise multiply
(B, 128, 250)
```

- 축소 비율: `4`
- 목적: 혈압 추정에 더 유용한 채널을 강조

### 7.2 브랜치별 post-SE 요약

기존 모델의 구조를 유지하되 PyTorch 축 순서에 맞춰 다음과 같이 구현한다.

```text
(B, 128, 250)
  → Conv1d(128 → 32, kernel=1)
  → AvgPool1d(2)                  -> (B, 32, 125)
  → BatchNorm1d(32) + ReLU + Dropout(0.2)

두 개의 요약 경로
  1. spatial summary:
     Conv1d(32 → 1, kernel=1)     -> (B, 1, 125)
  2. channel summary:
     AdaptiveAvgPool1d(1)         -> (B, 32, 1)

두 결과를 시간축 기준으로 펼쳐 붙이면
  125 + 32 = 157 길이의 1채널 표현으로 재구성
```

PyTorch 텐서 축은 `(B, C, L)`이므로 다음과 같이 구현한다.

```text
spatial: (B, 1, 125)
channel: (B, 32, 1) -> reshape -> (B, 1, 32)
concat along time -> (B, 1, 157)
```

## 8. 공유 사영 및 비대칭 채널 분기

두 브랜치의 요약 출력을 통합해 공통 표현을 만든 뒤, SBP/DBP 경로로 나눈다.

```text
short branch : (B, 1, 157)
long branch  : (B, 1, 157)
      │
      ▼ concat(channel)
   (B, 2, 157)
      │
      ▼ Conv1d(2 → 32, kernel=1)
   (B, 32, 157)
      │
      ▼ Conv1d(32 → 48, kernel=1)
   (B, 48, 157)
      │
      ├── sbp_feat = x[:, :32, :]   -> (B, 32, 157)
      └── dbp_feat = x[:, 32:, :]   -> (B, 16, 157)
```

비대칭 분기의 의도는 동일하다.

- SBP는 보통 DBP보다 변동 폭이 크고 추정 난도가 높다.
- 따라서 표현 차원을 `32 : 16`으로 나눠 SBP 쪽에 더 많은 용량을 할당한다.

## 9. SBP/DBP Temporal Self-Attention 헤드

각 분기에는 독립적인 attention encoder를 둔다.

### 9.1 텐서 배치 규약

PyTorch `nn.MultiheadAttention`은 기본적으로 `(B, T, C)`를 다루기 위해
`batch_first=True` 사용을 권장한다.

따라서 분기 특징은 attention 직전에 transpose 한다.

```text
SBP path: (B, 32, 157) -> (B, 157, 32)
DBP path: (B, 16, 157) -> (B, 157, 16)
```

### 9.2 SBP attention block

```text
Input             : (B, 157, 32)
MultiheadAttention: embed_dim=32, num_heads=4, dropout=0.1
Residual + LayerNorm
FFN: Linear(32→64) → ReLU → Linear(64→32)
Residual + LayerNorm
Output            : (B, 157, 32)
```

### 9.3 DBP attention block

```text
Input             : (B, 157, 16)
MultiheadAttention: embed_dim=16, num_heads=2, dropout=0.1
Residual + LayerNorm
FFN: Linear(16→32) → ReLU → Linear(32→16)
Residual + LayerNorm
Output            : (B, 157, 16)
```

## 10. 회귀 헤드

어텐션 블록 이후 텐서는 `(B, T, C)` 포맷이다. 회귀 헤드는 `nn.Linear`를 직접
사용하므로 `(B, C, T)` 포맷으로의 transpose가 불필요하다.

각 분기는 독립적으로 스칼라 혈압으로 회귀한다.

### 10.1 SBP 회귀 헤드

```text
(B, 157, 32)
  ├─ time summary    : Linear(32→1) applied per token -> (B, 157, 1)
  ├─ channel summary : mean over time                 -> (B, 32)
  └─ flatten and concat -> (B, 189)
        ↓
     Dropout(0.2)
     Linear(189 → 128) + ReLU
     Linear(128 → 64)  + ReLU
     Linear(64 → 32)   + ReLU
     Linear(32 → 1)
```

### 10.2 DBP 회귀 헤드

```text
(B, 157, 16)
  ├─ time summary    : Linear(16→1) applied per token -> (B, 157, 1)
  ├─ channel summary : mean over time                 -> (B, 16)
  └─ flatten and concat -> (B, 173)
        ↓
     Dropout(0.2)
     Linear(173 → 64) + ReLU
     Linear(64 → 32)  + ReLU
     Linear(32 → 1)
```

### 10.3 최종 출력 조립

```text
sbp_hat : (B, 1)
dbp_hat : (B, 1)
torch.cat([sbp_hat, dbp_hat], dim=1) -> (B, 2)
```

이 출력 형식은 `train-model.py`, `eval-model.py`, `print-model.py`의 현재 기대와
직접 호환된다.

## 11. 권장 PyTorch 구현 골격

```python
@register_model("bpnet_cf")
class BPNetCF(nn.Module):
    def __init__(self):
        super().__init__()
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_3d(x)                 # (B, L) -> (B, 1, L)
        short = self.short_encoder(x)
        long = self.long_encoder(x)
        fused = torch.cat([short, long], dim=1)
        shared = self.shared_proj(fused)
        split = self.asym_proj(shared)
        sbp_feat = split[:, :32, :]
        dbp_feat = split[:, 32:, :]
        sbp = self.sbp_head(sbp_feat)
        dbp = self.dbp_head(dbp_feat)
        return torch.cat([sbp, dbp], dim=1)
```

구현 시 재사용 가능한 공통 블록:

- `ensure_3d()` from `bpe/models/blocks.py`
- `register_model()` from `bpe/models/registry.py`
- depthwise separable block은 `conv_reg_ds.py` 스타일을 재사용 가능

## 12. 손실 함수와 학습 전략

이 프로젝트의 학습기(`Trainer`)는 기본적으로 **단일 출력 텐서**에 대해 동작하므로,
BPNet-CF도 별도 멀티아웃풋 래퍼 없이 학습할 수 있어야 한다.

권장 손실은 다음 두 가지 중 하나다.

### 12.1 기본 권장안: weighted Huber

```text
loss = 1.5 * Huber(pred[:, 0], target[:, 0])
     + 1.0 * Huber(pred[:, 1], target[:, 1])
```

이유:

- 레거시 설계의 SBP 우선 학습 의도를 유지할 수 있다.
- 이상치에 MSE보다 강인하다.

### 12.2 단순 시작안: MAE 또는 SmoothL1

처음 구현을 단순하게 가져가려면 `(B, 2)` 전체에 대해 `nn.SmoothL1Loss()`를 바로
적용해도 된다. 이후 필요하면 SBP/DBP 가중치를 추가한다.

## 13. 예상 장점과 리스크

### 13.1 장점

- 단일 PPG만으로 동작하는 calibration-free 구조
- 다중 시간 스케일 인코딩으로 local/global morphology 동시 반영
- 공유 백본으로 파라미터 효율 확보
- SBP/DBP 분기 이후 독립 attention으로 타깃별 표현 분리

### 13.2 리스크

- 구조가 현재 baseline들보다 복잡해 작은 데이터 분할에서 과적합 가능성 존재
- branch 후처리의 `157` 길이 재구성은 레거시 구조를 1000샘플 입력에 맞춘
  근사 변환이므로, 실제 구현 후 ablation이 필요하다
- attention/SE/비대칭 헤드가 모두 들어가므로 `conv_reg`, `resnet1d`류보다
  학습 안정성 튜닝 비용이 높을 수 있다

## 14. 구현 체크리스트

1. `bpe/models/bpnet_cf.py` 생성
2. `@register_model("bpnet_cf")` 추가
3. `bpe/models/__init__.py`에서 import/export 연결
4. `uv run python scripts/print-model.py --model bpnet_cf`로 shape 확인
5. `uv run python scripts/train-model.py --model bpnet_cf`로 학습 진입 확인
6. 필요 시 weighted Huber 손실 지원을 `Trainer` 또는 train script에 추가

## 15. 문서상의 가정

이 설계 문서는 아래 가정을 바탕으로 작성했다.

1. 새 모델도 이 저장소의 다른 회귀 모델과 동일하게 **입력 1개, 출력 1개(`(B, 2)`)**
   규약을 따라야 한다.
2. 데이터셋 기본 세그먼트 길이는 `1000`샘플이며, 레거시의 `800`샘플 구조는 그대로
   복제하지 않고 프로젝트 기준 길이에 맞게 조정해야 한다.
3. 우선 목표는 **문서화 가능한 구현 설계**이며, 실제 파라미터 수는 코드 작성 후
   `print-model.py` 결과로 최종 확정하는 것이 적절하다.
