# PulseWOQResNet1D 모델 상세 설계서

## 1. 개요

PulseWOQResNet1D(Pulse With Overlapping & Quality supervision ResNet 1D)는
8초 PPG 신호를 **중첩 1초 세그먼트**로 분할하고, 공유 PulseBackbone으로 각
세그먼트의 혈압과 품질 로짓을 동시에 예측한 뒤, 품질 점수 가중 평균으로 최종
혈압을 산출하는 모델이다.

훈련 시 별도의 품질 레이블 없이 **세그먼트별 BP 예측 오차에서 품질 목표값을
자동 유도**하여 품질 헤드를 함께 지도 학습한다.

- **구현 파일**: [`bpe/models/pulsewoq_resnet1d.py`](../bpe/models/pulsewoq_resnet1d.py)
- **모델 등록명**: `pulsewoq_resnet1d`
- **백본 의존**: [`bpe/models/pulse_resnet1d.py`](../bpe/models/pulse_resnet1d.py)의 `PulseBackbone`

### 공개 인터페이스

| 메서드                     | 입력                    | 출력                       | 용도                     |
| -------------------------- | ----------------------- | -------------------------- | ------------------------ |
| `forward(x)`               | (B, L) 또는 (B, 1, L)   | (B, 2) [SBP, DBP]          | 추론                     |
| `forward_with_quality(x)`  | (B, L) 또는 (B, 1, L)   | (B, 3) [SBP, DBP, q∈(0,1)] | 신호 품질 평가 포함 추론 |
| `compute_loss(x, y, crit)` | (B, L), (B, 2), loss fn | (loss, (B, 2))             | Trainer 자동 호출        |

## 2. 모델 계보

```text
PulseResNet1D           PulseWOResNet1D         PulseWOQResNet1D
─────────────────       ────────────────        ─────────────────────
비중첩 세그먼트          중첩 세그먼트            중첩 세그먼트
단순 평균 집계           소프트맥스 품질          소프트맥스 품질
품질 헤드 없음           가중 집계                가중 집계
                         품질 지도 학습 없음      품질 지도 학습 (자동)
```

| 항목          | PulseResNet1D             | PulseWOQResNet1D           |
| ------------- | ------------------------- | -------------------------- |
| 세그먼트 분할 | reshape (비중첩, 고정 수) | unfold (중첩, 고정 길이)   |
| 세그먼트 수   | 8 (기본)                  | 15 (1000샘플 기준, 계산됨) |
| 세그먼트 길이 | L // num_segments = 125   | seg_len=125 고정           |
| 집계 방법     | 단순 평균                 | softmax(q) 가중 평균       |
| 백본 출력     | F=2 (SBP, DBP)            | F+1=3 (SBP, DBP, q_logit)  |
| 품질 지도     | 없음                      | exp(-MAE/temp) 자동 유도   |

## 3. 전체 아키텍처

```text
입력: PPG 8초 세그먼트
      (B, L=1000)  또는  (B, 1, L=1000)
                   │
                   ▼  ensure_3d
             (B, 1, 1000)
                   │
                   ▼  unfold(dim=2, size=125, step=62)
         (B, 1, S=15, 125)
                   │
                   ▼  permute + view
         (B*15, 1, 125)
                   │
    ┌──────────────┴───────────────────────────────┐
    │  PulseBackbone(in=1, out=3, C=16)            │
    │                                              │
    │  stem:  ConvBnAct1d(1→16, k=7, s=2)          │
    │         MaxPool1d(k=3, s=2, pad=1)           │
    │         (B*15, 1, 125) → (B*15, 16, 32)      │
    │                                              │
    │  stage1: BasicBlock1D(16→16, s=1)            │
    │         (B*15, 16, 32) → (B*15, 16, 32)      │
    │                                              │
    │  stage2: BasicBlock1D(16→32, s=2)            │
    │         (B*15, 16, 32) → (B*15, 32, 16)      │
    │                                              │
    │  stage3: BasicBlock1D(32→64, s=2)            │
    │         (B*15, 32, 16) → (B*15, 64, 8)       │
    │                                              │
    │  head:  AdaptiveAvgPool1d(1) + Linear(64→3)  │
    │         (B*15, 64, 8) → (B*15, 3)            │
    └──────────────┬───────────────────────────────┘
                   │
                   ▼  view(B, S=15, 3)
              (B, 15, 3)
          ┌────────┴────────┐
          │                 │
   bp = [:,:,:2]      q = [:,:,2]
   (B, 15, 2)           (B, 15)    q_logit 비경계(unbounded)
   [SBP, DBP]         [품질 로짓]
          │                 │
          └────────┬────────┘
                   ▼  _weighted_bp
        w = softmax(q, dim=1)       (B, 15)
        pred = (w · bp).sum(dim=1)  (B, 2)
                   │
                (B, 2)
             [SBP, DBP]
```

## 4. 중첩 세그먼트 분할 상세

### unfold 연산

```python
x.unfold(dimension=2, size=seg_len, step=stride)
# (B, 1, 1000) → (B, 1, S, 125)
```

세그먼트 수 공식:

```text
S = floor((L - seg_len) / stride) + 1
  = floor((1000 - 125) / 62) + 1
  = floor(14.11) + 1
  = 15
```

### 세그먼트 위치 및 중첩 분석

| 세그먼트 | 시작 샘플 | 종료 샘플 | 시작 시각 | 종료 시각 |
| -------- | --------- | --------- | --------- | --------- |
| 0        | 0         | 124       | 0.000 s   | 0.992 s   |
| 1        | 62        | 186       | 0.496 s   | 1.488 s   |
| 2        | 124       | 248       | 0.992 s   | 1.984 s   |
| 3        | 186       | 310       | 1.488 s   | 2.480 s   |
| …        | …         | …         | …         | …         |
| 13       | 806       | 930       | 6.448 s   | 7.440 s   |
| 14       | 868       | 992       | 6.944 s   | 7.936 s   |

```text
샘플 인덱스
0        125       250       375  ...    875       1000
│─────────│         │         │          │         │
┌─────────────────┐                                   세그먼트 0 (0~124)
        ┌─────────────────┐                           세그먼트 1 (62~186)
                ┌─────────────────┐                   세그먼트 2 (124~248)
                        ...
                                             ┌───────────────── 세그먼트 14 (868~992)

중첩: 125 - 62 = 63 샘플 (0.504초, 약 50.4%)
미포함: 993~999 (7 샘플)
```

- **인접 세그먼트 중첩**: ~50% → 한 심박 주기 전후를 모두 포함하는 세그먼트 다수 확보
- **125 샘플 @ 125 Hz = 1초**: 평균 1~1.5 심박 주기 포함

## 5. PulseBackbone 상세 설계

`PulseBackbone(in_channels=1, out_features=3, base_channels=C=16)` 기준.

입력: `(B*S, 1, 125)` — 1초 PPG 세그먼트

### 5.1 Stem

```text
(B*S, 1, 125)
  │  ConvBnAct1d(1→16, k=7, stride=2)
  │  padding = (7-1)//2 = 3
  │  L_out = floor((125 + 6 - 7)/2) + 1 = floor(124/2) + 1 = 63
  ▼
(B*S, 16, 63)
  │  MaxPool1d(k=3, stride=2, padding=1)
  │  L_out = floor((63 + 2 - 3)/2) + 1 = floor(62/2) + 1 = 32
  ▼
(B*S, 16, 32)
```

`ConvBnAct1d = Conv1d(bias=False) → BatchNorm1d → ReLU`

### 5.2 BasicBlock1D 구조

```text
입력: (B*S, in_ch, L)
  │                       ┌── shortcut ──────────────────┐
  │                       │  in_ch == out_ch, stride=1:  │
  │  conv1: ConvBnAct1d   │    Identity()                │
  │  (in_ch→out_ch, k=7,  │  그 외:                       │
  │   stride=stride)      │    Conv1d(in→out, k=1, s)    │
  │                       │    + BN                       │
  ▼                       │                               │
(B*S, out_ch, L')         │                               │
  │                       │                               │
  │  conv2: Conv1d(out_ch→out_ch, k=7, pad=3, bias=False)│
  │         + BN                                          │
  ▼                       │                               │
(B*S, out_ch, L')         │                               │
  └───────── + ←──────────┘                               │
             │                                             │
             ▼  ReLU
       (B*S, out_ch, L')
```

shortcut: stride=1이고 채널이 같을 때만 `Identity()`; 나머지는 `Conv1d(k=1) + BN`.

### 5.3 Stage별 텐서 흐름

| Stage  | 블록                     | 입력 shape    | 출력 shape    | shortcut                 |
| ------ | ------------------------ | ------------- | ------------- | ------------------------ |
| stage1 | BasicBlock1D(16→16, s=1) | (B*S, 16, 32) | (B*S, 16, 32) | Identity                 |
| stage2 | BasicBlock1D(16→32, s=2) | (B*S, 16, 32) | (B*S, 32, 16) | Conv1d(16→32,k=1,s=2)+BN |
| stage3 | BasicBlock1D(32→64, s=2) | (B*S, 32, 16) | (B*S, 64, 8)  | Conv1d(32→64,k=1,s=2)+BN |

stage2, stage3의 시간 차원:

```text
stage2 conv1: floor((32 + 6 - 7)/2) + 1 = 16
stage3 conv1: floor((16 + 6 - 7)/2) + 1 = 8
```

### 5.4 RegressionHead

```text
(B*S, 64, 8)
  │  AdaptiveAvgPool1d(1)
  ▼
(B*S, 64, 1)
  │  flatten(1)
  ▼
(B*S, 64)
  │  Dropout(0.1)
  │  Linear(64 → 3)   ← out_features + 1 = 3
  ▼
(B*S, 3)
  ├── [:, :2] → per-segment [SBP, DBP]
  └── [:, 2]  → per-segment quality logit (unbounded)
```

## 6. 전체 텐서 흐름 추적표

기본값 `L=1000, seg_len=125, stride=62, C=16` 기준.

| 단계 | 처리               | 입력 shape        | 출력 shape          |
| ---- | ------------------ | ----------------- | ------------------- |
| 0    | ensure_3d          | (B, 1000)         | (B, 1, 1000)        |
| 1    | unfold(2, 125, 62) | (B, 1, 1000)      | (B, 1, 15, 125)     |
| 2    | permute(0,2,1,3)   | (B, 1, 15, 125)   | (B, 15, 1, 125)     |
| 3    | view(B×15, 1, 125) | (B, 15, 1, 125)   | (B×15, 1, 125)      |
| 4    | Stem: ConvBnAct1d  | (B×15, 1, 125)    | (B×15, 16, 63)      |
| 5    | Stem: MaxPool1d    | (B×15, 16, 63)    | (B×15, 16, 32)      |
| 6    | Stage1: BasicBlock | (B×15, 16, 32)    | (B×15, 16, 32)      |
| 7    | Stage2: BasicBlock | (B×15, 16, 32)    | (B×15, 32, 16)      |
| 8    | Stage3: BasicBlock | (B×15, 32, 16)    | (B×15, 64, 8)       |
| 9    | Head: AvgPool+fc   | (B×15, 64, 8)     | (B×15, 3)           |
| 10   | view(B, 15, 3)     | (B×15, 3)         | (B, 15, 3)          |
| 11   | 분리 bp / q        | (B, 15, 3)        | bp(B,15,2), q(B,15) |
| 12   | softmax(q, dim=1)  | (B, 15)           | w(B, 15)            |
| 13   | (w·bp).sum(dim=1)  | (B,15,1)·(B,15,2) | (B, 2)              |

## 7. 품질 지도 학습 설계

### 7.1 품질 목표값 자동 유도

```python
with torch.no_grad():
    bp_err   = (bp - y.unsqueeze(1)).abs().mean(dim=-1)   # (B, S)
    q_target = torch.exp(-bp_err / quality_temp)           # (B, S)
```

- `y.unsqueeze(1)`: (B, 2) → (B, 1, 2) — 브로드캐스트 적용
- `bp_err`: 세그먼트별 SBP·DBP 평균 절댓값 오차 (mmHg 단위, (B, S))
- `q_target`: 오차를 [0, 1]으로 사상한 소프트 목표값

#### quality_temp 민감도 분석

`q_target = exp(-bp_err / quality_temp)`

| bp_err (mmHg) | temp=2 | temp=5 (기본) | temp=10 |
| ------------- | ------ | ------------- | ------- |
| 0             | 1.000  | 1.000         | 1.000   |
| 2             | 0.368  | 0.670         | 0.819   |
| 5             | 0.082  | 0.368         | 0.607   |
| 10            | 0.007  | 0.135         | 0.368   |
| 20            | ~0     | 0.018         | 0.135   |

- `temp`가 작을수록: 소량의 오차에도 품질이 급격히 감소 → 엄격한 기준
- `temp`가 클수록: 오차에 둔감, 대부분 세그먼트가 높은 품질 목표값 → 완화된 기준

### 7.2 품질 손실 및 합산 손실

```python
pred    = self._weighted_bp(bp, q)                 # (B, F)
bp_loss = criterion(pred, y)                        # 스칼라

q_loss  = F.mse_loss(torch.sigmoid(q), q_target)   # 스칼라

loss = bp_loss + quality_weight * q_loss
```

#### 손실 공식

```text
loss = bp_loss + quality_weight × MSE(sigmoid(q_logit), q_target)
```

- `sigmoid(q_logit)` ∈ (0, 1): 예측 품질 확률
- `q_target` ∈ (0, 1]: 오차 기반 품질 목표값
- MSE: 두 값의 제곱 평균 차이

기본값 `quality_weight=0.5`:

```text
loss = bp_loss + 0.5 × q_loss
```

#### 그래디언트 흐름 경로

`q_logit`은 두 경로에서 그래디언트를 받는다:

```text
bp_loss → pred = (softmax(q)·bp).sum
               ├── → bp (BackboneHead BP 출력부)
               └── → q  (Backbone quality logit — softmax 가중치)

q_loss  → sigmoid(q) → q  (BackboneHead quality 출력부)
```

`q_target`은 `no_grad` 블록 내에서 계산되므로 `bp`·`q`의 현재 예측값을 기반으로
하지만 목표값 자체는 상수 취급된다.

### 7.3 `_weighted_bp` — softmax 품질 가중 집계

```python
w    = F.softmax(q, dim=1).unsqueeze(-1)   # (B, S, 1)
pred = (w * bp).sum(dim=1)                  # (B, F)
```

- `softmax(q, dim=1)`: 같은 배치 샘플 내 세그먼트들이 합 = 1이 되도록 정규화
- 모든 세그먼트 중 상대적으로 품질이 높은 세그먼트에 더 많은 기여 가중치
- sigmoid 가중치(합이 1 미만)와 달리 softmax는 항상 합이 1 → 예측 스케일 안정

## 8. forward_with_quality 인터페이스

```python
def forward_with_quality(x):
    bp, q = _segment_forward(x)
    pred  = _weighted_bp(bp, q)                        # (B, 2)
    w     = F.softmax(q, dim=1)                        # (B, 15)
    qual  = (w * torch.sigmoid(q)).sum(dim=1, keepdim=True)  # (B, 1)
    return torch.cat([pred, qual], dim=1)              # (B, 3)
```

`qual` 계산:

```text
qual = Σ_s [ softmax(q)_s × sigmoid(q_s) ]   ∈ (0, 1)
```

softmax 가중치(기여 비율)와 sigmoid 점수(개별 품질)의 기댓값이다.

- 모든 세그먼트가 품질이 높고(sigmoid → 1) 균일하게 기여할 때: qual → 1
- 일부 세그먼트만 품질이 높을 때: 그 세그먼트들이 높은 softmax 가중치를 가지므로
  qual은 개별 sigmoid 값들의 가중 평균 → 최대 세그먼트의 sigmoid에 가까워짐

**활용 예시**: 신호 품질 필터링

```python
out = model.forward_with_quality(ppg)   # (B, 3)
sbp, dbp, quality = out[:, 0], out[:, 1], out[:, 2]

# 품질 임계값 0.6 이상 샘플만 수집
mask = quality > 0.6
valid_sbp = sbp[mask]
```

## 9. 학습 가능 파라미터 목록

기본값 `base_channels=C=16, out_features=2` → 백본 출력 3 (SBP, DBP, q_logit).

### Stem

| 구성요소                                          | shape      | 파라미터 수 |
| ------------------------------------------------- | ---------- | ----------- |
| ConvBnAct1d(1→16, k=7): Conv1d(1,16,7,bias=False) | (16, 1, 7) | 112         |
| ConvBnAct1d(1→16, k=7): BN(16) weight+bias        | (16,)×2    | 32          |
| MaxPool1d: 없음                                   | —          | 0           |

Stem 합계: 144

### Stage1: BasicBlock1D(16→16, stride=1)

| 구성요소                      | shape     | 파라미터 수 |
| ----------------------------- | --------- | ----------- |
| conv1: Conv1d(16,16,7,bias=F) | (16,16,7) | 1,792       |
| conv1: BN(16)                 | —         | 32          |
| conv2: Conv1d(16,16,7,bias=F) | (16,16,7) | 1,792       |
| conv2: BN(16)                 | —         | 32          |
| shortcut: Identity            | —         | 0           |

Stage1 합계: 3,648

### Stage2: BasicBlock1D(16→32, stride=2)

| 구성요소                             | shape     | 파라미터 수 |
| ------------------------------------ | --------- | ----------- |
| conv1: Conv1d(16,32,7,bias=F)        | (32,16,7) | 3,584       |
| conv1: BN(32)                        | —         | 64          |
| conv2: Conv1d(32,32,7,bias=F)        | (32,32,7) | 7,168       |
| conv2: BN(32)                        | —         | 64          |
| shortcut: Conv1d(16,32,1,s=2,bias=F) | (32,16,1) | 512         |
| shortcut: BN(32)                     | —         | 64          |

Stage2 합계: 11,456

### Stage3: BasicBlock1D(32→64, stride=2)

| 구성요소                             | shape     | 파라미터 수 |
| ------------------------------------ | --------- | ----------- |
| conv1: Conv1d(32,64,7,bias=F)        | (64,32,7) | 14,336      |
| conv1: BN(64)                        | —         | 128         |
| conv2: Conv1d(64,64,7,bias=F)        | (64,64,7) | 28,672      |
| conv2: BN(64)                        | —         | 128         |
| shortcut: Conv1d(32,64,1,s=2,bias=F) | (64,32,1) | 2,048       |
| shortcut: BN(64)                     | —         | 128         |

Stage3 합계: 45,440

### RegressionHead

| 구성요소             | shape   | 파라미터 수 |
| -------------------- | ------- | ----------- |
| AdaptiveAvgPool1d(1) | —       | 0           |
| Dropout(0.1)         | —       | 0           |
| Linear(64→3): weight | (3, 64) | 192         |
| Linear(64→3): bias   | (3,)    | 3           |

Head 합계: 195

### 전체 합계

| 구성요소 | 파라미터 수        |
| -------- | ------------------ |
| Stem     | 144                |
| Stage1   | 3,648              |
| Stage2   | 11,456             |
| Stage3   | 45,440             |
| Head     | 195                |
| **총계** | **60,883 (~61 K)** |

백본이 단일 모듈이고 S=15개 세그먼트가 파라미터를 공유하므로,
15회 순전파(B*S 배치 차원)가 동일한 61 K 파라미터를 공유한다.

## 10. 하이퍼파라미터 참조표

| 파라미터         | 기본값 | 역할                                       |
| ---------------- | ------ | ------------------------------------------ |
| `in_channels`    | 1      | PPG 입력 채널 수                           |
| `out_features`   | 2      | 예측 혈압 지표 수 (SBP, DBP)               |
| `base_channels`  | 16     | PulseBackbone 기본 채널 수 (C)             |
| `seg_len`        | 125    | 세그먼트 길이 (샘플 수, 125Hz × 1초)       |
| `stride`         | 62     | unfold 이동 간격 (샘플 수); 중첩 비율 결정 |
| `dropout`        | 0.1    | RegressionHead Dropout 비율                |
| `quality_temp`   | 5.0    | 품질 목표 계산 온도 (mmHg 단위)            |
| `quality_weight` | 0.5    | 품질 손실 가중치                           |

### 세그먼트 수 공식

```text
S = floor((input_length - seg_len) / stride) + 1
```

다양한 설정에서의 세그먼트 수:

| input_length | seg_len | stride | S             |
| ------------ | ------- | ------ | ------------- |
| 1000         | 125     | 62     | **15** (기본) |
| 1000         | 125     | 125    | 8 (비중첩)    |
| 1000         | 250     | 125    | 7             |
| 1000         | 100     | 50     | 19            |

## 11. 설계 결정 사항

### 11.1 unfold 기반 중첩 분할 (vs 비중첩 reshape)

`PulseResNet1D`의 `reshape` 기반 비중첩 분할 대신 `unfold`를 사용하는 이유:

- **다양한 위상(phase)의 심박 캡처**: 비중첩 분할에서는 경계가 심박 주기 중간에
  걸릴 경우 해당 세그먼트의 BP 예측이 부정확할 수 있다. 중첩 분할은 같은 심박을
  다양한 오프셋에서 캡처한 세그먼트를 제공한다.
- **품질 추정 근거 다양화**: 서로 다른 시간 위치의 세그먼트들이 독립적으로
  예측값을 내므로, 품질이 낮은 구간(모션 아티팩트 등)을 식별하기 쉽다.
- **입력 길이 변화에 유연**: stride만 조정하면 다른 입력 길이에서 자동으로
  세그먼트 수가 계산된다.

### 11.2 품질 레이블 없는 자동 품질 지도 학습

품질 목표값을 별도 레이블 없이 **현재 BP 예측 오차에서 실시간 유도**한다.

이 접근의 장점:

- VitalDB 데이터셋에 신호 품질 레이블이 없어도 적용 가능
- 품질 헤드가 "좋은 예측을 내는 세그먼트 특성"을 학습함으로써 BP 오차와
  상관된 신호 품질 표현을 자동 획득

한계:

- 훈련 초기 BP 예측이 불안정할 때 품질 목표값도 불안정 → warm-up 기간 고려 필요
- 질적으로 나쁜 신호라도 우연히 BP 예측이 맞으면 높은 품질 목표를 받는 경우 있음

### 11.3 no_grad 블록에서 q_target 계산

```python
with torch.no_grad():
    bp_err   = (bp - y.unsqueeze(1)).abs().mean(dim=-1)
    q_target = torch.exp(-bp_err / self.quality_temp)
```

`bp`는 그래디언트 추적이 활성화된 텐서지만, `no_grad` 블록 안에서 파생된
`q_target`은 상수로 취급된다. 이를 통해:

- `q_loss`가 `bp`에 역전파되지 않음 → 품질 목표값이 BP 예측에 직접 간섭하지 않음
- `bp_loss`와 `q_loss`의 그래디언트 경로가 독립적으로 유지됨

### 11.4 softmax 가중 집계 (vs sigmoid 또는 attention)

```python
w = F.softmax(q, dim=1)   # 합 = 1 보장
```

- **sigmoid 가중 집계**: `w_s = σ(q_s)` → 합이 S × σ̄(q), 가중 합이 입력 스케일에
  의존. 세그먼트가 모두 낮은 품질이면 최종 예측 스케일이 줄어드는 문제.
- **softmax 가중 집계**: 합 = 1 항상 보장 → 혈압 단위(mmHg) 보존.
  세그먼트 간 상대적 우열로만 집계 → 경쟁적 집계.

### 11.5 단일 PulseBackbone 공유

15개 세그먼트 모두 동일한 PulseBackbone 파라미터를 공유한다.
`(B*S, 1, 125)` 형태로 배치 차원에 합쳐서 한 번에 순전파하므로:

- 메모리: 15 × B 크기의 피처 맵 동시 보관 (B가 크면 GPU 메모리 주의)
- 연산: 하나의 큰 배치 순전파 = DataParallel과 호환
- 세그먼트별 독립 파라미터보다 정규화 효과 (파라미터 공유가 일반화 유도)

### 11.6 exp(-error/temp) 목표 함수

오차-품질 매핑을 지수 함수로 구성한 이유:

- **단조 감소**: 오차가 크면 품질이 낮아지는 단조성 보장
- **부드러운 기울기**: 분계점 없이 연속적으로 변함 → MSE 손실과 결합 시
  안정적인 그래디언트
- **스케일 불변 파라미터**: `quality_temp`가 mmHg 단위이므로 BP 오차 범위에
  대한 직관적 해석 가능 (temp=5 → 5 mmHg 오차에서 품질 ≈ 0.37)

## 12. 관련 모델 비교

| 모델                | 세그먼트         | 집계         | 품질          | 파라미터  |
| ------------------- | ---------------- | ------------ | ------------- | --------- |
| `resnet1d`          | 없음 (전체 1000) | N/A          | 없음          | ~400 K    |
| `pulse_resnet1d`    | 비중첩 8개       | 단순 평균    | 없음          | ~61 K     |
| `pulsewoq_resnet1d` | **중첩 15개**    | softmax 가중 | **지도 학습** | **~61 K** |

`pulsewoq_resnet1d`는 `pulse_resnet1d`와 파라미터 수가 거의 같지만
(Linear 출력이 2→3으로 증가하는 65 파라미터 차이)
중첩 세그먼트와 품질 지도 학습을 통해 더 신뢰성 있는 예측과 신호 품질 점수를
제공한다.

## 13. 훈련 방법

### 기본 훈련

```bash
bin\train-model.bat --model pulsewoq_resnet1d
```

### quality_temp 조정

```bash
# 엄격한 품질 기준 (5 mmHg 오차에서 품질 ≈ 0.08)
bin\train-model.bat --model pulsewoq_resnet1d \
    --model-kwargs "quality_temp=2.0"

# 완화된 품질 기준 (5 mmHg 오차에서 품질 ≈ 0.61)
bin\train-model.bat --model pulsewoq_resnet1d \
    --model-kwargs "quality_temp=10.0"
```

### quality_weight 조정

```bash
# 품질 지도 없이 BP 회귀만
bin\train-model.bat --model pulsewoq_resnet1d \
    --model-kwargs "quality_weight=0.0"

# 품질 지도 강조
bin\train-model.bat --model pulsewoq_resnet1d \
    --model-kwargs "quality_weight=1.0"
```

### 세그먼트 설정 실험

```bash
# 비중첩 설정 (stride=seg_len)
bin\train-model.bat --model pulsewoq_resnet1d \
    --model-kwargs "stride=125"

# 더 높은 중첩 (~75%)
bin\train-model.bat --model pulsewoq_resnet1d \
    --model-kwargs "stride=31"
```

## 14. 모델 검사

```bash
bin\print-model.bat --model pulsewoq_resnet1d
```

출력 예시:

```text
PulseWOQResNet1D
  (backbone): PulseBackbone
    (stem): Sequential
      (0): ConvBnAct1d(1→16, k=7, stride=2)
      (1): MaxPool1d(k=3, stride=2, pad=1)
    (stage1): Sequential
      (0): BasicBlock1D(16→16, stride=1)
    (stage2): Sequential
      (0): BasicBlock1D(16→32, stride=2)
    (stage3): Sequential
      (0): BasicBlock1D(32→64, stride=2)
    (head): RegressionHead(64→3, dropout=0.1)

seg_len=125, stride=62
quality_temp=5.0, quality_weight=0.5
S=15 segments (L=1000)

Total params    : ~60,883  (~61 K)
Trainable params: ~60,883
Input shape     : (1, 1000)
```
