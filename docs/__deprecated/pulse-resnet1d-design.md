# PulseResNet1D 설계 문서

## 배경과 동기

기존 모델(`resnet1d` 등)은 8초 분량(1000 샘플, 125 Hz)의 PPG 신호 전체를
한 번에 처리한다. 그러나 혈압 추정에 실질적으로 필요한 정보는 단일 맥박
파형(~1초, ~125 샘플)에 이미 담겨 있다. 8초 구간에는 복수의 맥박이 포함되므로,
모델이 여러 맥박을 동시에 고려하면 오히려 학습이 어렵고 불필요한 파라미터가
생긴다.

이 모델은 1000 샘플 입력을 8개의 125 샘플 구간으로 분할하고, 공유 백본으로
각 구간을 독립적으로 처리한 뒤, 8개 추정값의 평균을 최종 출력으로 낸다.

**기대 효과**:

- 단일 맥박만 보므로 학습 신호가 더 명확해진다.
- 파라미터 수가 크게 줄어 과적합(overfitting) 위험이 줄어든다.
- 1개 입력 샘플이 8번의 그래디언트 기여를 하므로 epoch당 학습 속도가 빠르다.
- 8개 예측의 평균으로 출력하므로 이상 구간에 대한 robustness가 높아진다.

## 입출력 인터페이스

기존 모델과 완전히 동일하다.

```text
입력: x  float32  (B, 1000)  또는  (B, 1, 1000)   PPG 세그먼트
출력: y  float32  (B, 2)                           [SBP, DBP] (mmHg)
```

학습/평가 스크립트 수정 없이 `--model pulse_resnet1d`만으로 교체 가능하다.

## 아키텍처 개요

```text
입력 (B, 1, 1000)
    │
    ▼  reshape
(B×8, 1, 125)   ← 8개 비중첩(non-overlapping) 구간, 각 125 샘플
    │
    ▼  PulseBackbone (공유 가중치)
(B×8, 2)        ← 각 구간의 SBP/DBP 추정값
    │
    ▼  reshape + mean (AvgPool 동등)
(B, 2)          ← 8개 추정값의 평균 → 최종 출력
```

백본은 8개 구간이 **가중치를 공유**한다. 위치별로 독립된 백본을 두는 것은
파라미터를 8배 늘릴 뿐 이점이 없다.

## 구간 분할과 평균 집계 구현

사용자가 명시한 대로 표준 레이어(convolution layer, average pooling layer)로
구현 가능하다. 구체적으로는 텐서 `view` 연산과 `AvgPool1d`를 사용한다.

```python
# 1. 구간 분할: (B, 1, 1000) → (B*8, 1, 125)
x = x.view(B * NUM_SEGMENTS, 1, SEGMENT_LENGTH)

# 2. 공유 백본으로 각 구간 처리: (B*8, 1, 125) → (B*8, 2)
x = self.backbone(x)

# 3. 집계: (B*8, 2) → (B, 2, 8) → AvgPool1d → (B, 2)
x = x.view(B, NUM_SEGMENTS, 2).permute(0, 2, 1)   # (B, 2, 8)
x = F.avg_pool1d(x, NUM_SEGMENTS).squeeze(-1)      # (B, 2)
```

`view` 연산은 메모리 재배치 없이 O(1)이므로 추론/학습 모두 오버헤드가 없다.

## PulseBackbone 설계 (125 샘플 입력용)

125 샘플 입력에 맞게 스템(stem)과 스테이지(stage) 수를 줄인다.
기존 ResNet1D의 `BasicBlock1D`를 그대로 재사용한다.

```text
Layer                 Output shape    비고
─────────────────────────────────────────────────────
입력                  (B, 1,  125)
stem.Conv(k=7, s=2)   (B,C,   63)    C = base_channels
stem.MaxPool(s=2)     (B,C,   32)
stage1 (stride=1)     (B,C,   32)    1×BasicBlock1D(C→C)
stage2 (stride=2)     (B,2C,  16)    1×BasicBlock1D(C→2C)
stage3 (stride=2)     (B,4C,   8)    1×BasicBlock1D(2C→4C)
RegressionHead        (B,2)          AdaptiveAvgPool1d(1) + Linear
```

기본값 `base_channels=16`:

| 레이어    | 파라미터 수 (근사) |
| --------- | ------------------ |
| stem      | ~1,100             |
| stage1    | ~3,600             |
| stage2    | ~10,800            |
| stage3    | ~42,800            |
| head      | ~130               |
| **합계**  | **~58,000**        |

ResNet1D(2.18M)의 약 **2.7 %** 수준이다. `base_channels=32`로 키우면 ~230K이며,
이 경우에도 ResNet1D의 10 % 이하를 유지한다.

## 파일 레이아웃

```text
bpe/models/pulse_resnet1d.py   ← 새 파일 (PulseBackbone + PulseResNet1D)
bpe/models/__init__.py         ← PulseResNet1D import + __all__ 추가
```

`blocks.py`, `registry.py`, `resnet1d.py`는 수정하지 않는다.

## 구현 단계

1. `bpe/models/pulse_resnet1d.py` 작성
   - `PulseBackbone`: 3-stage ResNet, 125 샘플 입력 전용
   - `PulseResNet1D`: 분할 → 백본 → 평균 집계, `@register_model("pulse_resnet1d")`
2. `bpe/models/__init__.py`에 import 추가
3. `uv run python scripts/print-model.py --model pulse_resnet1d` 로 레이어 구조 확인
4. `uv run python scripts/train.py --model pulse_resnet1d` 로 단기 학습 실행하여
   기존 `resnet1d`와 val MAE 비교

## 열린 질문 (구현 전 확인 필요)

아래 두 가지는 현재 명세에서 결정되지 않은 사항이다. 기본값을 정하고 진행하지만,
다른 선호가 있으면 알려 달라.

1. **`base_channels` 기본값**: 16(~58K params)과 32(~230K params) 중 어느 쪽을
   기본값으로 할지. 제안은 `base_channels=16`으로 시작해 결과를 보는 것이다.

2. **구간 수 고정 여부**: 현재 설계는 8개 구간(125 샘플)을 하드코딩한다.
   `num_segments`를 생성자 인자로 받아 유연하게 할 수도 있으나, AGENTS.md의
   "섣부른 추상화 금지" 원칙에 따라 8로 고정하는 것을 제안한다.
