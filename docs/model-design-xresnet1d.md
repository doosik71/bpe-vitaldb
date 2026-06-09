# XResNet1D101 모델 상세 설계서

## 1. 개요

XResNet1D101은 2D 이미지 분류의 ResNet-101 깊이 구성 `(3, 4, 23, 3)`과
병목(bottleneck) 잔차 블록을 1D PPG 혈압 회귀에 적용한 **대형 깊이 모델**이다.

- **구현 파일**: [`bpe/models/xresnet1d.py`](../bpe/models/xresnet1d.py)
- **모델 등록명**: `xresnet1d` (별칭: `xresnet1d101`)
- **기반 클래스**: `ResNet1D` (상속) — Stem·스테이지 구성·헤드를 그대로 재사용
- **블록 타입**: `BottleneckBlock1D` (`expansion=4`)

```python
super().__init__(
    layers=(3, 4, 23, 3),
    block=BottleneckBlock1D,
    base_channels=32,
    dropout=0.1,
)
```

### ResNet1D 패밀리 내 위치

| 등록명           | 블록 타입      | layers            | 총 블록 수 | Conv 층 수 | 최종 채널 |
| ---------------- | -------------- | ----------------- | ---------- | ---------- | --------- |
| `resnet1d_micro` | BasicBlock     | (1,)              | 1          | 3          | 32        |
| `resnet1d_tiny`  | BasicBlock     | (1, 1)            | 2          | 5          | 64        |
| `resnet1d_mini`  | BasicBlock     | (1, 1, 1, 1)      | 4          | 9          | 256       |
| `resnet1d`       | BasicBlock     | (2, 2, 2, 2)      | 8          | 17         | 256       |
| **`xresnet1d`**  | **Bottleneck** | **(3, 4, 23, 3)** | **33**     | **100**    | **1024**  |

총 Conv 층 계산: Stem 1층 + 33 블록 × 3층 + FC 1층 = **101 층** (ResNet-101 기준)

## 2. 전체 아키텍처

```text
입력: PPG 세그먼트
                  (B, 1000) 또는 (B, 1, 1000)
                            │
                            ▼  ensure_3d
                       (B, 1, 1000)
                            │
┌───────────────────────────┴────────────────────────────┐
│  Stem  (ResNet1D에서 상속)                             │
│  ConvBnAct1d(1→32, k=15, stride=2)  → (B, 32, 500)     │
│  MaxPool1d(k=3, stride=2, padding=1) → (B, 32, 250)    │
└───────────────────────────┬────────────────────────────┘
                      (B, 32, 250)
                            │
         ┌──────────────────┴────────────────────┐
         │  Stage 1  (stride=1, 3 블록)          │
         │  BottleneckBlock1D(32→32, s=1)        │  첫 블록: 32→128
         │  BottleneckBlock1D(128→32, s=1) × 2   │
         │             (B, 128, 250)             │
         ├───────────────────────────────────────┤
         │  Stage 2  (stride=2, 4 블록)          │
         │  BottleneckBlock1D(128→64, s=2)       │  첫 블록: 128→256
         │  BottleneckBlock1D(256→64, s=1) × 3   │
         │             (B, 256, 125)             │
         ├───────────────────────────────────────┤
         │  Stage 3  (stride=2, 23 블록)         │
         │  BottleneckBlock1D(256→128, s=2)      │  첫 블록: 256→512
         │  BottleneckBlock1D(512→128, s=1) × 22 │
         │             (B, 512, 63)              │
         ├───────────────────────────────────────┤
         │  Stage 4  (stride=2, 3 블록)          │
         │  BottleneckBlock1D(512→256, s=2)      │  첫 블록: 512→1024
         │  BottleneckBlock1D(1024→256, s=1) × 2 │
         │             (B, 1024, 32)             │
         └─────────────────┬─────────────────────┘
                           │
          ┌────────────────┴────────────────────┐
          │  RegressionHead                     │
          │  AdaptiveAvgPool1d(1) → (B, 1024)   │
          │  Dropout(0.1) → Linear(1024→2)      │
          └────────────────┬────────────────────┘
                        (B, 2)
                      [SBP, DBP] (mmHg)
```

## 3. 텐서 흐름 요약

| 단계 | 처리                                         | 입력 shape    | 출력 shape    |
| ---- | -------------------------------------------- | ------------- | ------------- |
| 0    | ensure_3d                                    | (B, 1000)     | (B, 1, 1000)  |
| 1    | Stem Conv(k=15, s=2)                         | (B, 1, 1000)  | (B, 32, 500)  |
| 2    | Stem MaxPool(k=3, s=2)                       | (B, 32, 500)  | (B, 32, 250)  |
| 3    | Stage 1: 3× BottleneckBlock1D (s=1)          | (B, 32, 250)  | (B, 128, 250) |
| 4    | Stage 2: 4× BottleneckBlock1D (첫 블록 s=2)  | (B, 128, 250) | (B, 256, 125) |
| 5    | Stage 3: 23× BottleneckBlock1D (첫 블록 s=2) | (B, 256, 125) | (B, 512, 63)  |
| 6    | Stage 4: 3× BottleneckBlock1D (첫 블록 s=2)  | (B, 512, 63)  | (B, 1024, 32) |
| 7    | AdaptiveAvgPool1d(1) + flatten               | (B, 1024, 32) | (B, 1024)     |
| 8    | Dropout + Linear(1024→2)                     | (B, 1024)     | (B, 2)        |

## 4. BottleneckBlock1D 상세

> BottleneckBlock1D 전체 설계는 [`model-design-resnet1d.md`](model-design-resnet1d.md)
> 4.3절에 기술되어 있다. 여기서는 XResNet1D101 문맥에서의 동작만 보완한다.

### 4.1 블록 내부 구조

```text
입력 x : (B, C_in, L)
    │
    ├─ [잔차] shortcut(x)
    │     ┌─ Identity              C_in == C_in×4 이고 stride==1일 때
    │     └─ Conv1d(k=1) + BN     그 외 (채널 전환 또는 stride>1)
    │
    ├─ [주 경로]
    │   ConvBnAct1d(C_in  → hidden,    k=1)        채널 축소 + ReLU
    │   ConvBnAct1d(hidden → hidden,   k=7, s)     시계열 처리 + ReLU
    │   Conv1d     (hidden → hidden×4, k=1) + BN   채널 확장 (ReLU 없음)
    │
    ▼  ReLU(주경로 + shortcut)
출력 : (B, hidden×4, L')
```

생성자 파라미터 `out_channels`는 **병목 내부 채널(hidden)** 이며,
실제 출력 채널은 `out_channels × 4`이다.

### 4.2 스테이지별 채널 전이

| 스테이지   | 첫 블록 in→hidden→out | 나머지 블록 in→hidden→out | shortcut 조건                            |
| ---------- | --------------------- | ------------------------- | ---------------------------------------- |
| 1 (3블록)  | 32 → 32 → **128**     | 128 → 32 → 128            | 첫 블록: 프로젝션(s=1), 나머지: Identity |
| 2 (4블록)  | 128 → 64 → **256**    | 256 → 64 → 256            | 첫 블록: 프로젝션(s=2), 나머지: Identity |
| 3 (23블록) | 256 → 128 → **512**   | 512 → 128 → 512           | 첫 블록: 프로젝션(s=2), 나머지: Identity |
| 4 (3블록)  | 512 → 256 → **1024**  | 1024 → 256 → 1024         | 첫 블록: 프로젝션(s=2), 나머지: Identity |

각 스테이지 내 **첫 번째 블록만** 채널 수를 4배 확장하고(Identity shortcut 불가),
나머지 블록은 이미 확장된 채널에서 동일 채널을 유지한다(Identity shortcut 사용).

Identity shortcut이 가능한 조건: `stride==1` 이고 `in_channels == out_channels × 4`

Stage 3의 블록 수(23)가 압도적으로 많은 이유: 이 스테이지만 22개 블록이
Identity shortcut을 사용해 파라미터·연산 비용이 낮으면서도 표현 깊이를 극대화한다.

## 5. 층 깊이 계산

`layers=(3, 4, 23, 3)` 구성이 "101"이 되는 근거:

```text
Stem         :  1 Conv 층
Stage 1      :  3 블록 × 3 Conv = 9 층
Stage 2      :  4 블록 × 3 Conv = 12 층
Stage 3      : 23 블록 × 3 Conv = 69 층
Stage 4      :  3 블록 × 3 Conv =  9 층
─────────────────────────────────────────
Conv 합계    : 100 층
FC Head      :  1 층
─────────────────────────────────────────
총 학습 층수  : 101 층  →  ResNet-101
```

이 관례는 He et al. (2016)의 원 논문에서 각 Conv와 FC 층을 계산하는 방식을 따른다.
Shortcut의 프로젝션 Conv는 주 경로와 병렬이므로 별도 층으로 세지 않는다.

## 6. 파라미터 분포

`base_channels=32` 기준 각 스테이지의 파라미터 수:

| 구성 요소         | 파라미터 수 (근사) | 전체 비중 |
| ----------------- | ------------------ | --------- |
| Stem              | ~0.5 K             | < 0.1 %   |
| Stage 1 (3 블록)  | ~49 K              | ~0.5 %    |
| Stage 2 (4 블록)  | ~273 K             | ~2.9 %    |
| Stage 3 (23 블록) | ~5,782 K           | ~61.4 %   |
| Stage 4 (3 블록)  | ~3,353 K           | ~35.6 %   |
| Head              | ~2 K               | < 0.1 %   |
| **합계**          | **~9,460 K**       | 100 %     |

Stage 3이 전체 파라미터의 61 %를 차지한다. `base_channels`를 낮추면 전체 모델
크기가 2의 제곱 비율로 줄어든다 (채널이 절반이면 파라미터는 약 1/4).

## 7. ResNet1D (BasicBlock, layers=(2,2,2,2))과 비교

| 항목       | ResNet1D         | XResNet1D101               |
| ---------- | ---------------- | -------------------------- |
| 블록 타입  | BasicBlock1D     | BottleneckBlock1D          |
| Conv/블록  | 2                | 3                          |
| 총 블록 수 | 8                | 33                         |
| 총 Conv 층 | 17               | 100                        |
| 최종 채널  | 256              | 1024                       |
| 파라미터   | ~1.7 M           | ~9.5 M                     |
| 주요 용도  | 빠른 실험·기준선 | 강한 기준선·최적 성능 탐색 |

## 8. 하이퍼파라미터 참조표

| 파라미터        | 기본값              | 역할                              |
| --------------- | ------------------- | --------------------------------- |
| `in_channels`   | 1                   | 입력 채널 (단일 PPG)              |
| `out_features`  | 2                   | 출력 차원 ([SBP, DBP])            |
| `base_channels` | 32                  | Stem 및 Stage 1 병목 채널 기준값  |
| `dropout`       | 0.1                 | RegressionHead Dropout 비율       |
| `layers`        | (3, 4, 23, 3)       | 고정 (ResNet-101 구성, 변경 불가) |
| `block`         | `BottleneckBlock1D` | 고정, 변경 불가                   |

`layers`와 `block`은 `XResNet1D101.__init__`에서 부모 클래스에 하드코딩된다.
변경하려면 `ResNet1D`를 직접 사용한다:

```bash
# ResNet-50 구성 (3,4,6,3)
bin\train-model.bat --model resnet1d \
    --model-kwargs "block=BottleneckBlock1D,layers=(3,4,6,3)"

# ResNet-152 구성 (3,8,36,3)
bin\train-model.bat --model resnet1d \
    --model-kwargs "block=BottleneckBlock1D,layers=(3,8,36,3)"
```

### `base_channels` 조정 효과

| base_channels | Stage 채널 진행   | Head 입력 | 파라미터 (근사) |
| ------------- | ----------------- | --------- | --------------- |
| 16            | 64→128→256→512    | 512       | ~2.4 M          |
| 32 (기본)     | 128→256→512→1024  | 1024      | ~9.5 M          |
| 64            | 256→512→1024→2048 | 2048      | ~37 M           |

## 9. 설계 결정 사항

### 9.1 layers=(3, 4, 23, 3) 고정

ResNet-101의 `(3, 4, 23, 3)` 구성은 Stage 3에 블록을 집중시키는 비대칭 설계다.
이미지 분류에서 Stage 3이 중간 수준의 의미론적 특징을 형성하는 핵심 레이어임이
반복적으로 확인되었다. PPG 시계열에서도 Stage 3 수준의 해상도(63 샘플)는
심박 주기(~125 샘플, 125 Hz 기준)의 약 절반으로, 맥파 내 국소 패턴을 포착하기에
적합한 해상도다.

### 9.2 BasicBlock1D 대신 BottleneckBlock1D

깊이(층 수)를 늘릴 때 BasicBlock1D를 쌓으면 파라미터 증가 속도가 채널의 제곱에
비례한다. BottleneckBlock1D는 k=1 압축·복원으로 큰 커널 합성곱을 좁은 채널에서
수행해, 동일 표현력을 약 절반의 파라미터로 달성한다.

33개 블록을 BasicBlock1D로 구성하면:

- Stage 3 22블록: `Conv(512→512, k=7) × 2` = `512×512×7×2 × 22 ≈ 82M` 파라미터
  (현실적이지 않다)

BottleneckBlock1D로 구성하면:

- Stage 3 22블록: `Conv(512→128, k=1) + Conv(128→128, k=7) + Conv(128→512, k=1)` ≈ 5.4M

### 9.3 두 등록명 제공

```python
@register_model("xresnet1d")
@register_model("xresnet1d101")
```

짧은 `xresnet1d`는 기본 CLI 사용을 위해, 긴 `xresnet1d101`은 다른 깊이 변형
(예: `xresnet1d50`, `xresnet1d152`)이 추가될 경우 구분성을 위해 함께 등록한다.

## 10. 훈련 방법

### 기본 훈련

```bash
bin\train-model.bat --model xresnet1d
```

### 학습률 워밍업 권장

파라미터 수가 ~9.5 M이므로 학습 초기 발산 위험이 있다. 낮은 학습률로 시작하는
것이 안정적이다:

```bash
bin\train-model.bat --model xresnet1d \
    --lr 3e-4 \
    --epochs 100
```

### 경량화 실험

```bash
# base_channels=16으로 파라미터 약 1/4
bin\train-model.bat --model xresnet1d \
    --model-kwargs "base_channels=16"
```

## 11. 모델 검사

```bash
bin\print-model.bat --model xresnet1d
```

출력 예시:

```text
XResNet1D101
  (stem): Sequential
    (0): ConvBnAct1d(1→32, k=15, s=2)
    (1): MaxPool1d(k=3, s=2)
  (stage1): Sequential
    (0): BottleneckBlock1D(32→32→128)    ← 채널 전환 블록
    (1): BottleneckBlock1D(128→32→128)   ← Identity shortcut
    (2): BottleneckBlock1D(128→32→128)
  (stage2): Sequential
    (0): BottleneckBlock1D(128→64→256, s=2)
    (1-3): BottleneckBlock1D(256→64→256) × 3
  (stage3): Sequential
    (0): BottleneckBlock1D(256→128→512, s=2)
    (1-22): BottleneckBlock1D(512→128→512) × 22
  (stage4): Sequential
    (0): BottleneckBlock1D(512→256→1024, s=2)
    (1-2): BottleneckBlock1D(1024→256→1024) × 2
  (head): RegressionHead(1024→2)

Total params    : ~9,460,000  (~9.5 M)
Trainable params: ~9,460,000
Input shape     : (1, 1000)
```

## 12. 참고 문헌

- He, K., Zhang, X., Ren, S., and Sun, J. (2016). "Deep Residual Learning for
  Image Recognition." *CVPR 2016*, pp. 770–778.
  (ResNet-101 아키텍처 정의, layers=(3,4,23,3) 구성)

- He, K., Zhang, X., Ren, S., and Sun, J. (2016). "Identity Mappings in Deep
  Residual Networks." *ECCV 2016*.
  (Bottleneck 구조 및 잔차 연결 설계 원리)
