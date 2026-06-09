# ResNet1D 모델 상세 설계서

## 1. 개요

ResNet1D는 2D 이미지 분류에서 검증된 Residual Network(He et al., 2016)의
핵심 구조를 1D PPG 시계열 혈압 회귀에 적용한 모델 패밀리다.

- **구현 파일**: [`bpe/models/resnet1d.py`](../bpe/models/resnet1d.py)
- **파생 모델**: `resnet1d_mini.py`, `resnet1d_tiny.py`, `resnet1d_micro.py`

### 모델 패밀리 일람

| 등록명           | 스테이지 | 스테이지별 블록 | 최종 채널 | 구현 방식                     |
| ---------------- | -------- | --------------- | --------- | ----------------------------- |
| `resnet1d`       | 4        | (2, 2, 2, 2)    | 256       | `ResNet1D`                    |
| `resnet1d_mini`  | 4        | (1, 1, 1, 1)    | 256       | `ResNet1DMini(ResNet1D)` 상속 |
| `resnet1d_tiny`  | 2        | (1, 1)          | 64        | `ResNet1DTiny` 독립 클래스    |
| `resnet1d_micro` | 1        | (1,)            | 32        | `ResNet1DMicro` 독립 클래스   |

`ResNet1DMini`는 `ResNet1D`를 상속해 `layers=(1,1,1,1)`만 바꾼다.
`ResNet1DTiny`와 `ResNet1DMicro`는 `ResNet1D`를 상속하지 않고 필요한
스테이지만 직접 구성한다.

## 2. 전체 아키텍처 (ResNet1D 기본형)

```text
입력: PPG 세그먼트
                  (B, 1000)  또는  (B, 1, 1000)
                            │
                            ▼  ensure_3d
                       (B, 1, 1000)
                            │
┌───────────────────────────┴──────────────────────────────┐
│  Stem                                                    │
│  ConvBnAct1d(1→32, k=15, stride=2)  → (B, 32, 500)       │
│  MaxPool1d(k=3, stride=2, padding=1) → (B, 32, 250)      │
└───────────────────────────┬──────────────────────────────┘
                      (B, 32, 250)
                            │
              ┌─────────────┴─────────────┐
              │  Stage 1  (stride=1)      │
              │  BasicBlock1D(32→32) × 2  │
              │         (B, 32, 250)      │
              ├───────────────────────────┤
              │  Stage 2  (stride=2)      │
              │  BasicBlock1D(32→64)      │
              │  BasicBlock1D(64→64)      │
              │         (B, 64, 125)      │
              ├───────────────────────────┤
              │  Stage 3  (stride=2)      │
              │  BasicBlock1D(64→128)     │
              │  BasicBlock1D(128→128)    │
              │         (B, 128, 63)      │
              ├───────────────────────────┤
              │  Stage 4  (stride=2)      │
              │  BasicBlock1D(128→256)    │
              │  BasicBlock1D(256→256)    │
              │         (B, 256, 32)      │
              └────────────┬──────────────┘
                           │
┌──────────────────────────┴────────────────────────────┐
│  RegressionHead                                       │
│  AdaptiveAvgPool1d(1) → (B, 256)                      │
│  Dropout(0.1) → Linear(256→2)                         │
└──────────────────────────┬────────────────────────────┘
                         (B, 2)
                       [SBP, DBP] (mmHg)
```

## 3. 텐서 흐름 요약

### 3.1 BasicBlock1D 기준 (ResNet1D 기본형)

| 단계 | 처리                                                             | 입력 shape   | 출력 shape   |
| ---- | ---------------------------------------------------------------- | ------------ | ------------ |
| 0    | ensure_3d                                                        | (B, 1000)    | (B, 1, 1000) |
| 1    | Stem Conv(k=15, s=2) + BN + ReLU                                 | (B, 1, 1000) | (B, 32, 500) |
| 2    | Stem MaxPool(k=3, s=2)                                           | (B, 32, 500) | (B, 32, 250) |
| 3    | Stage 1: 2× BasicBlock1D(32→32, s=1)                             | (B, 32, 250) | (B, 32, 250) |
| 4    | Stage 2: BasicBlock1D(32→64, s=2) + BasicBlock1D(64→64, s=1)     | (B, 32, 250) | (B, 64, 125) |
| 5    | Stage 3: BasicBlock1D(64→128, s=2) + BasicBlock1D(128→128, s=1)  | (B, 64, 125) | (B, 128, 63) |
| 6    | Stage 4: BasicBlock1D(128→256, s=2) + BasicBlock1D(256→256, s=1) | (B, 128, 63) | (B, 256, 32) |
| 7    | AdaptiveAvgPool1d(1) + flatten                                   | (B, 256, 32) | (B, 256)     |
| 8    | Dropout + Linear(256→2)                                          | (B, 256)     | (B, 2)       |

### 3.2 BottleneckBlock1D 기준 (base_channels=32, expansion=4)

| 단계            | 출력 shape    | 채널 변화          |
| --------------- | ------------- | ------------------ |
| Stem            | (B, 32, 250)  | 1 → 32             |
| Stage 1 (2블록) | (B, 128, 250) | 32 → 128 (×4 확장) |
| Stage 2 (2블록) | (B, 256, 125) | 128 → 256          |
| Stage 3 (2블록) | (B, 512, 63)  | 256 → 512          |
| Stage 4 (2블록) | (B, 1024, 32) | 512 → 1024         |
| Head            | (B, 2)        | 1024 → 2           |

## 4. 모듈별 상세 설계

### 4.1 Stem

**역할**: 원시 PPG 입력의 해상도를 4배 줄이고 채널을 `base_channels`로 확장한다.
초기 긴 커널(k=15)은 고주파 잡음을 억제하고 저-중주파 PPG 패턴을 추출한다.

```text
입력 : (B, 1, 1000)
    │
    ▼  ConvBnAct1d(1→32, k=15, stride=2, padding=7) + BN + ReLU
(B, 32, 500)
    │
    ▼  MaxPool1d(k=3, stride=2, padding=1)
(B, 32, 250)
```

- `ConvBnAct1d` 내부 패딩: `(15-1)//2 = 7` → 스트라이드 이외 길이 보존
- 스트라이드 2 + MaxPool stride 2 → 총 4× 다운샘플: 1000 → 250
- 이후 스테이지들은 250 샘플(2초 분량, 125 Hz 기준) 위에서 작동

### 4.2 BasicBlock1D

**역할**: 표준 2-레이어 잔차 블록. 두 개의 k=7 합성곱과 잔차 연결로 구성된다.

```text
입력 x : (B, C_in, L)
    │
    ├─ [잔차 경로]
    │   shortcut(x) : Identity (C_in==C_out, stride==1)
    │                 또는 Conv1d(k=1, stride) + BN (그 외)
    │
    ├─ [주 경로]
    │   ConvBnAct1d(C_in→C_out, k=7, stride=stride)   [BN + ReLU 포함]
    │       → (B, C_out, L')
    │   Conv1d(C_out→C_out, k=7, padding=3, bias=False)
    │   BatchNorm1d(C_out)                              [ReLU 없음]
    │       → (B, C_out, L')
    │
    ▼  ReLU(주경로 + shortcut(x))
출력 : (B, C_out, L')
```

`conv2`에 ReLU가 없는 것이 핵심이다. 활성화는 잔차 합산 이후 한 번만 적용된다.

#### 잔차 연결 규칙

| 조건                           | shortcut                               |
| ------------------------------ | -------------------------------------- |
| `stride==1` 이고 `C_in==C_out` | `nn.Identity()` (파라미터 없음)        |
| 그 외                          | `Conv1d(C_in→C_out, k=1, stride) + BN` |

k=1 프로젝션은 공간 정보 없이 채널 수와 해상도만 맞춘다.

### 4.3 BottleneckBlock1D

**역할**: 3-레이어 병목 잔차 블록. k=1 합성곱으로 채널을 압축·복원하고
가운데 k=7로 시계열 특징을 추출한다.

```text
입력 x : (B, C_in, L)
    │
    ├─ [잔차 경로]
    │   shortcut(x) : Conv1d(C_in→C_out×4, k=1, stride) + BN
    │
    ├─ [주 경로]
    │   ConvBnAct1d(C_in → hidden, k=1)           ← 채널 축소 (squeeze)
    │       hidden = out_channels
    │       → (B, hidden, L)
    │   ConvBnAct1d(hidden → hidden, k=7, stride=stride) ← 시계열 처리
    │       → (B, hidden, L')
    │   Conv1d(hidden → hidden×4, k=1, bias=False)
    │   BatchNorm1d(hidden×4)                      ← ReLU 없음
    │       expanded = out_channels × 4
    │       → (B, expanded, L')
    │
    ▼  ReLU(주경로 + shortcut(x))
출력 : (B, expanded, L')  [= (B, out_channels×4, L')]
```

> **주의**: `BottleneckBlock1D`에서 생성자 파라미터 `out_channels`는
> **병목 내부 채널** 수이며, 실제 출력 채널은 `out_channels × expansion(=4)`이다.

#### BasicBlock1D vs BottleneckBlock1D

| 항목             | BasicBlock1D   | BottleneckBlock1D               |
| ---------------- | -------------- | ------------------------------- |
| `expansion`      | 1              | 4                               |
| 합성곱 레이어 수 | 2 (k=7, k=7)   | 3 (k=1, k=7, k=1)               |
| 출력 채널        | `out_channels` | `out_channels × 4`              |
| 주 용도          | 경량 모델      | 대형 모델 (깊이 증가 시 효율적) |

### 4.4 `_make_stage` 알고리즘

스테이지를 구성하는 팩토리 메서드. `self.in_channels` 상태를 업데이트하며
채널 수를 다음 스테이지로 전달한다.

```python
def _make_stage(self, block, out_channels, blocks, stride):
    # 첫 번째 블록: stride 적용 + 채널 전환
    layers = [block(self.in_channels, out_channels, stride)]
    self.in_channels = out_channels * block.expansion   # ← 상태 업데이트
    # 나머지 블록: stride=1, 채널 유지
    for _ in range(1, blocks):
        layers.append(block(self.in_channels, out_channels, 1))
    return nn.Sequential(*layers)
```

`self.in_channels`는 생성자 내에서 살아있는 전이 상태(mutable state)다.
스테이지 간 채널 정보를 인자로 전달하지 않고 이 상태를 통해 전파한다.

스테이지 내에서 채널 변화가 일어나는 경우:

- **첫 번째 블록**: `C_in(이전 스테이지) → C_out × expansion`
- **나머지 블록**: `C_out × expansion → C_out × expansion` (변화 없음)

## 5. 모델 패밀리 상세

### 5.1 ResNet1D (기본형)

**등록명**: `resnet1d`

```text
레이어 구성: layers=(2, 2, 2, 2), block=BasicBlock1D
채널 진행:  32 → 32 → 64 → 128 → 256
```

### 5.2 ResNet1DMini

**등록명**: `resnet1d_mini` | `ResNet1D` 상속

```python
super().__init__(layers=(1, 1, 1, 1), block=BasicBlock1D, ...)
```

`ResNet1D`와 Stem·채널 구성이 완전히 동일하며, 각 스테이지의 블록 수만 2→1로
줄인다. 클래스 상속으로 구현되므로 `ResNet1D`의 하이퍼파라미터 변경이 Mini에도
그대로 반영된다.

```text
채널 진행:  32 → 32 → 64 → 128 → 256  (ResNet1D와 동일)
블록 수:    8 blocks → 4 blocks (각 스테이지 절반)
```

### 5.3 ResNet1DTiny

**등록명**: `resnet1d_tiny` | 독립 클래스

2개 스테이지만 구성. 채널 진행은 Stage 2까지만 수행한다.

```text
Stem: 1→32, (B, 32, 250)
Stage 1: BasicBlock1D(32→32, s=1) → (B, 32, 250)
Stage 2: BasicBlock1D(32→64, s=2) → (B, 64, 125)
Head: Linear(64→2)
```

`ResNet1D`를 상속하지 않고 직접 구현한 이유: 4스테이지 구조를 상속받아
stage3, stage4를 제거하면 `_make_stage`의 `self.in_channels` 상태 관리가
복잡해진다. 독립 클래스가 더 명확하다.

### 5.4 ResNet1DMicro

**등록명**: `resnet1d_micro` | 독립 클래스

단일 스테이지. Stem 이후 BasicBlock1D 1개만 적용한다.

```text
Stem: 1→32, (B, 32, 250)
Stage 1: BasicBlock1D(32→32, s=1) → (B, 32, 250)
Head: Linear(32→2)
```

실험 기준선(baseline) 또는 최소 구성으로 사용한다.

### 5.5 패밀리 텐서 흐름 비교

|              | Micro      | Tiny       | Mini       | Base       |
| ------------ | ---------- | ---------- | ---------- | ---------- |
| Stem 출력    | (B,32,250) | (B,32,250) | (B,32,250) | (B,32,250) |
| Stage 1 후   | (B,32,250) | (B,32,250) | (B,32,250) | (B,32,250) |
| Stage 2 후   | —          | (B,64,125) | (B,64,125) | (B,64,125) |
| Stage 3 후   | —          | —          | (B,128,63) | (B,128,63) |
| Stage 4 후   | —          | —          | (B,256,32) | (B,256,32) |
| Head 입력 ch | 32         | 64         | 256        | 256        |

## 6. 하이퍼파라미터 참조표

### ResNet1D

| 파라미터        | 기본값         | 역할                               |
| --------------- | -------------- | ---------------------------------- |
| `in_channels`   | 1              | 입력 채널 (단일 PPG)               |
| `out_features`  | 2              | 출력 차원 ([SBP, DBP])             |
| `base_channels` | 32             | Stem 출력 채널 = Stage 1 기준 채널 |
| `layers`        | (2, 2, 2, 2)   | 각 스테이지의 잔차 블록 수         |
| `block`         | `BasicBlock1D` | 잔차 블록 타입                     |
| `dropout`       | 0.1            | RegressionHead Dropout 비율        |

### Stem 커널 크기와 수용 영역

Stem의 k=15는 VitalDB 125 Hz 기준 약 **120 ms** 범위를 포착한다.
이는 PPG 상승부(systolic upstroke) 전체에 해당하며, 초기 단계에서
심박 주기 내 중요 형태 정보를 압축하는 데 적합하다.

### `base_channels` 조정 효과

```bash
# 채널 절반 (더 가벼운 모델)
bin\train-model.bat --model resnet1d --model-kwargs "base_channels=16"
# Stage 채널 진행: 16 → 32 → 64 → 128, Head 입력 128

# 채널 두 배 (더 높은 표현력)
bin\train-model.bat --model resnet1d --model-kwargs "base_channels=64"
# Stage 채널 진행: 64 → 128 → 256 → 512, Head 입력 512
```

### BottleneckBlock1D 사용 예시

```python
from bpe.models.resnet1d import BottleneckBlock1D

# 코드에서 직접
model = ResNet1D(block=BottleneckBlock1D, layers=(3, 4, 6, 3))
# Stage 채널: 32×4=128 → 256 → 512 → 1024

# CLI (eval 사용)
bin\train-model.bat --model resnet1d \
    --model-kwargs "block=BottleneckBlock1D,layers=(3,4,6,3)"
```

## 7. 설계 결정 사항

### 7.1 커널 크기 k=7

2D ResNet에서 3×3 합성곱이 담당하는 역할을 1D에서는 더 큰 수용 영역이 맡는다.
PPG 신호의 특징(맥파 상승·하강, 반사파 등)은 수십~수백 ms에 걸쳐 펼쳐지므로
k=7(56 ms @ 125 Hz)이 k=3(24 ms)보다 적합하다.
Stem에서 k=15(120 ms)로 초기 패턴을 잡은 뒤, 스테이지 내부에서는 k=7로
세밀한 특징을 추출하는 2단계 구조다.

### 7.2 스테이지별 2× 채널 확장

2D ResNet과 동일하게 공간 해상도를 2배 줄일 때마다 채널을 2배로 늘린다.
`base_channels × [1, 2, 4, 8]` 패턴으로, 총 표현 용량(공간 크기 × 채널 수)을
스테이지 간에 일정하게 유지한다.

### 7.3 conv2에 ReLU 없음

`BasicBlock1D.conv2`와 `BottleneckBlock1D.conv3`에는 BN만 있고 ReLU가 없다.
이는 원본 ResNet 논문의 post-activation 설계를 따른 것이다:

```text
output = ReLU(F(x) + x)   ← 잔차 합산 후 단일 비선형화
```

마지막 합성곱 레이어에도 ReLU를 적용하면 잔차 합산 전후로 두 번 비선형화가
일어나 그래디언트 흐름이 방해받는다.

### 7.4 `_make_stage`의 `self.in_channels` 상태

`_make_stage`가 `self.in_channels`를 내부에서 갱신하는 구조는 PyTorch 2D ResNet
공식 구현(`torchvision.models.resnet`)의 관례를 그대로 따른다.
호출 순서가 바뀌면 잘못된 채널 구성이 생성되므로, 생성자에서 `_make_stage`는
**반드시 stage1 → stage2 → stage3 → stage4 순서**로 호출해야 한다.

### 7.5 ResNet1DTiny·Micro의 독립 클래스 구현

4스테이지 `ResNet1D`를 상속받아 일부 스테이지를 제거하는 방식은
`_make_stage`의 상태 부작용 때문에 안전하지 않다. 스테이지를 스킵해도
`self.in_channels`가 변경되어 이후 블록이 잘못 구성될 수 있다.
독립 클래스로 구현하면 이 문제를 회피하고 코드가 더 읽기 쉬워진다.

## 8. 훈련 방법

### 기본 훈련

```bash
bin\train-model.bat --model resnet1d
```

### 패밀리 비교 실험

```bash
# 파라미터 수 확인
bin\print-model.bat --model resnet1d_micro
bin\print-model.bat --model resnet1d_tiny
bin\print-model.bat --model resnet1d_mini
bin\print-model.bat --model resnet1d

# 각각 훈련
bin\train-model.bat --model resnet1d_micro
bin\train-model.bat --model resnet1d_tiny
bin\train-model.bat --model resnet1d_mini
bin\train-model.bat --model resnet1d
```

### Bottleneck 버전

```bash
bin\train-model.bat --model resnet1d \
    --model-kwargs "block=BottleneckBlock1D,layers=(3,4,6,3)"
```

## 9. 모델 검사

```bash
bin\print-model.bat --model resnet1d
```

출력 예시:

```text
ResNet1D
  (stem): Sequential
    (0): ConvBnAct1d(1→32, k=15, s=2)
    (1): MaxPool1d(k=3, s=2)
  (stage1): Sequential
    (0-1): BasicBlock1D(32→32)
  (stage2): Sequential
    (0): BasicBlock1D(32→64, s=2)
    (1): BasicBlock1D(64→64)
  (stage3): Sequential ...
  (stage4): Sequential ...
  (head): RegressionHead(256→2)

Total params    : <N>
Trainable params: <N>
Input shape     : (1, 1000)
```

## 10. 참고 문헌

- He, K., Zhang, X., Ren, S., and Sun, J. (2016). "Deep Residual Learning for
  Image Recognition." *CVPR 2016*, pp. 770–778.

- He, K., Zhang, X., Ren, S., and Sun, J. (2016). "Identity Mappings in Deep
  Residual Networks." *ECCV 2016*, pp. 630–645.
