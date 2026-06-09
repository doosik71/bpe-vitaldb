# MTAE_TR 모델 상세 설계서

## 1. 개요

MTAE_TR(Multi-Task AutoEncoder with Transformer)은 CNN 기반 MTAE의 인코더와
디코더를 Transformer로 교체한 변형 모델이다. PPG 신호를 비중첩 패치 시퀀스로
분할하고, CLS 토큰 기반 Transformer 인코더로 잠재 벡터를 추출한 뒤 두 가지
작업을 동시에 수행한다.

- **구현 파일**: [`bpe/models/mtae_tr.py`](../bpe/models/mtae_tr.py)
- **모델 등록명**: `mtae_tr`
- **설계 기반**: ViT(Vision Transformer) 인코더 + MAE(Masked AutoEncoder) 디코더

| 태스크     | 경로                                | 손실                  |
| ---------- | ----------------------------------- | --------------------- |
| 혈압 회귀  | PatchEmbed → Encoder(CLS) → bp_head | `criterion(pred, y)`  |
| PPG 재구성 | Encoder 잠재 → Decoder(mask tokens) | `criterion(recon, x)` |

### MTAE(CNN)과의 차이

| 항목           | MTAE             | MTAE_TR                             |
| -------------- | ---------------- | ----------------------------------- |
| 특징 추출      | stride Conv 스택 | Attention 기반 패치 처리            |
| 입력 표현      | 연속 시계열      | N개 패치 토큰 시퀀스                |
| 장거리 의존성  | 수용 영역 내     | 전체 시퀀스 Attention               |
| 디코더         | Upsample + Conv  | MAE 스타일 mask token + Transformer |
| 기본 잠재 차원 | 16               | 32                                  |
| 파라미터 수    | 소형             | 중형 (~109 K, 기본값)               |

## 2. 전체 아키텍처

```text
입력: PPG 세그먼트
                   (B, 1000)  또는  (B, 1, 1000)
                             │
    ┌────────────────────────┴─────────────────────────┐
    │  _PatchEmbed  (patch_size=25)                    │
    │  1000샘플 → 40개 패치 × 25샘플 → Linear(25→32)   │
    └────────────────────────┬─────────────────────────┘
                        (B, 40, 32)
                             │
    ┌────────────────────────┴─────────────────────────┐
    │  _TransformerEncoder                             │
    │  ┌─ CLS 토큰 선두 추가  → (B, 41, 32)            │
    │  ├─ + pos_embed(1, 41, 32) [학습 가능]           │
    │  ├─ TransformerEncoderLayer × 4                  │
    │  │    └─ MHSA(d=32, h=4) + FFN(32→128→32)        │
    │  └─ x[:, 0] → Linear(32→32) → Sigmoid            │
    └────────────────────────┬─────────────────────────┘
                      (B, 32) latent z
             ┌───────────────┴───────────┐
    bp_head  │                           │ Decoder (훈련 시)
Linear(32→2) │            ┌──────────────┴───────────────┐
             │            │  _TransformerDecoder         │
          (B, 2)          │  Linear(32→32) → (B, 1, 32)  │
        [SBP, DBP]        │  + mask_token(B, 40, 32)     │
                          │  → cat → (B, 41, 32)         │
                          │  + pos_embed(1, 41, 32)      │
                          │  TransformerEncoderLayer × 4 │
                          │  x[:, 1:] → (B, 40, 32)      │
                          │  patch_proj(32→25)           │
                          │  reshape(B, 1, 1000)         │
                          └──────────────┬───────────────┘
                                    (B, 1, 1000)
                                    [재구성 PPG]
```

## 3. 텐서 흐름 요약

### 3.1 추론 경로: forward(x)

| 단계 | 모듈       | 처리                    | 입력 shape   | 출력 shape  |
| ---- | ---------- | ----------------------- | ------------ | ----------- |
| 0    | —          | squeeze(1) if 3D        | (B, 1, 1000) | (B, 1000)   |
| 1    | PatchEmbed | reshape(B, N, P)        | (B, 1000)    | (B, 40, 25) |
| 2    | PatchEmbed | Linear(25→32)           | (B, 40, 25)  | (B, 40, 32) |
| 3    | Encoder    | CLS prepend + cat       | (B, 40, 32)  | (B, 41, 32) |
| 4    | Encoder    | + pos_embed             | (B, 41, 32)  | (B, 41, 32) |
| 5    | Encoder    | TransformerEncoder × 4  | (B, 41, 32)  | (B, 41, 32) |
| 6    | Encoder    | x[:, 0]                 | (B, 41, 32)  | (B, 32)     |
| 7    | Encoder    | Linear(32→32) + Sigmoid | (B, 32)      | (B, 32)     |
| 8    | bp_head    | Linear(32→2)            | (B, 32)      | (B, 2)      |

### 3.2 훈련 경로: compute_loss(x, y, criterion)

단계 0–7은 동일. 단계 7 이후:

| 단계 | 모듈    | 처리                         | 입력 shape               | 출력 shape   |
| ---- | ------- | ---------------------------- | ------------------------ | ------------ |
| 8a   | bp_head | Linear(32→2)                 | (B, 32)                  | (B, 2)       |
| 8b   | Decoder | Linear(32→32) + unsqueeze(1) | (B, 32)                  | (B, 1, 32)   |
| 9    | Decoder | mask_token.expand + cat      | (B, 1, 32) + (B, 40, 32) | (B, 41, 32)  |
| 10   | Decoder | + pos_embed                  | (B, 41, 32)              | (B, 41, 32)  |
| 11   | Decoder | TransformerEncoder × 4       | (B, 41, 32)              | (B, 41, 32)  |
| 12   | Decoder | x[:, 1:]                     | (B, 41, 32)              | (B, 40, 32)  |
| 13   | Decoder | patch_proj(32→25)            | (B, 40, 32)              | (B, 40, 25)  |
| 14   | Decoder | reshape(B, 1, -1)            | (B, 40, 25)              | (B, 1, 1000) |

## 4. 모듈별 상세 설계

### 4.1 _PatchEmbed

**역할**: 1D PPG 신호를 비중첩(non-overlapping) 고정 크기 패치로 분할하고
Transformer 입력 차원으로 선형 투영한다.

```text
입력: (B, L)  [3D이면 squeeze(1) 후]
    │
    ▼  reshape(B, L // patch_size, patch_size)
(B, N, P)     N = L // P = 1000 // 25 = 40
    │
    ▼  Linear(patch_size → d_model)   [학습 가능 투영 행렬]
(B, N, d_model)
```

```text
PPG (B, 1000) 샘플 배치
┌────────────────────────────────────────────┐
│  P0  │  P1  │  P2  │ ... │ P38 │ P39       │
│  25  │  25  │  25  │     │ 25  │  25 샘플  │
└────────────────────────────────────────────┘
 0    25    50    75       950   975  1000

125 Hz 기준 패치 1개 = 200 ms ≈ 0.2~0.25 심박 주기
```

**`patch_size` 설계 시 고려사항**

- `input_length % patch_size == 0` 조건이 생성자에서 `assert`로 강제됨
- 작은 `patch_size` → 패치 수 ↑, 시퀀스 길이 ↑, Attention 연산량 ↑, 고해상도 특징
- 큰 `patch_size` → 패치 수 ↓, 낮은 계산 비용, 각 패치가 더 넓은 시간 구간 포괄

`input_length=1000`에서 유효한 `patch_size` 값 (약수):
5, 8, 10, 20, **25**, 40, 50, 100, 125, 200, 250

CNN 패치 임베딩(stride Conv1d)을 사용하지 않고 단순 `reshape + Linear`을 사용하는
이유는 MAE 원 논문의 방식을 따르면서 패치 경계를 명확히 유지하기 위함이다.
stride Conv는 패치 간 중첩을 허용해 경계 해석이 불분명해진다.

### 4.2 _TransformerEncoder

**역할**: 패치 토큰 시퀀스에서 CLS 토큰을 통해 전체 PPG를 대표하는 잠재 벡터를
추출한다.

```text
입력 tokens : (B, N, d_model)   N = 40 패치
    │
    ├─ cls = cls_token.expand(B, 1, d_model)    (B, 1, 32)  [학습 가능]
    ├─ x = cat([cls, tokens], dim=1)            (B, 41, 32)
    │
    ├─ x = x + pos_embed                        (B, 41, 32) [학습 가능]
    │      pos_embed shape: (1, N+1, d_model) = (1, 41, 32)
    │      인덱스 0: CLS 위치, 1~40: 패치 0~39 위치
    │
    ├─ x = TransformerEncoder(x)                (B, 41, 32)
    │      num_layers=4개 TransformerEncoderLayer 순차 적용
    │
    ├─ cls_out = x[:, 0]                        (B, 32) ← CLS 위치만 추출
    │
    ▼  sigmoid(Linear(d_model → latent_dim)(cls_out))
출력 z : (B, latent_dim=32)  ← [0, 1] 범위의 잠재 벡터
```

#### CLS 토큰 (`nn.Parameter(zeros(1, 1, d_model))`)

시퀀스 앞에 삽입하는 학습 가능한 집계 토큰. ViT와 BERT의 `[CLS]` 토큰과 동일한
역할이다. Transformer의 Self-Attention이 CLS 위치와 모든 패치 위치 사이의 관계를
학습하므로, CLS 출력이 전체 시퀀스의 압축 표현이 된다.

초기화: `trunc_normal_(std=0.02)` — 매우 작은 값으로 초기화해 학습 초기에
CLS가 모든 패치에 균등하게 주목하도록 유도한다.

#### 위치 임베딩 (`nn.Parameter(zeros(1, N+1, d_model))`)

```text
pos_embed[0, 0,  :] ← CLS 위치 임베딩
pos_embed[0, 1,  :] ← 패치 P0 위치 임베딩
pos_embed[0, 2,  :] ← 패치 P1 위치 임베딩
...
pos_embed[0, 40, :] ← 패치 P39 위치 임베딩
```

`x + self.pos_embed` 형태의 합산(additive)으로 적용한다. Transformer는 입력 순서를
인식하지 못하므로 위치 임베딩이 없으면 패치 0과 패치 39가 동일하게 취급된다.

초기화: `trunc_normal_(std=0.02)` — 작은 무작위값으로 초기화.

### 4.3 TransformerEncoderLayer 내부 구조

인코더·디코더 모두 동일한 `nn.TransformerEncoderLayer` 설정을 사용한다.

```text
입력 x : (B, L_seq, d_model)   [batch_first=True]
    │
    ├─ [Multi-Head Self-Attention]
    │    head_dim = d_model // nhead = 32 // 4 = 8
    │    Q = x·W_Q,  K = x·W_K,  V = x·W_V    각 (B, L, 32)
    │    분할 → 4 헤드 각 (B, L, 8)
    │    Attn = softmax(QKᵀ / √8) · V          (B, 4, L, 8)
    │    병합 → (B, L, 32) → 출력 투영 W_O
    │    Dropout(0.1)
    │
    ├─ residual add + LayerNorm    (post-norm, PyTorch 기본값)
    │    x = LayerNorm(x + Attn_out)
    │
    ├─ [Feed-Forward Network]
    │    Linear(32 → 128) + ReLU + Dropout(0.1)
    │    Linear(128 → 32)
    │    Dropout(0.1)
    │
    ▼  residual add + LayerNorm
    x = LayerNorm(x + FFN_out)
출력 : (B, L_seq, d_model)
```

`dim_feedforward = d_model × 4 = 128` — Transformer 원 논문의 관례적 비율.

**Post-norm 구조**: 이 모델은 `norm_first=False`(PyTorch 기본값)를 사용한다.
잔차 합산 후 LayerNorm을 적용하는 Post-norm 방식이다.

#### 레이어별 파라미터 수 (d_model=32, nhead=4)

| 구성요소                                | 파라미터 수       |
| --------------------------------------- | ----------------- |
| MHSA: Q, K, V 투영 (각 32×32 + bias 32) | 3 × 1,056 = 3,168 |
| MHSA: 출력 투영 (32×32 + bias 32)       | 1,056             |
| FFN: Linear(32→128) + bias              | 4,224             |
| FFN: Linear(128→32) + bias              | 4,128             |
| LayerNorm × 2 (weight + bias, 각 32)    | 128               |
| **레이어 합계**                         | **12,704**        |

4개 레이어: 4 × 12,704 = **50,816 파라미터**

### 4.4 _TransformerDecoder

**역할**: 잠재 벡터 z에서 원본 PPG 패치들을 재구성하는 MAE 스타일 디코더.

```text
입력 z : (B, latent_dim=32)
    │
    ├─ lat = Linear(latent_dim → d_model)(z).unsqueeze(1)
    │        (B, 32) → (B, 32) → (B, 1, 32)   ← 잠재 토큰
    │
    ├─ mask = mask_token.expand(B, N, d_model)
    │         (B, 40, 32)   ← 40개 패치 위치의 학습 가능 대리 토큰
    │
    ├─ x = cat([lat, mask], dim=1)              (B, 41, 32)
    │
    ├─ x = x + pos_embed                        (B, 41, 32)
    │      [인코더와 독립된 학습 가능 위치 임베딩]
    │
    ├─ x = TransformerEncoder(x)                (B, 41, 32)
    │      (4 layers, 인코더와 동일 구조)
    │
    ├─ patches = patch_proj(x[:, 1:])
    │            (B, 40, 32) → Linear(32→25) → (B, 40, 25)
    │            ※ x[:, 0] (잠재 토큰 위치)는 버린다
    │
    ▼  patches.reshape(B, 1, -1)
출력 : (B, 1, 1000)   ← 재구성 PPG
```

#### mask_token (`nn.Parameter(zeros(1, N, d_model))`)

N개 패치 위치 각각에 할당된 학습 가능한 단일 토큰. 모든 패치 위치에서
**동일한** mask_token이 사용되므로, Transformer가 위치 임베딩과 잠재 토큰과의
상호작용을 통해 패치별 차이를 만들어내야 한다.

초기화: `trunc_normal_(std=0.02)`

#### 잠재 토큰 위치(`x[:, 0]`)를 재구성에 사용하지 않는 이유

디코더 시퀀스 `[lat, mask_0, mask_1, ..., mask_39]`에서:

- `x[:, 0]`: lat이 Attention을 통해 갱신된 위치 — z의 정보를 집약, PPG 신호 값이 아님
- `x[:, 1:]`: 각 패치 위치에서 z 정보를 흡수한 mask_token 갱신 결과 — 해당 위치의 PPG 값을 예측

`x[:, 0]`을 버리는 것은 MAE 원 논문의 관례를 따른 것으로, 디코더 내부에서
잠재 토큰이 "질의(query)를 받는 답변자" 역할을 하고 실제 출력은 마스크 위치에서
나온다는 설계 의도를 반영한다.

#### 인코더·디코더 위치 임베딩 독립성

| 위치 임베딩        | shape       | 의미                                |
| ------------------ | ----------- | ----------------------------------- |
| 인코더 `pos_embed` | (1, 41, 32) | CLS[0] + 패치 P0~P39 [1~40]         |
| 디코더 `pos_embed` | (1, 41, 32) | 잠재 토큰[0] + 마스크 M0~M39 [1~40] |

두 위치 임베딩은 독립적으로 학습된다. 인코더는 "각 패치 토큰이 어디에 있는가",
디코더는 "어느 위치의 PPG를 예측해야 하는가"를 각자 학습한다.

### 4.5 bp_head

```text
입력 z : (B, latent_dim=32)
    │
    ▼  Linear(latent_dim → 2)
출력 : (B, 2)   ← [SBP, DBP] (mmHg)
```

단층 선형 투영. 잠재 벡터가 Sigmoid 활성화로 이미 [0, 1] 범위 내에 있으므로
별도의 정규화나 활성화가 필요하지 않다.

## 5. compute_loss 훈련 인터페이스

Trainer가 `hasattr(model, "compute_loss")`로 자동 감지해 훈련/검증 루프에서
호출한다.

```python
def compute_loss(self, x, y, criterion):
    x3d = ensure_3d(x)
    z    = self.encoder(self.patch_embed(x3d))   # (B, latent_dim)

    pred  = self.bp_head(z)                       # (B, 2)
    recon = self.decoder(z)                       # (B, 1, 1000)

    bp_loss    = criterion(pred, y)               # 스칼라
    recon_loss = criterion(recon, x3d)            # 스칼라

    loss = (1 - self.recon_weight) * bp_loss
         +      self.recon_weight  * recon_loss
    return loss, pred
```

### 멀티태스크 손실 공식

```text
loss = (1 − recon_weight) × bp_loss + recon_weight × recon_loss
```

- `bp_loss`: `criterion(pred (B,2), y (B,2))` — 혈압 예측 오차
- `recon_loss`: `criterion(recon (B,1,1000), x3d (B,1,1000))` — 재구성 오차

기본 `criterion`은 MAE(Mean Absolute Error)다. MAE의 경우:

- `bp_loss` = (|pred_SBP - y_SBP| + |pred_DBP - y_DBP|) / 2 → mmHg 단위
- `recon_loss` = 1000 × 1개 채널에 대한 평균 절댓값 → 정규화 신호 단위

두 손실의 단위가 다르므로 `recon_weight`는 단순 중요도 비율이 아닌 스케일
조정 역할도 겸한다. 실험적으로 결정해야 한다.

### forward() vs compute_loss() 실행 경로

| 호출                       | 디코더 | 비고                   |
| -------------------------- | ------ | ---------------------- |
| `model(x)`                 | 미실행 | 추론, `print-model` 시 |
| `compute_loss(x, y, crit)` | 실행   | 훈련·검증 공통         |

## 6. 학습 가능 파라미터 목록

기본값 `patch_size=25, d_model=32, nhead=4, num_layers=4, latent_dim=32,
input_length=1000` (N=40 패치) 기준.

| 구성요소   | 파라미터                      | shape       | 개수         |
| ---------- | ----------------------------- | ----------- | ------------ |
| PatchEmbed | `proj.weight`                 | (32, 25)    | 800          |
| PatchEmbed | `proj.bias`                   | (32,)       | 32           |
| Encoder    | `cls_token`                   | (1, 1, 32)  | 32           |
| Encoder    | `pos_embed`                   | (1, 41, 32) | 1,312        |
| Encoder    | TransformerEncoder × 4 layers | —           | 50,816       |
| Encoder    | `fc.weight`                   | (32, 32)    | 1,024        |
| Encoder    | `fc.bias`                     | (32,)       | 32           |
| Decoder    | `fc.weight`                   | (32, 32)    | 1,024        |
| Decoder    | `fc.bias`                     | (32,)       | 32           |
| Decoder    | `mask_token`                  | (1, 40, 32) | 1,280        |
| Decoder    | `pos_embed`                   | (1, 41, 32) | 1,312        |
| Decoder    | TransformerEncoder × 4 layers | —           | 50,816       |
| Decoder    | `patch_proj.weight`           | (25, 32)    | 800          |
| Decoder    | `patch_proj.bias`             | (25,)       | 25           |
| bp_head    | `weight`                      | (2, 32)     | 64           |
| bp_head    | `bias`                        | (2,)        | 2            |
| **합계**   |                               |             | **~109,403** |

## 7. 하이퍼파라미터 참조표

| 파라미터       | 기본값 | 역할                                                |
| -------------- | ------ | --------------------------------------------------- |
| `patch_size`   | 25     | 패치 크기 (샘플 수); `input_length`의 약수 필수     |
| `d_model`      | 32     | Transformer 임베딩 차원                             |
| `nhead`        | 4      | Self-Attention 헤드 수; `d_model % nhead == 0` 필수 |
| `num_layers`   | 4      | 인코더·디코더 각각의 Transformer 레이어 수          |
| `latent_dim`   | 32     | Sigmoid 병목 잠재 벡터 차원                         |
| `recon_weight` | 0.5    | 재구성 손실 가중치 (0: 회귀만, 1: 재구성만)         |
| `input_length` | 1000   | PPG 세그먼트 샘플 수; 위치 임베딩 크기 결정         |

### 유효성 제약

```python
assert input_length % patch_size == 0   # 생성자에서 강제
# d_model % nhead == 0 은 PyTorch TransformerEncoderLayer 내부에서 강제
```

### 규모 조정 예시

```bash
# 경량 (빠른 실험, ~30 K 파라미터)
bin\train-model.bat --model mtae_tr \
    --model-kwargs "d_model=16,num_layers=2,latent_dim=16"

# 중형 (기본값, ~109 K 파라미터)
bin\train-model.bat --model mtae_tr

# 대형 (표현력 증가, ~600 K 파라미터)
bin\train-model.bat --model mtae_tr \
    --model-kwargs "d_model=64,nhead=8,num_layers=6,latent_dim=64"

# 큰 패치 (패치 수 20개, 각 50샘플)
bin\train-model.bat --model mtae_tr \
    --model-kwargs "patch_size=50"
```

## 8. 설계 결정 사항

### 8.1 패치 분할 방식: reshape 기반 비중첩 패치

중첩 패치(stride < patch_size인 Conv1d)나 연속 시계열 직접 입력 대신
비중첩 분할을 사용하는 이유:

- **재구성 목표와의 일치**: 디코더가 각 패치를 독립적으로 예측하므로, 인코더도
  동일한 패치 단위로 입력을 처리하는 것이 논리적으로 일관된다.
- **경계 명확성**: 패치 경계가 명확해 어떤 시간 범위를 예측하는지 추적 가능하다.
- **연산 단순성**: `reshape + Linear`는 Conv1d보다 파라미터가 적고 구현이 단순하다.

### 8.2 인코더·디코더 대칭 구조

인코더와 디코더가 동일한 `d_model`, `nhead`, `num_layers`를 공유한다.

MAE 원 논문(He et al. 2022)에서는 디코더를 인코더보다 가볍게 설계하는 경우가
많다. 이 구현에서 대칭 구조를 사용하는 이유:

- `num_layers` 하이퍼파라미터 하나로 전체 깊이를 일괄 제어할 수 있어 탐색이 용이함
- PPG 시퀀스 길이(40 패치)가 이미지보다 훨씬 짧아 대칭 디코더의 추가 비용이 작음

### 8.3 디코더에서 TransformerEncoder 사용

이름은 "Decoder"이지만 내부적으로 `nn.TransformerEncoder`(양방향 어텐션)를 사용한다.
`nn.TransformerDecoder`(cross-attention 포함)를 사용하지 않는 이유:

- 디코더 입력이 인코더 출력 시퀀스 전체가 아닌 단일 잠재 벡터 z이기 때문이다.
- z는 mask_token 시퀀스 앞에 잠재 토큰으로 삽입되어 Self-Attention을 통해
  mask_token들과 상호작용한다.
- Cross-Attention 기반 TransformerDecoder는 인코더 출력 시퀀스(key/value 소스)가
  별도로 필요한데, 여기서는 z 하나만 있어 적합하지 않다.

### 8.4 Sigmoid 병목

잠재 벡터를 [0, 1]로 제한한다. MTAE 설계서 4.4절 참조.

### 8.5 재구성 손실 스케일 문제

`recon_loss`는 1,000 샘플에 대한 평균 절댓값이고, `bp_loss`는 2개 값에 대한 평균이다.
PPG가 정규화(z-score)되어 있다면 두 손실은 비슷한 범위에 있다. 단, PPG 진폭
범위와 mmHg 범위가 다를 수 있으므로 `recon_weight` 조정이 중요하다.

실험 지침:

- `recon_weight=0.5`: 동등 가중, 기본 출발점
- `recon_weight=0.2~0.3`: 혈압 정확도 우선
- `recon_weight=0.7~0.8`: 재구성 품질 우선 (사전 학습 목적)

## 9. 훈련 방법

### 기본 훈련

```bash
bin\train-model.bat --model mtae_tr
```

### recon_weight 조정

```bash
# 혈압 회귀 우선
bin\train-model.bat --model mtae_tr --model-kwargs "recon_weight=0.2"

# 재구성 보조 없이 순수 회귀 (디코더는 파라미터만 존재, 그래디언트 없음)
bin\train-model.bat --model mtae_tr --model-kwargs "recon_weight=0.0"
```

### patch_size 실험

```bash
# 더 세밀한 패치 (50개 패치 × 20샘플)
bin\train-model.bat --model mtae_tr --model-kwargs "patch_size=20"

# 더 넓은 패치 (20개 패치 × 50샘플)
bin\train-model.bat --model mtae_tr --model-kwargs "patch_size=50"
```

## 10. 모델 검사

```bash
bin\print-model.bat --model mtae_tr
```

출력 예시:

```text
MTAE_TR
  (patch_embed): _PatchEmbed
    (proj): Linear(in=25, out=32)
  (encoder): _TransformerEncoder
    (cls_token): Parameter(1, 1, 32)
    (pos_embed): Parameter(1, 41, 32)
    (transformer): TransformerEncoder(4 layers, d=32, h=4, ff=128)
    (fc): Linear(32→32)
  (decoder): _TransformerDecoder
    (fc): Linear(32→32)
    (mask_token): Parameter(1, 40, 32)
    (pos_embed): Parameter(1, 41, 32)
    (transformer): TransformerEncoder(4 layers, d=32, h=4, ff=128)
    (patch_proj): Linear(32→25)
  (bp_head): Linear(32→2)

Total params    : ~109,403  (~109 K)
Trainable params: ~109,403
Input shape     : (1, 1000)
```

## 11. 참고 문헌

- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., and Girshick, R. (2022).
  "Masked Autoencoders Are Scalable Vision Learners." *CVPR 2022*.
  (MAE 디코더: mask token + Transformer 재구성, 패치 기반 입력)

- Dosovitskiy, A. et al. (2021). "An Image is Worth 16×16 Words: Transformers
  for Image Recognition at Scale." *ICLR 2021*.
  (ViT: CLS 토큰, 패치 임베딩, 학습 가능 위치 임베딩)

- Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). "BERT:
  Pre-training of Deep Bidirectional Transformers for Language Understanding."
  *NAACL-HLT 2019*.
  (CLS 토큰을 통한 시퀀스 수준 표현 집계)

- Vaswani, A. et al. (2017). "Attention Is All You Need." *NeurIPS 2017*.
  (Transformer 기본 구조: MHSA, FFN, Post-Norm)
