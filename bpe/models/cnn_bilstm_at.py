"""CNN-BiLSTM-AT: Hybrid CNN–BiLSTM with Additive Attention for BP estimation.

Reference:
    Mohammadi et al., "Cuff-less blood pressure monitoring via PPG signals
    using a hybrid CNN-BiLSTM deep learning model with attention mechanism",
    Scientific Reports, vol. 15, p. 22229, 2025.
    https://doi.org/10.1038/s41598-025-07087-2

Architecture (Section "Deep learning model" / Table 4):
    3 × [Conv1d → ReLU → MaxPool1d]    local feature extraction
    2 × BiLSTM(128 units)              temporal dependency capture
    1 × Additive attention             focus on relevant time steps
    1 × Linear(256 → 2)               SBP / DBP regression

Attention (Eq. 2–4):
    e_t   = tanh(W_a · h_t + b_a)
    α_t   = softmax(e_t)
    c     = Σ_t  α_t · h_t

Optimal hyperparameters (Table 4):
    filters     : [32, 64, 128]        (one per CNN layer)
    kernel_size : 3
    pool_size   : 2  (MaxPool, stride=1)
    lstm_units  : 128
    dropout     : 0.2
    optimizer   : Adam, lr=0.001

Adaptations for VitalDB:
    * Input: 1 000-sample PPG segments (8 s @ 125 Hz) vs. the paper's
      1 024-sample segments (8.192 s @ 125 Hz).  The architecture is
      length-agnostic so no structural change is required.
    * Output: [SBP, DBP] in mmHg, matching the VitalDB label format.
"""

import torch
from torch import nn

from bpe.models.blocks import ensure_3d
from bpe.models.registry import register_model


class _AdditiveSelfAttention(nn.Module):
    """Single-layer additive attention over a sequence of hidden states.

    Implements Equations 2–4 from the paper:

        e_t   = tanh(W_a · h_t + b_a)
        α_t   = softmax({e_t})
        c     = Σ_t  α_t · h_t

    Args:
        hidden_size: Dimension of each hidden state h_t.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted context vector.

        Args:
            h: Hidden states of shape (B, T, H).

        Returns:
            Context vector of shape (B, H).
        """
        e = torch.tanh(self.score(h))        # (B, T, 1)
        alpha = torch.softmax(e, dim=1)      # (B, T, 1)
        return (alpha * h).sum(dim=1)        # (B, H)


@register_model("cnn_bilstm_at")
class CNNBiLSTMAttention(nn.Module):
    """Hybrid CNN–BiLSTM with additive attention for cuffless BP estimation.

    Pipeline:
        PPG (B, L)
          → 3 × [Conv1d + ReLU + MaxPool1d]   (B, 128, L')
          → transpose                           (B, L', 128)
          → BiLSTM × 2                          (B, L', 256)
          → additive attention                  (B, 256)
          → Linear                              (B, 2) = [SBP, DBP]

    Sequence length after CNN:
        Each MaxPool1d(kernel_size=2, stride=1) reduces length by 1.
        Three pooling layers: L' = L − 3  (e.g. 1000 → 997).

    Args:
        filters:      Output channels for each CNN layer (default [32, 64, 128]).
        kernel_size:  Convolution kernel width (default 3).
        pool_size:    MaxPool1d kernel size (default 2, stride fixed at 1).
        lstm_units:   Hidden units per direction in each BiLSTM (default 128).
        dropout:      Dropout probability applied after each BiLSTM (default 0.2).
        out_features: Output dimension — 2 for [SBP, DBP] (default 2).
    """

    def __init__(
        self,
        filters: tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 3,
        pool_size: int = 2,
        lstm_units: int = 128,
        dropout: float = 0.2,
        out_features: int = 2,
    ) -> None:
        super().__init__()

        # ── CNN feature extraction ──────────────────────────────────────────
        # Three Conv1d + ReLU + MaxPool1d blocks.
        # Padding = kernel_size // 2 preserves length after convolution;
        # MaxPool1d(kernel_size=pool_size, stride=1) reduces length by
        # (pool_size - 1) per layer, matching the paper's pool_stride=1 choice.
        cnn_blocks: list[nn.Module] = []
        in_ch = 1
        for out_ch in filters:
            cnn_blocks += [
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                          padding=kernel_size // 2),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=pool_size, stride=1),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_blocks)

        # ── BiLSTM temporal modelling ───────────────────────────────────────
        # Two stacked BiLSTM layers.  The first receives CNN feature maps
        # (in_ch channels); its output size is lstm_units * 2 due to the
        # bidirectional concatenation, which becomes the input to the second.
        self.bilstm1 = nn.LSTM(
            in_ch, lstm_units, batch_first=True, bidirectional=True
        )
        self.drop1 = nn.Dropout(dropout)
        self.bilstm2 = nn.LSTM(
            lstm_units * 2, lstm_units, batch_first=True, bidirectional=True
        )
        self.drop2 = nn.Dropout(dropout)

        # ── Additive attention ──────────────────────────────────────────────
        self.attention = _AdditiveSelfAttention(lstm_units * 2)

        # ── Regression head ─────────────────────────────────────────────────
        self.head = nn.Linear(lstm_units * 2, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: PPG waveform tensor of shape (B, L) or (B, 1, L).

        Returns:
            Tensor of shape (B, 2) with [SBP, DBP] predictions in mmHg.
        """
        x = ensure_3d(x)                       # (B, 1, L)

        x = self.cnn(x)                        # (B, 128, L-3)
        x = x.transpose(1, 2)                  # (B, L-3, 128)

        h1, _ = self.bilstm1(x)                # (B, L-3, 256)
        h1 = self.drop1(h1)

        h2, _ = self.bilstm2(h1)               # (B, L-3, 256)
        h2 = self.drop2(h2)

        c = self.attention(h2)                 # (B, 256)
        return self.head(c)                    # (B, 2)
