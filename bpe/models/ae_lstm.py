"""AE-LSTM: Autoencoder-LSTM for blood pressure estimation from PPG.

Reference:
    Vanithamani R. et al., "Deep learning approaches for continuous blood
    pressure estimation from photoplethysmography signal," Measurement:
    Sensors, vol. 40, p. 101866, 2025.
    https://doi.org/10.1016/j.measen.2025.101866

Architecture (Section 3.5 + Table 1):
    Encoder LSTM  : PPG sequence (B, L, 1) → latent vector (B, hidden_size)
                    via final hidden state of a single LSTM layer + Dropout
    Decoder LSTM  : latent → expand to (B, L, hidden_size) → LSTM →
                    Linear(1) → reconstructed PPG (B, 1, L)
    BP Head       : Linear(hidden_size, 2) → [SBP, DBP]

Hyperparameters from Table 1:
    hidden_size : 64   (paper states ~100 hidden units; 64 chosen for efficiency)
    dropout     : 0.2
    recon_weight: 0.5  (paper does not specify; follows MTAE convention)

Adaptations for VitalDB:
    * Input: 1,000-sample PPG segments (8 s @ 125 Hz); paper uses 256 Hz wrist PPG.
    * Output: [SBP, DBP] scalar regression (mmHg) instead of paper's 3-class
      BP classification with SoftMax + cross-entropy.
    * Optimizer: Adam (paper uses SGDM; project-standard optimizer).
    * Loss: MSE/Huber for both reconstruction and BP regression.
    * The combined multi-task loss mirrors the MTAE convention used in this project:
          loss = (1 − recon_weight) * bp_loss + recon_weight * recon_loss
"""

import torch
from torch import nn

from bpe.models.blocks import ensure_3d
from bpe.models.registry import register_model


class _LSTMEncoder(nn.Module):
    """LSTM encoder: PPG sequence → latent vector via final hidden state."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L) → (B, L, 1)
        _, (h_n, _) = self.lstm(x.permute(0, 2, 1))
        return self.drop(h_n.squeeze(0))   # (B, hidden_size)


class _LSTMDecoder(nn.Module):
    """LSTM decoder: latent vector → PPG sequence reconstruction."""

    def __init__(self, hidden_size: int, seq_len: int = 1000) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, hidden_size) → (B, L, hidden_size)
        z_seq = z.unsqueeze(1).expand(-1, self.seq_len, -1).contiguous()
        h, _ = self.lstm(z_seq)           # (B, L, hidden_size)
        recon = self.out_proj(h)          # (B, L, 1)
        return recon.permute(0, 2, 1)     # (B, 1, L)


@register_model("ae_lstm")
class AE_LSTM(nn.Module):
    """Autoencoder-LSTM for cuffless BP estimation from PPG signals.

    An LSTM encoder compresses the full PPG waveform into a latent vector
    by processing the signal as a time series and retaining the final hidden
    state.  A decoder LSTM reconstructs the original signal from this latent
    representation as a self-supervised auxiliary task, which improves the
    encoder's feature quality.  A linear BP head regresses [SBP, DBP] from
    the same latent vector.

    The combined loss used during training::

        loss = (1 - recon_weight) * bp_loss + recon_weight * recon_loss

    Args:
        hidden_size:   LSTM hidden units.  Default: 64 (paper uses ~100).
        dropout:       Dropout probability applied to the encoder output.
                       Default: 0.2 (paper Table 1).
        seq_len:       Input/output sequence length (samples).  Default: 1000
                       (8 s @ 125 Hz as used by this project).
        recon_weight:  Reconstruction-loss weight in [0, 1].  Default: 0.5.
        out_features:  Output dimension — 2 for [SBP, DBP].  Default: 2.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        dropout: float = 0.2,
        seq_len: int = 1000,
        recon_weight: float = 0.5,
        out_features: int = 2,
    ) -> None:
        super().__init__()
        self.recon_weight = recon_weight
        self.encoder = _LSTMEncoder(hidden_size, dropout)
        self.decoder = _LSTMDecoder(hidden_size, seq_len)
        self.bp_head = nn.Linear(hidden_size, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return BP predictions, shape (B, 2) — [SBP, DBP]."""
        return self.bp_head(self.encoder(ensure_3d(x)))

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute joint BP regression + PPG reconstruction loss.

        Called by the Trainer when this method is present.

        Returns:
            (loss, pred) where ``pred`` has shape (B, 2).
        """
        x3d = ensure_3d(x)
        z = self.encoder(x3d)
        pred = self.bp_head(z)
        recon = self.decoder(z)

        bp_loss = criterion(pred, y)
        recon_loss = criterion(recon, x3d)

        loss = (1.0 - self.recon_weight) * bp_loss + self.recon_weight * recon_loss
        return loss, pred
