"""Multi-Task AutoEncoder with Transformer backbone (MTAE_TR).

Encoder: patch embedding → Transformer encoder (CLS token) → sigmoid latent
Decoder: MAE-style — latent token + learnable position queries → Transformer → patch projection
BP head: Linear(latent_dim, 2)
"""

import torch
from torch import nn

from bpe.models.blocks import ensure_3d
from bpe.models.registry import register_model


class _PatchEmbed(nn.Module):
    """Split a 1-D PPG segment into non-overlapping patches and project to d_model."""

    def __init__(self, patch_size: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.squeeze(1)                                    # (B, L)
        B, L = x.shape
        x = x.reshape(B, L // self.patch_size, self.patch_size) # (B, N, P)
        return self.proj(x)                                      # (B, N, d_model)


class _TransformerEncoder(nn.Module):
    """Transformer encoder with a prepended CLS token.

    Outputs a ``latent_dim``-dimensional sigmoid vector from the CLS position.
    """

    def __init__(
        self,
        num_patches: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        latent_dim: int,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, latent_dim)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B = tokens.size(0)
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, d_model)
        x = torch.cat([cls, tokens], dim=1)              # (B, N+1, d_model)
        x = self.transformer(x + self.pos_embed)
        return torch.sigmoid(self.fc(x[:, 0]))           # (B, latent_dim)


class _TransformerDecoder(nn.Module):
    """MAE-style decoder.

    The latent token is prepended to N learnable mask tokens (one per patch
    position).  A Transformer encoder then refines all positions jointly, and
    each patch token is projected back to ``patch_size`` values.
    """

    def __init__(
        self,
        num_patches: int,
        patch_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        latent_dim: int,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.fc = nn.Linear(latent_dim, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, num_patches, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.patch_proj = nn.Linear(d_model, patch_size)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        lat = self.fc(z).unsqueeze(1)                    # (B, 1, d_model)
        mask = self.mask_token.expand(B, -1, -1)         # (B, N, d_model)
        x = torch.cat([lat, mask], dim=1)                # (B, N+1, d_model)
        x = self.transformer(x + self.pos_embed)
        patches = self.patch_proj(x[:, 1:])              # (B, N, patch_size)
        return patches.reshape(B, 1, -1)                 # (B, 1, L)


@register_model("mtae_tr")
class MTAE_TR(nn.Module):
    """Multi-Task AutoEncoder with Transformer backbone.

    Replaces the CNN encoder/decoder of MTAE with Transformer layers.
    The encoder uses a CLS token to produce a sigmoid-activated latent vector;
    the decoder follows the MAE paradigm to reconstruct all patch positions.

    Args:
        patch_size:    Samples per patch.  Must divide input length.  Default: 25.
        d_model:       Transformer embedding dimension.  Default: 32.
        nhead:         Number of attention heads.  Default: 4.
        num_layers:    Transformer layers in encoder and decoder each.  Default: 4.
        latent_dim:    Bottleneck size.  Default: 32.
        recon_weight:  Reconstruction loss weight in [0, 1].  Default: 0.5.
        input_length:  PPG segment length in samples.  Default: 1000.
    """

    def __init__(
        self,
        patch_size: int = 25,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 4,
        latent_dim: int = 32,
        recon_weight: float = 0.5,
        input_length: int = 1000,
    ):
        super().__init__()
        assert input_length % patch_size == 0, (
            f"input_length ({input_length}) must be divisible by patch_size ({patch_size})"
        )
        num_patches = input_length // patch_size

        self.latent_dim = latent_dim
        self.recon_weight = recon_weight

        self.patch_embed = _PatchEmbed(patch_size, d_model)
        self.encoder = _TransformerEncoder(num_patches, d_model, nhead, num_layers, latent_dim)
        self.decoder = _TransformerDecoder(num_patches, patch_size, d_model, nhead, num_layers, latent_dim)
        self.bp_head = nn.Linear(latent_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return BP predictions, shape (B, 2) — [SBP, DBP]."""
        return self.bp_head(self.encoder(self.patch_embed(x)))

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-task loss; called by Trainer when this method exists.

        Returns:
            (loss, pred) where ``pred`` has shape (B, 2).
        """
        x3d = ensure_3d(x)
        z = self.encoder(self.patch_embed(x3d))
        pred = self.bp_head(z)
        recon = self.decoder(z)

        bp_loss = criterion(pred, y)
        recon_loss = criterion(recon, x3d)

        loss = (1.0 - self.recon_weight) * bp_loss + self.recon_weight * recon_loss
        return loss, pred
