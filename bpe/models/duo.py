"""Evaluation-only ensemble of two models with inter-model disagreement rejection.

DuoModel loads two pre-trained models, runs them in parallel on each input
segment, and accepts the measurement only when both models agree within a
configurable threshold (default: 5 mmHg) on *both* SBP and DBP.

Accepted prediction = average of the two model outputs.

This model does not support training: attempting to call ``.train(True)`` is a
no-op and all parameters have ``requires_grad=False``.
"""

import json
from pathlib import Path

import torch
from torch import nn

from bpe.models.registry import create_model


def _load_model(run_dir: Path, device: torch.device) -> tuple[nn.Module, str]:
    """Load a trained model from a run directory (config.json + best.pt).

    Returns:
        (model, model_name_string)
    """
    cfg_path  = run_dir / "config.json"
    ckpt_path = run_dir / "best.pt"

    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"best.pt not found: {ckpt_path}")

    model_name = json.loads(cfg_path.read_text(encoding="utf-8"))["model"]
    model = create_model(model_name).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, model_name


class DuoModel(nn.Module):
    """Evaluation-only ensemble with disagreement-based measurement rejection.

    Combines two independently trained models.  A segment is *accepted* when
    both models agree within ``threshold`` mmHg on **both** SBP and DBP.
    The final prediction for accepted segments is the simple average of the two
    model outputs.

    Args:
        model_a_id: Subdirectory name (within ``models_dir``) for model A.
        model_b_id: Subdirectory name (within ``models_dir``) for model B.
        models_dir: Root directory containing per-model run directories.
        threshold:  Rejection threshold in mmHg.  Reject when
                    ``|SBP_A - SBP_B| >= threshold`` **or**
                    ``|DBP_A - DBP_B| >= threshold``.
        device:     Torch device to load both models onto.

    Note:
        This class is intentionally never trainable.  Calling ``train(True)``
        is silently ignored and all parameters are frozen.
    """

    def __init__(
        self,
        model_a_id: str = "conv_reg_ds",
        model_b_id: str = "mtae",
        models_dir: Path | str = Path("data/models"),
        threshold: float = 5.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        self.model_a_id = model_a_id
        self.model_b_id = model_b_id
        self.threshold   = threshold

        _device = device or torch.device("cpu")
        _mdir   = Path(models_dir)

        self.model_a, _a = _load_model(_mdir / model_a_id, _device)
        self.model_b, _b = _load_model(_mdir / model_b_id, _device)

        # Lock to eval — no gradient ever needed
        super().train(False)

    # ── Prevent accidental training ───────────────────────────────────────────

    def train(self, mode: bool = True) -> "DuoModel":  # type: ignore[override]
        """Always stay in eval mode."""
        return super().train(False)

    # ── Inference ─────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return averaged predictions (B, 2) for all segments.

        For rejection-aware inference use :meth:`forward_with_mask` instead.
        """
        return (self.model_a(x) + self.model_b(x)) / 2

    def forward_with_mask(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(avg_pred, accepted)`` for a batch.

        A segment is accepted when both models agree within ``self.threshold``
        mmHg on **both** SBP and DBP:

        .. code-block:: text

            accepted_i = (|SBP_A_i - SBP_B_i| < threshold)
                       & (|DBP_A_i - DBP_B_i| < threshold)

        Returns:
            avg_pred:  Shape ``(B, 2)``.  Averaged SBP/DBP predictions.
            accepted:  Shape ``(B,)``.  Bool tensor — ``True`` = accepted.
        """
        pred_a = self.model_a(x)   # (B, 2)
        pred_b = self.model_b(x)   # (B, 2)
        diff   = torch.abs(pred_a - pred_b)                           # (B, 2)
        accepted = (diff[:, 0] < self.threshold) & (diff[:, 1] < self.threshold)
        return (pred_a + pred_b) / 2, accepted
