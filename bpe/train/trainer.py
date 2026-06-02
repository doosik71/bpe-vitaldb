"""Training loop, checkpointing, and early stopping for BPE models.

Typical usage::

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=nn.HuberLoss(delta=5.0),
        device=device,
        run_dir=Path("data/models/resnet1d/20260102_120000"),
    )
    result = trainer.fit(train_loader, val_loader, epochs=100, patience=15)
"""

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


class Trainer:
    """Train a direct-regression BP model and checkpoint the best weights.

    The trainer saves two checkpoints under ``run_dir``:

    - ``best.pt``  — state at the epoch with the lowest validation loss.
    - ``last.pt``  — state at the end of training (or when early stopping fires).

    Epoch metrics are appended to ``metrics.csv`` and printed to stdout.

    Args:
        model:      PyTorch module to train.
        optimizer:  Optimizer instance.
        scheduler:  LR scheduler updated once per epoch (after validation).
                    Pass ``None`` to skip scheduling.
        criterion:  Loss function, e.g. ``nn.HuberLoss(delta=5.0)``.
        device:     Target device (``torch.device``).
        run_dir:    Directory for checkpoints and ``metrics.csv``.
        grad_clip:  Maximum gradient norm (0 to disable).
    """

    def __init__(
        self,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        criterion: nn.Module,
        device:    torch.device,
        run_dir:   Path,
        grad_clip: float = 1.0,
    ):
        self.model     = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device    = device
        self.run_dir   = run_dir
        self.grad_clip = grad_clip

        run_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path  = run_dir / "metrics.csv"
        self._csv_file  = None
        self._csv_writer = None

    # ── Public interface ──────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        *,
        epochs:  int = 100,
        patience: int = 15,
    ) -> dict:
        """Run the full training loop.

        Returns:
            dict with keys ``best_epoch``, ``best_val_loss``,
            ``best_val_sbp_mae``, ``best_val_dbp_mae``.
        """
        self._open_csv()
        best_val_loss = float("inf")
        best_epoch    = 0
        best_metrics: dict = {}
        no_improve    = 0

        log.info("Starting training for up to %d epochs (patience=%d)", epochs, patience)
        log.info("Run directory: %s", self.run_dir)

        try:
            for epoch in range(1, epochs + 1):
                t0 = time.perf_counter()
                train_m = self._run_epoch(train_loader, training=True)
                val_m   = self._run_epoch(val_loader,   training=False)
                elapsed = time.perf_counter() - t0

                lr = self._current_lr()
                self._log_epoch(epoch, epochs, train_m, val_m, lr, elapsed)
                self._write_csv(epoch, train_m, val_m, lr)

                is_best = val_m["loss"] < best_val_loss
                if is_best:
                    best_val_loss  = val_m["loss"]
                    best_epoch     = epoch
                    best_metrics   = val_m
                    no_improve     = 0
                    self._save_checkpoint("best.pt", epoch, val_m)
                else:
                    no_improve += 1

                self._save_checkpoint("last.pt", epoch, val_m)

                if self.scheduler is not None:
                    self.scheduler.step()

                if no_improve >= patience:
                    log.info(
                        "Early stopping: no improvement for %d epochs (best epoch %d).",
                        patience, best_epoch,
                    )
                    break

        finally:
            self._close_csv()

        result = {
            "best_epoch":     best_epoch,
            "best_val_loss":  best_val_loss,
            "best_val_sbp_mae": best_metrics.get("sbp_mae", float("nan")),
            "best_val_dbp_mae": best_metrics.get("dbp_mae", float("nan")),
        }
        log.info(
            "Training complete.  Best epoch %d — val_loss=%.4f  SBP_MAE=%.2f  DBP_MAE=%.2f",
            result["best_epoch"],
            result["best_val_loss"],
            result["best_val_sbp_mae"],
            result["best_val_dbp_mae"],
        )
        return result

    # ── Epoch loop ────────────────────────────────────────────────────────────

    def _run_epoch(self, loader: DataLoader, *, training: bool) -> dict:
        self.model.train(training)
        total_loss = 0.0
        sbp_abs    = 0.0
        dbp_abs    = 0.0
        n          = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                pred = self.model(x)
                loss = self.criterion(pred, y)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    self.optimizer.step()

                bs          = x.size(0)
                total_loss += loss.item() * bs
                sbp_abs    += (pred[:, 0] - y[:, 0]).abs().sum().item()
                dbp_abs    += (pred[:, 1] - y[:, 1]).abs().sum().item()
                n          += bs

        return {
            "loss":    total_loss / n,
            "sbp_mae": sbp_abs    / n,
            "dbp_mae": dbp_abs    / n,
            "mae":     (sbp_abs + dbp_abs) / (2 * n),
        }

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, name: str, epoch: int, metrics: dict) -> None:
        path = self.run_dir / name
        torch.save(
            {
                "epoch":                epoch,
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss":             metrics["loss"],
                "val_sbp_mae":          metrics["sbp_mae"],
                "val_dbp_mae":          metrics["dbp_mae"],
            },
            path,
        )

    # ── Logging helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _fmt(metrics: dict, prefix: str) -> str:
        return (
            f"{prefix}_loss={metrics['loss']:.4f}  "
            f"{prefix}_SBP={metrics['sbp_mae']:.2f}  "
            f"{prefix}_DBP={metrics['dbp_mae']:.2f}"
        )

    def _log_epoch(
        self,
        epoch: int,
        total: int,
        train_m: dict,
        val_m:   dict,
        lr:      float,
        elapsed: float,
    ) -> None:
        log.info(
            "Epoch %3d/%d  %s  |  %s  lr=%.2e  [%.1fs]",
            epoch, total,
            self._fmt(train_m, "train"),
            self._fmt(val_m,   "val"),
            lr, elapsed,
        )

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    # ── CSV log ───────────────────────────────────────────────────────────────

    _CSV_FIELDS = [
        "epoch",
        "train_loss", "train_sbp_mae", "train_dbp_mae",
        "val_loss",   "val_sbp_mae",   "val_dbp_mae",
        "lr",
    ]

    def _open_csv(self) -> None:
        write_header = not self._csv_path.exists()
        self._csv_file   = open(self._csv_path, "a", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._CSV_FIELDS)
        if write_header:
            self._csv_writer.writeheader()

    def _close_csv(self) -> None:
        if self._csv_file:
            self._csv_file.close()
            self._csv_file   = None
            self._csv_writer = None

    def _write_csv(
        self, epoch: int, train_m: dict, val_m: dict, lr: float
    ) -> None:
        if self._csv_writer is None:
            return
        self._csv_writer.writerow(
            {
                "epoch":         epoch,
                "train_loss":    f"{train_m['loss']:.6f}",
                "train_sbp_mae": f"{train_m['sbp_mae']:.4f}",
                "train_dbp_mae": f"{train_m['dbp_mae']:.4f}",
                "val_loss":      f"{val_m['loss']:.6f}",
                "val_sbp_mae":   f"{val_m['sbp_mae']:.4f}",
                "val_dbp_mae":   f"{val_m['dbp_mae']:.4f}",
                "lr":            f"{lr:.2e}",
            }
        )
        self._csv_file.flush()
