"""
Show training progress for a BPE model run.

Reads metrics.csv from a run directory and writes two PNG graphs:
  loss_graph.png  — train_loss vs val_loss per epoch
  mae_graph.png   — SBP/DBP MAE (train + val) per epoch

Usage:
    uv run python scripts/train-status.py <run_dir>
    uv run python scripts/train-status.py data/models/resnet1d/20260101_120000
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot training metrics from a BPE run directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "run_dir", type=Path,
        help="Path to run directory (data/models/<model>/<datetime>)",
    )
    p.add_argument(
        "--no-save", action="store_true",
        help="Print summary only; do not write PNG files",
    )
    return p.parse_args()


def load_metrics(csv_path: Path) -> dict[str, list]:
    """Return columns as lists of floats."""
    cols: dict[str, list] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v))
    return cols


def plot_loss(cols: dict, out_path: Path) -> None:
    epochs = cols["epoch"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, cols["train_loss"], label="train_loss", color="#2196F3")
    ax.plot(epochs, cols["val_loss"],   label="val_loss",   color="#F44336")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Huber)")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mae(cols: dict, out_path: Path) -> None:
    epochs = cols["epoch"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, cols["train_sbp_mae"], label="train SBP MAE", color="#2196F3", linestyle="-")
    ax.plot(epochs, cols["train_dbp_mae"], label="train DBP MAE", color="#2196F3", linestyle="--")
    ax.plot(epochs, cols["val_sbp_mae"],   label="val SBP MAE",   color="#F44336", linestyle="-")
    ax.plot(epochs, cols["val_dbp_mae"],   label="val DBP MAE",   color="#F44336", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (mmHg)")
    ax.set_title("SBP / DBP Mean Absolute Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def print_summary(run_dir: Path, cols: dict) -> None:
    epochs = cols["epoch"]
    val_losses = cols["val_loss"]
    best_idx = val_losses.index(min(val_losses))
    best_ep = int(epochs[best_idx])
    total_ep = int(epochs[-1])

    print(f"\nRun directory : {run_dir}")
    print(f"Epochs logged : {total_ep}  (best epoch: {best_ep})")
    print()
    print(f"{'Metric':<22}  {'Last':>8}  {'Best':>8}")
    print("-" * 44)
    for label, key in [
        ("train_loss",    "train_loss"),
        ("val_loss",      "val_loss"),
        ("train_sbp_mae", "train_sbp_mae"),
        ("train_dbp_mae", "train_dbp_mae"),
        ("val_sbp_mae",   "val_sbp_mae"),
        ("val_dbp_mae",   "val_dbp_mae"),
    ]:
        last = cols[key][-1]
        best = min(cols[key]) if "loss" in key or "mae" in key else max(cols[key])
        print(f"  {label:<20}  {last:>8.4f}  {best:>8.4f}")
    print()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir

    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        print(f"ERROR: metrics.csv not found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    cols = load_metrics(csv_path)
    if not cols.get("epoch"):
        print("ERROR: metrics.csv is empty.", file=sys.stderr)
        sys.exit(1)

    print_summary(run_dir, cols)

    if not args.no_save:
        loss_path = run_dir / "loss_graph.png"
        mae_path  = run_dir / "mae_graph.png"
        plot_loss(cols, loss_path)
        plot_mae(cols, mae_path)
        print(f"Saved: {loss_path}")
        print(f"Saved: {mae_path}")


if __name__ == "__main__":
    main()
