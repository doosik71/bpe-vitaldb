"""
Evaluate a trained BPE model on the held-out test set.

Loads best.pt from a run directory, runs inference over the test split, and
writes evaluation results to the same directory:

  eval_results.json   — summary metrics (MAE, RMSE, ME, SD; BHS grade; AAMI)
  eval_plot.png       — predicted vs actual scatter plots for SBP and DBP
  error_hist.png      — error distribution histograms for SBP and DBP

Usage:
    uv run python scripts/eval.py <run_dir> [OPTIONS]
    uv run python scripts/eval.py data/models/resnet1d

Options:
    --dataset-dir   Root dataset directory  (default: data/dataset)
    --device        auto | cpu | cuda | cuda:N  (default: auto)
    --batch-size    Inference batch size    (default: 512)
    --no-normalize  Skip per-segment z-score normalization
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from bpe.models import create_model
from bpe.train.dataset import PPGDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a BPE model checkpoint on the test split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "run_dir", type=Path,
        help="Run directory containing best.pt and config.json",
    )
    p.add_argument(
        "--dataset-dir", type=Path, default=Path("data/dataset"),
        help="Root dataset directory (default: data/dataset)",
    )
    p.add_argument(
        "--device", default="auto",
        help="Device: auto | cpu | cuda | cuda:N  (default: auto)",
    )
    p.add_argument(
        "--batch-size", type=int, default=512,
        help="Inference batch size (default: 512)",
    )
    p.add_argument(
        "--no-normalize", action="store_true",
        help="Skip per-segment z-score normalization of PPG",
    )
    return p.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Return a dict of clinical BP estimation metrics for one channel."""
    err = pred - true
    mae  = float(np.mean(np.abs(err)))
    me   = float(np.mean(err))
    sd   = float(np.std(err, ddof=1))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # BHS cumulative error distribution (≤5, ≤10, ≤15 mmHg)
    n = len(err)
    bhs_5  = float(np.sum(np.abs(err) <= 5)  / n * 100)
    bhs_10 = float(np.sum(np.abs(err) <= 10) / n * 100)
    bhs_15 = float(np.sum(np.abs(err) <= 15) / n * 100)

    # BHS grade: A ≥60%/85%/95%, B ≥50%/75%/90%, C ≥40%/65%/85%, D below C
    if bhs_5 >= 60 and bhs_10 >= 85 and bhs_15 >= 95:
        bhs_grade = "A"
    elif bhs_5 >= 50 and bhs_10 >= 75 and bhs_15 >= 90:
        bhs_grade = "B"
    elif bhs_5 >= 40 and bhs_10 >= 65 and bhs_15 >= 85:
        bhs_grade = "C"
    else:
        bhs_grade = "D"

    # AAMI criterion: |ME| ≤ 5 mmHg and SD ≤ 8 mmHg
    aami_pass = abs(me) <= 5.0 and sd <= 8.0

    return {
        "mae":       mae,
        "me":        me,
        "sd":        sd,
        "rmse":      rmse,
        "bhs_pct_5":  bhs_5,
        "bhs_pct_10": bhs_10,
        "bhs_pct_15": bhs_15,
        "bhs_grade":  bhs_grade,
        "aami_pass":  aami_pass,
        "n_samples":  n,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_scatter(
    pred_sbp: np.ndarray,
    true_sbp: np.ndarray,
    pred_dbp: np.ndarray,
    true_dbp: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, pred, true, label in [
        (axes[0], pred_sbp, true_sbp, "SBP"),
        (axes[1], pred_dbp, true_dbp, "DBP"),
    ]:
        lim_min = min(true.min(), pred.min()) - 5
        lim_max = max(true.max(), pred.max()) + 5
        ax.scatter(true, pred, alpha=0.15, s=4, color="#2196F3", rasterized=True)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1, label="y = x")
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel(f"Actual {label} (mmHg)")
        ax.set_ylabel(f"Predicted {label} (mmHg)")
        ax.set_title(f"{label}: Predicted vs Actual")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_hist(
    err_sbp: np.ndarray,
    err_dbp: np.ndarray,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, err, label in [
        (axes[0], err_sbp, "SBP"),
        (axes[1], err_dbp, "DBP"),
    ]:
        ax.hist(err, bins=80, color="#2196F3", edgecolor="none", alpha=0.8)
        ax.axvline(0,            color="black",  linewidth=1.2, linestyle="--")
        ax.axvline(err.mean(),   color="#F44336", linewidth=1.2, linestyle="-",
                   label=f"ME = {err.mean():.2f} mmHg")
        ax.set_xlabel(f"{label} Error (mmHg)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} Error Distribution")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predictions, targets) as float32 numpy arrays of shape (N, 2)."""
    model.eval()
    preds_list, targets_list = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x)
            preds_list.append(pred.cpu().numpy())
            targets_list.append(y.numpy())

    preds   = np.concatenate(preds_list,   axis=0)
    targets = np.concatenate(targets_list, axis=0)
    return preds.astype(np.float32), targets.astype(np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    run_dir = args.run_dir
    device  = resolve_device(args.device)

    # ── Validate run directory ────────────────────────────────────────────────
    cfg_path  = run_dir / "config.json"
    ckpt_path = run_dir / "best.pt"

    for path in (cfg_path, ckpt_path):
        if not path.exists():
            print(f"ERROR: {path} not found.", file=sys.stderr)
            sys.exit(1)

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    model_name = cfg.get("model")
    if not model_name:
        print("ERROR: 'model' key missing from config.json.", file=sys.stderr)
        sys.exit(1)

    print(f"Run directory : {run_dir}")
    print(f"Model         : {model_name}")
    print(f"Checkpoint    : {ckpt_path}")
    print(f"Device        : {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = create_model(model_name).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    best_epoch = ckpt.get("epoch", "?")
    print(f"Loaded best.pt (epoch {best_epoch}, val_loss={ckpt.get('val_loss', float('nan')):.4f})")

    # ── Test dataset ──────────────────────────────────────────────────────────
    test_dir = args.dataset_dir / "test"
    try:
        test_ds = PPGDataset(
            test_dir,
            normalize=not args.no_normalize,
            preload=False,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(f"Run bin/construct-dataset first to build the dataset.", file=sys.stderr)
        sys.exit(1)

    print(f"Test set      : {len(test_ds):,} segments from {test_ds.n_files} cases")

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    print("Running inference …")
    preds, targets = run_inference(model, loader, device)

    pred_sbp, pred_dbp = preds[:, 0],   preds[:, 1]
    true_sbp, true_dbp = targets[:, 0], targets[:, 1]
    err_sbp = pred_sbp - true_sbp
    err_dbp = pred_dbp - true_dbp

    # ── Compute metrics ───────────────────────────────────────────────────────
    sbp_m = compute_metrics(pred_sbp, true_sbp)
    dbp_m = compute_metrics(pred_dbp, true_dbp)

    print()
    print(f"{'Metric':<18}  {'SBP':>10}  {'DBP':>10}")
    print("-" * 42)
    for key, fmt in [
        ("n_samples", "d"),
        ("mae",       ".2f"),
        ("me",        ".2f"),
        ("sd",        ".2f"),
        ("rmse",      ".2f"),
        ("bhs_pct_5",  ".1f"),
        ("bhs_pct_10", ".1f"),
        ("bhs_pct_15", ".1f"),
        ("bhs_grade",  "s"),
        ("aami_pass",  "s"),
    ]:
        sv = sbp_m[key]
        dv = dbp_m[key]
        if fmt == "d":
            print(f"  {key:<16}  {sv:>10d}  {dv:>10d}")
        elif fmt == "s":
            print(f"  {key:<16}  {str(sv):>10}  {str(dv):>10}")
        else:
            print(f"  {key:<16}  {sv:>10{fmt}}  {dv:>10{fmt}}")
    print()

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "run_dir":    str(run_dir),
        "model":      model_name,
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "test_dir":   str(test_dir),
        "n_segments": int(len(test_ds)),
        "n_cases":    int(test_ds.n_files),
        "sbp":        sbp_m,
        "dbp":        dbp_m,
    }
    results_path = run_dir / "eval_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {results_path}")

    scatter_path = run_dir / "eval_plot.png"
    plot_scatter(pred_sbp, true_sbp, pred_dbp, true_dbp, scatter_path)
    print(f"Saved: {scatter_path}")

    hist_path = run_dir / "error_hist.png"
    plot_error_hist(err_sbp, err_dbp, hist_path)
    print(f"Saved: {hist_path}")


if __name__ == "__main__":
    main()
