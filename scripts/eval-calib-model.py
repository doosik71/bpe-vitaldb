"""
Calibrate a trained BPE model on a single test case and evaluate.

Loads best.pt from a run directory, freezes all layers except the last
(BP output) linear layer, fine-tunes that layer for 1 epoch on a small
subset of segments from the specified test case, then evaluates the
calibrated model on all segments of that case.

Files written to data/models/<model_id>/<case_id>/:
  calib.pt          — calibrated model checkpoint
  eval_results.json — MAE, RMSE, ME, SD; BHS grade; AAMI pass/fail
  eval_plot.png     — predicted vs actual scatter plots for SBP and DBP
  error_hist.png    — error distribution histograms for SBP and DBP
  bland_altman.png  — Bland-Altman plots for SBP and DBP

Usage:
    uv run python scripts/eval-calib-model.py <run_dir> [OPTIONS]
    uv run python scripts/eval-calib-model.py data/models/resnet1d
    uv run python scripts/eval-calib-model.py data/models/resnet1d --case-id 1234 --n-calib 20
    uv run python scripts/eval-calib-model.py data/models/resnet1d --n-calib 0.1  # 10 % of segments

Options:
    --case-id       Test case ID (default: first test case by numeric sort)
    --n-calib       Segments for calibration: integer count (>=1) or fraction
                    (0 < n < 1).  (default: 10)
    --dataset-dir   Root dataset directory  (default: data/dataset)
    --device        auto | cpu | cuda | cuda:N  (default: auto)
    --batch-size    Inference / calibration batch size  (default: 512)
    --lr            Learning rate for calibration  (default: 1e-4)
    --seed          Random seed for segment selection  (default: 42)
    --no-normalize  Skip per-segment z-score normalization
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from bpe.models import create_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrate a BPE model on a single test case and evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "run_dir", type=Path,
        help="Run directory (contains best.pt + config.json)",
    )
    p.add_argument(
        "--case-id", default=None,
        help="Test case ID (default: first test case by numeric sort)",
    )
    p.add_argument(
        "--n-calib", type=float, default=10.0,
        help="Segments for calibration: count (>=1) or fraction (0<n<1)  "
             "(default: 10)",
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
        help="Inference / calibration batch size (default: 512)",
    )
    p.add_argument(
        "--lr", type=float, default=1e-4,
        help="Calibration learning rate (default: 1e-4)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for segment selection (default: 42)",
    )
    p.add_argument(
        "--no-normalize", action="store_true",
        help="Skip per-segment z-score normalization",
    )
    return p.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


# ── Single-case dataset ────────────────────────────────────────────────────────

class CaseDataset(Dataset):
    """Minimal Dataset wrapping one case NPZ file.

    Applies the same per-segment z-score normalization as PPGDataset.
    """

    def __init__(
        self,
        npz_path: Path,
        normalize: bool = True,
        indices: np.ndarray | None = None,
    ):
        data = np.load(npz_path)
        x = data["x"].astype(np.float32)  # (N, seg_len)
        y = data["y"].astype(np.float32)  # (N, 2)
        if indices is not None:
            x = x[indices]
            y = y[indices]
        self._x = x
        self._y = y
        self._normalize = normalize

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self._x[idx].copy())
        y = torch.from_numpy(self._y[idx].copy())
        if self._normalize:
            std = x.std()
            x = (x - x.mean()) / std.clamp_min(1e-6)
        return x, y


# ── Last-layer utilities ───────────────────────────────────────────────────────

def get_bp_output_layer(model: nn.Module) -> tuple[str, nn.Linear]:
    """Return (name, module) of the last nn.Linear with out_features == 2.

    Falls back to the last nn.Linear of any size if none output exactly 2.
    """
    found_name: str | None = None
    found_module: nn.Linear | None = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.out_features == 2:
            found_name = name
            found_module = module
    if found_name is None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                found_name = name
                found_module = module
    if found_name is None:
        raise RuntimeError("No nn.Linear found in model")
    return found_name, found_module  # type: ignore[return-value]


def freeze_all_except(model: nn.Module, layer_name: str) -> None:
    """Freeze all parameters, then unfreeze those in the named module."""
    for param in model.parameters():
        param.requires_grad_(False)
    for name, module in model.named_modules():
        if name == layer_name:
            for param in module.parameters():
                param.requires_grad_(True)


# ── Metrics (same as eval-model.py) ───────────────────────────────────────────

def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    err  = pred - true
    mae  = float(np.mean(np.abs(err)))
    me   = float(np.mean(err))
    sd   = float(np.std(err, ddof=1))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    n      = len(err)
    bhs_5  = float(np.sum(np.abs(err) <= 5)  / n * 100)
    bhs_10 = float(np.sum(np.abs(err) <= 10) / n * 100)
    bhs_15 = float(np.sum(np.abs(err) <= 15) / n * 100)

    if bhs_5 >= 60 and bhs_10 >= 85 and bhs_15 >= 95:
        bhs_grade = "A"
    elif bhs_5 >= 50 and bhs_10 >= 75 and bhs_15 >= 90:
        bhs_grade = "B"
    elif bhs_5 >= 40 and bhs_10 >= 65 and bhs_15 >= 85:
        bhs_grade = "C"
    else:
        bhs_grade = "D"

    aami_pass = abs(me) <= 5.0 and sd <= 8.0

    return {
        "mae":        mae,
        "me":         me,
        "sd":         sd,
        "rmse":       rmse,
        "bhs_pct_5":  bhs_5,
        "bhs_pct_10": bhs_10,
        "bhs_pct_15": bhs_15,
        "bhs_grade":  bhs_grade,
        "aami_pass":  aami_pass,
        "n_samples":  n,
    }


# ── Plots (same as eval-model.py) ─────────────────────────────────────────────

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
        ax.scatter(true, pred, alpha=0.3, s=8, color="#2196F3", rasterized=True)
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
        ax.axvline(0,          color="black",  linewidth=1.2, linestyle="--")
        ax.axvline(err.mean(), color="#F44336", linewidth=1.2, linestyle="-",
                   label=f"ME = {err.mean():.2f} mmHg")
        ax.set_xlabel(f"{label} Error (mmHg)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} Error Distribution")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bland_altman(
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
        mean_vals = (pred + true) / 2.0
        diff_vals = pred - true
        me        = float(np.mean(diff_vals))
        sd        = float(np.std(diff_vals, ddof=1))
        loa_upper = me + 1.96 * sd
        loa_lower = me - 1.96 * sd
        ax.scatter(mean_vals, diff_vals, alpha=0.3, s=8, color="#2196F3", rasterized=True)
        ax.axhline(0,         color="black",   linewidth=0.8, linestyle=":")
        ax.axhline(me,        color="#F44336", linewidth=1.5, linestyle="-",
                   label=f"Bias = {me:.2f} mmHg")
        ax.axhline(loa_upper, color="#FF9800", linewidth=1.2, linestyle="--",
                   label=f"+1.96 SD = {loa_upper:.2f} mmHg")
        ax.axhline(loa_lower, color="#FF9800", linewidth=1.2, linestyle="--",
                   label=f"−1.96 SD = {loa_lower:.2f} mmHg")
        ax.set_xlabel(f"Mean of Actual and Predicted {label} (mmHg)")
        ax.set_ylabel(f"Predicted − Actual {label} (mmHg)")
        ax.set_title(f"{label}: Bland-Altman Plot")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    preds_list, targets_list = [], []
    elapsed = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            t0 = time.perf_counter()
            pred = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - t0
            preds_list.append(pred.cpu().numpy())
            targets_list.append(y.numpy())
    preds   = np.concatenate(preds_list,   axis=0)
    targets = np.concatenate(targets_list, axis=0)
    return preds.astype(np.float32), targets.astype(np.float32), elapsed


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate(
    model: nn.Module,
    calib_ds: Dataset,
    device: torch.device,
    lr: float,
    batch_size: int,
) -> float:
    """Freeze all layers except the last BP output linear; train 1 epoch.

    Keeps BatchNorm and Dropout in eval mode (model.eval()) to avoid
    corrupting running statistics with only a handful of calibration
    segments.  Gradients flow only through the unfrozen linear layer.

    Returns the average MSE loss over the epoch.
    """
    layer_name, _ = get_bp_output_layer(model)
    freeze_all_except(model, layer_name)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Fine-tuning layer : '{layer_name}'  ({n_trainable:,} trainable params)")

    loader = DataLoader(calib_ds, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    criterion = nn.MSELoss()

    model.eval()  # keep BN running stats; disable dropout
    total_loss = 0.0
    n_batches  = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


# ── Metrics table printer ─────────────────────────────────────────────────────

def print_metrics(sbp_m: dict, dbp_m: dict, avg_ms: float) -> None:
    print(f"\n{'Metric':<18}  {'SBP':>10}  {'DBP':>10}")
    print("-" * 42)
    for key, fmt in [
        ("n_samples",  "d"),
        ("mae",        ".2f"),
        ("me",         ".2f"),
        ("sd",         ".2f"),
        ("rmse",       ".2f"),
        ("bhs_pct_5",  ".1f"),
        ("bhs_pct_10", ".1f"),
        ("bhs_pct_15", ".1f"),
        ("bhs_grade",  "s"),
        ("aami_pass",  "s"),
    ]:
        sv, dv = sbp_m[key], dbp_m[key]
        if fmt == "d":
            print(f"  {key:<16}  {sv:>10d}  {dv:>10d}")
        elif fmt == "s":
            print(f"  {key:<16}  {str(sv):>10}  {str(dv):>10}")
        else:
            print(f"  {key:<16}  {sv:>10{fmt}}  {dv:>10{fmt}}")
    print(f"  {'avg_ms/sample':<16}  {avg_ms:>10.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    run_dir = args.run_dir
    device  = resolve_device(args.device)

    # ── Load model ────────────────────────────────────────────────────────────
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

    model = create_model(model_name).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    best_epoch = ckpt.get("epoch", "?")
    print(f"Loaded best.pt (epoch {best_epoch})")

    # ── Resolve test case ─────────────────────────────────────────────────────
    test_dir = args.dataset_dir / "test"
    if not test_dir.exists():
        print(f"ERROR: {test_dir} not found.", file=sys.stderr)
        sys.exit(1)

    test_files = sorted(
        test_dir.glob("*.npz"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )
    if not test_files:
        print(f"ERROR: No .npz files in {test_dir}.", file=sys.stderr)
        sys.exit(1)

    if args.case_id is None:
        case_file = test_files[0]
        case_id   = case_file.stem
    else:
        case_id   = str(args.case_id)
        case_file = test_dir / f"{case_id}.npz"
        if not case_file.exists():
            print(f"ERROR: Case file not found: {case_file}", file=sys.stderr)
            sys.exit(1)

    print(f"Case          : {case_id}")

    # ── Determine calibration segment count ───────────────────────────────────
    case_data = np.load(case_file)
    n_total   = int(case_data["x"].shape[0])
    print(f"Total segments: {n_total}")

    n_calib_arg = args.n_calib
    if n_calib_arg < 1.0:
        n_calib = max(1, int(round(n_total * n_calib_arg)))
    else:
        n_calib = min(n_total, int(n_calib_arg))

    rng           = np.random.default_rng(args.seed)
    calib_indices = np.sort(rng.choice(n_total, size=n_calib, replace=False))

    print(f"Calib segments: {n_calib} / {n_total}  (seed={args.seed})")
    print(f"Learning rate : {args.lr}")

    normalize = not args.no_normalize
    calib_ds  = CaseDataset(case_file, normalize=normalize, indices=calib_indices)

    # ── Output directory ──────────────────────────────────────────────────────
    calib_dir = run_dir / case_id
    calib_dir.mkdir(parents=True, exist_ok=True)

    # ── Calibrate ─────────────────────────────────────────────────────────────
    print("\nCalibrating …")
    calib_loss = calibrate(model, calib_ds, device, lr=args.lr, batch_size=args.batch_size)
    print(f"  Calib MSE loss: {calib_loss:.6f}")

    # ── Save calibrated checkpoint ────────────────────────────────────────────
    calib_ckpt_path = calib_dir / "calib.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name":       model_name,
            "base_checkpoint":  str(ckpt_path),
            "case_id":          case_id,
            "n_calib":          n_calib,
            "calib_indices":    calib_indices.tolist(),
            "calib_loss":       round(calib_loss, 6),
            "lr":               args.lr,
            "seed":             args.seed,
        },
        calib_ckpt_path,
    )
    print(f"  Saved: {calib_ckpt_path}")

    # ── Evaluate on ALL segments of the case ──────────────────────────────────
    print("\nEvaluating on all segments …")
    all_ds = CaseDataset(case_file, normalize=normalize)
    all_loader = DataLoader(
        all_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )

    preds, targets, inference_sec = run_inference(model, all_loader, device)

    n_samples = len(preds)
    avg_ms    = inference_sec / n_samples * 1000 if n_samples > 0 else 0.0
    pred_sbp, pred_dbp = preds[:, 0],   preds[:, 1]
    true_sbp, true_dbp = targets[:, 0], targets[:, 1]

    sbp_m = compute_metrics(pred_sbp, true_sbp)
    dbp_m = compute_metrics(pred_dbp, true_dbp)

    print_metrics(sbp_m, dbp_m, avg_ms)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "run_dir":           str(run_dir),
        "model":             model_name,
        "base_checkpoint":   str(ckpt_path),
        "calib_checkpoint":  str(calib_ckpt_path),
        "case_id":           case_id,
        "n_calib_segments":  n_calib,
        "n_total_segments":  n_total,
        "calib_lr":          args.lr,
        "calib_loss":        round(calib_loss, 6),
        "inference_sec":     round(inference_sec, 4),
        "avg_ms_per_sample": round(avg_ms, 4),
        "sbp":               sbp_m,
        "dbp":               dbp_m,
    }
    json_path = calib_dir / "eval_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {json_path}")

    plot_scatter(pred_sbp, true_sbp, pred_dbp, true_dbp, calib_dir / "eval_plot.png")
    print(f"Saved: {calib_dir / 'eval_plot.png'}")

    plot_error_hist(pred_sbp - true_sbp, pred_dbp - true_dbp, calib_dir / "error_hist.png")
    print(f"Saved: {calib_dir / 'error_hist.png'}")

    plot_bland_altman(pred_sbp, true_sbp, pred_dbp, true_dbp, calib_dir / "bland_altman.png")
    print(f"Saved: {calib_dir / 'bland_altman.png'}")


if __name__ == "__main__":
    main()
