"""
Evaluate a trained BPE model on the held-out test set.

Loads best.pt from a run directory, runs inference over the test split, and
writes evaluation results to the same directory:

  eval_results.json     — summary metrics (MAE, RMSE, ME, SD; BHS grade; AAMI)
  eval_plot.png         — predicted vs actual scatter plots for SBP and DBP
  error_hist.png        — error distribution histograms for SBP and DBP
  bland_altman.png      — Bland-Altman plots for SBP and DBP

Usage:
    uv run python scripts/eval-model.py <run_dir> [OPTIONS]
    uv run python scripts/eval-model.py data/models/resnet1d

Duo mode (two-model ensemble with disagreement rejection):
    uv run python scripts/eval-model.py <output_dir> --duo [OPTIONS]
    uv run python scripts/eval-model.py data/models/duo_conv_reg_ds_mtae --duo

    Measures are rejected when either model disagrees by >= threshold mmHg on
    SBP or DBP.  Accepted prediction = average of both models.

Options:
    --dataset-dir   Root dataset directory  (default: data/dataset)
    --device        auto | cpu | cuda | cuda:N  (default: auto)
    --batch-size    Inference batch size    (default: 512)
    --no-normalize  Skip per-segment z-score normalization

Duo-only options:
    --duo               Enable duo evaluation mode
    --duo-models A B    Two model IDs (default: conv_reg_ds mtae)
    --duo-threshold T   Rejection threshold in mmHg (default: 5.0)
    --models-dir DIR    Root models directory (default: data/models)
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
        help="Run directory (single mode: contains best.pt + config.json; "
             "duo mode: output directory, created if needed)",
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
    # ── Duo mode ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--duo", action="store_true",
        help="Enable duo evaluation mode (two-model ensemble with rejection)",
    )
    p.add_argument(
        "--duo-models", nargs=2, default=["conv_reg_ds", "mtae"],
        metavar=("MODEL_A", "MODEL_B"),
        help="Two model IDs for duo mode (default: conv_reg_ds mtae)",
    )
    p.add_argument(
        "--duo-threshold", type=float, default=5.0,
        help="Rejection threshold in mmHg (default: 5.0)",
    )
    p.add_argument(
        "--models-dir", type=Path, default=Path("data/models"),
        help="Root models directory used to locate checkpoints in duo mode "
             "(default: data/models)",
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

        me = float(np.mean(diff_vals))
        sd = float(np.std(diff_vals, ddof=1))
        loa_upper = me + 1.96 * sd
        loa_lower = me - 1.96 * sd

        ax.scatter(mean_vals, diff_vals, alpha=0.15, s=4, color="#2196F3", rasterized=True)
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


# ── Per-case stats ────────────────────────────────────────────────────────────

def compute_per_case_stats(
    preds: np.ndarray,
    targets: np.ndarray,
    segs: list[tuple[int, int]],
    files: list[Path],
) -> dict:
    """Return best_case_id and worst_case_id by average (SBP+DBP) MAE."""
    from collections import defaultdict

    file_to_indices: dict[int, list[int]] = defaultdict(list)
    for seg_idx, (file_idx, _) in enumerate(segs):
        file_to_indices[file_idx].append(seg_idx)

    best_case_id = None
    worst_case_id = None
    best_avg_mae  = float("inf")
    worst_avg_mae = float("-inf")

    for file_idx, indices in file_to_indices.items():
        idx = np.array(indices)
        sbp_mae = float(np.mean(np.abs(preds[idx, 0] - targets[idx, 0])))
        dbp_mae = float(np.mean(np.abs(preds[idx, 1] - targets[idx, 1])))
        avg_mae = (sbp_mae + dbp_mae) / 2.0

        stem = files[file_idx].stem
        case_id = int(stem) if stem.isdigit() else stem

        if avg_mae < best_avg_mae:
            best_avg_mae  = avg_mae
            best_case_id  = case_id
        if avg_mae > worst_avg_mae:
            worst_avg_mae = avg_mae
            worst_case_id = case_id

    return {
        "best_case_id":      best_case_id,
        "best_case_avg_mae": round(best_avg_mae,  4),
        "worst_case_id":     worst_case_id,
        "worst_case_avg_mae": round(worst_avg_mae, 4),
    }


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (predictions, targets, elapsed_sec) where elapsed_sec is pure inference wall time."""
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


def run_duo_inference(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Run two models over the test set and collect per-segment predictions.

    Returns:
        preds_a:   (N, 2) predictions from model A.
        preds_b:   (N, 2) predictions from model B.
        targets:   (N, 2) ground-truth SBP/DBP.
        elapsed_a: pure inference wall time for model A (seconds).
        elapsed_b: pure inference wall time for model B (seconds).
    """
    model_a.eval()
    model_b.eval()
    preds_a_list, preds_b_list, targets_list = [], [], []
    elapsed_a = elapsed_b = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            t0 = time.perf_counter()
            pa = model_a(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_a += time.perf_counter() - t0

            t0 = time.perf_counter()
            pb = model_b(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_b += time.perf_counter() - t0

            preds_a_list.append(pa.cpu().numpy())
            preds_b_list.append(pb.cpu().numpy())
            targets_list.append(y.numpy())

    preds_a = np.concatenate(preds_a_list, axis=0).astype(np.float32)
    preds_b = np.concatenate(preds_b_list, axis=0).astype(np.float32)
    targets  = np.concatenate(targets_list,  axis=0).astype(np.float32)
    return preds_a, preds_b, targets, elapsed_a, elapsed_b


# ── Shared helper ─────────────────────────────────────────────────────────────

def _print_and_save(
    run_dir: Path,
    preds: np.ndarray,
    targets: np.ndarray,
    inference_sec: float,
    extra_json: dict,
    suffix: str = "",
) -> None:
    """Print metrics table, save JSON + plots.  suffix appended to filenames."""
    pred_sbp, pred_dbp = preds[:, 0],   preds[:, 1]
    true_sbp, true_dbp = targets[:, 0], targets[:, 1]
    err_sbp = pred_sbp - true_sbp
    err_dbp = pred_dbp - true_dbp

    sbp_m  = compute_metrics(pred_sbp, true_sbp)
    dbp_m  = compute_metrics(pred_dbp, true_dbp)
    n_samples = len(preds)
    avg_ms = inference_sec / n_samples * 1000 if n_samples > 0 else 0.0

    label = f" [{suffix.strip('_')}]" if suffix else ""
    print(f"\n{'Metric':<18}  {'SBP':>10}  {'DBP':>10}{label}")
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
    if not suffix:
        print(f"  {'avg_ms/sample':<16}  {avg_ms:>10.3f}")

    json_name  = f"eval_results{suffix}.json"
    plot_name  = f"eval_plot{suffix}.png"
    hist_name  = f"error_hist{suffix}.png"

    results = {
        **extra_json,
        "n_samples":        n_samples,
        "inference_sec":    round(inference_sec, 4),
        "avg_ms_per_sample": round(avg_ms, 4),
        "sbp":              sbp_m,
        "dbp":              dbp_m,
    }
    (run_dir / json_name).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved: {run_dir / json_name}")

    plot_scatter(pred_sbp, true_sbp, pred_dbp, true_dbp, run_dir / plot_name)
    print(f"Saved: {run_dir / plot_name}")

    plot_error_hist(err_sbp, err_dbp, run_dir / hist_name)
    print(f"Saved: {run_dir / hist_name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.duo:
        _main_duo(args)
    else:
        _main_single(args)


def _main_single(args: argparse.Namespace) -> None:
    """Standard single-model evaluation (original behaviour)."""
    run_dir = args.run_dir
    device  = resolve_device(args.device)

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
    print(f"Loaded best.pt (epoch {best_epoch}, val_loss={ckpt.get('val_loss', float('nan')):.4f})")

    test_dir = args.dataset_dir / "test"
    try:
        test_ds = PPGDataset(test_dir, normalize=not args.no_normalize, preload=False)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Run bin/construct-dataset first to build the dataset.", file=sys.stderr)
        sys.exit(1)

    print(f"Test set      : {len(test_ds):,} segments from {test_ds.n_files} cases")

    loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=device.type == "cuda",
    )

    print("Running inference …")
    preds, targets, inference_sec = run_inference(model, loader, device)

    case_m = compute_per_case_stats(preds, targets, test_ds._segs, test_ds._files)
    n_samples = len(preds)
    avg_ms = inference_sec / n_samples * 1000

    sbp_m = compute_metrics(preds[:, 0], targets[:, 0])
    dbp_m = compute_metrics(preds[:, 1], targets[:, 1])

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
    print()
    print(f"  best  case: {case_m['best_case_id']}  (avg MAE {case_m['best_case_avg_mae']:.2f} mmHg)")
    print(f"  worst case: {case_m['worst_case_id']}  (avg MAE {case_m['worst_case_avg_mae']:.2f} mmHg)")

    results = {
        "run_dir":           str(run_dir),
        "model":             model_name,
        "checkpoint":        str(ckpt_path),
        "best_epoch":        best_epoch,
        "test_dir":          str(test_dir),
        "n_segments":        int(len(test_ds)),
        "n_cases":           int(test_ds.n_files),
        "inference_sec":     round(inference_sec, 4),
        "avg_ms_per_sample": round(avg_ms, 4),
        "sbp":               sbp_m,
        "dbp":               dbp_m,
        "best_case_id":       case_m["best_case_id"],
        "best_case_avg_mae":  case_m["best_case_avg_mae"],
        "worst_case_id":      case_m["worst_case_id"],
        "worst_case_avg_mae": case_m["worst_case_avg_mae"],
    }
    results_path = run_dir / "eval_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {results_path}")

    plot_scatter(preds[:, 0], targets[:, 0], preds[:, 1], targets[:, 1],
                 run_dir / "eval_plot.png")
    print(f"Saved: {run_dir / 'eval_plot.png'}")

    plot_error_hist(preds[:, 0] - targets[:, 0], preds[:, 1] - targets[:, 1],
                    run_dir / "error_hist.png")
    print(f"Saved: {run_dir / 'error_hist.png'}")

    ba_path = run_dir / "bland_altman.png"
    plot_bland_altman(preds[:, 0], targets[:, 0], preds[:, 1], targets[:, 1], ba_path)
    print(f"Saved: {ba_path}")


def _main_duo(args: argparse.Namespace) -> None:
    """Duo evaluation: two-model ensemble with disagreement-based rejection."""
    from bpe.models.duo import _load_model

    out_dir    = args.run_dir
    device     = resolve_device(args.device)
    model_a_id, model_b_id = args.duo_models
    threshold  = args.duo_threshold
    models_dir = args.models_dir

    print(f"Mode          : duo")
    print(f"Model A       : {model_a_id}")
    print(f"Model B       : {model_b_id}")
    print(f"Threshold     : {threshold} mmHg")
    print(f"Models dir    : {models_dir}")
    print(f"Output dir    : {out_dir}")
    print(f"Device        : {device}")

    # ── Load both models ──────────────────────────────────────────────────────
    try:
        model_a, _ = _load_model(models_dir / model_a_id, device)
        model_b, _ = _load_model(models_dir / model_b_id, device)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {model_a_id} and {model_b_id}")

    # ── Test dataset ──────────────────────────────────────────────────────────
    test_dir = args.dataset_dir / "test"
    try:
        test_ds = PPGDataset(test_dir, normalize=not args.no_normalize, preload=False)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Run bin/construct-dataset first to build the dataset.", file=sys.stderr)
        sys.exit(1)

    print(f"Test set      : {len(test_ds):,} segments from {test_ds.n_files} cases")

    loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=device.type == "cuda",
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    print("Running inference for both models …")
    preds_a, preds_b, targets, elapsed_a, elapsed_b = run_duo_inference(
        model_a, model_b, loader, device
    )

    avg_preds  = (preds_a + preds_b) / 2
    diff       = np.abs(preds_a - preds_b)                    # (N, 2)
    accepted   = (diff[:, 0] < threshold) & (diff[:, 1] < threshold)  # (N,)

    n_total    = len(avg_preds)
    n_accepted = int(accepted.sum())
    n_rejected = n_total - n_accepted
    accept_rate = n_accepted / n_total * 100

    print()
    print(f"  Total segments   : {n_total:,}")
    print(f"  Accepted         : {n_accepted:,}  ({accept_rate:.1f}%)")
    print(f"  Rejected         : {n_rejected:,}  ({100 - accept_rate:.1f}%)")

    # ── Disagreement distribution ─────────────────────────────────────────────
    print()
    sbp_diff_mean = float(np.mean(diff[:, 0]))
    dbp_diff_mean = float(np.mean(diff[:, 1]))
    sbp_diff_p95  = float(np.percentile(diff[:, 0], 95))
    dbp_diff_p95  = float(np.percentile(diff[:, 1], 95))
    print(f"  SBP inter-model diff  mean={sbp_diff_mean:.2f} mmHg  p95={sbp_diff_p95:.2f} mmHg")
    print(f"  DBP inter-model diff  mean={dbp_diff_mean:.2f} mmHg  p95={dbp_diff_p95:.2f} mmHg")

    # ── Metrics: ALL segments ─────────────────────────────────────────────────
    sbp_all_m = compute_metrics(avg_preds[:, 0], targets[:, 0])
    dbp_all_m = compute_metrics(avg_preds[:, 1], targets[:, 1])

    print(f"\n{'Metric':<18}  {'SBP (all)':>12}  {'DBP (all)':>12}")
    print("-" * 46)
    for key, fmt in [
        ("n_samples", "d"), ("mae", ".2f"), ("me", ".2f"),
        ("sd", ".2f"), ("rmse", ".2f"),
        ("bhs_pct_5", ".1f"), ("bhs_pct_10", ".1f"), ("bhs_pct_15", ".1f"),
        ("bhs_grade", "s"), ("aami_pass", "s"),
    ]:
        sv, dv = sbp_all_m[key], dbp_all_m[key]
        if fmt == "d":
            print(f"  {key:<16}  {sv:>12d}  {dv:>12d}")
        elif fmt == "s":
            print(f"  {key:<16}  {str(sv):>12}  {str(dv):>12}")
        else:
            print(f"  {key:<16}  {sv:>12{fmt}}  {dv:>12{fmt}}")

    # ── Metrics: ACCEPTED segments ────────────────────────────────────────────
    if n_accepted > 0:
        acc_preds   = avg_preds[accepted]
        acc_targets = targets[accepted]
        sbp_acc_m   = compute_metrics(acc_preds[:, 0], acc_targets[:, 0])
        dbp_acc_m   = compute_metrics(acc_preds[:, 1], acc_targets[:, 1])

        print(f"\n{'Metric':<18}  {'SBP (accepted)':>14}  {'DBP (accepted)':>14}")
        print("-" * 50)
        for key, fmt in [
            ("n_samples", "d"), ("mae", ".2f"), ("me", ".2f"),
            ("sd", ".2f"), ("rmse", ".2f"),
            ("bhs_pct_5", ".1f"), ("bhs_pct_10", ".1f"), ("bhs_pct_15", ".1f"),
            ("bhs_grade", "s"), ("aami_pass", "s"),
        ]:
            sv, dv = sbp_acc_m[key], dbp_acc_m[key]
            if fmt == "d":
                print(f"  {key:<16}  {sv:>14d}  {dv:>14d}")
            elif fmt == "s":
                print(f"  {key:<16}  {str(sv):>14}  {str(dv):>14}")
            else:
                print(f"  {key:<16}  {sv:>14{fmt}}  {dv:>14{fmt}}")

        # ── Improvement summary ───────────────────────────────────────────────
        print()
        print("  Improvement (all → accepted):")
        for bp, m_all, m_acc in [("SBP", sbp_all_m, sbp_acc_m),
                                   ("DBP", dbp_all_m, dbp_acc_m)]:
            d_mae  = m_all["mae"]  - m_acc["mae"]
            d_sd   = m_all["sd"]   - m_acc["sd"]
            d_rmse = m_all["rmse"] - m_acc["rmse"]
            print(f"    {bp}: MAE {d_mae:+.2f}  SD {d_sd:+.2f}  RMSE {d_rmse:+.2f} mmHg")
    else:
        sbp_acc_m = dbp_acc_m = None
        print("\n  (no segments accepted — all rejected)")

    # ── Timing ────────────────────────────────────────────────────────────────
    total_elapsed = elapsed_a + elapsed_b
    avg_ms_total  = total_elapsed / n_total * 1000
    avg_ms_a      = elapsed_a / n_total * 1000
    avg_ms_b      = elapsed_b / n_total * 1000
    print()
    print(f"  Inference time: {model_a_id} {avg_ms_a:.4f} ms/sample + "
          f"{model_b_id} {avg_ms_b:.4f} ms/sample = {avg_ms_total:.4f} ms/sample total")

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model":              "duo",
        "model_a":            model_a_id,
        "model_b":            model_b_id,
        "threshold_mmhg":     threshold,
        "test_dir":           str(test_dir),
        "n_cases":            int(test_ds.n_files),
        "n_segments_total":   n_total,
        "n_segments_accepted": n_accepted,
        "n_segments_rejected": n_rejected,
        "acceptance_rate_pct": round(accept_rate, 4),
        "sbp_diff_mean":      round(sbp_diff_mean, 4),
        "dbp_diff_mean":      round(dbp_diff_mean, 4),
        "sbp_diff_p95":       round(sbp_diff_p95, 4),
        "dbp_diff_p95":       round(dbp_diff_p95, 4),
        "inference_sec_a":    round(elapsed_a, 4),
        "inference_sec_b":    round(elapsed_b, 4),
        "avg_ms_per_sample":  round(avg_ms_total, 4),
        "sbp_all":            sbp_all_m,
        "dbp_all":            dbp_all_m,
        "sbp_accepted":       sbp_acc_m,
        "dbp_accepted":       dbp_acc_m,
    }
    results_path = out_dir / "eval_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {results_path}")

    # Plots for all and accepted segments
    plot_scatter(avg_preds[:, 0], targets[:, 0], avg_preds[:, 1], targets[:, 1],
                 out_dir / "eval_plot_all.png")
    print(f"Saved: {out_dir / 'eval_plot_all.png'}")

    plot_error_hist(avg_preds[:, 0] - targets[:, 0], avg_preds[:, 1] - targets[:, 1],
                    out_dir / "error_hist_all.png")
    print(f"Saved: {out_dir / 'error_hist_all.png'}")

    ba_all_path = out_dir / "bland_altman_all.png"
    plot_bland_altman(avg_preds[:, 0], targets[:, 0], avg_preds[:, 1], targets[:, 1],
                      ba_all_path)
    print(f"Saved: {ba_all_path}")

    if n_accepted > 0:
        acc_p, acc_t = avg_preds[accepted], targets[accepted]
        plot_scatter(acc_p[:, 0], acc_t[:, 0], acc_p[:, 1], acc_t[:, 1],
                     out_dir / "eval_plot.png")
        print(f"Saved: {out_dir / 'eval_plot.png'}")

        plot_error_hist(acc_p[:, 0] - acc_t[:, 0], acc_p[:, 1] - acc_t[:, 1],
                        out_dir / "error_hist.png")
        print(f"Saved: {out_dir / 'error_hist.png'}")

        # Disagreement distribution plot
        _plot_duo_diff(diff, accepted, threshold, out_dir / "diff_dist.png")
        print(f"Saved: {out_dir / 'diff_dist.png'}")

        ba_acc_path = out_dir / "bland_altman_accepted.png"
        plot_bland_altman(acc_p[:, 0], acc_t[:, 0], acc_p[:, 1], acc_t[:, 1],
                          ba_acc_path)
        print(f"Saved: {ba_acc_path}")


def _plot_duo_diff(
    diff: np.ndarray,
    accepted: np.ndarray,
    threshold: float,
    out_path: Path,
) -> None:
    """Plot inter-model disagreement distribution with acceptance/rejection split."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, label in [(axes[0], 0, "SBP"), (axes[1], 1, "DBP")]:
        d = diff[:, col]
        ax.hist(d[accepted],  bins=60, color="#4CAF50", alpha=0.7, label="Accepted")
        ax.hist(d[~accepted], bins=60, color="#F44336", alpha=0.7, label="Rejected")
        ax.axvline(threshold, color="black", linewidth=1.2, linestyle="--",
                   label=f"Threshold = {threshold} mmHg")
        ax.set_xlabel(f"|{label}_A − {label}_B| (mmHg)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label} Inter-model Disagreement")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
