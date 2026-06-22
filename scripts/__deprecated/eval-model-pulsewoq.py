"""
Evaluate a trained pulsewoq_resnet1d model on the held-out test set.

Two evaluation scenarios:

  Scenario A - Quality threshold filtering
    Evaluates on three subsets: all segments, quality >= 0.2, quality >= 0.3.
    For each subset, reports count, ratio, and clinical metrics (MAE, RMSE,
    ME, SD, BHS grade, AAMI pass/fail).  The intent is to check whether
    filtering out low-quality predictions and re-measuring improves accuracy.

  Scenario B - Best-of-N repeated-measurement simulation
    For each case in the test set, simulates N repeated measurements by
    randomly sampling N segments.  The segment with the highest quality score
    is selected as the final BP estimate.  Plots MAE vs. N so the benefit of
    quality-guided selection grows visible as N increases.

Output files written to the run directory:
  eval_quality_results.json   - Scenario A metrics (all / q>=0.2 / q>=0.3)
  eval_quality_scatter.png    - 2x3 scatter plots (SBP / DBP) x (3 thresholds)
  eval_quality_dist.png       - Quality score distribution histogram
  eval_best_of_n.png          - Scenario B: MAE vs. number of measurements

Usage:
    uv run python scripts/eval-model-pulsewoq.py <run_dir> [OPTIONS]

Options:
    --dataset-dir   Root dataset directory  (default: data/dataset)
    --device        auto | cpu | cuda | cuda:N  (default: auto)
    --batch-size    Inference batch size    (default: 512)
    --no-normalize  Skip per-segment z-score normalization
    --max-n         Max measurements per case for Scenario B  (default: 16)
    --n-trials      Sampling trials per N value  (default: 200)
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


QUALITY_THRESHOLDS = [
    ("All",    None),
    ("q>=0.2",  0.2),
    ("q>=0.3",  0.3),
]


# -- CLI -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quality-aware evaluation of a pulsewoq_resnet1d checkpoint",
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
    p.add_argument(
        "--max-n", type=int, default=16,
        help="Maximum repeated measurements per case for Scenario B (default: 16)",
    )
    p.add_argument(
        "--n-trials", type=int, default=200,
        help="Sampling trials per N value for Scenario B (default: 200)",
    )
    return p.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


# -- Metrics -------------------------------------------------------------------

def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """Return a dict of clinical BP estimation metrics for one channel."""
    err  = pred - true
    mae  = float(np.mean(np.abs(err)))
    me   = float(np.mean(err))
    sd   = float(np.std(err, ddof=1)) if len(err) > 1 else 0.0
    rmse = float(np.sqrt(np.mean(err ** 2)))

    n = len(err)
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
        "n_samples":  n,
        "mae":        mae,
        "me":         me,
        "sd":         sd,
        "rmse":       rmse,
        "bhs_pct_5":  bhs_5,
        "bhs_pct_10": bhs_10,
        "bhs_pct_15": bhs_15,
        "bhs_grade":  bhs_grade,
        "aami_pass":  aami_pass,
    }


# -- Inference -----------------------------------------------------------------

def run_inference_with_quality(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference via forward_with_quality.

    Returns:
        preds     float32 (N, 2)   [SBP, DBP] predictions in mmHg
        qualities float32 (N,)     quality scores in (0, 1)
        targets   float32 (N, 2)   [SBP, DBP] ground truth in mmHg
    """
    model.eval()
    preds_list, qual_list, targets_list = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x   = x.to(device)
            out = model.forward_with_quality(x)         # (B, 3)
            preds_list.append(out[:, :2].cpu().numpy())
            qual_list.append(out[:, 2].cpu().numpy())
            targets_list.append(y.numpy())

    preds     = np.concatenate(preds_list,   axis=0).astype(np.float32)
    qualities = np.concatenate(qual_list,    axis=0).astype(np.float32)
    targets   = np.concatenate(targets_list, axis=0).astype(np.float32)
    return preds, qualities, targets


# -- Scenario A ----------------------------------------------------------------

def print_threshold_table(label: str, n_total: int, sbp_m: dict, dbp_m: dict) -> None:
    n      = sbp_m["n_samples"]
    ratio  = n / n_total * 100
    print(f"\n{'-'*62}")
    print(f"  {label}  (n = {n:,} / {n_total:,} = {ratio:.1f} %)")
    print(f"{'-'*62}")
    print(f"  {'Metric':<20}  {'SBP':>10}  {'DBP':>10}")
    print(f"  {'-'*44}")
    for key, fmt in [
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
        if fmt == "s":
            print(f"  {key:<20}  {str(sv):>10}  {str(dv):>10}")
        else:
            print(f"  {key:<20}  {sv:>10{fmt}}  {dv:>10{fmt}}")


# -- Scenario B ----------------------------------------------------------------

def simulate_best_of_n(
    qualities:  np.ndarray,
    pred_sbp:   np.ndarray,
    pred_dbp:   np.ndarray,
    true_sbp:   np.ndarray,
    true_dbp:   np.ndarray,
    case_groups: dict[int, list[int]],
    max_n:      int,
    n_trials:   int,
    rng:        np.random.Generator,
) -> tuple[list, list, list, list, list]:
    """Simulate best-of-N quality selection for each case.

    For each N and each trial:
      - Randomly sample N segments from each case (with replacement).
      - Select the segment with the highest quality score.
      - Compare its prediction against its own ground-truth BP.
    Averages MAE over all cases per trial, then reports mean +/- SD over trials.

    Returns:
        n_values, sbp_maes, sbp_stds, dbp_maes, dbp_stds
    """
    case_segs_list = [np.array(segs) for segs in case_groups.values()]
    n_values  = list(range(1, max_n + 1))
    sbp_maes, sbp_stds = [], []
    dbp_maes, dbp_stds = [], []

    for N in n_values:
        # For every case, generate (n_trials x N) samples at once.
        # Shapes per case: choices (n_trials, N) -> best_idxs (n_trials,)
        sbp_per_case = []   # each entry: (n_trials,)
        dbp_per_case = []
        sbp_t_per_case = []
        dbp_t_per_case = []

        for segs_arr in case_segs_list:
            choices     = rng.choice(segs_arr, size=(n_trials, N), replace=True)  # (n_trials, N)
            q_chosen    = qualities[choices]                                        # (n_trials, N)
            best_pos    = q_chosen.argmax(axis=1)                                  # (n_trials,)
            best_idxs   = choices[np.arange(n_trials), best_pos]                  # (n_trials,)
            sbp_per_case.append(pred_sbp[best_idxs])
            dbp_per_case.append(pred_dbp[best_idxs])
            sbp_t_per_case.append(true_sbp[best_idxs])
            dbp_t_per_case.append(true_dbp[best_idxs])

        # Stack: (n_cases, n_trials) -> MAE over cases per trial -> (n_trials,)
        sbp_arr   = np.stack(sbp_per_case)    # (n_cases, n_trials)
        dbp_arr   = np.stack(dbp_per_case)
        sbp_t_arr = np.stack(sbp_t_per_case)
        dbp_t_arr = np.stack(dbp_t_per_case)

        sbp_trial_maes = np.abs(sbp_arr - sbp_t_arr).mean(axis=0)   # (n_trials,)
        dbp_trial_maes = np.abs(dbp_arr - dbp_t_arr).mean(axis=0)

        sbp_maes.append(float(sbp_trial_maes.mean()))
        sbp_stds.append(float(sbp_trial_maes.std()))
        dbp_maes.append(float(dbp_trial_maes.mean()))
        dbp_stds.append(float(dbp_trial_maes.std()))

    return n_values, sbp_maes, sbp_stds, dbp_maes, dbp_stds


# -- Plots ---------------------------------------------------------------------

def plot_scatter_by_threshold(
    pred_sbp:  np.ndarray,
    pred_dbp:  np.ndarray,
    true_sbp:  np.ndarray,
    true_dbp:  np.ndarray,
    qualities: np.ndarray,
    out_path:  Path,
) -> None:
    """2x3 scatter grid: (SBP, DBP) rows x (all, q>=0.2, q>=0.3) columns."""
    n_total = len(qualities)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # Shared axis limits per row, derived from the full unfiltered data so all
    # three columns in the same row use identical scales.
    pad = 5.0
    row_lims = [
        (float(np.concatenate([pred_sbp, true_sbp]).min()) - pad,
         float(np.concatenate([pred_sbp, true_sbp]).max()) + pad),
        (float(np.concatenate([pred_dbp, true_dbp]).min()) - pad,
         float(np.concatenate([pred_dbp, true_dbp]).max()) + pad),
    ]

    for col, (label, threshold) in enumerate(QUALITY_THRESHOLDS):
        mask  = np.ones(n_total, dtype=bool) if threshold is None else (qualities >= threshold)
        n     = int(mask.sum())
        ratio = n / n_total * 100

        for row, (p, t, bplabel) in enumerate([
            (pred_sbp[mask], true_sbp[mask], "SBP"),
            (pred_dbp[mask], true_dbp[mask], "DBP"),
        ]):
            ax = axes[row, col]
            if n == 0:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=13, color="gray")
                ax.set_title(f"{bplabel} - {label}  (n=0, 0.0%)", fontsize=10)
                ax.set_axis_off()
                continue

            lim_min, lim_max = row_lims[row]
            mae     = float(np.mean(np.abs(p - t)))

            ax.scatter(t, p, alpha=0.15, s=3, color="#2196F3", rasterized=True)
            ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1, label="y = x")
            ax.set_xlim(lim_min, lim_max)
            ax.set_ylim(lim_min, lim_max)
            ax.set_aspect("equal")
            ax.set_xlabel(f"Actual {bplabel} (mmHg)")
            ax.set_ylabel(f"Predicted {bplabel} (mmHg)")
            ax.set_title(
                f"{bplabel} - {label}  (n={n:,}, {ratio:.1f}%)\n"
                f"MAE = {mae:.2f} mmHg",
                fontsize=10,
            )
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Predicted vs Actual by Quality Threshold", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_quality_distribution(qualities: np.ndarray, out_path: Path) -> None:
    """Histogram of quality scores with threshold markers at 0.2 and 0.3."""
    n   = len(qualities)
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(qualities, bins=100, color="#2196F3", edgecolor="none", alpha=0.8)
    for thr, color, ls in [(0.2, "#FF5722", "--"), (0.3, "#4CAF50", "-.")]:
        n_above = int((qualities >= thr).sum())
        ax.axvline(
            thr, color=color, linewidth=1.8, linestyle=ls,
            label=f"q >= {thr}: {n_above:,} ({n_above / n * 100:.1f}%)",
        )

    ax.set_xlabel("Quality score")
    ax.set_ylabel("Segment count")
    ax.set_title("Quality Score Distribution - test set")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_best_of_n(
    n_values:  list[int],
    sbp_maes:  list[float],
    sbp_stds:  list[float],
    dbp_maes:  list[float],
    dbp_stds:  list[float],
    out_path:  Path,
) -> None:
    """Plot MAE +/- 1 SD vs. number of repeated measurements (Scenario B)."""
    n  = np.array(n_values)
    sm = np.array(sbp_maes)
    ss = np.array(sbp_stds)
    dm = np.array(dbp_maes)
    ds = np.array(dbp_stds)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, mean, std, label, color in [
        (axes[0], sm, ss, "SBP", "#2196F3"),
        (axes[1], dm, ds, "DBP", "#FF5722"),
    ]:
        ax.fill_between(n, mean - std, mean + std, color=color, alpha=0.2, label="+/-1 SD")
        ax.plot(n, mean, "o-", color=color, linewidth=2, markersize=5, label="Mean MAE")
        # annotate N=1 (random selection baseline) and the last point
        ax.annotate(
            f"N=1\n{mean[0]:.2f}",
            xy=(n[0], mean[0]),
            xytext=(n[0] + 0.3, mean[0] + 0.05 * (mean[0] - mean[-1])),
            fontsize=8, color=color,
        )
        ax.annotate(
            f"N={n[-1]}\n{mean[-1]:.2f}",
            xy=(n[-1], mean[-1]),
            xytext=(n[-1] - 1.5, mean[-1] + 0.05 * (mean[0] - mean[-1])),
            fontsize=8, color=color,
        )
        ax.set_xlabel("Number of measurements (N)")
        ax.set_ylabel("MAE (mmHg)")
        ax.set_title(f"{label} MAE vs. Measurement Count")
        ax.set_xticks(n)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Best-of-N Quality Selection: Effect on BP Estimation Accuracy",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -- Main ----------------------------------------------------------------------

def main() -> None:
    args    = parse_args()
    run_dir = args.run_dir
    device  = resolve_device(args.device)

    # -- Validate run directory ------------------------------------------------
    cfg_path  = run_dir / "config.json"
    ckpt_path = run_dir / "best.pt"
    for path in (cfg_path, ckpt_path):
        if not path.exists():
            print(f"ERROR: {path} not found.", file=sys.stderr)
            sys.exit(1)

    cfg        = json.loads(cfg_path.read_text(encoding="utf-8"))
    model_name = cfg.get("model", "")

    print(f"Run directory : {run_dir}")
    print(f"Model         : {model_name}")
    print(f"Device        : {device}")

    if "pulsewoq" not in model_name:
        print(f"WARNING: model '{model_name}' is not a pulsewoq variant - "
              "forward_with_quality may not be available.")

    # -- Load model ------------------------------------------------------------
    model = create_model(model_name).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(
        f"Loaded best.pt  (epoch {ckpt.get('epoch', '?')}, "
        f"val_loss={ckpt.get('val_loss', float('nan')):.4f})"
    )

    if not hasattr(model, "forward_with_quality"):
        print("ERROR: model does not implement forward_with_quality.", file=sys.stderr)
        sys.exit(1)

    # -- Test dataset ----------------------------------------------------------
    test_dir = args.dataset_dir / "test"
    try:
        test_ds = PPGDataset(test_dir, normalize=not args.no_normalize, preload=False)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print("Run bin/construct-dataset first to build the dataset.", file=sys.stderr)
        sys.exit(1)

    print(f"Test set      : {len(test_ds):,} segments from {test_ds.n_files} cases")

    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    # -- Inference -------------------------------------------------------------
    print("Running inference ...")
    preds, qualities, targets = run_inference_with_quality(model, loader, device)
    pred_sbp, pred_dbp = preds[:, 0],   preds[:, 1]
    true_sbp, true_dbp = targets[:, 0], targets[:, 1]
    n_total = len(qualities)

    print(
        f"Quality scores : min={qualities.min():.4f}  "
        f"max={qualities.max():.4f}  mean={qualities.mean():.4f}  "
        f"median={float(np.median(qualities)):.4f}"
    )

    # -- Scenario A -------------------------------------------------------------
    print("\n" + "=" * 62)
    print("  Scenario A: Quality Threshold Filtering")
    print("=" * 62)

    scenario_a: dict = {}
    for label, threshold in QUALITY_THRESHOLDS:
        mask  = (
            np.ones(n_total, dtype=bool)
            if threshold is None
            else (qualities >= threshold)
        )
        if mask.sum() == 0:
            print(f"\n  {label}: no segments pass this threshold - skipped.")
            continue
        sbp_m = compute_metrics(pred_sbp[mask], true_sbp[mask])
        dbp_m = compute_metrics(pred_dbp[mask], true_dbp[mask])
        print_threshold_table(label, n_total, sbp_m, dbp_m)
        scenario_a[label] = {
            "threshold": threshold,
            "n_samples": int(mask.sum()),
            "n_total":   n_total,
            "ratio_pct": float(mask.sum() / n_total * 100),
            "sbp":       sbp_m,
            "dbp":       dbp_m,
        }

    # -- Scenario B -------------------------------------------------------------
    print("\n\n" + "=" * 62)
    print(
        f"  Scenario B: Best-of-N Simulation  "
        f"(max_n={args.max_n}, trials={args.n_trials})"
    )
    print("=" * 62)

    # Group segment indices by case (file_idx)
    case_groups: dict[int, list[int]] = {}
    for i, (file_idx, _) in enumerate(test_ds._segs):
        case_groups.setdefault(file_idx, []).append(i)
    print(f"\n  Cases: {len(case_groups)}  "
          f"(avg {n_total / len(case_groups):.1f} segments/case)")

    rng = np.random.default_rng(42)
    print(f"  Simulating ... ", end="", flush=True)
    n_values, sbp_maes, sbp_stds, dbp_maes, dbp_stds = simulate_best_of_n(
        qualities, pred_sbp, pred_dbp, true_sbp, true_dbp,
        case_groups, args.max_n, args.n_trials, rng,
    )
    print("done.")

    print(f"\n  {'N':>3}  {'SBP MAE':>10}  {'SBP SD':>8}  {'DBP MAE':>10}  {'DBP SD':>8}")
    print(f"  {'-'*3}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}")
    for nv, sm, ss, dm, ds in zip(n_values, sbp_maes, sbp_stds, dbp_maes, dbp_stds):
        print(f"  {nv:>3}  {sm:>10.4f}  {ss:>8.4f}  {dm:>10.4f}  {ds:>8.4f}")

    # -- Save results ----------------------------------------------------------
    results = {
        "run_dir":    str(run_dir),
        "model":      model_name,
        "n_segments": n_total,
        "n_cases":    len(case_groups),
        "quality": {
            "min":    float(qualities.min()),
            "max":    float(qualities.max()),
            "mean":   float(qualities.mean()),
            "median": float(np.median(qualities)),
        },
        "scenario_a": scenario_a,
        "scenario_b": {
            "max_n":    args.max_n,
            "n_trials": args.n_trials,
            "n_values": n_values,
            "sbp_mae":  [round(v, 6) for v in sbp_maes],
            "sbp_std":  [round(v, 6) for v in sbp_stds],
            "dbp_mae":  [round(v, 6) for v in dbp_maes],
            "dbp_std":  [round(v, 6) for v in dbp_stds],
        },
    }
    results_path = run_dir / "eval_quality_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {results_path}")

    scatter_path = run_dir / "eval_quality_scatter.png"
    plot_scatter_by_threshold(pred_sbp, pred_dbp, true_sbp, true_dbp, qualities, scatter_path)
    print(f"Saved: {scatter_path}")

    dist_path = run_dir / "eval_quality_dist.png"
    plot_quality_distribution(qualities, dist_path)
    print(f"Saved: {dist_path}")

    bon_path = run_dir / "eval_best_of_n.png"
    plot_best_of_n(n_values, sbp_maes, sbp_stds, dbp_maes, dbp_stds, bon_path)
    print(f"Saved: {bon_path}")


if __name__ == "__main__":
    main()
