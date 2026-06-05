"""
Analyse train / val / test dataset splits and produce summary statistics.

Reads every <case>.npz in data/dataset/{train,val,test}, computes per-split
BP statistics and per-case segment-count distributions, then writes:

  data/dataset/statistic.json          -- numerical summary (JSON)
  data/dataset/bp_distribution.png     -- SBP / DBP density histograms per split
  data/dataset/segments_per_case.png   -- per-case segment-count distribution

Purpose
-------
* bp_distribution.png   — check whether BP values are evenly distributed
                          across train / val / test
* segments_per_case.png — spot whether a small number of patients dominates
                          one split (data concentration check)
* statistic.json        — machine-readable numbers for downstream analysis

Usage:
    uv run python scripts/dataset-statistic.py [OPTIONS]
    uv run python scripts/dataset-statistic.py --dataset-dir data/dataset

Options:
    --dataset-dir   Root dataset directory  (default: data/dataset)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SPLITS = ("train", "val", "test")
SPLIT_COLORS = {"train": "#2196F3", "val": "#FF9800", "test": "#4CAF50"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute dataset statistics and generate distribution plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset-dir", type=Path, default=Path("data/dataset"),
        help="Root dataset directory (default: data/dataset)",
    )
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(split_dir: Path) -> dict:
    """Load BP labels (y) from all NPZ files in a split directory."""
    npz_files = sorted(split_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {split_dir}")

    seg_counts: list[int] = []
    sbp_list:   list[np.ndarray] = []
    dbp_list:   list[np.ndarray] = []

    for path in npz_files:
        f = np.load(path)
        y = f["y"]  # (N, 2): [SBP, DBP]
        seg_counts.append(len(y))
        sbp_list.append(y[:, 0])
        dbp_list.append(y[:, 1])

    return {
        "n_cases":   len(npz_files),
        "case_ids":  [p.stem for p in npz_files],
        "seg_counts": np.array(seg_counts, dtype=np.int64),
        "sbp": np.concatenate(sbp_list).astype(np.float32),
        "dbp": np.concatenate(dbp_list).astype(np.float32),
    }


# ── Statistics helpers ────────────────────────────────────────────────────────

def _bp_stats(arr: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr, ddof=1)),
        "min":  float(np.min(arr)),
        "max":  float(np.max(arr)),
        "p25":  float(np.percentile(arr, 25)),
        "p50":  float(np.percentile(arr, 50)),
        "p75":  float(np.percentile(arr, 75)),
    }


def _seg_stats(counts: np.ndarray, case_ids: list[str]) -> dict:
    n_total = int(counts.sum())
    sorted_idx = np.argsort(counts)[::-1]

    # What fraction of total segments do the top-10 % of cases hold?
    top10_n = max(1, int(np.ceil(len(counts) * 0.10)))
    top10_segs = int(counts[sorted_idx[:top10_n]].sum())

    top5 = [
        {"case_id": case_ids[i], "n_segments": int(counts[i])}
        for i in sorted_idx[:5]
    ]
    return {
        "mean": float(np.mean(counts)),
        "std":  float(np.std(counts, ddof=1)),
        "min":  int(counts.min()),
        "max":  int(counts.max()),
        "p25":  float(np.percentile(counts, 25)),
        "p50":  float(np.percentile(counts, 50)),
        "p75":  float(np.percentile(counts, 75)),
        "max_to_median_ratio": round(float(counts.max()) / float(np.median(counts)), 1),
        "top10pct_cases_hold_pct_segments": round(top10_segs / n_total * 100, 1),
        "top5_cases": top5,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_bp_distribution(raw: dict[str, dict], out_path: Path) -> None:
    """Overlaid SBP / DBP density histograms for train / val / test."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("BP Value Distribution Across Splits", fontsize=13)

    sbp_bins = np.linspace(60, 220, 80)
    dbp_bins = np.linspace(30, 140, 80)

    for ax, bp_key, label, bins in [
        (axes[0], "sbp", "SBP", sbp_bins),
        (axes[1], "dbp", "DBP", dbp_bins),
    ]:
        for split in SPLITS:
            if split not in raw:
                continue
            arr = raw[split][bp_key]
            ax.hist(
                arr, bins=bins,
                density=True,
                histtype="step",
                linewidth=1.8,
                color=SPLIT_COLORS[split],
                label=f"{split}  (n={len(arr):,})",
            )
        ax.set_xlabel(f"{label} (mmHg)")
        ax.set_ylabel("Density")
        ax.set_title(f"{label} Distribution")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_segments_per_case(raw: dict[str, dict], out_path: Path) -> None:
    """Per-split histogram of how many segments each case contributes."""
    present = [s for s in SPLITS if s in raw]
    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 5))
    if len(present) == 1:
        axes = [axes]
    fig.suptitle("Segments per Case — Concentration Check", fontsize=13)

    for ax, split in zip(axes, present):
        counts = raw[split]["seg_counts"]
        color  = SPLIT_COLORS[split]

        ax.hist(counts, bins=40, color=color, alpha=0.8, edgecolor="none")

        mean_val   = float(np.mean(counts))
        median_val = float(np.median(counts))
        ax.axvline(mean_val,   color="black", linewidth=1.2, linestyle="--",
                   label=f"Mean   = {mean_val:,.0f}")
        ax.axvline(median_val, color="red",   linewidth=1.2, linestyle="-",
                   label=f"Median = {median_val:,.0f}")

        # Annotate the single most data-heavy case
        top_idx  = int(np.argmax(counts))
        top_case = raw[split]["case_ids"][top_idx]
        top_val  = int(counts[top_idx])
        ax.annotate(
            f"max: {top_val:,}\n(case {top_case})",
            xy=(top_val, 0),
            xytext=(top_val, ax.get_ylim()[1] * 0.5 if ax.get_ylim()[1] > 0 else 1),
            fontsize=7,
            color="darkred",
            arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8),
            ha="right",
        )

        n_total = int(counts.sum())
        ax.set_xlabel("Segments per case")
        ax.set_ylabel("Number of cases")
        ax.set_title(
            f"{split.capitalize()}  "
            f"({len(counts):,} cases · {n_total:,} segments)"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    dataset_dir: Path = args.dataset_dir

    if not dataset_dir.exists():
        print(f"ERROR: dataset directory not found: {dataset_dir}")
        print("Run bin/construct-dataset first to build the dataset.")
        return

    # ── Load splits ───────────────────────────────────────────────────────────
    print("Loading dataset splits …")
    raw: dict[str, dict] = {}
    for split in SPLITS:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            print(f"  {split}: directory not found, skipping.")
            continue
        print(f"  {split}: ", end="", flush=True)
        d = load_split(split_dir)
        raw[split] = d
        print(f"{d['n_cases']} cases · {int(d['seg_counts'].sum()):,} segments")

    if not raw:
        print("ERROR: No split data loaded.")
        return

    # ── Compute statistics ────────────────────────────────────────────────────
    summary: dict = {}
    for split, d in raw.items():
        summary[split] = {
            "n_cases":            d["n_cases"],
            "n_segments":         int(d["seg_counts"].sum()),
            "sbp":                _bp_stats(d["sbp"]),
            "dbp":                _bp_stats(d["dbp"]),
            "segments_per_case":  _seg_stats(d["seg_counts"], d["case_ids"]),
        }

    # ── Print table ───────────────────────────────────────────────────────────
    print()
    cols = (
        f"  {'Split':<8}  {'Cases':>6}  {'Segments':>12}"
        f"  {'SBP mean±std':>16}  {'DBP mean±std':>16}  {'max/median':>10}"
        f"  {'top10% holds':>13}"
    )
    print(cols)
    print("-" * len(cols))
    for split in SPLITS:
        if split not in summary:
            continue
        s   = summary[split]
        sbp = s["sbp"]
        dbp = s["dbp"]
        spc = s["segments_per_case"]
        print(
            f"  {split:<8}  {s['n_cases']:>6}  {s['n_segments']:>12,}"
            f"  {sbp['mean']:>7.1f} ± {sbp['std']:<6.1f}"
            f"  {dbp['mean']:>7.1f} ± {dbp['std']:<6.1f}"
            f"  {spc['max_to_median_ratio']:>10.1f}"
            f"  {spc['top10pct_cases_hold_pct_segments']:>12.1f}%"
        )
    print()

    # ── Save JSON ─────────────────────────────────────────────────────────────
    stat_path = dataset_dir / "statistic.json"
    stat_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {stat_path}")

    # ── Save plots ────────────────────────────────────────────────────────────
    bp_path  = dataset_dir / "bp_distribution.png"
    plot_bp_distribution(raw, bp_path)
    print(f"Saved: {bp_path}")

    seg_path = dataset_dir / "segments_per_case.png"
    plot_segments_per_case(raw, seg_path)
    print(f"Saved: {seg_path}")


if __name__ == "__main__":
    main()
