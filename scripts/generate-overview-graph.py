"""
Generate overview graphs comparing all models by parameter count vs metric.

For each metric (MAE, ME, SD, RMSE), produces one PNG with two subplots:
  Left:  SBP (systolic blood pressure)
  Right: DBP (diastolic blood pressure)

x-axis: trainable parameter count (log scale)
y-axis: metric value (mmHg)

Data sources:
  data/models/<model>/struct.txt           — trainable parameter count
  data/models/<model>/eval_results.json    — metric values

Output:
  images/mae.png
  images/me.png
  images/sd.png
  images/rmse.png

Usage:
    uv run python scripts/generate-overview-graph.py
    uv run python scripts/generate-overview-graph.py --models-dir data/models --output-dir images
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── colour palette (tab20, cycling) ──────────────────────────────────────────
import matplotlib.cm as cm
_PALETTE = [cm.tab20(i) for i in range(20)]


EXCLUDE_MODELS: set[str] = {"naive"}

METRICS: list[tuple[str, str, bool]] = [
    ("mae",  "MAE (mmHg)",  False),
    ("me",   "ME (mmHg)",   True),   # True → draw y = 0 reference line
    ("sd",   "SD (mmHg)",   False),
    ("rmse", "RMSE (mmHg)", False),
]


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate parameter-count vs metric overview graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--models-dir", type=Path, default=Path("data/models"),
        help="Root directory containing model subdirectories (default: data/models)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("images"),
        help="Directory to write PNG files (default: images)",
    )
    return p.parse_args()


# ── data loading ──────────────────────────────────────────────────────────────

def _parse_param_count(struct_path: Path) -> int | None:
    text = struct_path.read_text(encoding="utf-8")
    m = re.search(r"Trainable params:\s*([\d,]+)", text)
    return int(m.group(1).replace(",", "")) if m else None


def load_model_data(models_dir: Path) -> list[dict]:
    """Return list of dicts with model name, param count, and eval metrics."""
    records: list[dict] = []

    for model_dir in sorted(models_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        if model_dir.name in EXCLUDE_MODELS:
            continue

        struct_path = model_dir / "struct.txt"
        if not struct_path.exists():
            continue

        n_params = _parse_param_count(struct_path)
        if n_params is None:
            print(f"  [warn] could not parse param count from {struct_path}")
            continue

        eval_path = model_dir / "eval_results.json"
        if not eval_path.exists():
            print(f"  [warn] no eval_results.json found for {model_dir.name}")
            continue
        with open(eval_path, encoding="utf-8") as f:
            eval_data = json.load(f)

        records.append({
            "model":    model_dir.name,
            "n_params": n_params,
            "sbp":      eval_data["sbp"],
            "dbp":      eval_data["dbp"],
        })

    return records


# ── formatting helpers ────────────────────────────────────────────────────────

def _param_formatter(x: float, _pos) -> str:
    """Format parameter counts as 2, 15K, 440K, 2.18M, etc."""
    if x < 1_000:
        return f"{int(x)}"
    if x < 1_000_000:
        v = x / 1_000
        return f"{v:.0f}K" if v == int(v) else f"{v:.1f}K"
    v = x / 1_000_000
    return f"{v:.0f}M" if v == int(v) else f"{v:.2f}M"


# ── plotting ──────────────────────────────────────────────────────────────────

def _annotate(ax, x: float, y: float, label: str) -> None:
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(5, 4),
        textcoords="offset points",
        fontsize=7.5,
        clip_on=True,
    )


def plot_metric(
    data: list[dict],
    metric: str,
    ylabel: str,
    zero_line: bool,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, bp, bp_label in zip(axes, ["sbp", "dbp"], ["SBP", "DBP"]):
        for i, rec in enumerate(data):
            x = rec["n_params"]
            y = rec[bp][metric]
            color = _PALETTE[i % len(_PALETTE)]
            ax.scatter(x, y, s=70, color=color, zorder=5, label=rec["model"])
            _annotate(ax, x, y, rec["model"])

        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_param_formatter))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

        ax.set_xlabel("Trainable Parameters (log scale)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{bp_label} — {metric.upper()}", fontsize=11)
        ax.grid(True, which="major", linestyle="--", alpha=0.4)
        ax.grid(True, which="minor", linestyle=":",  alpha=0.2)

        if zero_line:
            ax.axhline(0, color="gray", linewidth=0.9, linestyle="--")

    fig.suptitle(
        f"Model Comparison: {metric.upper()} vs Parameter Count",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()

    out_path = output_dir / f"{metric}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    data = load_model_data(args.models_dir)
    if not data:
        print("No model data found — nothing to plot.")
        return

    print(f"Loaded {len(data)} models.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for metric, ylabel, zero_line in METRICS:
        plot_metric(data, metric, ylabel, zero_line, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
