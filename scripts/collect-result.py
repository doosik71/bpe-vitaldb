"""
Collect result files from all model directories.

Scans data/models/<model>/ and copies files from each model directory.

Files collected:
  loss_graph.png             -> results/loss_graph/<model>.png
  mae_graph.png              -> results/mae_graph/<model>.png
  error_hist.png             -> results/error_hist/<model>.png
  eval_plot.png              -> results/eval_plot/<model>.png
  bland_altman.png           -> results/bland_altman/<model>.png
  bland_altman_all.png       -> results/bland_altman_all/<model>.png
  bland_altman_accepted.png  -> results/bland_altman_accepted/<model>.png
  eval_results.json          -> results/eval_results/<model>.json
  metrics.csv                -> results/metrics/<model>.csv

Usage:
    uv run python scripts/collect-result.py
    uv run python scripts/collect-result.py --models-dir data/models --results-dir data/results
"""

import argparse
import shutil
from pathlib import Path

GRAPH_NAMES = [
    "loss_graph.png",
    "mae_graph.png",
    "error_hist.png",
    "eval_plot.png",
    "bland_altman.png",
    "bland_altman_all.png",
    "bland_altman_accepted.png",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect result files from all model run directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--models-dir", type=Path, default=Path("data/models"),
        help="Root directory containing model run subdirectories (default: data/models)",
    )
    p.add_argument(
        "--results-dir", type=Path, default=Path("data/results"),
        help="Output directory for all collected files (default: data/results)",
    )
    return p.parse_args()


def collect(models_dir: Path, results_dir: Path) -> None:
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return

    model_dirs = sorted(d for d in models_dir.iterdir() if d.is_dir())
    if not model_dirs:
        print("No model directories found.")
        return

    copied = 0
    for model_dir in model_dirs:
        model_name = model_dir.name

        for graph_name in GRAPH_NAMES:
            src = model_dir / graph_name
            if not src.exists():
                continue
            graph_type = graph_name.removesuffix(".png")
            dest_dir = results_dir / graph_type
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{model_name}.png"
            shutil.copy2(src, dest)
            print(f"  {src}  ->  {dest}")
            copied += 1

        for src_name, dest_subdir, dest_ext in [
            ("eval_results.json", results_dir / "eval_results", ".json"),
            ("metrics.csv",       results_dir / "metrics",      ".csv"),
        ]:
            src = model_dir / src_name
            if not src.exists():
                continue
            dest_subdir.mkdir(parents=True, exist_ok=True)
            dest = dest_subdir / f"{model_name}{dest_ext}"
            shutil.copy2(src, dest)
            print(f"  {src}  ->  {dest}")
            copied += 1

    print(f"\n{copied} file(s) copied.")


def main() -> None:
    args = parse_args()
    collect(args.models_dir, args.results_dir)


if __name__ == "__main__":
    main()
