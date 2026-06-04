"""
Collect result PNG files from all model run directories into images/.

Scans data/models/<model>/<run-id>/ for result images and copies the
most-recent run's files to images/<graph_type>/<model>.png.

Files collected:
  loss_graph.png  → images/loss_graph/<model>.png
  mae_graph.png   → images/mae_graph/<model>.png
  error_hist.png  → images/error_hist/<model>.png
  eval_plot.png   → images/eval_plot/<model>.png

Usage:
    uv run python scripts/collect-result.py
    uv run python scripts/collect-result.py --models-dir data/models --output-dir images
"""

import argparse
import shutil
from pathlib import Path

GRAPH_NAMES = ["loss_graph.png", "mae_graph.png", "error_hist.png", "eval_plot.png"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect result PNG files from all model run directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--models-dir", type=Path, default=Path("data/models"),
        help="Root directory containing model run subdirectories (default: data/models)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("images"),
        help="Root output directory for collected images (default: images)",
    )
    return p.parse_args()


def collect(models_dir: Path, output_dir: Path) -> None:
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        return

    # group run directories by model name
    model_runs: dict[str, list[Path]] = {}
    for run_dir in models_dir.glob("*/*"):
        if not run_dir.is_dir():
            continue
        model_name = run_dir.parent.name
        model_runs.setdefault(model_name, []).append(run_dir)

    if not model_runs:
        print("No run directories found.")
        return

    copied = 0
    for model_name, run_dirs in sorted(model_runs.items()):
        # pick the most recent run (run directories are timestamp-named)
        latest_run = sorted(run_dirs)[-1]

        for graph_name in GRAPH_NAMES:
            src = latest_run / graph_name
            if not src.exists():
                continue

            graph_type = graph_name.removesuffix(".png")
            dest_dir = output_dir / graph_type
            dest_dir.mkdir(parents=True, exist_ok=True)

            dest = dest_dir / f"{model_name}.png"
            shutil.copy2(src, dest)
            print(f"  {src}  →  {dest}")
            copied += 1

    print(f"\n{copied} file(s) copied.")


def main() -> None:
    args = parse_args()
    collect(args.models_dir, args.output_dir)


if __name__ == "__main__":
    main()
