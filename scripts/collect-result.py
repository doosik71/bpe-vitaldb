"""
Collect result files from all model directories.

Scans data/models/<model>/ and copies files from each model directory.

Files collected:
  loss_graph.png    → images/loss_graph/<model>.png
  mae_graph.png     → images/mae_graph/<model>.png
  error_hist.png    → images/error_hist/<model>.png
  eval_plot.png     → images/eval_plot/<model>.png
  eval_results.json → logs/eval_results/<model>.json
  metrics.csv       → logs/metrics/<model>.csv
  best.pt           → models/<model>.pt

Usage:
    uv run python scripts/collect-result.py
    uv run python scripts/collect-result.py --models-dir data/models --images-dir images --logs-dir logs --pt-dir models
"""

import argparse
import shutil
from pathlib import Path

GRAPH_NAMES = ["loss_graph.png", "mae_graph.png", "error_hist.png", "eval_plot.png"]


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
        "--images-dir", type=Path, default=Path("images"),
        help="Output directory for PNG images (default: images)",
    )
    p.add_argument(
        "--logs-dir", type=Path, default=Path("logs"),
        help="Output directory for eval_results.json and metrics.csv (default: logs)",
    )
    p.add_argument(
        "--pt-dir", type=Path, default=Path("models"),
        help="Output directory for best.pt model files (default: models)",
    )
    return p.parse_args()


def collect(models_dir: Path, images_dir: Path, logs_dir: Path, pt_dir: Path) -> None:
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
            dest_dir = images_dir / graph_type
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{model_name}.png"
            shutil.copy2(src, dest)
            print(f"  {src}  →  {dest}")
            copied += 1

        for src_name, dest_subdir, dest_ext in [
            ("eval_results.json", logs_dir / "eval_results", ".json"),
            ("metrics.csv",       logs_dir / "metrics",      ".csv"),
        ]:
            src = model_dir / src_name
            if not src.exists():
                continue
            dest_subdir.mkdir(parents=True, exist_ok=True)
            dest = dest_subdir / f"{model_name}{dest_ext}"
            shutil.copy2(src, dest)
            print(f"  {src}  →  {dest}")
            copied += 1

        src = model_dir / "best.pt"
        if src.exists():
            pt_dir.mkdir(parents=True, exist_ok=True)
            dest = pt_dir / f"{model_name}.pt"
            shutil.copy2(src, dest)
            print(f"  {src}  →  {dest}")
            copied += 1

    print(f"\n{copied} file(s) copied.")


def main() -> None:
    args = parse_args()
    collect(args.models_dir, args.images_dir, args.logs_dir, args.pt_dir)


if __name__ == "__main__":
    main()
