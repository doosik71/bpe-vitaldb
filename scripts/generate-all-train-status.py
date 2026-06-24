"""
Generate training status graphs for all registered BPE models.

For each model in bpe.models.list_models(), runs generate-train-status.py if
metrics.csv exists in <models-dir>/<model>/. Models without metrics.csv are
skipped with a warning.

Usage:
    uv run python scripts/generate-all-train-status.py
    uv run python scripts/generate-all-train-status.py --models-dir data/models-v1
    uv run python scripts/generate-all-train-status.py --no-save

Options:
    --models-dir    Root directory of trained models  (default: data/models)
    --dry-run       Print commands without executing them
    Additional options are forwarded to generate-train-status.py (e.g. --no-save)
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
from pathlib import Path

from bpe.models import list_models

ROOT = Path(__file__).parent.parent
STATUS_SCRIPT = ROOT / "scripts" / "generate-train-status.py"
DEFAULT_MODELS_DIR = Path("data/models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Generate training status graphs for all registered BPE models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Root directory of trained models (default: data/models)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    args, forward_args = parser.parse_known_args()

    if forward_args and forward_args[0] == "--":
        forward_args = forward_args[1:]

    return args, forward_args


def build_command(run_dir: Path, forward_args: list[str]) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        str(STATUS_SCRIPT),
        str(run_dir),
        *forward_args,
    ]


def main() -> None:
    args, forward_args = parse_args()

    all_models = list(list_models())
    if not all_models:
        log.error("No models are registered in bpe.models.list_models().")
        sys.exit(1)

    models_dir: Path = args.models_dir

    models: list[str] = []
    skipped: list[str] = []
    for model in all_models:
        if (models_dir / model / "metrics.csv").exists():
            models.append(model)
        else:
            skipped.append(model)

    if skipped:
        log.warning("Skipping (no metrics.csv): %s", ", ".join(skipped))

    if not models:
        log.error("No trained models found under %s.", models_dir)
        sys.exit(1)

    log.info("Models to process: %s", ", ".join(models))
    log.info("Models dir       : %s", models_dir)
    log.info("Forward args     : %s", shlex.join(forward_args) if forward_args else "(none)")

    if args.dry_run:
        for model in models:
            run_dir = models_dir / model
            log.info("Dry run: %s", shlex.join(build_command(run_dir, forward_args)))
        return

    failed = False
    completed: list[tuple[str, bool]] = []

    for model in models:
        run_dir = models_dir / model
        cmd = build_command(run_dir, forward_args)

        log.info("Generate %-25s ...", model)

        result = subprocess.run(cmd, cwd=ROOT)

        ok = result.returncode == 0
        completed.append((model, ok))
        if ok:
            log.info("Done     %-25s", model)
        else:
            failed = True
            log.error("Failed   %-25s (exit %d)", model, result.returncode)

    log.info("=" * 60)
    for model, ok in completed:
        status = "ok    " if ok else "FAILED"
        log.info("%-25s  %s", model, status)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
