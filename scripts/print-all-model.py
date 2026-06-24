"""
Print layer structure for all registered BPE models.

For each model in bpe.models.list_models(), runs print-model.py and saves
the output to <models-dir>/<model>/struct.txt.

Usage:
    uv run python scripts/print-all-model.py
    uv run python scripts/print-all-model.py --models-dir data/models-v1
    uv run python scripts/print-all-model.py --input-length 1000 --batch-size 1

Options:
    --models-dir    Root directory for struct.txt outputs  (default: data/models)
    --dry-run       Print commands without executing them
    Additional options are forwarded to print-model.py (e.g. --input-length, --batch-size, --device)
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
PRINT_SCRIPT = ROOT / "scripts" / "print-model.py"
DEFAULT_MODELS_DIR = Path("data/models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Print layer structure for all registered BPE models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Root directory for struct.txt outputs (default: data/models)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them",
    )
    args, forward_args = parser.parse_known_args()

    if forward_args and forward_args[0] == "--":
        forward_args = forward_args[1:]

    for tok in forward_args:
        if tok == "--model" or tok.startswith("--model="):
            parser.error("Do not pass --model; print-all-model.py runs all models automatically.")

    return args, forward_args


def build_command(model: str, forward_args: list[str]) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        str(PRINT_SCRIPT),
        "--model",
        model,
        *forward_args,
    ]


def main() -> None:
    args, forward_args = parse_args()

    models = list(list_models())
    if not models:
        log.error("No models are registered in bpe.models.list_models().")
        sys.exit(1)

    models_dir: Path = args.models_dir
    log.info("Models to print: %s", ", ".join(models))
    log.info("Output dir     : %s", models_dir)
    log.info("Forward args   : %s", shlex.join(forward_args) if forward_args else "(none)")

    if args.dry_run:
        for model in models:
            out_path = models_dir / model / "struct.txt"
            cmd = build_command(model, forward_args)
            log.info("Dry run: %s > %s", shlex.join(cmd), out_path)
        return

    failed = False
    completed: list[tuple[str, bool, Path]] = []

    for model in models:
        out_dir = models_dir / model
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "struct.txt"
        cmd = build_command(model, forward_args)

        log.info("Print %-25s -> %s", model, out_path)

        result = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            out_path.write_text(result.stdout, encoding="utf-8")
            completed.append((model, True, out_path))
            log.info("Done  %-25s (%d lines)", model, result.stdout.count("\n"))
        else:
            failed = True
            log.error("Failed %-24s (exit %d)", model, result.returncode)
            if result.stderr:
                log.error("stderr: %s", result.stderr.strip())
            completed.append((model, False, out_path))

    log.info("=" * 60)
    for model, ok, path in completed:
        status = "ok    " if ok else "FAILED"
        log.info("%-25s  %s  %s", model, status, path)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
