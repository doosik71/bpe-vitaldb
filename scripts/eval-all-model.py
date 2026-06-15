"""
Evaluate every registered BPE model by dispatching eval-model.py jobs.

This script enumerates all models from `bpe.models.list_models()` and launches
`eval-model.py` once per model. It decides the run directory and `--device`
automatically; every other CLI argument is forwarded to `eval-model.py`
unchanged.

Scheduling policy:
  - if CUDA devices are available, run up to one evaluation job per CUDA device
    in parallel and queue the remaining models
  - if CUDA is unavailable, run models sequentially on CPU

Models whose run directory does not contain `best.pt` are skipped with a warning.

Usage:
    uv run python scripts/eval-all-model.py [EVAL-MODEL OPTIONS...]

Examples:
    uv run python scripts/eval-all-model.py --models-dir data/models-v1
    uv run python scripts/eval-all-model.py --batch-size 256 --no-normalize
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import torch

from bpe.models import list_models

ROOT = Path(__file__).parent.parent
EVAL_SCRIPT = ROOT / "scripts" / "eval-model.py"
DEFAULT_MODELS_DIR = Path("data/models")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class ActiveJob:
    def __init__(
        self,
        *,
        model: str,
        device: str,
        process: subprocess.Popen,
        log_path: Path,
        started_at: float,
    ):
        self.model = model
        self.device = device
        self.process = process
        self.log_path = log_path
        self.started_at = started_at


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate all registered BPE models with automatic device scheduling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        add_help=True,
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Root directory containing trained model subdirectories (default: data/models)",
    )
    parser.add_argument(
        "--poll-sec",
        type=float,
        default=2.0,
        help="Scheduler polling interval in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the evaluation commands without launching them",
    )
    args, forward_args = parser.parse_known_args()

    if forward_args and forward_args[0] == "--":
        forward_args = forward_args[1:]

    for tok in forward_args:
        if tok == "--device" or tok.startswith("--device="):
            parser.error("Do not pass --device; eval-all-model.py assigns devices automatically.")
        if tok == "--duo" or tok.startswith("--duo"):
            parser.error("Do not pass --duo; eval-all-model.py only supports single-model evaluation.")

    return args, forward_args


def detect_devices() -> list[str]:
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count > 0:
            return [f"cuda:{i}" for i in range(count)]
    return ["cpu"]


def build_command(run_dir: Path, device: str, forward_args: list[str]) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        str(EVAL_SCRIPT),
        str(run_dir),
        "--device",
        device,
        *forward_args,
    ]


def launch_job(model: str, run_dir: Path, device: str, forward_args: list[str]) -> ActiveJob:
    log_path = run_dir / "eval-all.log"
    command = build_command(run_dir, device, forward_args)

    log.info("Launch %-20s on %-7s -> %s", model, device, log_path)
    log.info("Command: %s", shlex.join(command))

    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write(f"\n=== eval-all launch at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log_file.write(f"device={device}\n")
    log_file.write(f"command={shlex.join(command)}\n\n")
    log_file.flush()

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process._eval_all_log_file = log_file  # type: ignore[attr-defined]

    return ActiveJob(
        model=model,
        device=device,
        process=process,
        log_path=log_path,
        started_at=time.time(),
    )


def close_job_log(job: ActiveJob) -> None:
    log_file = getattr(job.process, "_eval_all_log_file", None)
    if log_file is not None:
        log_file.flush()
        log_file.close()


def main() -> None:
    args, forward_args = parse_args()

    all_models = list(list_models())
    if not all_models:
        log.error("No models are registered in bpe.models.list_models().")
        sys.exit(1)

    models_dir: Path = args.models_dir

    # Filter to models that have a trained checkpoint
    models: list[str] = []
    skipped: list[str] = []
    for model in all_models:
        run_dir = models_dir / model
        if (run_dir / "best.pt").exists():
            models.append(model)
        else:
            skipped.append(model)

    if skipped:
        log.warning("Skipping (no best.pt): %s", ", ".join(skipped))

    if not models:
        log.error("No trained models found under %s.", models_dir)
        sys.exit(1)

    devices = detect_devices()
    max_parallel = len(devices)

    log.info("Models to evaluate: %s", ", ".join(models))
    log.info("Detected devices  : %s", ", ".join(devices))
    log.info("Parallel slots    : %d", max_parallel)
    log.info("Forward args      : %s", shlex.join(forward_args) if forward_args else "(none)")

    if args.dry_run:
        for idx, model in enumerate(models):
            device = devices[idx % len(devices)]
            run_dir = models_dir / model
            log.info("Dry run: %s", shlex.join(build_command(run_dir, device, forward_args)))
        return

    pending = deque(models)
    active: dict[str, ActiveJob] = {}
    completed: list[tuple[str, str, int, float, Path]] = []
    failed = False

    try:
        while pending or active:
            for device in devices:
                if device in active:
                    continue
                if not pending:
                    break
                model = pending.popleft()
                run_dir = models_dir / model
                active[device] = launch_job(model, run_dir, device, forward_args)

            if not active:
                break

            time.sleep(max(args.poll_sec, 0.1))

            finished_devices: list[str] = []
            for device, job in active.items():
                ret = job.process.poll()
                if ret is None:
                    continue
                elapsed = time.time() - job.started_at
                close_job_log(job)
                completed.append((job.model, job.device, ret, elapsed, job.log_path))
                if ret == 0:
                    log.info(
                        "Done   %-20s on %-7s in %.1fs",
                        job.model,
                        job.device,
                        elapsed,
                    )
                else:
                    failed = True
                    log.error(
                        "Failed %-20s on %-7s with exit code %d in %.1fs (log: %s)",
                        job.model,
                        job.device,
                        ret,
                        elapsed,
                        job.log_path,
                    )
                finished_devices.append(device)

            for device in finished_devices:
                active.pop(device, None)

    except KeyboardInterrupt:
        log.warning("Interrupted. Terminating active evaluation jobs...")
        for job in active.values():
            job.process.terminate()
        for job in active.values():
            try:
                job.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                job.process.kill()
            close_job_log(job)
        raise

    log.info("=" * 60)
    for model, device, ret, elapsed, log_path in completed:
        status = "ok" if ret == 0 else f"exit {ret}"
        log.info("%-20s  %-7s  %-8s  %.1fs  %s", model, device, status, elapsed, log_path)

    missing = [model for model in models if model not in {m for m, *_ in completed}]
    if missing:
        failed = True
        log.error("Not completed: %s", ", ".join(missing))

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
