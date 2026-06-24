"""
Train every registered BPE model by dispatching train-model.py jobs.

This script enumerates all models from `bpe.models.list_models()` and launches
`train-model.py` once per model. It decides `--model` and `--device`
automatically; every other CLI argument is forwarded to `train-model.py`
unchanged.

Scheduling policy:
  - if CUDA devices are available, run up to one training job per CUDA device
    in parallel and queue the remaining models
  - if CUDA is unavailable, run models sequentially on CPU

Usage:
    uv run python scripts/train-all-model.py [TRAIN-MODEL OPTIONS...]

Examples:
    uv run python scripts/train-all-model.py --dataset-dir data/dataset-v1 --models-dir data/models-v1
    uv run python scripts/train-all-model.py --epochs 150 --batch-size 128 --workers 2
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
TRAIN_SCRIPT = ROOT / "scripts" / "train-model.py"
DEFAULT_OUTPUT_DIR = Path("data/models")

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
        description="Train all registered BPE models with automatic device scheduling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
        add_help=True,
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
        help="Print the training commands without launching them",
    )
    args, forward_args = parser.parse_known_args()

    if forward_args and forward_args[0] == "--":
        forward_args = forward_args[1:]

    for tok in forward_args:
        if tok == "--model" or tok.startswith("--model="):
            parser.error("Do not pass --model; train-all-model.py assigns models automatically.")
        if tok == "--device" or tok.startswith("--device="):
            parser.error("Do not pass --device; train-all-model.py assigns devices automatically.")

    return args, forward_args


def extract_option_path(forward_args: list[str], flag: str, default: Path) -> Path:
    for i, tok in enumerate(forward_args):
        if tok == flag and i + 1 < len(forward_args):
            return Path(forward_args[i + 1])
        if tok.startswith(flag + "="):
            return Path(tok.split("=", 1)[1])
    return default


def detect_devices() -> list[str]:
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count > 0:
            return [f"cuda:{i}" for i in range(count)]
    return ["cpu"]


def build_command(model: str, device: str, forward_args: list[str]) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        str(TRAIN_SCRIPT),
        "--model",
        model,
        "--device",
        device,
        *forward_args,
    ]


def launch_job(model: str, device: str, forward_args: list[str], output_dir: Path) -> ActiveJob:
    model_dir = output_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / "train-all.log"
    command = build_command(model, device, forward_args)

    log.info("Launch %-20s on %-7s -> %s", model, device, log_path)
    log.info("Command: %s", shlex.join(command))

    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write(f"\n=== train-all launch at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
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
    process._train_all_log_file = log_file  # type: ignore[attr-defined]

    return ActiveJob(
        model=model,
        device=device,
        process=process,
        log_path=log_path,
        started_at=time.time(),
    )


def close_job_log(job: ActiveJob) -> None:
    log_file = getattr(job.process, "_train_all_log_file", None)
    if log_file is not None:
        log_file.flush()
        log_file.close()


def main() -> None:
    args, forward_args = parse_args()

    models = list(list_models())
    if not models:
        log.error("No models are registered in bpe.models.list_models().")
        sys.exit(1)

    output_dir = extract_option_path(forward_args, "--models-dir", DEFAULT_OUTPUT_DIR)
    devices = detect_devices()
    max_parallel = len(devices)

    log.info("Registered models: %s", ", ".join(models))
    log.info("Detected devices : %s", ", ".join(devices))
    log.info("Parallel slots   : %d", max_parallel)
    log.info("Forward args     : %s", shlex.join(forward_args) if forward_args else "(none)")

    if args.dry_run:
        for idx, model in enumerate(models):
            device = devices[idx % len(devices)]
            log.info("Dry run: %s", shlex.join(build_command(model, device, forward_args)))
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
                active[device] = launch_job(model, device, forward_args, output_dir)

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
        log.warning("Interrupted. Terminating active training jobs...")
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
