"""
Train a direct-regression BP model on the VitalDB NPZ dataset.

Saves checkpoints and a metrics CSV under data/models/<model>/<run-id>/.

Usage:
    uv run python scripts/train.py --model resnet1d [OPTIONS]

Options:
    --model              Model name from the registry (required)
                         Available: resnet1d, st_resnet, minception,
                                    xresnet1d
    --dataset-dir        Root dataset directory        (default: data/dataset)
    --output-dir         Root models directory         (default: data/models)
    --epochs             Maximum training epochs       (default: 100)
    --batch-size         Mini-batch size               (default: 256)
    --lr                 Initial learning rate         (default: 1e-3)
    --weight-decay       AdamW weight decay            (default: 1e-4)
    --patience           Early-stopping patience       (default: 15)
    --seed               Random seed                   (default: 42)
    --device             auto | cpu | cuda | cuda:N    (default: auto)
    --workers            DataLoader worker processes   (default: 4)
    --preload            Load all segments into RAM before training
    --no-normalize       Skip per-segment z-score normalization
    --resume             Path to a checkpoint .pt to resume from

Augmentation (all enabled by default; use --no-* to disable):
    --no-aug-noise       Disable Gaussian noise (std=0.01)
    --no-aug-scale       Disable amplitude scaling (×0.9~1.1)
    --no-aug-shift       Disable circular time shift (±25 samples)
    --no-aug-mask        Disable random masking (5~10% of samples)

Patient balancing (enabled by default):
    --no-patient-balance Disable per-patient WeightedRandomSampler.
                         By default each patient contributes equally to
                         every epoch regardless of segment count.
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from bpe.models import create_model, list_models
from bpe.train.augment import (
    AmplitudeScaling,
    GaussianNoise,
    PPGAugment,
    RandomMasking,
    TimeShift,
)
from bpe.train.dataset import PPGDataset
from bpe.train.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a BPE model on VitalDB NPZ segments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model", required=True,
        help=f"Model name. Available: {', '.join(list_models())}",
    )
    p.add_argument(
        "--dataset-dir", type=Path, default=Path("data/dataset"),
        help="Root dataset directory (default: data/dataset)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("data/models"),
        help="Root directory for saved runs (default: data/models)",
    )
    p.add_argument(
        "--epochs", type=int, default=100,
        help="Maximum training epochs (default: 100)",
    )
    p.add_argument(
        "--batch-size", type=int, default=256,
        help="Mini-batch size (default: 256)",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial learning rate (default: 1e-3)",
    )
    p.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="AdamW weight decay (default: 1e-4)",
    )
    p.add_argument(
        "--patience", type=int, default=15,
        help="Early-stopping patience in epochs (default: 15)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    p.add_argument(
        "--device", default="auto",
        help="Device: auto | cpu | cuda | cuda:N (default: auto)",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="DataLoader worker processes (default: 4)",
    )
    p.add_argument(
        "--preload", action="store_true",
        help="Load all segment arrays into RAM before training",
    )
    p.add_argument(
        "--no-normalize", action="store_true",
        help="Skip per-segment z-score normalization of PPG",
    )
    p.add_argument(
        "--resume", type=Path, default=None,
        help="Path to a checkpoint .pt file to resume training from",
    )

    # ── Augmentation flags (all default ON; --no-* to disable) ────────────────
    aug = p.add_argument_group("augmentation (all enabled by default)")
    aug.add_argument(
        "--no-aug-noise", dest="aug_noise", action="store_false", default=True,
        help="Disable Gaussian noise augmentation (std=0.01)",
    )
    aug.add_argument(
        "--no-aug-scale", dest="aug_scale", action="store_false", default=True,
        help="Disable amplitude scaling augmentation (×0.9~1.1)",
    )
    aug.add_argument(
        "--no-aug-shift", dest="aug_shift", action="store_false", default=True,
        help="Disable circular time-shift augmentation (±25 samples)",
    )
    aug.add_argument(
        "--no-aug-mask", dest="aug_mask", action="store_false", default=True,
        help="Disable random masking augmentation (5~10%% of samples)",
    )

    # ── Patient balancing ─────────────────────────────────────────────────────
    p.add_argument(
        "--no-patient-balance", dest="patient_balance", action="store_false",
        default=True,
        help=(
            "Disable per-patient WeightedRandomSampler. "
            "By default every patient contributes equally per epoch."
        ),
    )

    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(output_dir: Path, model_name: str) -> Path:
    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / model_name / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(run_dir: Path, args: argparse.Namespace) -> None:
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (run_dir / "config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )


def load_resume(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Path,
    device: torch.device,
) -> int:
    """Load checkpoint; return the epoch to resume from."""
    ckpt  = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt.get("epoch", 0)
    log.info(
        "Resumed from %s (epoch %d, val_loss=%.4f)",
        path, epoch, ckpt.get("val_loss", float("nan")),
    )
    return epoch


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    log.info("Model     : %s", args.model)
    log.info("Device    : %s", device)
    log.info("Dataset   : %s", args.dataset_dir)
    log.info("Batch size: %d  |  LR: %.2e  |  Epochs: %d", args.batch_size, args.lr, args.epochs)

    # ── Augmentation pipeline ─────────────────────────────────────────────────
    aug_transforms = []
    if args.aug_noise:
        aug_transforms.append(GaussianNoise(std=0.01))
    if args.aug_scale:
        aug_transforms.append(AmplitudeScaling(lo=0.9, hi=1.1))
    if args.aug_shift:
        aug_transforms.append(TimeShift(max_shift=25))
    if args.aug_mask:
        aug_transforms.append(RandomMasking(lo_frac=0.05, hi_frac=0.10))
    augment = PPGAugment(aug_transforms) if aug_transforms else None

    active = [n for n, f in [
        ("noise", args.aug_noise), ("scale", args.aug_scale),
        ("shift", args.aug_shift), ("mask",  args.aug_mask),
    ] if f]
    log.info(
        "Augmentation: %s",
        ", ".join(active) if active else "disabled",
    )
    log.info(
        "Patient balance: %s",
        "enabled (WeightedRandomSampler)" if args.patient_balance else "disabled",
    )

    # ── Datasets ─────────────────────────────────────────────────────────────
    normalize = not args.no_normalize
    try:
        train_ds = PPGDataset(
            args.dataset_dir / "train",
            normalize=normalize,
            preload=args.preload,
            augment=augment,
        )
        val_ds = PPGDataset(
            args.dataset_dir / "val",
            normalize=normalize,
            preload=args.preload,
        )
    except FileNotFoundError as e:
        log.error("%s", e)
        log.error("Run bin/construct-dataset.bat first to build the dataset.")
        sys.exit(1)

    log.info(
        "Train: %d segments from %d cases  |  Val: %d segments from %d cases",
        len(train_ds), train_ds.n_files,
        len(val_ds),   val_ds.n_files,
    )

    # Determine segment length for sanity check
    seg_len = train_ds.segment_length()
    log.info("Segment length: %d samples", seg_len)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    loader_kwargs = dict(
        batch_size  = args.batch_size,
        num_workers = args.workers,
        pin_memory  = device.type == "cuda",
    )
    if args.patient_balance:
        from torch.utils.data import WeightedRandomSampler
        weights = train_ds.sample_weights()
        sampler = WeightedRandomSampler(
            weights, num_samples=len(train_ds), replacement=True
        )
        train_loader = DataLoader(train_ds, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # ── Model ─────────────────────────────────────────────────────────────────
    try:
        model = create_model(args.model).to(device)
    except KeyError as e:
        log.error("%s", e)
        sys.exit(1)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Parameters: %s", f"{n_params:,}")

    # ── Optimiser and scheduler ───────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 1e-2,
    )

    # ── Optional resume ───────────────────────────────────────────────────────
    if args.resume:
        load_resume(model, optimizer, args.resume, device)

    # ── Run directory and config ──────────────────────────────────────────────
    run_dir = make_run_dir(args.output_dir, args.model)
    save_config(run_dir, args)
    log.info("Run directory: %s", run_dir)

    # ── Loss function ─────────────────────────────────────────────────────────
    # HuberLoss (delta=5 mmHg): quadratic for |error| < 5, linear beyond.
    # More robust to BP outliers than plain MSE; more stable than MAE.
    criterion = nn.HuberLoss(delta=5.0)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        run_dir=run_dir,
    )
    result = trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        patience=args.patience,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Best epoch  : %d", result["best_epoch"])
    log.info("Val loss    : %.4f", result["best_val_loss"])
    log.info("Val SBP MAE : %.2f mmHg", result["best_val_sbp_mae"])
    log.info("Val DBP MAE : %.2f mmHg", result["best_val_dbp_mae"])
    log.info("Checkpoints : %s", run_dir)

    # Append a summary JSON for quick comparison across runs
    summary_path = args.output_dir / args.model / "runs.jsonl"
    with open(summary_path, "a", encoding="utf-8") as f:
        import json as _json
        _json.dump(
            {
                "run_dir":       str(run_dir),
                **result,
                "model":         args.model,
                "epochs":        args.epochs,
                "batch_size":    args.batch_size,
                "lr":            args.lr,
                "weight_decay":  args.weight_decay,
                "seed":          args.seed,
            },
            f,
        )
        f.write("\n")


if __name__ == "__main__":
    main()
