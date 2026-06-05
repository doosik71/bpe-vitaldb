"""
Print layer structure and output tensor shapes for BPE model architectures.

Runs a single forward pass with a dummy input and captures the output shape
of every named module via forward hooks.

Usage:
    uv run python scripts/print_model.py --model resnet1d
    uv run python scripts/print_model.py --model all
    uv run python scripts/print_model.py           # same as --model all

Options:
    --model         Model name or "all"                (default: all)
    --input-length  PPG segment length in samples      (default: 1000, 8 s @ 125 Hz)
    --batch-size    Batch size for the dummy forward    (default: 1)
    --device        cpu | cuda | auto                  (default: cpu)
"""

import argparse
import sys
from collections import OrderedDict

import torch
from torch import nn

from bpe.models import create_model, list_models


# ── helpers ──────────────────────────────────────────────────────────────────

def _fmt_shape(shape) -> str:
    if shape is None:
        return "-"
    return "(" + ", ".join(str(d) for d in shape) + ")"


def _param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters(recurse=False))


def _fmt_params(n: int) -> str:
    if n == 0:
        return ""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} M"
    if n >= 1_000:
        return f"{n / 1_000:.1f} K"
    return str(n)


def _total_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── forward-hook capture ──────────────────────────────────────────────────────

def _collect_output_shapes(
    model: nn.Module,
    dummy_inputs: tuple,
    dummy_kwargs: dict,
) -> OrderedDict:
    """Return {qualified_name: output_shape} for every named module."""
    shapes: OrderedDict = OrderedDict()
    hooks = []

    for name, module in model.named_modules():
        if name == "":
            continue  # skip the root module itself

        def _make_hook(n):
            def hook(_, __, output):
                if isinstance(output, torch.Tensor):
                    shapes[n] = tuple(output.shape)
                elif isinstance(output, (list, tuple)):
                    # take the first tensor found
                    for item in output:
                        if isinstance(item, torch.Tensor):
                            shapes[n] = tuple(item.shape)
                            break
                # non-tensor outputs (e.g. Identity on skip path) → None
                if n not in shapes:
                    shapes[n] = None
            return hook

        hooks.append(module.register_forward_hook(_make_hook(name)))

    model.eval()
    with torch.no_grad():
        try:
            model(*dummy_inputs, **dummy_kwargs)
        except Exception as exc:
            print(f"  [warn] forward pass failed: {exc}", file=sys.stderr)

    for h in hooks:
        h.remove()

    return shapes


# ── table printer ─────────────────────────────────────────────────────────────

_COL_NAME   = 52
_COL_TYPE   = 28
_COL_SHAPE  = 28
_COL_PARAMS = 10

def _header() -> str:
    return (
        f"{'Layer (name)':<{_COL_NAME}}"
        f"{'Type':<{_COL_TYPE}}"
        f"{'Output shape':<{_COL_SHAPE}}"
        f"{'Params':>{_COL_PARAMS}}"
    )

def _separator() -> str:
    return "-" * (_COL_NAME + _COL_TYPE + _COL_SHAPE + _COL_PARAMS)

def _row(name: str, module: nn.Module, shape) -> str:
    type_name = type(module).__name__
    return (
        f"{name:<{_COL_NAME}}"
        f"{type_name:<{_COL_TYPE}}"
        f"{_fmt_shape(shape):<{_COL_SHAPE}}"
        f"{_fmt_params(_param_count(module)):>{_COL_PARAMS}}"
    )


# ── per-model dummy input factory ─────────────────────────────────────────────

def _make_inputs(
    model_name: str,
    batch: int,
    length: int,
    device: torch.device,
) -> tuple[tuple, dict]:
    """Return (args, kwargs) for a forward call appropriate to each model."""
    ppg = torch.randn(batch, length, device=device)

    return (ppg,), {}


# ── main display function ─────────────────────────────────────────────────────

def print_model_summary(
    model_name: str,
    *,
    input_length: int = 1000,
    batch_size: int = 1,
    device: torch.device,
) -> None:
    width = _COL_NAME + _COL_TYPE + _COL_SHAPE + _COL_PARAMS
    print(f"\n{'=' * width}")
    print(f"  Model: {model_name}")
    print(f"{'=' * width}")

    try:
        model = create_model(model_name).to(device)
    except Exception as exc:
        print(f"  [error] Could not instantiate model: {exc}")
        return

    args, kwargs = _make_inputs(model_name, batch_size, input_length, device)

    shapes = _collect_output_shapes(model, args, kwargs)

    print(_header())
    print(_separator())

    for name, module in model.named_modules():
        if name == "":
            continue
        shape = shapes.get(name)
        print(_row(name, module, shape))

    print(_separator())
    total   = _total_params(model)
    trainable = _trainable_params(model)
    print(f"  Total params    : {total:,}  ({_fmt_params(total)})")
    print(f"  Trainable params: {trainable:,}  ({_fmt_params(trainable)})")
    print(
        f"  Input shape     : ({batch_size}, {input_length})"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print layer structure and output shapes for BPE models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model",
        default=None,
        help=(
            f"Model name or 'all'. Available: {', '.join(list_models())}. "
            "Omit to list available models."
        ),
    )
    p.add_argument(
        "--input-length",
        type=int,
        default=1000,
        metavar="N",
        help="PPG segment length in samples (default: 1000 = 8 s @ 125 Hz)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help="Batch size for the dummy forward pass (default: 1)",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device to run the forward pass on (default: cpu)",
    )
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()

    if args.model is None:
        models = list_models()
        print(f"Available models ({len(models)}):")
        for name in models:
            print(f"  {name}")
        print("\nRun with --model <name> or --model all to inspect a model.")
        return

    device = resolve_device(args.device)

    if args.model.lower() == "all":
        targets = list(list_models())
    else:
        targets = [args.model.lower()]

    for name in targets:
        print_model_summary(
            name,
            input_length=args.input_length,
            batch_size=args.batch_size,
            device=device,
        )

    print()


if __name__ == "__main__":
    main()
