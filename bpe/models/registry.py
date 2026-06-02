"""String registry for model construction."""

from collections.abc import Callable

from torch import nn

ModelFactory = Callable[..., nn.Module]

_MODELS: dict[str, ModelFactory] = {}


def register_model(name: str, factory: ModelFactory | None = None):
    """Register a model factory by name."""

    def decorator(inner: ModelFactory) -> ModelFactory:
        key = name.strip().lower().replace("-", "_")
        if not key:
            raise ValueError("model name must not be empty")
        if key in _MODELS:
            raise ValueError(f"model already registered: {key}")
        _MODELS[key] = inner
        return inner

    if factory is None:
        return decorator
    return decorator(factory)


def get_model_class(name: str) -> ModelFactory:
    key = name.strip().lower().replace("-", "_")
    try:
        return _MODELS[key]
    except KeyError as exc:
        options = ", ".join(list_models())
        raise KeyError(f"Unknown model '{name}'. Available models: {options}") from exc


def create_model(name: str, **kwargs) -> nn.Module:
    return get_model_class(name)(**kwargs)


def list_models() -> tuple[str, ...]:
    return tuple(sorted(_MODELS))

