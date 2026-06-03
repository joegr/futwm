"""
Adapter registry.

Adapters register themselves at import time via :func:`register_adapter`,
which lets us declare 20 vendors in a single place and instantiate any
of them by ``provider_id`` without depending on every adapter module
upfront.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from .base import ProviderAdapter

__all__ = [
    "AdapterError",
    "register_adapter",
    "get_adapter",
    "list_adapters",
]


class AdapterError(RuntimeError):
    """Raised when adapter registration or lookup fails."""


_REGISTRY: dict[str, type[ProviderAdapter]] = {}

T = TypeVar("T", bound=ProviderAdapter)


def register_adapter(provider_id: str):
    """
    Class decorator. Register an adapter under *provider_id*.

    >>> @register_adapter("statsbomb")
    ... class StatsBombAdapter(ProviderAdapter):
    ...     info = ProviderInfo(...)
    ...     def load_match(self, source, **kw): ...
    """
    if not provider_id:
        raise AdapterError("provider_id must be non-empty")

    def _decorate(cls: type[T]) -> type[T]:
        if not isinstance(cls, type) or not issubclass(cls, ProviderAdapter):
            raise AdapterError(
                f"@register_adapter target must be a ProviderAdapter subclass; "
                f"got {cls!r}"
            )
        if not hasattr(cls, "info"):
            raise AdapterError(
                f"{cls.__name__}: must declare a class-level `info` ProviderInfo"
            )
        if cls.info.provider_id != provider_id:                       # type: ignore[attr-defined]
            raise AdapterError(
                f"{cls.__name__}: decorator provider_id={provider_id!r} "
                f"does not match info.provider_id={cls.info.provider_id!r}"  # type: ignore[attr-defined]
            )
        if provider_id in _REGISTRY:
            raise AdapterError(
                f"provider_id {provider_id!r} is already registered to "
                f"{_REGISTRY[provider_id].__name__}"
            )
        _REGISTRY[provider_id] = cls
        return cls

    return _decorate


def get_adapter(provider_id: str, **adapter_kwargs) -> ProviderAdapter:
    """Instantiate the adapter registered under *provider_id*."""
    try:
        cls = _REGISTRY[provider_id]
    except KeyError as e:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise AdapterError(
            f"unknown provider_id {provider_id!r}; known: {known}"
        ) from e
    return cls(**adapter_kwargs)


def list_adapters() -> list[str]:
    """Return the sorted list of registered provider ids."""
    return sorted(_REGISTRY)


# Internal — for tests only.
def _registered_class(provider_id: str) -> type[ProviderAdapter]:
    return _REGISTRY[provider_id]


def _all_registered_classes() -> Iterable[type[ProviderAdapter]]:
    return tuple(_REGISTRY.values())
