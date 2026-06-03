"""
Provider adapters — read third-party soccer data into ``MatchEventStream``.

This sub-package is opt-in: importing :mod:`soccer_model` does **not**
pull adapters in, and importing one adapter does not pull in the others.

Public API
----------
* :class:`ProviderAdapter`     — ABC every adapter implements
* :class:`ProviderInfo`        — declarative metadata per provider
* :func:`register_adapter`     — decorator / registry hook
* :func:`get_adapter`          — instantiate by provider id
* :func:`list_adapters`        — every registered provider id

Importing this package triggers registration of all 20 first-party
adapters declared in :mod:`soccer_model.adapters._registry`.

See ``docs/PROVIDERS.md`` for the research synthesis that motivates the
contract.
"""

from __future__ import annotations

# Trigger registration of every shipped adapter. Each module side-effects
# a ``@register_adapter("...")`` call when imported.
from . import _registry  # noqa: F401  (import for side effects)
from .base import (
    CoordinateFrame,
    DirectionConvention,
    Encoding,
    IdentityScheme,
    ProviderAdapter,
    ProviderInfo,
    TimeBasis,
    Transport,
)
from .registry import (
    AdapterError,
    get_adapter,
    list_adapters,
    register_adapter,
)

__all__ = [
    # ABCs / value types
    "ProviderAdapter",
    "ProviderInfo",
    "Transport",
    "Encoding",
    "CoordinateFrame",
    "DirectionConvention",
    "IdentityScheme",
    "TimeBasis",
    # registry
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "AdapterError",
]
