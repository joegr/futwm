"""Kernel ABC — common contract for every kernel in the pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class KernelError(RuntimeError):
    """Raised when a kernel cannot process its input."""


class Kernel(ABC):
    """
    Common interface for every kernel.

    A *kernel* is any deterministic transformation from one tensor-like
    representation to another. Concrete subclasses cover:

    * 1-D / 2-D Gaussian smoothing (``GaussianKernel*``)
    * spatial density estimation (``GaussianKDE2D``)
    * camera-to-pitch projective transforms (``PlanarHomography``)
    * pitch / non-pitch image segmentation (``HSVPitchMask``)

    Subclasses must implement :meth:`apply`. The :meth:`__call__` shortcut
    forwards to :meth:`apply` so that kernels can be composed naturally:

    >>> mask = HSVPitchMask()(frame)
    >>> heat = GaussianKDE2D(...)(touches)
    """

    name: str = "Kernel"

    @abstractmethod
    def apply(self, *args: Any, **kwargs: Any) -> Any:
        """Run the kernel. Subclass-specific signature."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:        # pragma: no cover
        return self.apply(*args, **kwargs)

    def __repr__(self) -> str:                                   # pragma: no cover
        return f"<{self.__class__.__name__} name={self.name!r}>"
