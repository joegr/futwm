"""
Pure-numpy image and spatial kernels.

This sub-package contains every CV / spatial operation that does **not**
require OpenCV, PyTorch, or any other heavy dependency. Each kernel
implements the :class:`~soccer_vision.kernels.base.Kernel` ABC so that
the pipeline can compose them uniformly.
"""

from __future__ import annotations

from .base import Kernel, KernelError
from .gaussian import GaussianKernel1D, GaussianKernel2D
from .homography import HomographyError, PlanarHomography
from .kde import GaussianKDE2D
from .pitch_mask import HSVPitchMask, PitchMaskConfig

__all__ = [
    "Kernel",
    "KernelError",
    "GaussianKernel1D",
    "GaussianKernel2D",
    "GaussianKDE2D",
    "HSVPitchMask",
    "PitchMaskConfig",
    "PlanarHomography",
    "HomographyError",
]
