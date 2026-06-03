"""
Gaussian smoothing kernels — pure numpy, no OpenCV / scipy dependency.

The 2-D kernel is implemented as two passes of a 1-D kernel (separability),
which is O(N · k) per pixel rather than O(N · k²).
"""

from __future__ import annotations

import math

import numpy as np

from .base import Kernel, KernelError


def _gaussian_1d(sigma: float, radius: int | None = None) -> np.ndarray:
    """
    Return a normalised 1-D Gaussian kernel.

    Parameters
    ----------
    sigma : float
        Standard deviation in pixels. Must be > 0.
    radius : int, optional
        Half-width of the kernel. Defaults to ``ceil(4 * sigma)`` which
        captures ~99.99% of the mass.
    """
    if sigma <= 0:
        raise KernelError(f"sigma must be > 0, got {sigma!r}")
    r = int(math.ceil(4.0 * sigma)) if radius is None else int(radius)
    if r < 1:
        raise KernelError(f"radius must be >= 1, got {r}")
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k


class GaussianKernel1D(Kernel):
    """
    1-D Gaussian smoothing along a chosen axis.

    Examples
    --------
    >>> g = GaussianKernel1D(sigma=2.0)
    >>> smoothed = g.apply(signal)
    """

    name = "GaussianKernel1D"

    def __init__(self, sigma: float, radius: int | None = None) -> None:
        self.sigma = float(sigma)
        self.kernel = _gaussian_1d(self.sigma, radius)

    def apply(self, signal: np.ndarray, axis: int = -1) -> np.ndarray:
        """Convolve *signal* with the kernel along *axis* using ``mode='reflect'``."""
        arr = np.asarray(signal, dtype=np.float64)
        if arr.ndim == 0:
            raise KernelError("signal must be at least 1-D")
        return _convolve_axis_reflect(arr, self.kernel, axis=axis)


class GaussianKernel2D(Kernel):
    """
    Separable 2-D Gaussian smoothing for image / heat-map inputs.

    Operates on the last two axes of the input array (typical for grayscale
    images of shape ``(H, W)`` or multichannel images of shape ``(H, W, C)``).
    Color channels are smoothed independently.
    """

    name = "GaussianKernel2D"

    def __init__(self, sigma: float, radius: int | None = None) -> None:
        self.sigma = float(sigma)
        self._k = _gaussian_1d(self.sigma, radius)

    def apply(self, image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image, dtype=np.float64)
        if arr.ndim < 2:
            raise KernelError(f"image must be 2-D or 3-D, got shape {arr.shape}")
        # smooth along H (axis -2) then W (axis -1)
        smooth_h = _convolve_axis_reflect(arr,        self._k, axis=-2)
        smooth_w = _convolve_axis_reflect(smooth_h,   self._k, axis=-1)
        return smooth_w


# ── helpers ──────────────────────────────────────────────────────────────────

def _convolve_axis_reflect(arr: np.ndarray, kernel: np.ndarray, *, axis: int) -> np.ndarray:
    """
    1-D convolution of *arr* with *kernel* along *axis* with reflect padding.

    Pure numpy implementation — avoids a scipy dependency. Reflect padding
    matches ``scipy.ndimage.convolve1d(mode='reflect')`` and is the standard
    choice for image smoothing.
    """
    if kernel.ndim != 1:
        raise KernelError("kernel must be 1-D")
    if kernel.size % 2 != 1:
        raise KernelError("kernel must have odd length")
    n = arr.shape[axis]
    if n == 0:
        return arr.copy()
    r = kernel.size // 2

    # Pad along the chosen axis with 'reflect' mode (without repeating the
    # edge sample, matching scipy's reflect / numpy.pad('reflect')).
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (r, r)
    padded = np.pad(arr, pad_width, mode="reflect")

    # Vectorised sliding-window dot-product: build a view of shape
    # (..., n, kernel_size) and contract against kernel.
    swapped = np.moveaxis(padded, axis, -1)
    # shape after moveaxis: (..., n + 2r)
    windows = np.lib.stride_tricks.sliding_window_view(swapped, kernel.size, axis=-1)
    # windows shape: (..., n, kernel_size)
    out_swapped = np.tensordot(windows, kernel, axes=([-1], [0]))
    out = np.moveaxis(out_swapped, -1, axis)
    return out
