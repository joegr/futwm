"""
HSV-based pitch segmentation kernel.

Cheap, deterministic, and useful as a *prior* to feed into a heavier
segmentation model: anywhere the pixel is roughly grass-coloured (a wide
HSV band), the pixel is on-pitch. This filters out stands, sky, players,
and the ball before object detection runs.

The implementation is pure numpy. It accepts RGB or BGR frames; specify
the channel order via :attr:`PitchMaskConfig.channel_order`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import Kernel, KernelError


@dataclass(slots=True)
class PitchMaskConfig:
    """Tuning parameters for the HSV pitch segmentation."""

    # Hue band that covers most grass shades (turfgrass, dyed turf, drier
    # pitches). Hue is on [0, 180] to match the standard CV convention.
    hue_min: float = 30.0
    hue_max: float = 95.0
    # Saturation must be high enough to exclude grey concrete / advertising.
    sat_min: float = 25.0
    # Value (brightness) must be in a "lit grass" range — excludes dark
    # shadows and over-exposed touch-line paint.
    val_min: float = 25.0
    val_max: float = 240.0
    # Frame channel order: 'rgb' (matplotlib / Pillow) or 'bgr' (OpenCV).
    channel_order: str = "rgb"

    # Optional morphological close radius (pixels) to fill small holes in
    # the mask. 0 disables. Implemented with a simple square kernel for
    # numpy purity — for high-quality results, plug in OpenCV in `detection/`.
    close_radius: int = 0

    # Validation
    def __post_init__(self) -> None:
        if self.channel_order not in ("rgb", "bgr"):
            raise KernelError(f"channel_order must be 'rgb' or 'bgr', got {self.channel_order!r}")
        if self.close_radius < 0:
            raise KernelError("close_radius must be >= 0")


@dataclass(slots=True)
class _HSV:
    h: np.ndarray = field(default_factory=lambda: np.empty(0))
    s: np.ndarray = field(default_factory=lambda: np.empty(0))
    v: np.ndarray = field(default_factory=lambda: np.empty(0))


def _rgb_to_hsv(rgb: np.ndarray) -> _HSV:
    """
    Convert an HxWx3 RGB image (uint8 or float in [0, 255]) to OpenCV-style HSV:

        H ∈ [0, 180]  (degree / 2)
        S ∈ [0, 255]
        V ∈ [0, 255]

    Pure numpy — vectorised, no Python-level pixel loop.
    """
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise KernelError(f"RGB image must be HxWx3, got shape {rgb.shape}")
    rgb_f = rgb.astype(np.float64)

    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc

    delta = maxc - minc
    s = np.where(maxc > 0, (delta / np.where(maxc == 0, 1, maxc)) * 255.0, 0.0)

    # Hue (0..360°, then divided by 2 for OpenCV convention)
    rc = np.where(delta == 0, 0, (maxc - r) / np.where(delta == 0, 1, delta))
    gc = np.where(delta == 0, 0, (maxc - g) / np.where(delta == 0, 1, delta))
    bc = np.where(delta == 0, 0, (maxc - b) / np.where(delta == 0, 1, delta))

    h360 = np.where(maxc == r, bc - gc,
           np.where(maxc == g, 2.0 + rc - bc, 4.0 + gc - rc)) * 60.0
    h360 = (h360 + 360.0) % 360.0
    h = np.where(delta == 0, 0, h360) / 2.0

    return _HSV(h=h, s=s, v=v)


def _square_close(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Naive morphological closing with a square structuring element.

    Implemented via boolean max-pooling (dilation) then min-pooling
    (erosion) using stride_tricks. Costs O(N · r) and is intended for
    *small* radii (≤ 5). For larger radii, install opencv-python-headless
    and use ``cv2.morphologyEx``.
    """
    if radius == 0:
        return mask
    if radius > 25:
        raise KernelError(
            f"close_radius={radius} too large for pure-numpy path; "
            f"install OpenCV for production use"
        )
    pad = radius
    padded = np.pad(mask.astype(np.uint8), pad, mode="constant", constant_values=0)
    h, w = mask.shape
    # Dilation: any True in the (2r+1)² neighbourhood
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (2 * radius + 1, 2 * radius + 1)
    )
    dilated = windows.max(axis=(-2, -1)).astype(bool)
    # Erosion of the dilation
    padded2 = np.pad(dilated.astype(np.uint8), pad, mode="constant", constant_values=0)
    windows2 = np.lib.stride_tricks.sliding_window_view(
        padded2, (2 * radius + 1, 2 * radius + 1)
    )
    eroded = windows2.min(axis=(-2, -1)).astype(bool)
    # Crop back to original shape
    return eroded[:h, :w]


class HSVPitchMask(Kernel):
    """
    Produces a boolean pitch-mask (True = on-pitch) from an RGB / BGR frame.

    >>> mask = HSVPitchMask()(frame)   # shape (H, W), dtype bool
    """

    name = "HSVPitchMask"

    def __init__(self, config: PitchMaskConfig | None = None) -> None:
        self.config = config or PitchMaskConfig()

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise KernelError(f"frame must be HxWx3, got shape {frame.shape}")

        if self.config.channel_order == "bgr":
            frame = frame[..., ::-1]

        hsv = _rgb_to_hsv(frame)
        c = self.config
        mask = (
            (hsv.h >= c.hue_min) & (hsv.h <= c.hue_max) &
            (hsv.s >= c.sat_min) &
            (hsv.v >= c.val_min) & (hsv.v <= c.val_max)
        )
        if c.close_radius > 0:
            mask = _square_close(mask, c.close_radius)
        return mask
