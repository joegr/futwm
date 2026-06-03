"""
2-D Gaussian Kernel Density Estimation over the pitch.

Used for heat maps, press density surfaces, defensive shape, and any
spatial visualisation that needs a smooth field from a set of point
events (touches, shots, recoveries, …).
"""

from __future__ import annotations

import math

import numpy as np

from .base import Kernel, KernelError


class GaussianKDE2D(Kernel):
    """
    Bandwidth-controlled 2-D Gaussian KDE on a regular pitch grid.

    Coordinates are expressed in pitch metres (FIFA frame: x ∈ [0, length],
    y ∈ [0, width]). Output is a density field shaped ``(ny, nx)`` with
    ``y`` increasing along axis 0 and ``x`` along axis 1, matching the
    typical image convention used by visualisation code.

    Parameters
    ----------
    pitch_length : float, default 105.0
        Pitch length in metres (x extent).
    pitch_width : float, default 68.0
        Pitch width in metres (y extent).
    nx, ny : int, default 105, 68
        Grid resolution along x and y. Defaults give 1 m² cells.
    bandwidth : float, default 3.0
        Gaussian bandwidth (σ) in metres.
    normalise : bool, default True
        If True, divide the output so it integrates to 1 over the pitch
        (i.e. a probability density). If False, preserve the raw kernel
        sum (counts × area-weighting).
    """

    name = "GaussianKDE2D"

    def __init__(
        self,
        *,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        nx: int = 105,
        ny: int = 68,
        bandwidth: float = 3.0,
        normalise: bool = True,
    ) -> None:
        if pitch_length <= 0 or pitch_width <= 0:
            raise KernelError("pitch dimensions must be positive")
        if nx < 2 or ny < 2:
            raise KernelError("grid resolution must be >= 2 on each axis")
        if bandwidth <= 0:
            raise KernelError("bandwidth must be > 0")

        self.pitch_length = float(pitch_length)
        self.pitch_width = float(pitch_width)
        self.nx = int(nx)
        self.ny = int(ny)
        self.bandwidth = float(bandwidth)
        self.normalise = bool(normalise)

        # Cell centres in pitch metres
        self._xs = np.linspace(0.0, self.pitch_length, self.nx)
        self._ys = np.linspace(0.0, self.pitch_width, self.ny)
        # Cell area for proper integral normalisation
        self._dx = self.pitch_length / (self.nx - 1)
        self._dy = self.pitch_width / (self.ny - 1)

    # ── primary kernel API ───────────────────────────────────────────────────

    def apply(
        self,
        points: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute the density field for *points*.

        Parameters
        ----------
        points : array-like, shape (N, 2)
            (x, y) point coordinates in pitch metres. Points lying outside
            the pitch are clipped to the boundary before contributing.
        weights : array-like, shape (N,), optional
            Per-point weight (e.g. shot xG). Defaults to uniform 1.0.

        Returns
        -------
        ndarray, shape (ny, nx)
            Density field. When ``normalise=True``, ``sum(density) * dx * dy ≈ 1``.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if pts.size == 0:
            return np.zeros((self.ny, self.nx), dtype=np.float64)

        if weights is None:
            w = np.ones(pts.shape[0], dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w.shape[0] != pts.shape[0]:
                raise KernelError(
                    f"weights length {w.shape[0]} != points length {pts.shape[0]}"
                )

        # Clip to pitch
        pts[:, 0] = np.clip(pts[:, 0], 0.0, self.pitch_length)
        pts[:, 1] = np.clip(pts[:, 1], 0.0, self.pitch_width)

        # Outer-product Gaussian on a grid: ρ(x, y) = Σ w_i · g(x − x_i) · g(y − y_i)
        # where g(z) = exp(−z²/2σ²) / (σ √(2π)).
        sigma = self.bandwidth
        inv_2s2 = 1.0 / (2.0 * sigma * sigma)
        norm = 1.0 / (sigma * math.sqrt(2.0 * math.pi))

        # shape: (N, nx)
        gx = norm * np.exp(-((self._xs[None, :] - pts[:, 0, None]) ** 2) * inv_2s2)
        # shape: (N, ny)
        gy = norm * np.exp(-((self._ys[None, :] - pts[:, 1, None]) ** 2) * inv_2s2)

        # density(y, x) = Σ_i w_i · gy[i, y] · gx[i, x]
        weighted_gy = gy * w[:, None]              # (N, ny)
        density = weighted_gy.T @ gx               # (ny, nx)

        if self.normalise:
            total = density.sum() * self._dx * self._dy
            if total > 0:
                density = density / total
        return density

    # ── convenience ──────────────────────────────────────────────────────────

    @property
    def grid_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the (xs, ys) cell-centre coordinate arrays."""
        return self._xs, self._ys

    def to_dict(self) -> dict:
        """Serialise the kernel configuration (for API responses)."""
        return {
            "name": self.name,
            "pitch_length": self.pitch_length,
            "pitch_width": self.pitch_width,
            "nx": self.nx,
            "ny": self.ny,
            "bandwidth": self.bandwidth,
            "normalise": self.normalise,
        }
