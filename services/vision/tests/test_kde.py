"""Unit tests for the 2-D Gaussian KDE kernel."""

from __future__ import annotations

import numpy as np
import pytest

from soccer_vision.kernels.base import KernelError
from soccer_vision.kernels.kde import GaussianKDE2D


def test_kde_empty_input_returns_zero_field():
    k = GaussianKDE2D()
    out = k.apply(np.empty((0, 2)))
    assert out.shape == (k.ny, k.nx)
    assert out.sum() == 0.0


def test_kde_integrates_to_one_when_normalised():
    k = GaussianKDE2D(nx=210, ny=136, bandwidth=2.0, normalise=True)  # 0.5m cells
    out = k.apply([(52.5, 34.0)])    # single touch at pitch centre
    integral = out.sum() * (k.pitch_length / (k.nx - 1)) * (k.pitch_width / (k.ny - 1))
    assert integral == pytest.approx(1.0, abs=1e-2)


def test_kde_peaks_near_input_point():
    k = GaussianKDE2D(bandwidth=2.0)
    out = k.apply([(52.5, 34.0)])     # centre of pitch
    iy, ix = np.unravel_index(out.argmax(), out.shape)
    # Peak cell centre should be ~(52.5, 34.0)
    assert abs(k._xs[ix] - 52.5) < 1.0
    assert abs(k._ys[iy] - 34.0) < 1.0


def test_kde_weights_affect_relative_mass():
    k = GaussianKDE2D(bandwidth=2.0, normalise=False)
    out_uniform  = k.apply([(20.0, 34.0), (80.0, 34.0)])
    out_weighted = k.apply([(20.0, 34.0), (80.0, 34.0)], weights=[1.0, 3.0])
    # Right hot-spot weighted 3× should dominate
    left_mass_w  = out_weighted[:, :k.nx // 2].sum()
    right_mass_w = out_weighted[:, k.nx // 2:].sum()
    assert right_mass_w > 2.5 * left_mass_w
    # Uniform weights produce roughly equal masses
    left_mass_u  = out_uniform[:, :k.nx // 2].sum()
    right_mass_u = out_uniform[:, k.nx // 2:].sum()
    assert abs(left_mass_u - right_mass_u) / max(left_mass_u, right_mass_u) < 0.05


def test_kde_clips_points_outside_pitch():
    k = GaussianKDE2D(bandwidth=2.0, normalise=False)
    # Far outside the pitch — should be clipped to the corner
    out = k.apply([(-50.0, -50.0)])
    iy, ix = np.unravel_index(out.argmax(), out.shape)
    assert (ix, iy) == (0, 0)


def test_kde_rejects_bad_inputs():
    with pytest.raises(KernelError):
        GaussianKDE2D(bandwidth=0.0)
    with pytest.raises(KernelError):
        GaussianKDE2D(nx=1)
    with pytest.raises(KernelError):
        GaussianKDE2D(pitch_length=-1)


def test_kde_weights_length_mismatch():
    k = GaussianKDE2D()
    with pytest.raises(KernelError):
        k.apply([(1.0, 1.0), (2.0, 2.0)], weights=[1.0])


def test_kde_grid_xy_consistent():
    k = GaussianKDE2D(nx=10, ny=8)
    xs, ys = k.grid_xy
    assert xs.shape == (10,)
    assert ys.shape == (8,)
    assert xs[0] == 0.0 and xs[-1] == pytest.approx(k.pitch_length)
    assert ys[0] == 0.0 and ys[-1] == pytest.approx(k.pitch_width)
