"""Unit tests for the pure-numpy kernels."""

from __future__ import annotations

import numpy as np
import pytest

from soccer_vision.kernels.base import KernelError
from soccer_vision.kernels.gaussian import GaussianKernel1D, GaussianKernel2D


# ── 1-D ──────────────────────────────────────────────────────────────────────

def test_gaussian1d_preserves_constant_signal():
    g = GaussianKernel1D(sigma=2.0)
    sig = np.ones(64)
    out = g.apply(sig)
    np.testing.assert_allclose(out, np.ones_like(out), atol=1e-12)


def test_gaussian1d_reduces_noise_variance():
    rng = np.random.default_rng(seed=7)
    noise = rng.standard_normal(2048)
    out = GaussianKernel1D(sigma=4.0).apply(noise)
    # smoothed variance must be strictly less than raw variance
    assert out.var() < noise.var() * 0.5


def test_gaussian1d_kernel_sums_to_one():
    g = GaussianKernel1D(sigma=3.0)
    assert g.kernel.sum() == pytest.approx(1.0, abs=1e-12)


def test_gaussian1d_rejects_bad_sigma():
    with pytest.raises(KernelError):
        GaussianKernel1D(sigma=0.0)
    with pytest.raises(KernelError):
        GaussianKernel1D(sigma=-1.0)


# ── 2-D ──────────────────────────────────────────────────────────────────────

def test_gaussian2d_preserves_constant_image():
    img = np.full((32, 48), 0.7)
    out = GaussianKernel2D(sigma=1.5).apply(img)
    np.testing.assert_allclose(out, img, atol=1e-12)


def test_gaussian2d_smooths_impulse_into_blob():
    img = np.zeros((33, 33))
    img[16, 16] = 1.0
    out = GaussianKernel2D(sigma=2.0).apply(img)
    # mass is preserved (reflect padding) and peaked at the impulse
    assert out.sum() == pytest.approx(1.0, abs=1e-9)
    assert out[16, 16] == out.max()


def test_gaussian2d_supports_3channel():
    img = np.zeros((20, 20, 3))
    img[10, 10] = [1.0, 2.0, 3.0]
    out = GaussianKernel2D(sigma=1.0).apply(img)
    assert out.shape == img.shape
    # channels are smoothed independently and each preserves its mass
    np.testing.assert_allclose(out.sum(axis=(0, 1)), [1.0, 2.0, 3.0], atol=1e-9)


def test_gaussian2d_rejects_1d_input():
    with pytest.raises(KernelError):
        GaussianKernel2D(sigma=1.0).apply(np.zeros(10))
