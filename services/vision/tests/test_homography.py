"""Unit tests for the projective transform (DLT) homography kernel."""

from __future__ import annotations

import numpy as np
import pytest

from soccer_vision.kernels.homography import HomographyError, PlanarHomography


def _broadcast_camera_rectangle() -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """
    Four pixel↔pitch correspondences for a typical broadcast camera angle.

    Pixel coords (1920x1080 frame):
      - bottom-left  pitch corner  → pixel (200,  900)
      - bottom-right pitch corner  → pixel (1720, 900)
      - top-left     pitch corner  → pixel (500,  300)
      - top-right    pitch corner  → pixel (1420, 300)

    Pitch coords (105 x 68):
      - (0, 0), (105, 0), (0, 68), (105, 68)
    """
    src = [(200.0, 900.0), (1720.0, 900.0), (500.0, 300.0), (1420.0, 300.0)]
    dst = [(0.0, 0.0), (105.0, 0.0), (0.0, 68.0), (105.0, 68.0)]
    return src, dst


def test_homography_fit_exact_for_four_points():
    src, dst = _broadcast_camera_rectangle()
    h = PlanarHomography.fit(src, dst)
    proj = h.apply(src)
    np.testing.assert_allclose(proj, np.array(dst), atol=1e-6)


def test_homography_round_trip():
    src, dst = _broadcast_camera_rectangle()
    h = PlanarHomography.fit(src, dst)
    h_inv = h.inverse()
    # Forward then inverse must return to source within numerical tolerance
    src_arr = np.array(src)
    round_trip = h_inv.apply(h.apply(src_arr))
    np.testing.assert_allclose(round_trip, src_arr, atol=1e-6)


def test_homography_rejects_fewer_than_four_points():
    with pytest.raises(HomographyError):
        PlanarHomography.fit([(0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (0, 1)])


def test_homography_rejects_mismatched_shapes():
    with pytest.raises(HomographyError):
        PlanarHomography.fit(
            [(0, 0), (1, 0), (0, 1), (1, 1)],
            [(0, 0), (1, 0), (0, 1)],
        )


def test_homography_rejects_collinear_points():
    # All four src points on the y=0 line — degenerate for a 2D->2D projection
    with pytest.raises(HomographyError):
        PlanarHomography.fit(
            [(0, 0), (1, 0), (2, 0), (3, 0)],
            [(0, 0), (1, 1), (2, 2), (3, 3)],
        )


def test_homography_reprojection_error_small():
    src, dst = _broadcast_camera_rectangle()
    h = PlanarHomography.fit(src, dst)
    rms = h.reprojection_error(src, dst)
    assert rms < 1e-6


def test_homography_serialisation_roundtrip():
    src, dst = _broadcast_camera_rectangle()
    h = PlanarHomography.fit(src, dst)
    blob = h.to_dict()
    restored = PlanarHomography.from_dict(blob)
    np.testing.assert_allclose(restored.matrix, h.matrix, atol=1e-12)


def test_homography_handles_overdetermined_system():
    # 8 correspondences -> least-squares fit, should still be very accurate
    src, dst = _broadcast_camera_rectangle()
    # Add four midpoints
    extra_src = [(960.0, 600.0), (300.0, 600.0), (1600.0, 600.0), (960.0, 800.0)]
    extra_dst = [(52.5, 34.0), (10.0, 34.0), (95.0, 34.0), (52.5, 10.0)]
    # Generate the exact projections of extra_src using the 4-point homography
    h4 = PlanarHomography.fit(src, dst)
    exact_extra = h4.apply(extra_src).tolist()
    h8 = PlanarHomography.fit(src + extra_src, dst + exact_extra)
    assert h8.reprojection_error(src, dst) < 1e-5
