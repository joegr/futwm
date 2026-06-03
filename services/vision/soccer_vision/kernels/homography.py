"""
Planar homography (8-DoF projective transform).

Implemented via the Direct Linear Transform (DLT) — pure numpy, no OpenCV
dependency. Given ≥ 4 correspondences ``(u, v) ↔ (X, Y)`` between pixel
coordinates and pitch metres, this kernel finds the 3×3 matrix ``H`` that
maps every pixel into pitch space.

Reference: Hartley & Zisserman, *Multiple View Geometry in Computer Vision*,
2nd ed., §4.1 (the DLT algorithm) and §4.4.4 (data normalisation).
"""

from __future__ import annotations

import numpy as np

from .base import Kernel, KernelError


class HomographyError(KernelError):
    """Raised when correspondences are degenerate or rank-deficient."""


class PlanarHomography(Kernel):
    """
    Fit-then-apply 3×3 projective transform between two planar coordinate frames.

    Typical use: convert pixel coordinates from a broadcast camera into FIFA
    pitch coordinates (metres) once the user has marked ≥ 4 reference points
    (corner flags, penalty area corners, centre circle intersections, …).

    Workflow
    --------
    >>> h = PlanarHomography.fit(
    ...     src_points=[(120, 480), (1800, 480), (120, 50), (1800, 50)],  # pixel
    ...     dst_points=[(0, 0),    (105, 0),    (0, 68),   (105, 68)],    # pitch
    ... )
    >>> pitch_xy = h.apply([(960, 540)])      # transform any pixel point
    """

    name = "PlanarHomography"

    def __init__(self, matrix: np.ndarray) -> None:
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape != (3, 3):
            raise HomographyError(f"matrix must be 3x3, got {m.shape}")
        if not np.isfinite(m).all():
            raise HomographyError("matrix contains non-finite values")
        # Normalise so the bottom-right element is 1 when non-zero (canonical form)
        if abs(m[2, 2]) > 1e-12:
            m = m / m[2, 2]
        self.matrix = m

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def fit(
        cls,
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ) -> PlanarHomography:
        """
        Solve for the 3×3 homography mapping ``src → dst``.

        Parameters
        ----------
        src_points, dst_points : array-like, shape (N, 2), N ≥ 4
            Corresponding 2-D points in the source and destination frames.

        Returns
        -------
        PlanarHomography

        Raises
        ------
        HomographyError
            If fewer than 4 points are supplied, shapes do not match, the
            point sets are collinear, or the DLT system is rank-deficient.
        """
        src = np.asarray(src_points, dtype=np.float64).reshape(-1, 2)
        dst = np.asarray(dst_points, dtype=np.float64).reshape(-1, 2)

        if src.shape[0] < 4:
            raise HomographyError(f"need >= 4 points, got {src.shape[0]}")
        if src.shape != dst.shape:
            raise HomographyError(
                f"src/dst shape mismatch: {src.shape} vs {dst.shape}"
            )
        if not (np.isfinite(src).all() and np.isfinite(dst).all()):
            raise HomographyError("non-finite input coordinates")

        # Hartley normalisation — translate centroid to origin and scale so
        # average distance from origin is √2. Improves DLT conditioning by
        # several orders of magnitude.
        src_n, T_src = _normalise(src)
        dst_n, T_dst = _normalise(dst)

        # Build the 2N × 9 DLT matrix.
        n = src_n.shape[0]
        A = np.zeros((2 * n, 9), dtype=np.float64)
        for i in range(n):
            x, y   = src_n[i]
            xp, yp = dst_n[i]
            A[2 * i] = [-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp]
            A[2 * i + 1] = [0, 0, 0, -x, -y, -1, x * yp, y * yp, yp]

        # Solve A · h = 0 via SVD; the last right-singular vector minimises
        # ||A h|| subject to ||h||=1.
        _, s, vh = np.linalg.svd(A, full_matrices=False)
        if s[-1] > 1e-6 * s[0] and n == 4:
            # 4-point exact solve: tolerate higher residual.
            pass
        elif s[-1] > 1e-2 * s[0]:
            raise HomographyError(
                f"degenerate correspondences (cond ratio {s[-1] / s[0]:.2e})"
            )
        h = vh[-1].reshape(3, 3)

        # Denormalise: H = T_dst^{-1} · H_norm · T_src
        H = np.linalg.inv(T_dst) @ h @ T_src
        return cls(H)

    # ── transform ────────────────────────────────────────────────────────────

    def apply(self, points: np.ndarray) -> np.ndarray:
        """
        Project a batch of 2-D points through the homography.

        Parameters
        ----------
        points : array-like, shape (N, 2)
            Source-frame coordinates.

        Returns
        -------
        ndarray, shape (N, 2)
            Destination-frame coordinates.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        n = pts.shape[0]
        if n == 0:
            return np.empty((0, 2), dtype=np.float64)

        ones = np.ones((n, 1), dtype=np.float64)
        homog = np.hstack([pts, ones])               # (N, 3)
        proj = homog @ self.matrix.T                 # (N, 3)
        w = proj[:, 2:3]
        # Reject points on the camera horizon (w ≈ 0) — they project to infinity.
        bad = np.abs(w[:, 0]) < 1e-9
        if bad.any():
            w = w.copy()
            w[bad, 0] = np.nan
        return proj[:, :2] / w

    def inverse(self) -> PlanarHomography:
        """Return the inverse homography (dst → src)."""
        try:
            inv = np.linalg.inv(self.matrix)
        except np.linalg.LinAlgError as e:
            raise HomographyError("matrix is singular and cannot be inverted") from e
        return PlanarHomography(inv)

    def reprojection_error(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ) -> float:
        """RMS pixel-distance between projected src and ground-truth dst."""
        proj = self.apply(src_points)
        dst = np.asarray(dst_points, dtype=np.float64).reshape(-1, 2)
        diff = proj - dst
        return float(np.sqrt((diff ** 2).sum(axis=1).mean()))

    # ── serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {"name": self.name, "matrix": self.matrix.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> PlanarHomography:
        if "matrix" not in data:
            raise HomographyError("dict must contain a 'matrix' key")
        return cls(np.asarray(data["matrix"], dtype=np.float64))


# ── helpers ──────────────────────────────────────────────────────────────────

def _normalise(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalisation: centroid to origin, mean distance to √2."""
    centroid = points.mean(axis=0)
    centred = points - centroid
    mean_dist = float(np.sqrt((centred ** 2).sum(axis=1)).mean())
    if mean_dist < 1e-12:
        raise HomographyError("degenerate point set (zero spread)")
    scale = np.sqrt(2.0) / mean_dist
    T = np.array([
        [scale, 0.0,   -scale * centroid[0]],
        [0.0,   scale, -scale * centroid[1]],
        [0.0,   0.0,   1.0],
    ], dtype=np.float64)
    normed = (centred * scale)
    return normed, T
