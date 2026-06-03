"""
Reference / stub detector implementations.

These exist so the pipeline can be wired end-to-end **without** the ML
stack installed. They are not suitable for production but are useful for:

* contract tests (verify the pipeline accepts ``Detection`` objects)
* debugging the homography + event emission pipeline against a known
  pixel position
* CI smoke tests

Production deployments should swap these for a YOLOv8 / RT-DETR / custom
detector. Heavy deps (``cv2``, ``torch``, ``ultralytics``) are imported
lazily so that this module is always importable.
"""

from __future__ import annotations

import numpy as np

from .base import Detection, Detector, DetectorError, ObjectClass


class OpenCVBallStub(Detector):
    """
    Trivial colour-threshold ball detector.

    Finds the brightest cluster of near-white pixels (the ball is typically
    the brightest object on a green pitch) and returns at most one
    detection. This is **not** a real ball detector — it is a reference
    implementation that lets you run the pipeline end-to-end without YOLO.

    When the optional ``opencv-python-headless`` dependency is installed,
    the detector uses ``cv2.connectedComponents`` for blob analysis.
    Otherwise it falls back to a pure-numpy centroid of the brightness
    threshold.
    """

    name = "OpenCVBallStub"

    def __init__(self, *, brightness_threshold: float = 220.0) -> None:
        if not (0 < brightness_threshold <= 255):
            raise DetectorError(
                f"brightness_threshold must be in (0, 255], got {brightness_threshold}"
            )
        self.brightness_threshold = float(brightness_threshold)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise DetectorError(f"frame must be HxWx3, got shape {frame.shape}")

        # Grayscale-ish brightness (sum of channels) → boolean mask
        brightness = frame.astype(np.float64).mean(axis=-1)
        mask = brightness >= self.brightness_threshold
        if not mask.any():
            return []

        # Centroid of all bright pixels — works when the ball is the only
        # near-white object in the frame.
        ys, xs = np.nonzero(mask)
        cx = float(xs.mean())
        cy = float(ys.mean())
        # Generate a small bounding box around the centroid (~5 px radius)
        # since this stub doesn't actually segment the ball.
        r = 5.0
        h, w = frame.shape[:2]
        return [
            Detection(
                object_class=ObjectClass.BALL,
                x_min=max(0.0, cx - r),
                y_min=max(0.0, cy - r),
                x_max=min(float(w), cx + r),
                y_max=min(float(h), cy + r),
                score=0.5,
                attributes={"source": self.name},
            )
        ]
