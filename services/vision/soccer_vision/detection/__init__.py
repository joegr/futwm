"""
Object detection scaffolds.

The package exposes the :class:`Detector` ABC and the :class:`Detection`
data model. Concrete implementations either rely on heavy ML deps
(OpenCV, PyTorch, Ultralytics) — which are **lazily imported** inside the
constructor — or pure-numpy stubs that exist mainly so the pipeline can
run end-to-end without the ML stack present.
"""

from __future__ import annotations

from .base import Detection, Detector, DetectorError, ObjectClass
from .opencv_stub import OpenCVBallStub

__all__ = [
    "Detection",
    "Detector",
    "DetectorError",
    "ObjectClass",
    "OpenCVBallStub",
]
