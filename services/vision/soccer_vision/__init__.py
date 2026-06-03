"""
soccer-vision-service
=====================

Computer-vision microservice that ingests video / frames of a soccer match,
runs a configurable pipeline of CV + spatial kernels, and emits canonical
events conforming to the ``soccer_model.MatchEventStream`` schema.

The package is deliberately structured so that:

* the **HTTP / WebSocket API** can boot without any ML dependencies
* pure-numpy **kernels** (Gaussian, KDE, homography, pitch mask) are
  always available
* heavy ML stacks (OpenCV, PyTorch, YOLO) are **lazily imported** inside
  the corresponding detector / source classes, and are only required when
  those classes are actually instantiated.

See ``README.md`` for the layout and contract.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Surface the most useful entry points at the top level.
from .events.emitter import EventEmitter, EmitterConfig
from .kernels.gaussian import GaussianKernel1D, GaussianKernel2D
from .kernels.homography import HomographyError, PlanarHomography
from .kernels.kde import GaussianKDE2D
from .kernels.pitch_mask import HSVPitchMask, PitchMaskConfig
from .pipeline import VisionPipeline

__all__ = [
    "__version__",
    # kernels
    "GaussianKernel1D",
    "GaussianKernel2D",
    "GaussianKDE2D",
    "HSVPitchMask",
    "PitchMaskConfig",
    "PlanarHomography",
    "HomographyError",
    # events / pipeline
    "EventEmitter",
    "EmitterConfig",
    "VisionPipeline",
]
