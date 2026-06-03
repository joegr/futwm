"""
End-to-end vision pipeline orchestration.

A :class:`VisionPipeline` composes one detector, one tracker, one
homography, and one event emitter, then exposes two operating modes:

* :meth:`process_frame` — synchronous, one frame in, list of events out
* :meth:`process_stream` — iterator-driven (any :class:`FrameSource`)
  producing a single flat event stream
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from .detection.base import Detector
from .events.emitter import EmitterConfig, EventEmitter
from .kernels.homography import PlanarHomography
from .streaming.source import Frame, FrameSource
from .tracking.tracker import Tracker


@dataclass(slots=True)
class FrameResult:
    """Per-frame pipeline output."""

    frame_index: int
    timestamp: float
    n_detections: int
    n_tracks: int
    events: list[BaseModel]


class VisionPipeline:
    """
    Configurable detector → tracker → emitter pipeline.

    The constructor takes already-instantiated components so callers can
    inject mocks or swap implementations.
    """

    def __init__(
        self,
        *,
        detector: Detector,
        homography: PlanarHomography,
        tracker: Tracker | None = None,
        emitter_config: EmitterConfig | None = None,
    ) -> None:
        self.detector = detector
        self.homography = homography
        self.tracker = tracker or Tracker()
        self.emitter = EventEmitter(homography=homography, config=emitter_config)

    # ── per-frame ────────────────────────────────────────────────────────────

    def process_frame(self, frame: Frame | np.ndarray, *, timestamp: float | None = None) -> FrameResult:
        """
        Run one frame through detect → track → emit.

        Accepts either a :class:`Frame` (preferred — carries its own
        timestamp / index) or a raw numpy array (timestamp required).
        """
        if isinstance(frame, Frame):
            pixels = frame.pixels
            ts = frame.timestamp
            idx = frame.index
        else:
            if timestamp is None:
                raise ValueError("timestamp required when passing a raw numpy frame")
            pixels = frame
            ts = float(timestamp)
            idx = -1

        detections = self.detector.detect(pixels)
        tracks = self.tracker.step(detections)
        events = self.emitter.emit(tracks, timestamp_s=ts)
        return FrameResult(
            frame_index=idx,
            timestamp=ts,
            n_detections=len(detections),
            n_tracks=len(tracks),
            events=events,
        )

    # ── streaming ────────────────────────────────────────────────────────────

    def process_stream(self, source: FrameSource | Iterable[Frame]) -> Iterator[FrameResult]:
        """
        Iterate over a source, yielding one :class:`FrameResult` per frame.

        Caller is responsible for closing the underlying source if it
        owns network / file resources.
        """
        for frame in source:
            yield self.process_frame(frame)

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset tracker and emitter state (e.g. at half-time)."""
        self.tracker.reset()
        self.emitter.reset()
