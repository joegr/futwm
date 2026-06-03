"""
Frame source / sink scaffolds.

The package defines the :class:`FrameSource` ABC and shipping
implementations:

* ``InMemorySource`` — emit a list of pre-loaded numpy frames (testing)
* ``DirectorySource`` — read frames from a directory of image files
* ``FFmpegSource``    — stub that lazy-imports ``av`` (PyAV) when the
                       ``streaming`` extra is installed

Real RTSP / mp4 / WebSocket transports live behind :class:`FrameSource`
so the pipeline never needs to know how frames arrived.
"""

from __future__ import annotations

from .source import (
    DirectorySource,
    FFmpegSource,
    Frame,
    FrameSource,
    FrameSourceError,
    InMemorySource,
)

__all__ = [
    "DirectorySource",
    "FFmpegSource",
    "Frame",
    "FrameSource",
    "FrameSourceError",
    "InMemorySource",
]
