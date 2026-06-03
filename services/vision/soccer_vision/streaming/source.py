"""
Frame source ABC + concrete implementations.

Sources are iterables of :class:`Frame` objects. The pipeline never sees
the underlying transport — it just consumes ``(timestamp, pixels)``
tuples — so RTSP, file, in-memory, and WebSocket sources are
interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class FrameSourceError(RuntimeError):
    """Raised when a source cannot deliver a frame."""


@dataclass(slots=True)
class Frame:
    """A single decoded frame with monotonically increasing timestamp."""

    index: int
    timestamp: float          # seconds since session start
    pixels: np.ndarray        # HxWx3 uint8 (RGB)

    def __post_init__(self) -> None:
        if self.pixels.ndim != 3 or self.pixels.shape[-1] != 3:
            raise FrameSourceError(
                f"pixels must be HxWx3, got shape {self.pixels.shape}"
            )
        if self.timestamp < 0:
            raise FrameSourceError(f"timestamp must be >= 0, got {self.timestamp}")


class FrameSource(ABC):
    """Common contract for any frame-producing source."""

    @abstractmethod
    def __iter__(self) -> Iterator[Frame]:
        ...

    def close(self) -> None:                  # pragma: no cover - default noop
        """Release any underlying resources (sockets, decoders, …)."""
        return None


# ── concrete sources ─────────────────────────────────────────────────────────

class InMemorySource(FrameSource):
    """Yield a pre-loaded sequence of numpy frames. Intended for tests."""

    def __init__(self, frames: list[np.ndarray], *, fps: float = 25.0) -> None:
        if fps <= 0:
            raise FrameSourceError(f"fps must be > 0, got {fps}")
        self._frames = frames
        self._dt = 1.0 / float(fps)

    def __iter__(self) -> Iterator[Frame]:
        for i, px in enumerate(self._frames):
            yield Frame(index=i, timestamp=i * self._dt, pixels=px)


class DirectorySource(FrameSource):
    """
    Read frames from a directory of image files in lexicographic order.

    Uses ``Pillow`` if available (PIL is a ``soccer-world-model`` transitive
    dep via ``numpy.imageio``-style ecosystems), else raises a clear error.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        fps: float = 25.0,
        pattern: str = "*.png",
    ) -> None:
        self.path = Path(path)
        if not self.path.is_dir():
            raise FrameSourceError(f"{self.path!s}: not a directory")
        self._files = sorted(self.path.glob(pattern))
        if not self._files:
            raise FrameSourceError(
                f"no files matching {pattern!r} in {self.path!s}"
            )
        if fps <= 0:
            raise FrameSourceError(f"fps must be > 0, got {fps}")
        self._dt = 1.0 / float(fps)

    def __iter__(self) -> Iterator[Frame]:
        try:
            from PIL import Image
        except ImportError as e:                       # pragma: no cover
            raise FrameSourceError(
                "Pillow is required for DirectorySource; "
                "install it with `pip install pillow`"
            ) from e
        for i, fp in enumerate(self._files):
            with Image.open(fp) as im:
                arr = np.asarray(im.convert("RGB"))
            yield Frame(index=i, timestamp=i * self._dt, pixels=arr)


class FFmpegSource(FrameSource):
    """
    RTSP / mp4 / WebRTC source via PyAV (ffmpeg bindings).

    Only available when the optional ``streaming`` extra is installed:
        ``pip install soccer-vision-service[streaming]``

    PyAV is imported lazily so this module remains importable in
    environments without ffmpeg.
    """

    def __init__(self, url: str, *, fps: float | None = None) -> None:
        self.url = url
        self.target_fps = fps

    def __iter__(self) -> Iterator[Frame]:                # pragma: no cover - I/O
        try:
            import av  # type: ignore[import-not-found]
        except ImportError as e:
            raise FrameSourceError(
                "PyAV is required for FFmpegSource; install the streaming extra: "
                "`pip install soccer-vision-service[streaming]`"
            ) from e

        with av.open(self.url) as container:
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "DEFAULT"
            time_base = float(stream.time_base) if stream.time_base else 1.0 / 25.0
            idx = 0
            for packet in container.demux(stream):
                for frame in packet.decode():
                    arr = frame.to_ndarray(format="rgb24")
                    ts = float(frame.pts or idx) * time_base
                    yield Frame(index=idx, timestamp=ts, pixels=arr)
                    idx += 1
