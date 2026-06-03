"""
Minimal greedy multi-object tracker — IoU + centroid distance association.

Real production would use SORT / ByteTrack / DeepSORT (with appearance
embeddings). The scaffold here exposes the contract so those can be
slotted in without touching the rest of the pipeline.

The tracker maintains stable IDs across frames so downstream event
emission (passes, possession changes) can reason about *which* player
just touched the ball.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..detection.base import Detection, ObjectClass

TrackId = int


class TrackerError(RuntimeError):
    """Raised when tracker state is inconsistent."""


@dataclass(slots=True)
class Track:
    """A single object track across multiple frames."""

    track_id: TrackId
    object_class: ObjectClass
    last_bbox: tuple[float, float, float, float]   # (x_min, y_min, x_max, y_max)
    last_seen_frame: int
    age: int = 0
    history: list[tuple[int, tuple[float, float]]] = field(default_factory=list)
    attributes: dict[str, object] = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.last_bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


class Tracker:
    """
    Greedy IoU-based tracker. Single-frame Hungarian assignment is not
    required at this scaffold tier; greedy nearest-neighbour is sufficient
    for ≤ 22 players + 1 ball + 4 officials per frame.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU for a detection to associate with an existing track.
    max_missed : int
        Number of consecutive frames a track may be unmatched before it
        is dropped.
    """

    def __init__(self, *, iou_threshold: float = 0.3, max_missed: int = 30) -> None:
        if not 0.0 < iou_threshold <= 1.0:
            raise TrackerError(f"iou_threshold must be in (0, 1], got {iou_threshold}")
        if max_missed < 1:
            raise TrackerError(f"max_missed must be >= 1, got {max_missed}")
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self._tracks: dict[TrackId, Track] = {}
        self._next_id: TrackId = 1
        self._frame_idx: int = -1

    # ── primary API ──────────────────────────────────────────────────────────

    def step(self, detections: list[Detection]) -> dict[TrackId, Track]:
        """
        Advance the tracker by one frame, ingesting the latest detections.

        Returns the active track table keyed by ``track_id``.
        """
        self._frame_idx += 1
        unmatched_dets = list(range(len(detections)))
        matched_track_ids: set[TrackId] = set()

        # Greedy assignment: for each existing track, pick the detection
        # with highest IoU above threshold (and same object_class).
        for tid, track in list(self._tracks.items()):
            best_i = -1
            best_iou = self.iou_threshold
            for i in unmatched_dets:
                det = detections[i]
                if det.object_class != track.object_class:
                    continue
                iou = _iou(track.last_bbox, (det.x_min, det.y_min, det.x_max, det.y_max))
                if iou >= best_iou:
                    best_iou = iou
                    best_i = i
            if best_i >= 0:
                det = detections[best_i]
                track.last_bbox = (det.x_min, det.y_min, det.x_max, det.y_max)
                track.last_seen_frame = self._frame_idx
                track.age += 1
                track.history.append((self._frame_idx, det.center))
                unmatched_dets.remove(best_i)
                matched_track_ids.add(tid)

        # Spawn new tracks for unmatched detections.
        for i in unmatched_dets:
            det = detections[i]
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = Track(
                track_id=tid,
                object_class=det.object_class,
                last_bbox=(det.x_min, det.y_min, det.x_max, det.y_max),
                last_seen_frame=self._frame_idx,
                age=1,
                history=[(self._frame_idx, det.center)],
                attributes=dict(det.attributes),
            )

        # Prune tracks that have not been seen recently.
        stale = [
            tid for tid, t in self._tracks.items()
            if self._frame_idx - t.last_seen_frame > self.max_missed
        ]
        for tid in stale:
            del self._tracks[tid]

        return dict(self._tracks)

    def reset(self) -> None:
        """Drop all tracks and rewind the frame counter."""
        self._tracks.clear()
        self._next_id = 1
        self._frame_idx = -1

    @property
    def tracks(self) -> dict[TrackId, Track]:
        return dict(self._tracks)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Standard axis-aligned IoU between two boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    a_area = max(0.0, (ax1 - ax0)) * max(0.0, (ay1 - ay0))
    b_area = max(0.0, (bx1 - bx0)) * max(0.0, (by1 - by0))
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return inter / union
