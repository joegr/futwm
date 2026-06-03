"""
Convert vision detections / tracks into canonical ``soccer_model`` events.

This is the *single point of contact* with the core library's schema —
keeping it isolated here means any change in detection or tracking code
cannot leak into the published event format, and any change in the
schema requires a deliberate update to this module.

Current scope (scaffold tier):

* :meth:`EventEmitter.touch_from_track` — emits a ``TouchEvent`` whenever
  the ball track lies within ``contact_radius`` metres of a player track.

Future extensions (intentionally not implemented yet, but the structure
supports them):

* possession-change detection → pass / dribble / tackle emissions
* foot-position vs ball position → ``foot`` field inference
* shot detection from ball trajectory + goal-line crossings
* set-piece detection from referee whistle + ball-stopped heuristic
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pydantic import BaseModel

# Lazy-imported below to avoid a hard dependency at module-import time;
# this makes the vision package usable in environments that haven't yet
# installed soccer-world-model (e.g. very early scaffolding stages).
try:
    from soccer_model import Foot, TouchEvent  # type: ignore[attr-defined]
    _SCHEMA_AVAILABLE = True
except ImportError:                              # pragma: no cover
    _SCHEMA_AVAILABLE = False

from ..detection.base import ObjectClass
from ..kernels.homography import PlanarHomography
from ..tracking.tracker import Track, TrackId


class EmitterError(RuntimeError):
    """Raised when the emitter is mis-configured or asked the impossible."""


@dataclass(slots=True)
class EmitterConfig:
    """Per-session configuration for the event emitter."""

    home_team: str = "home"
    away_team: str = "away"
    # Mapping ``track_id -> (team_label, player_label)``. Populated by the
    # downstream player-identification model; tracks without an entry are
    # emitted with placeholder labels.
    player_index: dict[int, tuple[str, str]] = field(default_factory=dict)
    # Metres: max distance between ball foot-point (pitch space) and a
    # player foot-point that still counts as a "touch".
    contact_radius_m: float = 1.5
    # Pitch dimensions for clamping.
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0

    def __post_init__(self) -> None:
        if self.contact_radius_m <= 0:
            raise EmitterError("contact_radius_m must be > 0")
        if self.pitch_length_m <= 0 or self.pitch_width_m <= 0:
            raise EmitterError("pitch dimensions must be positive")


class _GenericTouch(BaseModel):
    """Minimal stand-in used when ``soccer_model`` is not importable."""
    timestamp: float
    team: str
    player: str
    x: float
    y: float
    event_type: str = "touch"


class EventEmitter:
    """
    Stateful emitter that turns each frame's track table into zero or more
    schema-validated events.

    Parameters
    ----------
    homography : PlanarHomography
        Calibrated pixel → pitch projective transform.
    config : EmitterConfig
        Team labels, player-identity map, contact radius.
    """

    def __init__(
        self,
        homography: PlanarHomography,
        config: EmitterConfig | None = None,
    ) -> None:
        self.homography = homography
        self.config = config or EmitterConfig()
        # Track the last player to have touched the ball so we can
        # de-duplicate identical contacts on consecutive frames.
        self._last_touch_track: TrackId | None = None

    # ── primary API ──────────────────────────────────────────────────────────

    def emit(
        self,
        tracks: dict[TrackId, Track],
        *,
        timestamp_s: float,
    ) -> list[BaseModel]:
        """
        Produce schema-validated events from the current frame's tracks.

        Returns a list (possibly empty) of ``soccer_model.TouchEvent``
        instances, or :class:`_GenericTouch` stand-ins when ``soccer_model``
        is not importable.
        """
        if timestamp_s < 0:
            raise EmitterError(f"timestamp_s must be >= 0, got {timestamp_s}")

        ball = self._select_ball(tracks)
        if ball is None:
            return []

        # Project the ball's foot point (bottom-centre of bbox) into pitch coords.
        ball_pitch = self._project_foot(ball)
        if not _is_finite_xy(ball_pitch):
            return []

        # Find the nearest player whose foot-point is within contact_radius.
        nearest_id, nearest_dist = self._nearest_player(tracks, ball_pitch)
        if nearest_id is None or nearest_dist > self.config.contact_radius_m:
            self._last_touch_track = None
            return []

        # Suppress identical-player touches in consecutive frames (the ball
        # remains attached during dribbles). A real implementation would
        # use velocity / change-in-control; this is the cheapest heuristic.
        if nearest_id == self._last_touch_track:
            return []
        self._last_touch_track = nearest_id

        team, player = self._label_for_track(nearest_id)
        return [self._build_touch(timestamp_s, team, player, ball_pitch)]

    def reset(self) -> None:
        """Forget last-touch state (e.g. on session restart)."""
        self._last_touch_track = None

    # ── helpers ──────────────────────────────────────────────────────────────

    def _select_ball(self, tracks: dict[TrackId, Track]) -> Track | None:
        balls = [t for t in tracks.values() if t.object_class == ObjectClass.BALL]
        if not balls:
            return None
        # If multiple ball tracks (false positives), pick the youngest /
        # most-recently-seen as the most likely real ball.
        balls.sort(key=lambda t: -t.last_seen_frame)
        return balls[0]

    def _project_foot(self, track: Track) -> tuple[float, float]:
        x0, _y0, x1, y1 = track.last_bbox
        # foot-point = bottom-centre of the bbox (in pixels)
        px = ((x0 + x1) / 2.0, y1)
        proj = self.homography.apply([px])[0]
        # Clamp into pitch bounds (homography can produce slightly
        # out-of-bounds coords for points right on the touchline).
        x = max(0.0, min(float(proj[0]), self.config.pitch_length_m))
        y = max(0.0, min(float(proj[1]), self.config.pitch_width_m))
        return x, y

    def _nearest_player(
        self,
        tracks: dict[TrackId, Track],
        ball_pitch: tuple[float, float],
    ) -> tuple[TrackId | None, float]:
        best_id: TrackId | None = None
        best_dist = math.inf
        for tid, t in tracks.items():
            if t.object_class not in (ObjectClass.PLAYER, ObjectClass.GOALKEEPER):
                continue
            p = self._project_foot(t)
            d = math.hypot(p[0] - ball_pitch[0], p[1] - ball_pitch[1])
            if d < best_dist:
                best_dist = d
                best_id = tid
        return best_id, best_dist

    def _label_for_track(self, track_id: TrackId) -> tuple[str, str]:
        if track_id in self.config.player_index:
            return self.config.player_index[track_id]
        # Fall back to placeholder labels — caller can populate
        # ``player_index`` after a re-identification step.
        return (self.config.home_team, f"track_{track_id}")

    def _build_touch(
        self,
        ts: float,
        team: str,
        player: str,
        pitch_xy: tuple[float, float],
    ) -> BaseModel:
        if not _SCHEMA_AVAILABLE:                  # pragma: no cover
            return _GenericTouch(
                timestamp=ts, team=team, player=player,
                x=pitch_xy[0], y=pitch_xy[1],
            )
        return TouchEvent(                         # type: ignore[name-defined]
            timestamp=ts,
            team=team,
            player=player,
            x=pitch_xy[0],
            y=pitch_xy[1],
            foot=Foot.RIGHT,                       # type: ignore[name-defined]
        )


def _is_finite_xy(xy: tuple[float, float]) -> bool:
    return math.isfinite(xy[0]) and math.isfinite(xy[1])
