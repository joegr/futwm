"""Tests for the vision → soccer_model event emitter."""

from __future__ import annotations

import numpy as np
import pytest

from soccer_vision.detection.base import Detection, ObjectClass
from soccer_vision.events.emitter import EmitterConfig, EmitterError, EventEmitter
from soccer_vision.kernels.homography import PlanarHomography
from soccer_vision.tracking.tracker import Tracker


# A perfectly-flat homography (pixels == pitch metres) for unit tests so we
# can reason about coordinates directly.
@pytest.fixture
def identity_homography() -> PlanarHomography:
    return PlanarHomography(np.eye(3))


@pytest.fixture
def emitter(identity_homography: PlanarHomography) -> EventEmitter:
    cfg = EmitterConfig(
        home_team="Arsenal", away_team="Chelsea",
        contact_radius_m=2.0,
        pitch_length_m=105.0, pitch_width_m=68.0,
    )
    return EventEmitter(homography=identity_homography, config=cfg)


def _track_ball_and_player(ball_xy: tuple[float, float], player_xy: tuple[float, float]) -> dict:
    """
    Build a tracker state with one ball and one player at the supplied
    pitch-metre coordinates. Box width = 2m, height = 4m (player), 0.4m for ball.
    """
    tracker = Tracker(iou_threshold=0.01, max_missed=30)
    bx, by = ball_xy
    px, py = player_xy
    detections = [
        Detection(
            object_class=ObjectClass.BALL,
            x_min=bx - 0.2, y_min=by - 0.2, x_max=bx + 0.2, y_max=by + 0.2,
            score=0.9,
        ),
        Detection(
            object_class=ObjectClass.PLAYER,
            x_min=px - 1.0, y_min=py - 2.0, x_max=px + 1.0, y_max=py,
            score=0.9,
        ),
    ]
    tracks = tracker.step(detections)
    return tracks


def test_emitter_emits_touch_when_player_within_contact_radius(emitter):
    tracks = _track_ball_and_player(ball_xy=(50.0, 34.0), player_xy=(50.0, 34.0))
    events = emitter.emit(tracks, timestamp_s=10.0)
    assert len(events) == 1
    ev = events[0]
    # Schema-validated TouchEvent fields
    assert ev.team == "Arsenal"
    assert ev.timestamp == 10.0
    assert abs(ev.x - 50.0) < 0.1
    # y coord = bottom of player bbox = 34.0 (foot point)
    assert abs(ev.y - 34.0) < 0.1


def test_emitter_no_touch_when_player_too_far(emitter):
    # Ball at (50,34), player at (80,34) — 30m apart, far outside contact radius
    tracks = _track_ball_and_player(ball_xy=(50.0, 34.0), player_xy=(80.0, 34.0))
    events = emitter.emit(tracks, timestamp_s=10.0)
    assert events == []


def test_emitter_deduplicates_repeated_touches(emitter):
    """Same player touching the ball in consecutive frames yields one event, not two."""
    tracks1 = _track_ball_and_player((50.0, 34.0), (50.0, 34.0))
    tracks2 = _track_ball_and_player((50.5, 34.0), (50.5, 34.0))  # same player, slight move

    ev1 = emitter.emit(tracks1, timestamp_s=1.0)
    ev2 = emitter.emit(tracks2, timestamp_s=1.1)
    # The emitter's de-dup is keyed on track_id, not pitch xy. Each call
    # above goes through a *fresh* tracker, so track_id == 1 in both cases.
    # The second call must suppress the duplicate.
    assert len(ev1) == 1
    assert ev2 == []


def test_emitter_returns_no_events_when_ball_missing(emitter):
    tracker = Tracker(iou_threshold=0.01)
    detections = [Detection(
        object_class=ObjectClass.PLAYER,
        x_min=10.0, y_min=10.0, x_max=12.0, y_max=14.0, score=0.9,
    )]
    tracks = tracker.step(detections)
    assert emitter.emit(tracks, timestamp_s=0.0) == []


def test_emitter_uses_player_index_when_provided(identity_homography):
    cfg = EmitterConfig(player_index={2: ("Arsenal", "Saka")})
    em = EventEmitter(homography=identity_homography, config=cfg)
    # The tracker assigns track ids 1 (ball) and 2 (player) given input order
    tracks = _track_ball_and_player((40.0, 34.0), (40.0, 34.0))
    events = em.emit(tracks, timestamp_s=0.0)
    assert len(events) == 1
    assert events[0].player == "Saka"
    assert events[0].team == "Arsenal"


def test_emitter_rejects_negative_timestamp(emitter):
    with pytest.raises(EmitterError):
        emitter.emit({}, timestamp_s=-1.0)


def test_emitter_reset_clears_dedup_state(emitter):
    tracks1 = _track_ball_and_player((50.0, 34.0), (50.0, 34.0))
    tracks2 = _track_ball_and_player((50.5, 34.0), (50.5, 34.0))
    assert len(emitter.emit(tracks1, timestamp_s=1.0)) == 1
    assert emitter.emit(tracks2, timestamp_s=1.1) == []
    emitter.reset()
    # After reset, a new touch from the same track_id should fire again
    tracks3 = _track_ball_and_player((50.0, 34.0), (50.0, 34.0))
    assert len(emitter.emit(tracks3, timestamp_s=2.0)) == 1
