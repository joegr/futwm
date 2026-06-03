"""Tests for the greedy IoU multi-object tracker."""

from __future__ import annotations

import pytest

from soccer_vision.detection.base import Detection, ObjectClass
from soccer_vision.tracking.tracker import Tracker, TrackerError


def _det(cls: ObjectClass, x: float, y: float, *, w: float = 4.0, h: float = 8.0) -> Detection:
    return Detection(
        object_class=cls,
        x_min=x - w / 2, y_min=y - h / 2, x_max=x + w / 2, y_max=y + h / 2, score=0.9,
    )


def test_tracker_assigns_stable_ids_across_frames():
    tr = Tracker(iou_threshold=0.1)
    f1 = tr.step([_det(ObjectClass.PLAYER, 100, 100), _det(ObjectClass.BALL, 200, 100)])
    f2 = tr.step([_det(ObjectClass.PLAYER, 101, 100), _det(ObjectClass.BALL, 201, 100)])
    assert set(f1.keys()) == set(f2.keys())


def test_tracker_spawns_new_id_for_new_object():
    tr = Tracker(iou_threshold=0.1)
    f1 = tr.step([_det(ObjectClass.PLAYER, 100, 100)])
    f2 = tr.step([_det(ObjectClass.PLAYER, 100, 100), _det(ObjectClass.PLAYER, 500, 500)])
    assert len(f1) == 1 and len(f2) == 2


def test_tracker_drops_track_after_max_missed():
    tr = Tracker(iou_threshold=0.1, max_missed=2)
    tr.step([_det(ObjectClass.PLAYER, 100, 100)])
    tr.step([])
    tr.step([])
    tracks = tr.step([])  # 3 frames missed → dropped
    assert tracks == {}


def test_tracker_respects_class_separation():
    """A player track must not steal a ball detection at the same location."""
    tr = Tracker(iou_threshold=0.1)
    tr.step([_det(ObjectClass.PLAYER, 100, 100)])
    f2 = tr.step([_det(ObjectClass.BALL, 100, 100)])
    # Ball gets its own new track_id; player's track will not be matched
    classes = sorted(t.object_class for t in f2.values())
    assert ObjectClass.BALL in classes


def test_tracker_rejects_bad_config():
    with pytest.raises(TrackerError):
        Tracker(iou_threshold=0.0)
    with pytest.raises(TrackerError):
        Tracker(iou_threshold=1.5)
    with pytest.raises(TrackerError):
        Tracker(max_missed=0)


def test_tracker_reset_clears_state():
    tr = Tracker()
    tr.step([_det(ObjectClass.PLAYER, 100, 100)])
    tr.reset()
    assert tr.tracks == {}
    # Next step starts fresh with id = 1 again
    f = tr.step([_det(ObjectClass.PLAYER, 100, 100)])
    assert list(f.keys()) == [1]
