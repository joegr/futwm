"""Tests for the formal ontology, schema validation, and interchange round-trips."""

import json
import tempfile
from pathlib import Path

import pytest

import soccer_model as sm
from soccer_model.ontology import SCHEMA_VERSION, EventType, VALID_OUTCOMES
from soccer_model.schema import (
    TouchEvent,
    PassEvent,
    ShotEvent,
    DribbleEvent,
    TackleEvent,
    HeaderEvent,
    FoulEvent,
    GoalkeeperEvent,
    SetPieceEvent,
    MatchEventStream,
    MatchMetadata,
    export_json_schema,
    export_event_schemas,
)
from soccer_model.interchange import (
    read_json,
    write_json,
    read_csv,
    write_csv,
    validate_event_stream,
    validate_single_event,
    to_json_dict,
)


# ── ontology ─────────────────────────────────────────────────────────────────

def test_schema_version_format():
    parts = SCHEMA_VERSION.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_valid_outcomes_covers_all_event_types():
    for et in EventType:
        assert et in VALID_OUTCOMES, f"Missing VALID_OUTCOMES for {et}"
        assert len(VALID_OUTCOMES[et]) > 0


# ── schema construction ──────────────────────────────────────────────────────

def test_touch_event_valid():
    ev = TouchEvent(timestamp=10.0, team="Arsenal", player="Saka", x=50.0, y=34.0)
    assert ev.event_type == EventType.TOUCH
    assert ev.schema_version == SCHEMA_VERSION


def test_shot_event_xg_bounds():
    ev = ShotEvent(
        timestamp=100.0, team="Chelsea", player="Palmer",
        x=90.0, y=34.0, end_x=105.0, end_y=34.0, xg=0.15,
    )
    assert 0.0 <= ev.xg <= 1.0


def test_shot_event_invalid_outcome():
    with pytest.raises(Exception):
        ShotEvent(
            timestamp=100.0, team="Chelsea", player="Palmer",
            x=90.0, y=34.0, end_x=105.0, end_y=34.0,
            outcome="success",  # not valid for shots
        )


def test_pass_event_valid():
    ev = PassEvent(
        timestamp=20.0, team="Arsenal", player="Odegaard",
        x=40.0, y=30.0, end_x=60.0, end_y=40.0, to_player="Saka",
    )
    assert ev.pass_type.value == "short"


def test_coordinate_validation_rejects_out_of_bounds():
    with pytest.raises(Exception):
        TouchEvent(timestamp=1.0, team="A", player="B", x=200.0, y=34.0)


# ── JSON Schema export ───────────────────────────────────────────────────────

def test_export_json_schema_structure():
    schema = export_json_schema()
    assert "properties" in schema or "$defs" in schema
    assert "title" in schema


def test_export_event_schemas_all_types():
    schemas = export_event_schemas()
    assert len(schemas) == len(EventType)
    for et in EventType:
        assert et.value in schemas


# ── load_sample ──────────────────────────────────────────────────────────────

def test_load_sample():
    stream = sm.load_sample()
    assert isinstance(stream, MatchEventStream)
    assert len(stream.events) > 0
    assert stream.metadata.schema_version == SCHEMA_VERSION


# ── interchange round-trips ──────────────────────────────────────────────────

def test_json_round_trip(tmp_path):
    stream = sm.load_sample()
    jpath = tmp_path / "test.json"
    write_json(stream, jpath)
    stream2 = read_json(jpath)
    assert len(stream2.events) == len(stream.events)
    assert stream2.metadata.home_team == stream.metadata.home_team


def test_csv_round_trip(tmp_path):
    stream = sm.load_sample()
    cpath = tmp_path / "test.csv"
    write_csv(stream, cpath)
    stream2 = read_csv(cpath)
    assert len(stream2.events) == len(stream.events)


def test_validate_event_stream_valid():
    stream = sm.load_sample()
    data = to_json_dict(stream)
    result = validate_event_stream(data)
    assert result.valid
    assert len(result.errors) == 0


def test_validate_single_event_valid():
    ev = TouchEvent(timestamp=5.0, team="Arsenal", player="Saka", x=50.0, y=34.0)
    data = ev.model_dump(mode="json")
    result = validate_single_event(data)
    assert result.valid


def test_validate_single_event_invalid():
    data = {"event_type": "shot", "x": 200.0, "y": 34.0, "timestamp": 1.0,
            "team": "A", "player": "B", "end_x": 105.0, "end_y": 34.0}
    result = validate_single_event(data)
    assert not result.valid
