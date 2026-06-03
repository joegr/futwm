"""Tests for the row-level preprocessing pipeline."""

from __future__ import annotations

import pytest

from soccer_model.pitch import Pitch
from soccer_model.preprocess import (
    PreprocessError,
    apply_operation,
    apply_pipeline,
    operation_specs,
    validate_row,
)


# ── fixtures ─────────────────────────────────────────────────────────────────

def _row(**kw):
    base = {
        "timestamp": "1.0", "team": "A", "player": "p1",
        "event_type": "pass", "x": "10", "y": "20",
        "end_x": "30", "end_y": "40",
        "to_player": "p2", "pass_type": "short", "outcome": "success",
    }
    base.update(kw)
    return base


@pytest.fixture
def rows():
    return [
        _row(timestamp="0.5", team="A", player="p1", end_x="60"),
        _row(timestamp="1.5", team="B", player="b1", end_x="70"),
        _row(timestamp="2.5", team="A", player="p1", end_x="80", event_type="shot",
             xg="0.3", outcome="off_target"),
    ]


# ── op specs introspection ───────────────────────────────────────────────────

def test_operation_specs_are_listed():
    specs = operation_specs()
    names = {s.op for s in specs}
    # core set must always be present
    for required in {"rename_player", "rename_team", "swap_teams", "mirror",
                     "time_shift", "clamp_to_pitch", "transform_coords",
                     "keep_team", "drop_event_type", "filter_time_range",
                     "sort_by_time", "snap_to_zone", "deduplicate", "validate"}:
        assert required in names


# ── apply_operation: each handler ────────────────────────────────────────────

def test_rename_player_also_renames_recipient(rows):
    out = apply_operation(rows, "rename_player", **{"from": "p2", "to": "P2"})
    assert all(r["to_player"] != "p2" for r in out)
    assert any(r["to_player"] == "P2" for r in out)


def test_rename_team_propagates(rows):
    out = apply_operation(rows, "rename_team", **{"from": "A", "to": "ARS"})
    teams = {r["team"] for r in out}
    assert "A" not in teams
    assert "ARS" in teams


def test_swap_teams_swaps_labels_and_mirrors_x(rows):
    out = apply_operation(rows, "swap_teams", pitch_length=105.0)
    # labels swapped
    assert {r["team"] for r in out} == {"A", "B"}
    a_count = sum(1 for r in out if r["team"] == "A")
    b_count = sum(1 for r in out if r["team"] == "B")
    # rows 0,2 were team A (2 rows), row 1 was team B (1 row)
    assert a_count == 1 and b_count == 2
    # x was mirrored: original x=10 → 105 - 10 = 95
    assert float(out[0]["x"]) == pytest.approx(95.0)
    assert float(out[0]["end_x"]) == pytest.approx(105 - 60.0)


def test_mirror_x_only(rows):
    out = apply_operation(rows, "mirror", axis="x", pitch_length=105.0)
    assert float(out[0]["x"]) == pytest.approx(95.0)
    # y unchanged
    assert float(out[0]["y"]) == pytest.approx(20.0)


def test_mirror_y_only(rows):
    out = apply_operation(rows, "mirror", axis="y", pitch_width=68.0)
    assert float(out[0]["y"]) == pytest.approx(48.0)
    assert float(out[0]["x"]) == pytest.approx(10.0)


def test_mirror_rejects_bad_axis(rows):
    with pytest.raises(PreprocessError):
        apply_operation(rows, "mirror", axis="z")


def test_time_shift_clamps_at_zero(rows):
    out = apply_operation(rows, "time_shift", delta_s=-1.0)
    times = sorted(float(r["timestamp"]) for r in out)
    assert times[0] == pytest.approx(0.0)   # row at 0.5 - 1.0 clamped to 0
    assert times[1] == pytest.approx(0.5)
    assert times[2] == pytest.approx(1.5)


def test_time_scale_rejects_non_positive(rows):
    with pytest.raises(PreprocessError):
        apply_operation(rows, "time_scale", factor=0)
    with pytest.raises(PreprocessError):
        apply_operation(rows, "time_scale", factor=-1.0)


def test_round_time(rows):
    rows[0]["timestamp"] = 1.234567
    out = apply_operation(rows, "round_time", ndigits=2)
    assert float(out[0]["timestamp"]) == pytest.approx(1.23)


def test_clamp_to_pitch_clips_outliers():
    bad = [_row(x="-5", y="200", end_x="1e6", end_y="-10")]
    out = apply_operation(bad, "clamp_to_pitch", pitch_length=105, pitch_width=68)
    assert float(out[0]["x"]) == 0.0
    assert float(out[0]["y"]) == 68.0
    assert float(out[0]["end_x"]) == 105.0
    assert float(out[0]["end_y"]) == 0.0


def test_transform_coords_from_opta_percent():
    rs = [_row(x="50", y="50", end_x="100", end_y="100")]
    out = apply_operation(rs, "transform_coords", from_frame="opta_percent")
    assert float(out[0]["x"]) == pytest.approx(52.5)
    assert float(out[0]["y"]) == pytest.approx(34.0)
    assert float(out[0]["end_x"]) == pytest.approx(105.0)
    assert float(out[0]["end_y"]) == pytest.approx(68.0)


def test_transform_coords_rejects_unknown_frame():
    with pytest.raises(PreprocessError, match="unknown coordinate frame"):
        apply_operation([_row()], "transform_coords", from_frame="not_a_frame")


def test_transform_coords_from_fifa_is_noop():
    rs = [_row(x="10", y="20")]
    out = apply_operation(rs, "transform_coords", from_frame="fifa_metres")
    assert out[0]["x"] == "10"


def test_keep_team_drops_others(rows):
    out = apply_operation(rows, "keep_team", team="A")
    assert all(r["team"] == "A" for r in out)
    assert len(out) == 2


def test_drop_team(rows):
    out = apply_operation(rows, "drop_team", team="B")
    assert all(r["team"] != "B" for r in out)


def test_drop_event_type_case_insensitive(rows):
    out = apply_operation(rows, "drop_event_type", event_type="SHOT")
    assert all(r["event_type"] != "shot" for r in out)
    assert len(out) == 2


def test_filter_time_range(rows):
    out = apply_operation(rows, "filter_time_range", min_s=1.0, max_s=2.0)
    assert len(out) == 1
    assert float(out[0]["timestamp"]) == 1.5


def test_filter_time_range_rejects_inverted_window():
    with pytest.raises(PreprocessError):
        apply_operation([_row()], "filter_time_range", min_s=5.0, max_s=1.0)


def test_sort_by_time():
    shuffled = [_row(timestamp="3.0"), _row(timestamp="1.0"), _row(timestamp="2.0")]
    out = apply_operation(shuffled, "sort_by_time")
    assert [float(r["timestamp"]) for r in out] == [1.0, 2.0, 3.0]


def test_set_field_with_where_clauses(rows):
    out = apply_operation(rows, "set_field", field="pass_type", value="long",
                          where_team="A", where_event_type="pass")
    # Only rows where both filters match must be updated.
    for r in out:
        if r["team"] == "A" and r["event_type"] == "pass":
            assert r["pass_type"] == "long"


def test_snap_to_zone():
    rs = [_row(x="10", y="10", end_x="70", end_y="50")]
    out = apply_operation(rs, "snap_to_zone", nx=3, ny=3, pitch_length=105, pitch_width=68)
    # 10 m / 35 m → zone 0 → centre at 17.5
    assert float(out[0]["x"]) == pytest.approx(17.5)
    # 70 m / 35 m → zone 2 → centre at 87.5
    assert float(out[0]["end_x"]) == pytest.approx(87.5)


def test_deduplicate_consecutive():
    rs = [_row(timestamp="1.0"), _row(timestamp="1.0"), _row(timestamp="2.0"),
          _row(timestamp="1.0")]   # not adjacent to a 1.0 row → kept
    out = apply_operation(rs, "deduplicate")
    assert len(out) == 3


def test_validate_passes_on_clean_rows(rows):
    # rows have all required fields and within pitch — should NOT raise.
    apply_operation(rows, "validate", pitch_length=105.0, pitch_width=68.0)


def test_validate_fails_on_bad_rows():
    bad = [_row(timestamp="-1.0")]
    with pytest.raises(PreprocessError, match="validation failed"):
        apply_operation(bad, "validate")


# ── pipeline ─────────────────────────────────────────────────────────────────

def test_pipeline_runs_steps_in_order(rows):
    result = apply_pipeline(rows, [
        {"op": "keep_team", "team": "A"},
        {"op": "rename_team", "from": "A", "to": "ARS"},
        {"op": "time_shift", "delta_s": 10.0},
    ])
    assert result.errors == []
    assert len(result.log) == 3
    assert all(r["team"] == "ARS" for r in result.rows)
    assert min(float(r["timestamp"]) for r in result.rows) >= 10.0


def test_pipeline_stops_at_first_error(rows):
    result = apply_pipeline(rows, [
        {"op": "rename_team", "from": "A", "to": "X"},
        {"op": "mirror", "axis": "z"},          # error here
        {"op": "time_shift", "delta_s": 99.0},  # never executes
    ])
    assert len(result.errors) == 1
    assert result.errors[0]["op"] == "mirror"
    # The earlier rename DID apply on the working copy
    assert any(r["team"] == "X" for r in result.rows)
    # The later shift did NOT run
    assert float(result.rows[0]["timestamp"]) < 90


def test_pipeline_does_not_mutate_caller_rows(rows):
    snapshot = [dict(r) for r in rows]
    apply_pipeline(rows, [{"op": "rename_team", "from": "A", "to": "X"}])
    assert rows == snapshot


def test_pipeline_rejects_unknown_op(rows):
    result = apply_pipeline(rows, [{"op": "fly_to_the_moon"}])
    assert result.errors and "unknown operation" in result.errors[0]["error"]


def test_pipeline_rejects_non_dict_step(rows):
    result = apply_pipeline(rows, ["not a dict"])
    assert result.errors and "must be a dict" in result.errors[0]["error"]


# ── validate_row ─────────────────────────────────────────────────────────────

def test_validate_row_ok():
    assert validate_row(_row()) == []


def test_validate_row_reports_missing_required():
    errs = validate_row({"event_type": "pass"})
    assert any("missing required field" in e for e in errs)


def test_validate_row_rejects_out_of_pitch():
    errs = validate_row(_row(x="999"), pitch=Pitch(length=105, width=68))
    assert any("outside pitch" in e for e in errs)


def test_validate_row_rejects_unknown_enum_values():
    errs = validate_row(_row(event_type="teleport"))
    assert any("unknown event_type" in e for e in errs)
    errs = validate_row(_row(foot="left_arm"))
    assert any("unknown foot" in e for e in errs)


def test_validate_row_rejects_negative_timestamp():
    errs = validate_row(_row(timestamp="-1"))
    assert any("timestamp must be >= 0" in e for e in errs)
