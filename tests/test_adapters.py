"""Tests for the provider-adapter framework + the StatsBomb reference adapter."""

from __future__ import annotations

import pytest

from soccer_model.adapters import (
    AdapterError,
    CoordinateFrame,
    DirectionConvention,
    Encoding,
    IdentityScheme,
    ProviderAdapter,
    ProviderInfo,
    TimeBasis,
    Transport,
    get_adapter,
    list_adapters,
    register_adapter,
)
from soccer_model.adapters.coordinates import transform_xy


# ── coordinate transforms ────────────────────────────────────────────────────

def test_coordinates_round_trip_through_every_frame():
    pts = [(0.0, 0.0), (105.0, 68.0), (52.5, 34.0), (10.0, 50.0)]
    for frame in CoordinateFrame:
        for x, y in pts:
            xx, yy = transform_xy(x, y, src=CoordinateFrame.FIFA_METRES, dst=frame)
            xb, yb = transform_xy(xx, yy, src=frame, dst=CoordinateFrame.FIFA_METRES)
            assert abs(xb - x) < 1e-9, f"{frame}: x round-trip {x} -> {xx} -> {xb}"
            assert abs(yb - y) < 1e-9, f"{frame}: y round-trip {y} -> {yy} -> {yb}"


def test_coordinates_statsbomb_y_is_flipped():
    # StatsBomb has y running TOP-down; (0, 0) = top-left.
    # (60, 40) at the centre of a 120x80 yards pitch ≈ centre of 105x68 m.
    x_m, y_m = transform_xy(60.0, 40.0, src=CoordinateFrame.STATSBOMB, dst=CoordinateFrame.FIFA_METRES)
    assert abs(x_m - 52.5) < 0.01
    assert abs(y_m - 34.0) < 0.01


def test_coordinates_opta_percent_origin_bottom_left():
    x_m, y_m = transform_xy(0.0, 0.0, src=CoordinateFrame.OPTA_PERCENT, dst=CoordinateFrame.FIFA_METRES)
    assert (x_m, y_m) == (0.0, 0.0)
    x_m, y_m = transform_xy(100.0, 100.0, src=CoordinateFrame.OPTA_PERCENT, dst=CoordinateFrame.FIFA_METRES)
    assert abs(x_m - 105.0) < 1e-9
    assert abs(y_m - 68.0) < 1e-9


def test_coordinates_centred_cm_origin_centre():
    x_m, y_m = transform_xy(0.0, 0.0, src=CoordinateFrame.CENTRED_CM, dst=CoordinateFrame.FIFA_METRES)
    assert abs(x_m - 52.5) < 1e-9
    assert abs(y_m - 34.0) < 1e-9


# ── registry ─────────────────────────────────────────────────────────────────

def test_all_20_providers_registered():
    expected = {
        "statsbomb", "wyscout_v3", "opta_f24", "sportec", "impect", "pff_fc",
        "metrica", "skillcorner", "second_spectrum", "tracab", "signality",
        "hawkeye_2d", "sportradar", "api_football", "football_data_org",
        "sportmonks", "understat", "epts_fifa", "fifa_connect", "spadl",
    }
    assert set(list_adapters()) == expected
    assert len(list_adapters()) == 20


def test_get_adapter_returns_correct_class():
    a = get_adapter("statsbomb")
    assert isinstance(a, ProviderAdapter)
    assert a.info.provider_id == "statsbomb"
    assert a.info.coordinate_frame is CoordinateFrame.STATSBOMB


def test_get_adapter_unknown_raises():
    with pytest.raises(AdapterError, match="unknown provider_id"):
        get_adapter("not_a_real_provider")


def test_register_rejects_duplicate_provider_id():
    dup_info = ProviderInfo(
        provider_id="statsbomb",  # already registered
        display_name="dup",
        transport=Transport.FILE,
        encoding=Encoding.JSON,
        coordinate_frame=CoordinateFrame.FIFA_METRES,
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_ALWAYS,
        time_basis=TimeBasis.MATCH_SECONDS,
        identity_scheme=IdentityScheme.VENDOR_INTERNAL,
    )

    class Dup(ProviderAdapter):
        info = dup_info
        def load_match(self, source, **kw): ...

    with pytest.raises(AdapterError, match="already registered"):
        register_adapter("statsbomb")(Dup)


def test_register_rejects_provider_id_mismatch():
    mismatch_info = ProviderInfo(
        provider_id="brand_new",
        display_name="x",
        transport=Transport.FILE,
        encoding=Encoding.JSON,
        coordinate_frame=CoordinateFrame.FIFA_METRES,
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_ALWAYS,
        time_basis=TimeBasis.MATCH_SECONDS,
        identity_scheme=IdentityScheme.VENDOR_INTERNAL,
    )

    class Mismatch(ProviderAdapter):
        info = mismatch_info
        def load_match(self, source, **kw): ...

    with pytest.raises(AdapterError, match="does not match"):
        register_adapter("different_id")(Mismatch)


def test_every_registered_adapter_has_complete_info():
    for pid in list_adapters():
        a = get_adapter(pid)
        assert a.info.provider_id == pid
        assert a.info.display_name
        assert a.info.url or not a.info.public  # public providers must have a URL


def test_scaffold_adapters_raise_not_implemented():
    """The 19 non-StatsBomb adapters must raise NotImplementedError on load."""
    for pid in list_adapters():
        if pid == "statsbomb":
            continue
        a = get_adapter(pid)
        with pytest.raises(NotImplementedError, match="scaffold"):
            a.load_match(source=None)


# ── StatsBomb reference adapter ──────────────────────────────────────────────

def _statsbomb_event(type_name, x, y, player="Lionel Messi", team="Argentina",
                     period=1, ts="00:00:10.000"):
    return {
        "type": {"id": 0, "name": type_name},
        "team": {"id": 1, "name": team},
        "player": {"id": 1, "name": player},
        "period": period,
        "timestamp": ts,
        "location": [x, y],
    }


def test_statsbomb_loads_minimal_match_inline():
    raw = [
        _statsbomb_event("Pass", 60, 40),
        _statsbomb_event("Ball Receipt*", 70, 38, player="Julián Álvarez"),
        _statsbomb_event("Shot", 110, 40, player="Julián Álvarez"),
    ]
    a = get_adapter("statsbomb")
    stream = a.load_match(raw, home_team="Argentina", away_team="Brazil", match_id="test-1")
    assert stream.metadata.match_id == "test-1"
    assert stream.metadata.home_team == "Argentina"
    assert len(stream.events) == 3
    # Coords should now be in FIFA metres (105 x 68)
    assert all(0.0 <= e.x <= 105.0 for e in stream.events)
    assert all(0.0 <= e.y <= 68.0 for e in stream.events)


def test_statsbomb_drops_events_without_location():
    raw = [
        {"type": {"name": "Half Start"}, "team": {"name": "A"},
         "player": {"name": "x"}, "period": 1, "timestamp": "00:00:00.000"},
        _statsbomb_event("Pass", 60, 40),
    ]
    a = get_adapter("statsbomb")
    stream = a.load_match(raw, home_team="A", away_team="B")
    assert len(stream.events) == 1


def test_statsbomb_drops_unknown_type():
    raw = [
        _statsbomb_event("Pass", 60, 40),
        _statsbomb_event("Some Future Event", 60, 40),
        _statsbomb_event("Shot", 110, 40),
    ]
    a = get_adapter("statsbomb")
    stream = a.load_match(raw, home_team="A", away_team="B")
    assert len(stream.events) == 2


def test_statsbomb_lifts_period_to_match_seconds():
    raw = [
        _statsbomb_event("Pass", 60, 40, period=1, ts="00:01:00.000"),
        _statsbomb_event("Pass", 60, 40, period=2, ts="00:01:00.000"),
    ]
    a = get_adapter("statsbomb")
    stream = a.load_match(raw, home_team="A", away_team="B")
    # First period kicks off at t=0; second period adds 45 minutes.
    ts1, ts2 = (e.timestamp for e in stream.events)
    assert ts1 == 60.0
    assert ts2 == 60.0 + 45 * 60


def test_statsbomb_round_trips_through_validator():
    """The adapter's _finalise() runs validate_event_stream — verify success."""
    from soccer_model.interchange import to_json_dict, validate_event_stream

    raw = [_statsbomb_event("Pass", 60, 40), _statsbomb_event("Shot", 110, 40)]
    a = get_adapter("statsbomb")
    stream = a.load_match(raw, home_team="A", away_team="B")
    result = validate_event_stream(to_json_dict(stream))
    assert result.valid
