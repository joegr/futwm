"""HTTP contract tests for the editor / preprocess endpoints."""

from __future__ import annotations

import json

import pytest

flask_client = pytest.importorskip("flask")  # ensures Flask is installed

from soccer_model import server as srv  # noqa: E402


# ── fixtures ─────────────────────────────────────────────────────────────────

SAMPLE_CSV = (
    "timestamp,team,player,event_type,x,y,end_x,end_y,to_player,pass_type,outcome\n"
    "0.5,Home,p1,pass,30,40,50,30,p2,short,success\n"
    "1.0,Home,p2,touch,50,30,,,,,success\n"
    "2.0,Home,p2,shot,90,34,105,34,,,off_target\n"
    "3.0,Away,a1,pass,60,30,40,40,a2,short,success\n"
)


@pytest.fixture
def client():
    srv.app.config["TESTING"] = True
    with srv.app.test_client() as c:
        # reset module-level state between tests
        srv._match = None
        srv._rows  = []
        yield c
        srv._match = None
        srv._rows  = []


def _upload(client, csv_text=SAMPLE_CSV):
    return client.post(
        "/api/upload",
        data=json.dumps({"csv_text": csv_text}),
        content_type="application/json",
    )


# ── ontology endpoint ────────────────────────────────────────────────────────

def test_ontology_returns_enums_and_per_type_fields(client):
    r = client.get("/api/ontology")
    assert r.status_code == 200
    body = r.get_json()
    assert "pass" in body["enums"]["event_type"]
    assert "left"  in body["enums"]["foot"]
    # passes have required to_player + end_x + end_y
    pass_fields = body["fields_per_type"]["pass"]
    assert "to_player" in pass_fields["required"]
    assert "end_x"     in pass_fields["required"]
    # valid outcomes per type (shot can be goal, off_target, saved, blocked)
    shot_outs = set(body["valid_outcomes"]["shot"])
    assert {"goal", "off_target"} <= shot_outs


def test_preprocess_ops_endpoint_lists_specs(client):
    r = client.get("/api/preprocess/ops")
    assert r.status_code == 200
    ops = {o["op"] for o in r.get_json()["operations"]}
    for required in {"rename_player", "swap_teams", "transform_coords", "validate"}:
        assert required in ops


# ── add / edit / delete event ────────────────────────────────────────────────

def test_create_event_requires_upload_first_or_works_standalone(client):
    # POST /api/events without any prior upload should still validate fields and
    # accept a well-formed row (creating the first row from scratch).
    payload = {
        "timestamp": 0.1, "team": "Home", "player": "p1",
        "event_type": "touch", "x": 50, "y": 34, "outcome": "success",
    }
    r = client.post("/api/events", data=json.dumps(payload), content_type="application/json")
    assert r.status_code == 201, r.get_json()
    body = r.get_json()
    assert body["n_events"] == 1


def test_create_event_rejects_missing_required_field(client):
    payload = {"team": "Home", "player": "p1", "event_type": "pass", "x": 1, "y": 1}
    r = client.post("/api/events", data=json.dumps(payload), content_type="application/json")
    assert r.status_code == 422
    body = r.get_json()
    assert "missing required field" in " ".join(body["errors"])


def test_create_event_rejects_out_of_pitch(client):
    payload = {
        "timestamp": 1.0, "team": "Home", "player": "p1",
        "event_type": "touch", "x": 9999, "y": 0,
    }
    r = client.post("/api/events", data=json.dumps(payload), content_type="application/json")
    assert r.status_code == 422


def test_update_and_delete_event(client):
    _upload(client)
    # initial state
    r = client.get("/api/match")
    n = len(r.get_json()["events"])
    assert n == 4

    # PUT — change row 0 from a pass to a touch
    new_row = {
        "timestamp": 0.5, "team": "Home", "player": "p1",
        "event_type": "touch", "x": 30, "y": 40, "outcome": "success",
    }
    r = client.put("/api/events/0", data=json.dumps(new_row), content_type="application/json")
    assert r.status_code == 200
    assert r.get_json()["events"][0]["event_type"] == "touch"

    # DELETE — drop the last row
    r = client.delete(f"/api/events/{n - 1}")
    assert r.status_code == 200
    assert len(r.get_json()["events"]) == n - 1

    # Out-of-range
    assert client.delete("/api/events/9999").status_code == 400
    assert client.put(
        "/api/events/9999",
        data=json.dumps(new_row), content_type="application/json",
    ).status_code == 400


def test_update_with_invalid_row_rolls_back_state(client):
    _upload(client)
    before = client.get("/api/match").get_json()["events"]
    bad = {"timestamp": "not-a-number", "team": "Home", "player": "p1",
           "event_type": "pass", "x": 0, "y": 0}
    r = client.put("/api/events/0", data=json.dumps(bad), content_type="application/json")
    assert r.status_code == 422
    after = client.get("/api/match").get_json()["events"]
    assert before == after, "state must be untouched after a rejected edit"


# ── preprocess endpoint ──────────────────────────────────────────────────────

def test_preprocess_requires_loaded_match(client):
    r = client.post("/api/preprocess", data=json.dumps({"operations": []}),
                    content_type="application/json")
    assert r.status_code == 400


def test_preprocess_apply_changes_event_list(client):
    _upload(client)
    ops = [
        {"op": "keep_team", "team": "Home"},
        {"op": "rename_team", "from": "Home", "to": "Arsenal"},
    ]
    r = client.post("/api/preprocess",
                    data=json.dumps({"operations": ops}),
                    content_type="application/json")
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["ok"] is True
    assert {e["team"] for e in body["events"]} == {"Arsenal"}


def test_preprocess_dry_run_does_not_commit(client):
    _upload(client)
    before = client.get("/api/match").get_json()["events"]
    r = client.post("/api/preprocess",
                    data=json.dumps({
                        "operations": [{"op": "keep_team", "team": "Home"}],
                        "dry_run": True,
                    }),
                    content_type="application/json")
    body = r.get_json()
    assert body["ok"] is True
    assert body["dry_run"] is True
    assert "preview_rows" in body
    # The committed event list is unchanged.
    after = client.get("/api/match").get_json()["events"]
    assert before == after


def test_preprocess_reports_step_errors(client):
    _upload(client)
    r = client.post("/api/preprocess",
                    data=json.dumps({"operations": [{"op": "mirror", "axis": "z"}]}),
                    content_type="application/json")
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert body["errors"][0]["op"] == "mirror"


def test_preprocess_transform_coords_roundtrips_into_metres(client):
    csv = (
        "timestamp,team,player,event_type,x,y,end_x,end_y,to_player,pass_type,outcome\n"
        "0.5,Home,p1,pass,50,50,100,100,p2,short,success\n"
    )
    _upload(client, csv)
    r = client.post("/api/preprocess",
                    data=json.dumps({
                        "operations": [{"op": "transform_coords", "from_frame": "opta_percent"}],
                    }),
                    content_type="application/json")
    body = r.get_json()
    assert body["ok"] is True
    ev = body["events"][0]
    assert ev["x"] == pytest.approx(52.5)
    assert ev["y"] == pytest.approx(34.0)
    assert ev["end_x"] == pytest.approx(105.0)
    assert ev["end_y"] == pytest.approx(68.0)
