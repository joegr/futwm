"""Contract tests for the FastAPI surface."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from soccer_vision.api.server import app
from soccer_vision.api import routes as routes_module


@pytest.fixture(autouse=True)
def _isolate_sessions():
    """Each test gets a clean in-memory session store."""
    routes_module._SESSIONS.clear()
    yield
    routes_module._SESSIONS.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _png_b64(width: int = 64, height: int = 48) -> str:
    """Synthesise a tiny PNG with a bright spot the ball-stub will find."""
    pytest.importorskip("PIL")
    from PIL import Image
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[height // 2, width // 2] = (255, 255, 255)   # the "ball" pixel
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _session_body() -> dict:
    return {
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "homography": {
            "src_points": [[0, 0], [64, 0], [0, 48], [64, 48]],
            "dst_points": [[0, 0], [105, 0], [0, 68], [105, 68]],
        },
        "pitch_length_m": 105.0,
        "pitch_width_m": 68.0,
        "contact_radius_m": 1.5,
    }


# ── health ───────────────────────────────────────────────────────────────────

def test_healthz_ok(client):
    r = client.get("/v1/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["service"] == "soccer-vision-service"
    assert "ml_available" in body


def test_readyz_ok(client):
    r = client.get("/v1/readyz")
    assert r.status_code == 200


# ── sessions ─────────────────────────────────────────────────────────────────

def test_create_session_returns_id_and_rms(client):
    r = client.post("/v1/sessions", json=_session_body())
    assert r.status_code == 201, r.text
    info = r.json()
    assert "session_id" in info
    assert info["homography_rms_px"] < 1e-3
    assert info["home_team"] == "Arsenal"


def test_create_session_rejects_collinear_homography(client):
    body = _session_body()
    body["homography"]["src_points"] = [[0, 0], [1, 0], [2, 0], [3, 0]]
    body["homography"]["dst_points"] = [[0, 0], [1, 1], [2, 2], [3, 3]]
    r = client.post("/v1/sessions", json=body)
    assert r.status_code == 400


def test_create_session_rejects_mismatched_lengths(client):
    body = _session_body()
    body["homography"]["dst_points"] = [[0, 0], [105, 0], [0, 68]]
    r = client.post("/v1/sessions", json=body)
    assert r.status_code == 422   # pydantic validation


def test_get_session_404_for_unknown(client):
    r = client.get("/v1/sessions/does-not-exist")
    assert r.status_code == 404


def test_delete_session_idempotent(client):
    r1 = client.delete("/v1/sessions/does-not-exist")
    assert r1.status_code == 204
    r2 = client.delete("/v1/sessions/does-not-exist")
    assert r2.status_code == 204


# ── frame analysis ───────────────────────────────────────────────────────────

def test_analyze_frame_404_for_unknown_session(client):
    r = client.post("/v1/frames/analyze", json={
        "session_id": "nope",
        "timestamp_s": 0.0,
        "frame_b64": _png_b64(),
        "image_format": "png",
    })
    assert r.status_code == 404


def test_analyze_frame_returns_valid_response(client):
    s = client.post("/v1/sessions", json=_session_body()).json()
    r = client.post("/v1/frames/analyze", json={
        "session_id": s["session_id"],
        "timestamp_s": 1.0,
        "frame_b64": _png_b64(),
        "image_format": "png",
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session_id"] == s["session_id"]
    assert body["timestamp_s"] == 1.0
    # No player track on this frame, but the ball-stub will still detect
    # the bright pixel — and the emitter should NOT emit events because
    # there's no player.
    assert body["n_detections"] >= 1
    assert body["events"] == []


def test_analyze_frame_rejects_bad_base64(client):
    s = client.post("/v1/sessions", json=_session_body()).json()
    r = client.post("/v1/frames/analyze", json={
        "session_id": s["session_id"],
        "timestamp_s": 1.0,
        "frame_b64": "!!!not base64!!!",
        "image_format": "png",
    })
    assert r.status_code == 400


def test_session_counters_update_after_analyze(client):
    s = client.post("/v1/sessions", json=_session_body()).json()
    client.post("/v1/frames/analyze", json={
        "session_id": s["session_id"],
        "timestamp_s": 1.0,
        "frame_b64": _png_b64(),
        "image_format": "png",
    })
    s2 = client.get(f"/v1/sessions/{s['session_id']}").json()
    assert s2["frames_processed"] == 1


# ── OpenAPI surface ──────────────────────────────────────────────────────────

def test_openapi_schema_includes_all_routes(client):
    r = client.get("/v1/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    assert "/v1/healthz" in paths
    assert "/v1/readyz" in paths
    assert "/v1/sessions" in paths
    assert "/v1/sessions/{session_id}" in paths
    assert "/v1/frames/analyze" in paths
