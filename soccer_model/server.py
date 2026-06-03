"""
Flask API server for the soccer world-model dashboard.

Endpoints
---------
GET  /                  → serve frontend (index.html)
POST /api/upload        → upload CSV, parse, return match JSON
GET  /api/match         → current loaded match data
GET  /api/predict/<idx> → next-event prediction at event index
GET  /api/simulate      → Monte Carlo rollout from current state
GET  /api/pitch         → pitch geometry
POST /api/fit           → fit transition model on loaded data
"""

from __future__ import annotations

import io
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory
from flask_cors import CORS

from dataclasses import asdict

from . import ontology as ont
from .csv_loader import MatchCSV, load_csv
from .events import EventType
from .pitch import Pitch
from .predictor import EventPredictor
from .preprocess import (
    PreprocessError,
    apply_pipeline,
    operation_specs,
    validate_row,
)

# ── app setup ─────────────────────────────────────────────────────────────────

FRONTEND_DIR  = Path(__file__).resolve().parent.parent / "frontend"
EXAMPLES_DIR  = Path(__file__).resolve().parent.parent / "examples"

app = Flask(__name__, static_folder=str(FRONTEND_DIR))
CORS(app)

# ── state ─────────────────────────────────────────────────────────────────────

_pitch     = Pitch()
_match:    MatchCSV | None = None
_rows:     list[dict] = []           # raw CSV-style row dicts, kept in sync with _match
_predictor = EventPredictor(_pitch, seed=42)
_model     = _predictor.model


def _rebuild_match_from_rows() -> tuple[bool, str | None]:
    """Re-parse ``_rows`` into a fresh ``MatchCSV`` and refit the model.

    Returns ``(ok, error_message)`` so callers can surface failures back
    to the client without raising.
    """
    global _match
    if not _rows:
        _match = None
        return True, None
    try:
        # Build CSV text in-memory from the row dicts and pass it through
        # the canonical loader so every event is constructed identically
        # to the upload path.
        import csv as _csv

        buf = io.StringIO()
        # union of keys, preserving the canonical column order first
        canonical = [
            "timestamp", "team", "player", "event_type", "x", "y",
            "end_x", "end_y", "foot", "to_player", "outcome", "pass_type",
            "body_part", "xg", "card", "set_piece_type", "action",
            "gk_action_type", "tackled_player", "fouled_player",
        ]
        extra = sorted({k for r in _rows for k in r.keys() if k not in canonical})
        fieldnames = canonical + extra
        writer = _csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in _rows:
            writer.writerow({k: (r.get(k) if r.get(k) is not None else "") for k in fieldnames})
        buf.seek(0)
        _match = load_csv(buf, pitch=_pitch)
        _match.fit_model(_model)
        return True, None
    except Exception as e:
        return False, f"rebuild failed: {e}"


def _row_from_payload(payload: dict) -> dict:
    """Coerce a JSON payload from the frontend into a CSV-style row dict.

    Strings are stripped; numerics stay numeric; missing keys are omitted
    (the loader treats absent keys as defaults).
    """
    out: dict = {}
    for k, v in payload.items():
        if v is None or v == "":
            continue
        key = str(k).strip().lower().replace(" ", "_")
        if isinstance(v, str):
            out[key] = v.strip()
        else:
            out[key] = v
    return out


# ── static file serving ──────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/examples/<path:path>")
def example_files(path):
    full = EXAMPLES_DIR / path
    if full.is_file():
        return send_from_directory(str(EXAMPLES_DIR), path)
    abort(404)


@app.route("/<path:path>")
def static_files(path):
    full = FRONTEND_DIR / path
    if full.is_file():
        return send_from_directory(str(FRONTEND_DIR), path)
    abort(404)


# ── API ───────────────────────────────────────────────────────────────────────

@app.route("/api/pitch", methods=["GET"])
def api_pitch():
    return jsonify({
        "length": _pitch.length,
        "width":  _pitch.width,
        "goal_width": _pitch._goal_width,
        "penalty_depth": _pitch._penalty_depth,
        "goal_area_depth": _pitch._goal_area_depth,
        "centre_radius": _pitch._centre_radius,
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    global _match, _model, _predictor, _rows

    f = request.files.get("file")
    text = None
    if f:
        text = f.read().decode("utf-8")
    elif request.is_json:
        text = request.json.get("csv_text", "")
    elif request.form.get("csv_text"):
        text = request.form["csv_text"]

    if not text:
        return jsonify({"error": "No CSV data provided"}), 400

    # Snapshot the raw rows for downstream editing / preprocessing.
    import csv as _csv
    try:
        reader = _csv.DictReader(io.StringIO(text))
        new_rows = [
            {(k.strip().lower().replace(" ", "_") if k else k): (v.strip() if isinstance(v, str) else v)
             for k, v in r.items()}
            for r in reader
        ]
    except Exception as e:
        return jsonify({"error": f"CSV read error: {e}"}), 400

    try:
        _match = load_csv(io.StringIO(text), pitch=_pitch)
    except Exception as e:
        return jsonify({"error": f"CSV parse error: {e}"}), 400

    _rows = new_rows
    # auto-fit model on uploaded data
    _match.fit_model(_model)

    return jsonify({
        "ok": True,
        "n_events": len(_match.events),
        "teams": _match.teams,
        "players": _match.players,
        "events": _match.to_json(),
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Human-readable input / preprocessing endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/ontology", methods=["GET"])
def api_ontology():
    """
    Single source of truth that drives the dynamic Add/Edit Event form on
    the frontend.

    Returns every enum value, the valid-outcome set per event_type, and
    the list of fields a row of each type should populate. The frontend
    uses this to render a form whose options stay aligned with the schema
    without duplicate definitions.
    """
    def _values(enum_cls) -> list[str]:
        return [m.value for m in enum_cls]

    # Per-type required & optional row columns. These mirror the CSV
    # loader's per-type handling so any row this form produces will
    # round-trip through ``load_csv`` cleanly.
    fields_per_type = {
        ont.EventType.TOUCH.value:
            {"required": [], "optional": ["foot", "outcome"]},
        ont.EventType.PASS.value:
            {"required": ["to_player", "end_x", "end_y"], "optional": ["foot", "pass_type", "outcome"]},
        ont.EventType.SHOT.value:
            {"required": ["end_x", "end_y"], "optional": ["body_part", "xg", "outcome"]},
        ont.EventType.DRIBBLE.value:
            {"required": ["end_x", "end_y"], "optional": ["foot", "outcome"]},
        ont.EventType.TACKLE.value:
            {"required": ["tackled_player"], "optional": ["outcome"]},
        ont.EventType.HEADER.value:
            {"required": ["end_x", "end_y"], "optional": ["action", "outcome"]},
        ont.EventType.FOUL.value:
            {"required": ["fouled_player"], "optional": ["card"]},
        ont.EventType.GOALKEEPER_ACTION.value:
            {"required": [], "optional": ["gk_action_type", "foot", "end_x", "end_y", "outcome"]},
        ont.EventType.SET_PIECE.value:
            {"required": [], "optional": ["set_piece_type", "to_player", "foot", "end_x", "end_y", "outcome"]},
    }

    return jsonify({
        "schema_version": ont.SCHEMA_VERSION,
        "pitch": {"length": _pitch.length, "width": _pitch.width},
        "enums": {
            "event_type":      _values(ont.EventType),
            "foot":             _values(ont.Foot),
            "body_part":        _values(ont.BodyPart),
            "outcome":          _values(ont.EventOutcome),
            "pass_type":        _values(ont.PassType),
            "set_piece_type":   _values(ont.SetPieceType),
            "gk_action_type":   _values(ont.GoalkeeperActionType),
            "header_action":    _values(ont.HeaderAction),
            "card":             _values(ont.Card),
        },
        "valid_outcomes": {
            et.value: sorted(o.value for o in outs)
            for et, outs in ont.VALID_OUTCOMES.items()
        },
        "fields_per_type": fields_per_type,
        "common_fields": ["timestamp", "team", "player", "x", "y"],
    })


@app.route("/api/preprocess/ops", methods=["GET"])
def api_preprocess_ops():
    """Return the catalogue of preprocessing operations for the UI."""
    return jsonify({"operations": [asdict(s) for s in operation_specs()]})


@app.route("/api/events", methods=["POST"])
def api_events_create():
    """Append a single new event from a human-readable form payload."""
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400
    row = _row_from_payload(request.json or {})

    # Field-level validation before mutating state.
    errs = validate_row(row, pitch=_pitch)
    if errs:
        return jsonify({"error": "validation failed", "errors": errs, "row": row}), 422

    _rows.append(row)
    ok, err = _rebuild_match_from_rows()
    if not ok:
        # Roll back the speculative append so we never leave broken state.
        _rows.pop()
        _rebuild_match_from_rows()
        return jsonify({"error": err}), 400

    return jsonify({
        "ok": True,
        "n_events": len(_match.events) if _match else 0,
        "events": _match.to_json() if _match else [],
    }), 201


@app.route("/api/events/<int:idx>", methods=["PUT"])
def api_events_update(idx: int):
    """Replace the event at ``idx`` with the form payload."""
    if _match is None or idx < 0 or idx >= len(_rows):
        return jsonify({"error": f"Index {idx} out of range"}), 400
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    new_row = _row_from_payload(request.json or {})
    errs = validate_row(new_row, pitch=_pitch)
    if errs:
        return jsonify({"error": "validation failed", "errors": errs, "row": new_row}), 422

    prev = _rows[idx]
    _rows[idx] = new_row
    ok, err = _rebuild_match_from_rows()
    if not ok:
        _rows[idx] = prev
        _rebuild_match_from_rows()
        return jsonify({"error": err}), 400

    return jsonify({"ok": True, "events": _match.to_json() if _match else []})


@app.route("/api/events/<int:idx>", methods=["DELETE"])
def api_events_delete(idx: int):
    """Drop the event at ``idx``."""
    if _match is None or idx < 0 or idx >= len(_rows):
        return jsonify({"error": f"Index {idx} out of range"}), 400
    removed = _rows.pop(idx)
    ok, err = _rebuild_match_from_rows()
    if not ok:
        _rows.insert(idx, removed)
        _rebuild_match_from_rows()
        return jsonify({"error": err}), 400
    return jsonify({"ok": True, "events": _match.to_json() if _match else []})


@app.route("/api/preprocess", methods=["POST"])
def api_preprocess():
    """Run a pipeline of operations against the current row buffer.

    Body:
      ``{"operations": [{"op": "name", "param1": ..., "param2": ...}, ...],
         "dry_run": false}``

    Returns the new event list (after the typed loader re-builds the match)
    plus the per-step audit log. When ``dry_run`` is true, the row buffer is
    NOT updated — only the *what would change* preview is returned.
    """
    global _rows
    if _match is None:
        return jsonify({"error": "No match loaded"}), 400
    if not request.is_json:
        return jsonify({"error": "JSON body required"}), 400

    body = request.json or {}
    operations = body.get("operations") or []
    if not isinstance(operations, list):
        return jsonify({"error": "'operations' must be a list"}), 400
    dry_run = bool(body.get("dry_run", False))

    try:
        result = apply_pipeline(_rows, operations)
    except PreprocessError as e:                  # pragma: no cover - defensive
        return jsonify({"error": str(e)}), 400

    if result.errors:
        return jsonify({
            "ok": False,
            "log": result.log,
            "errors": result.errors,
        }), 400

    if dry_run:
        return jsonify({
            "ok": True,
            "dry_run": True,
            "rows_after": len(result.rows),
            "log": result.log,
            "preview_rows": result.rows[:50],
        })

    # Commit the new rows and rebuild.
    prev_rows = _rows
    _rows = result.rows
    ok, err = _rebuild_match_from_rows()
    if not ok:
        _rows = prev_rows
        _rebuild_match_from_rows()
        return jsonify({"error": err, "log": result.log}), 400

    return jsonify({
        "ok": True,
        "log": result.log,
        "n_events": len(_match.events) if _match else 0,
        "events": _match.to_json() if _match else [],
    })


@app.route("/api/match", methods=["GET"])
def api_match():
    if _match is None:
        return jsonify({"error": "No match loaded. Upload a CSV first."}), 400
    return jsonify({
        "teams":   _match.teams,
        "players": _match.players,
        "events":  _match.to_json(),
    })


@app.route("/api/predict/<int:idx>", methods=["GET"])
def api_predict(idx: int):
    if _match is None:
        return jsonify({"error": "No match loaded"}), 400
    if idx < 0 or idx >= len(_match.states):
        return jsonify({"error": f"Index {idx} out of range"}), 400

    state = _match.states[idx]
    pred  = _predictor.predict_next(state, top_k=5)

    top = []
    for ev, p in zip(pred.top_events, pred.top_probs):
        d = {
            "event_type": ev.event_type.value,
            "probability": round(p, 4),
            "x": ev.pitch_x,
            "y": ev.pitch_y,
        }
        if hasattr(ev, "dest_x"):
            d["end_x"] = ev.dest_x
            d["end_y"] = ev.dest_y
        if hasattr(ev, "xg"):
            d["xg"] = ev.xg
        top.append(d)

    dist = pred.distribution
    full_probs = {
        et.value: round(float(p), 4)
        for et, p in zip(dist.event_types, dist.probs)
    }

    return jsonify({
        "event_index": idx,
        "top_predictions": top,
        "full_distribution": full_probs,
        "expected_xg": round(pred.expected_xg, 6),
        "dest_mean": list(dist.dest_mean),
    })


@app.route("/api/simulate", methods=["GET"])
def api_simulate():
    if _match is None:
        return jsonify({"error": "No match loaded"}), 400

    idx     = request.args.get("from_index", 0, type=int)
    horizon = request.args.get("horizon", 8, type=int)
    samples = request.args.get("samples", 20, type=int)

    if idx < 0 or idx >= len(_match.states):
        return jsonify({"error": f"Index {idx} out of range"}), 400

    state = _match.states[idx]
    seqs  = _predictor.predict_sequence(state, horizon=horizon, samples=samples)

    sim_results = []
    for seq in seqs:
        sim_results.append([
            {
                "event_type": e.event_type.value,
                "x": e.pitch_x, "y": e.pitch_y,
                "outcome": getattr(e, "outcome", None) and getattr(e, "outcome").value,
                "xg": getattr(e, "xg", None),
            }
            for e in seq
        ])

    xg_stats = _predictor.cumulative_xg(seqs)

    return jsonify({
        "from_index": idx,
        "horizon": horizon,
        "samples": len(sim_results),
        "sequences": sim_results,
        "xg_stats": xg_stats,
    })


@app.route("/api/fit", methods=["POST"])
def api_fit():
    global _model
    if _match is None:
        return jsonify({"error": "No match loaded"}), 400
    _match.fit_model(_model)
    return jsonify({"ok": True, "message": "Model fitted on loaded match data"})


@app.route("/api/xg_timeline", methods=["GET"])
def api_xg_timeline():
    """Cumulative xG per team over the event timeline."""
    if _match is None:
        return jsonify({"error": "No match loaded"}), 400

    timeline = []
    cum_xg: dict[str, float] = {t: 0.0 for t in _match.teams}
    for i, ev in enumerate(_match.events):
        xg = getattr(ev, "xg", 0.0) or 0.0
        if ev.event_type == EventType.SHOT and ev.team in cum_xg:
            cum_xg[ev.team] += xg
        timeline.append({
            "index": i,
            "timestamp": ev.timestamp,
            **{f"xg_{t}": round(cum_xg.get(t, 0.0), 4) for t in _match.teams},
        })
    return jsonify({"teams": _match.teams, "timeline": timeline})


@app.route("/api/pass_network", methods=["GET"])
def api_pass_network():
    """Pass counts and average positions per player for the pass network."""
    if _match is None:
        return jsonify({"error": "No match loaded"}), 400

    team_filter = request.args.get("team", _match.teams[0])

    positions: dict[str, list[tuple[float, float]]] = {}
    edges:     dict[tuple[str, str], int] = {}

    for ev in _match.events:
        if ev.team != team_filter:
            continue
        pid = ev.player
        if pid not in positions:
            positions[pid] = []
        positions[pid].append((ev.pitch_x, ev.pitch_y))

        if ev.event_type == EventType.PASS:
            to_p = getattr(ev, "to_player", "")
            if to_p and getattr(ev, "outcome", None) and ev.outcome.value == "success":
                key = (pid, to_p)
                edges[key] = edges.get(key, 0) + 1

    nodes = []
    for pid, pts in positions.items():
        avg_x = sum(p[0] for p in pts) / len(pts)
        avg_y = sum(p[1] for p in pts) / len(pts)
        nodes.append({
            "player": pid,
            "team": team_filter,
            "avg_x": round(avg_x, 2),
            "avg_y": round(avg_y, 2),
            "touches": len(pts),
        })

    links = [
        {"source": s, "target": t, "count": c}
        for (s, t), c in sorted(edges.items(), key=lambda x: -x[1])
    ]

    return jsonify({"team": team_filter, "nodes": nodes, "links": links})


@app.route("/api/heatmap", methods=["GET"])
def api_heatmap():
    """Grid-based touch density heatmap."""
    if _match is None:
        return jsonify({"error": "No match loaded"}), 400

    team_filter = request.args.get("team", "")
    nx, ny = 21, 14   # grid cells

    grid = [[0] * ny for _ in range(nx)]
    for ev in _match.events:
        if team_filter and ev.team != team_filter:
            continue
        cx = int(ev.pitch_x / _pitch.length * (nx - 1))
        cy = int(ev.pitch_y / _pitch.width  * (ny - 1))
        cx = max(0, min(nx - 1, cx))
        cy = max(0, min(ny - 1, cy))
        grid[cx][cy] += 1

    cells = []
    for i in range(nx):
        for j in range(ny):
            if grid[i][j] > 0:
                cells.append({
                    "x": round(i / (nx - 1) * _pitch.length, 2),
                    "y": round(j / (ny - 1) * _pitch.width, 2),
                    "count": grid[i][j],
                })

    return jsonify({"team": team_filter or "all", "cells": cells, "nx": nx, "ny": ny})


# ── entry point ───────────────────────────────────────────────────────────────

def run(host: str = "127.0.0.1", port: int = 5050, debug: bool = True) -> None:
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run()
