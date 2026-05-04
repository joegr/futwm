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

import json
import io
import os
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, abort
from flask_cors import CORS

from .pitch import Pitch
from .csv_loader import load_csv, MatchCSV
from .stochastic import TransitionModel
from .predictor import EventPredictor
from .simulation import MatchSimulator
from .events import EventType

# ── app setup ─────────────────────────────────────────────────────────────────

FRONTEND_DIR  = Path(__file__).resolve().parent.parent / "frontend"
EXAMPLES_DIR  = Path(__file__).resolve().parent.parent / "examples"

app = Flask(__name__, static_folder=str(FRONTEND_DIR))
CORS(app)

# ── state ─────────────────────────────────────────────────────────────────────

_pitch     = Pitch()
_match:    MatchCSV | None = None
_predictor = EventPredictor(_pitch, seed=42)
_model     = _predictor.model


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
    global _match, _model, _predictor

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

    try:
        _match = load_csv(io.StringIO(text), pitch=_pitch)
    except Exception as e:
        return jsonify({"error": f"CSV parse error: {e}"}), 400

    # auto-fit model on uploaded data
    _match.fit_model(_model)

    return jsonify({
        "ok": True,
        "n_events": len(_match.events),
        "teams": _match.teams,
        "players": _match.players,
        "events": _match.to_json(),
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
