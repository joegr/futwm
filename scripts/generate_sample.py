#!/usr/bin/env python3
"""
Generate a realistic bundled sample match CSV using the world-model simulator.

Output: soccer_model/data/sample_match.csv
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
from pathlib import Path

from soccer_model.pitch import Pitch
from soccer_model.simulation import MatchSimulator
from soccer_model.events import (
    EventType, Pass, Shot, Dribble, Tackle, Header, Foul,
    GoalkeeperAction, SetPiece, Touch,
)

OUTPUT = Path(__file__).resolve().parent.parent / "soccer_model" / "data" / "sample_match.csv"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

# ── teams ────────────────────────────────────────────────────────────────────
HOME = "Arsenal"
AWAY = "Chelsea"

HOME_PLAYERS = [
    "Ramsdale", "White", "Saliba", "Gabriel", "Zinchenko",
    "Odegaard", "Rice", "Havertz", "Saka", "Martinelli", "Jesus",
]
AWAY_PLAYERS = [
    "Sanchez", "James", "Thiago_Silva", "Colwill", "Cucurella",
    "Caicedo", "Enzo", "Palmer", "Sterling", "Mudryk", "Jackson",
]

# ── simulate ─────────────────────────────────────────────────────────────────
pitch = Pitch()
sim = MatchSimulator(
    pitch,
    home_team=HOME,
    away_team=AWAY,
    home_players=HOME_PLAYERS,
    away_players=AWAY_PLAYERS,
    dt=3.5,
    seed=2024,
)

print("Simulating 90-minute match...")
result = sim.simulate_match(duration=5400.0)
print(f"  Score: {result.home_team} {result.home_score}–{result.away_score} {result.away_team}")
print(f"  Events: {len(result.events)}")
print(f"  xG: {result.total_xg}")
print(f"  Shots: {result.n_shots}")

# ── write CSV ────────────────────────────────────────────────────────────────
FIELDS = [
    "timestamp", "team", "player", "event_type", "x", "y",
    "end_x", "end_y", "foot", "to_player", "outcome",
    "pass_type", "body_part", "xg", "card",
    "set_piece_type", "action", "tackled_player", "fouled_player",
    "gk_action_type",
]


def event_to_row(ev) -> dict:
    row = {f: "" for f in FIELDS}
    row["timestamp"] = f"{ev.timestamp:.1f}"
    row["team"] = ev.team
    row["player"] = ev.player
    row["event_type"] = ev.event_type.value
    row["x"] = f"{ev.pitch_x:.1f}"
    row["y"] = f"{ev.pitch_y:.1f}"

    if isinstance(ev, Pass):
        row["end_x"] = f"{ev.dest_x:.1f}"
        row["end_y"] = f"{ev.dest_y:.1f}"
        row["foot"] = ev.foot.value
        row["to_player"] = ev.to_player
        row["outcome"] = ev.outcome.value
        row["pass_type"] = ev.pass_type.value
    elif isinstance(ev, Shot):
        row["end_x"] = f"{ev.target_x:.1f}"
        row["end_y"] = f"{ev.target_y:.1f}"
        row["body_part"] = ev.body_part.value
        row["xg"] = f"{ev.xg:.4f}"
        row["outcome"] = ev.outcome.value
    elif isinstance(ev, Dribble):
        row["end_x"] = f"{ev.end_x:.1f}"
        row["end_y"] = f"{ev.end_y:.1f}"
        row["foot"] = ev.foot.value
        row["outcome"] = ev.outcome.value
    elif isinstance(ev, Tackle):
        row["tackled_player"] = ev.tackled_player
        row["outcome"] = ev.outcome.value
    elif isinstance(ev, Header):
        row["end_x"] = f"{ev.dest_x:.1f}"
        row["end_y"] = f"{ev.dest_y:.1f}"
        row["action"] = ev.action
        row["outcome"] = ev.outcome.value
    elif isinstance(ev, Foul):
        row["fouled_player"] = ev.fouled_player
        row["outcome"] = ev.outcome.value
        row["card"] = ev.card or ""
    elif isinstance(ev, GoalkeeperAction):
        row["gk_action_type"] = ev.action_type.value
        row["end_x"] = f"{ev.dest_x:.1f}"
        row["end_y"] = f"{ev.dest_y:.1f}"
        row["outcome"] = ev.outcome.value
    elif isinstance(ev, SetPiece):
        row["set_piece_type"] = ev.set_piece_type.value
        row["end_x"] = f"{ev.dest_x:.1f}"
        row["end_y"] = f"{ev.dest_y:.1f}"
        row["foot"] = ev.foot.value if ev.foot else ""
        row["to_player"] = ev.to_player
        row["outcome"] = ev.outcome.value
    elif isinstance(ev, Touch):
        row["foot"] = ev.foot.value
        row["outcome"] = ev.outcome.value

    return row


with open(OUTPUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    for ev in result.events:
        writer.writerow(event_to_row(ev))

print(f"\nWritten: {OUTPUT}")
print(f"  Rows: {len(result.events)}")
