"""
Demo — stochastic soccer world-model.

Run from the repo root:
    python -m examples.demo
or
    python examples/demo.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from soccer_model import (
    Pitch, PitchZone,
    Pass, Shot,
    PlayerState, BallState, GameState, WorldState,
    TransitionModel, EventPredictor,
    MatchSimulator,
)
from soccer_model.world_model import PossessionPhase, GamePhase


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── 1. Pitch geometry ─────────────────────────────────────────────────────────

section("1. Pitch geometry")

pitch = Pitch()
print(pitch)
print(f"  Area at (10, 34)  : {pitch.area(10, 34).name}")
print(f"  Area at (95, 34)  : {pitch.area(95, 34).name}")
print(f"  Area at (52, 34)  : {pitch.area(52, 34).name}")
print(f"  Zone at x=20      : {pitch.zone(20).name}")
print(f"  Zone at x=80      : {pitch.zone(80).name}")
print(f"  Dist to away goal from (80, 34): {pitch.distance_to_goal(80, 34):.2f} m")
print(f"  Goal angle from (80, 34)        : {pitch.goal_angle(80, 34):.4f} rad")
print(f"  Normalise (52.5, 34)            : {pitch.normalise(52.5, 34)}")


# ── 2. Manual event construction ─────────────────────────────────────────────

section("2. Manual event construction")

from soccer_model.events import Foot, EventOutcome, PassType, BodyPart

t1 = Pass(
    timestamp=120.0,
    team="home", player="H10",
    pitch_x=55.0, pitch_y=34.0,
    foot=Foot.RIGHT,
    to_player="H9",
    origin_x=55.0, origin_y=34.0,
    dest_x=82.0,   dest_y=28.0,
    pass_type=PassType.THROUGH,
    outcome=EventOutcome.SUCCESS,
)
print(f"  Pass  : {t1.player} → {t1.to_player}  |  vector {t1.vector()}  |  {t1.outcome.value}")

s1 = Shot(
    timestamp=125.0,
    team="home", player="H9",
    pitch_x=82.0, pitch_y=28.0,
    body_part=BodyPart.RIGHT_FOOT,
    origin_x=82.0, origin_y=28.0,
    target_x=105.0, target_y=35.0, target_z=1.2,
    xg=0.18,
    outcome=EventOutcome.SAVED,
)
print(f"  Shot  : {s1.player}  xG={s1.xg:.2f}  outcome={s1.outcome.value}")


# ── 3. World state + feature vector ──────────────────────────────────────────

section("3. World state + feature vector")

gs    = GameState(home_team="home", away_team="away")
ball  = BallState(x=82.0, y=28.0, in_play=True, possessing_team="home", possessing_player="H9")
state = WorldState(
    pitch=pitch,
    game_state=gs,
    ball=ball,
    possession_team="home",
    possession_phase=PossessionPhase.FINAL_THIRD,
)

# add a minimal squad
for pid, (x, y) in zip(
    ["H9", "H10", "H7", "A5", "A6"],
    [(82, 28), (65, 34), (78, 44), (88, 30), (84, 38)],
):
    team = "home" if pid.startswith("H") else "away"
    state.add_player(PlayerState(player_id=pid, team=team, x=x, y=y))

feats = state.feature_vector()
print("  Feature vector:")
for k, v in feats.items():
    print(f"    {k:25s}: {v:.4f}")

print(f"  Pressure on ball  : {state.pressure_on_ball():.4f}")
print(f"  Pitch area        : {state.pitch_area().name}")


# ── 4. Single-step prediction ─────────────────────────────────────────────────

section("4. Single-step prediction (EventPredictor)")

predictor = EventPredictor(pitch, seed=42)
prediction = predictor.predict_next(state, top_k=4)
print(prediction.summary())


# ── 5. Beam search over event-type sequences ──────────────────────────────────

section("5. Beam search (horizon=4, beam_width=3)")

beams = predictor.beam_search(state, horizon=4, beam_width=3)
for i, (log_p, seq) in enumerate(beams, 1):
    types = " → ".join(et.value for et in seq)
    print(f"  Beam {i}  log_p={log_p:.2f}  {types}")


# ── 6. Monte Carlo rollout ─────────────────────────────────────────────────────

section("6. Monte Carlo sequence rollout (horizon=6, samples=10)")

sequences = predictor.predict_sequence(state, horizon=6, samples=10)
print(f"  Generated {len(sequences)} sequences")
for i, seq in enumerate(sequences[:3], 1):
    types = " → ".join(e.event_type.value for e in seq)
    print(f"  Seq {i}: {types}")

xg_stats = predictor.cumulative_xg(sequences)
print(f"  xG across sequences: mean={xg_stats['mean_xg']:.4f}  "
      f"max={xg_stats['max_xg']:.4f}  std={xg_stats['std_xg']:.4f}")


# ── 7. Possession chain simulation ────────────────────────────────────────────

section("7. Possession chain simulation")

sim    = MatchSimulator(pitch, home_team="home", away_team="away", seed=99)
state2 = sim._initial_state()
chain, final_state = sim.simulate_possession(state2, max_events=15)
print(f"  Possession chain length : {len(chain)} events")
print(f"  Final possession        : {final_state.possession_team}")
for e in chain:
    loc = f"({e.pitch_x:.0f},{e.pitch_y:.0f})"
    out = getattr(e, "outcome", None)
    out_str = f"  [{out.value}]" if out else ""
    print(f"    {e.event_type.value:22s} {loc}{out_str}")


# ── 8. Short match simulation (15 minutes) ────────────────────────────────────

section("8. Short match simulation (15 min)")

result = sim.simulate_match(duration=900.0)   # 15 min
print(result.summary())
print(f"  Total events : {len(result.events)}")

# event type breakdown
from collections import Counter
counts = Counter(e.event_type.value for e in result.events)
for et, n in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"    {et:22s}: {n}")
