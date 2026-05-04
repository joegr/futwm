# Soccer World Model

Stochastic, world-model-based prediction of soccer events with a **formal event ontology** and validated **interchange formats** (JSON / CSV).

[![PyPI](https://img.shields.io/pypi/v/soccer-world-model)](https://pypi.org/project/soccer-world-model/)
[![Python](https://img.shields.io/pypi/pyversions/soccer-world-model)](https://pypi.org/project/soccer-world-model/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Install

```bash
pip install soccer-world-model            # core
pip install soccer-world-model[server]    # includes Flask dashboard
```

## Quick start

```python
import soccer_model as sm

# Load bundled sample match (Arsenal vs Chelsea, 58 events)
stream = sm.load_sample()
print(stream.metadata)          # MatchMetadata
print(stream.events[0])         # validated Pydantic event model

# Validate any data against the schema
result = sm.validate_event_stream(sm.to_json_dict(stream))
assert result.valid

# Export JSON Schema for downstream tooling
schema = sm.export_json_schema()

# Read your own CSV
stream = sm.read_csv("my_match.csv", home_team="TeamA", away_team="TeamB")

# Run stochastic predictions
predictor = sm.EventPredictor(sm.Pitch(), seed=42)
# ... (see Prediction section below)
```

## Architecture

```
soccer_model/
├── ontology.py      — SCHEMA_VERSION, enum taxonomy, field semantics, constraints
├── schema.py        — Pydantic v2 formal event models (canonical interchange layer)
├── interchange.py   — read/write JSON & CSV with strict validation
├── pitch.py         — Pitch bounds, zones, areas, spatial helpers
├── events.py        — Lightweight dataclass events (internal engine use)
├── world_model.py   — Mutable world state (PlayerState, BallState, GameState, WorldState)
├── stochastic.py    — Parametric transition model P(e_{t+1} | world_state_t)
├── predictor.py     — EventPredictor: top-k, beam search, Monte Carlo rollouts
└── simulation.py    — MatchSimulator: possession chains & full-match simulation
```

## Ontology & Schema

The library defines a **versioned event ontology** (`SCHEMA_VERSION = "0.1.0"`) as the single source of truth for all data entering or leaving the system.

### Event types

| Type                | Pydantic model       | Key fields |
|---------------------|----------------------|------------|
| `touch`             | `TouchEvent`         | `foot`, `outcome` |
| `pass`              | `PassEvent`          | `foot`, `to_player`, `end_x/y`, `pass_type`, `outcome`, `aerial` |
| `shot`              | `ShotEvent`          | `body_part`, `end_x/y`, `end_z`, `xg`, `outcome`, `first_time` |
| `dribble`           | `DribbleEvent`       | `foot`, `end_x/y`, `outcome` |
| `tackle`            | `TackleEvent`        | `tackled_player`, `outcome` |
| `header`            | `HeaderEvent`        | `header_action`, `end_x/y`, `outcome`, `aerial_duel` |
| `foul`              | `FoulEvent`          | `fouled_player`, `card`, `outcome` |
| `goalkeeper_action` | `GoalkeeperEvent`    | `gk_action_type`, `end_x/y`, `outcome` |
| `set_piece`         | `SetPieceEvent`      | `set_piece_type`, `to_player`, `end_x/y`, `outcome` |

Every model carries common fields: `schema_version`, `event_id` (UUID4), `timestamp`, `team`, `player`, `event_type`, `x`, `y`, `sequence_id`, `under_pressure`.

Outcome values are **constrained per event type** (e.g., shots can only be `goal`, `saved`, `off_target`, `blocked`).

### Coordinate system

FIFA-standard reference frame: `x ∈ [0, 105]` (longitudinal), `y ∈ [0, 68]` (lateral), origin at bottom-left facing from behind the home goal. All coordinates are validated at construction time.

### JSON Schema export

```python
schema = sm.export_json_schema()       # full MatchEventStream schema
schemas = sm.export_event_schemas()    # per-event-type schemas
```

Pre-exported schemas are in the `schema/` directory.

## Interchange formats

### JSON (canonical, lossless)

```python
sm.write_json(stream, "match.json")
stream = sm.read_json("match.json")
```

### CSV (tabular, widely compatible)

```python
sm.write_csv(stream, "match.csv")
stream = sm.read_csv("match.csv", home_team="Arsenal", away_team="Chelsea")
```

CSV files include a metadata comment header: `# schema_version=0.1.0,home_team=...,away_team=...`

### Validation

```python
result = sm.validate_event_stream(data_dict)
print(result.valid, result.errors, result.warnings)

result = sm.validate_single_event(event_dict)
```

## Pitch

FIFA standard (105 × 68 m by default). Key methods:

```python
pitch = Pitch()                         # or Pitch(length=105, width=68)
pitch.contains(x, y)                    # bounds check
pitch.validate(x, y)                    # raises ValueError if out of bounds
pitch.zone(x)                           # DEFENSIVE / MIDDLE / ATTACKING_THIRD
pitch.area(x, y)                        # OWN_PENALTY_AREA, OPP_GOAL_AREA, …
pitch.distance_to_goal(x, y)
pitch.goal_angle(x, y)
pitch.pressure_index(pos, opponent_positions)
pitch.normalise(x, y)                   # → [0,1]²
```

## Stochastic Model

`TransitionModel` maps `WorldState` → `EventDistribution`:

- **Event-type probs**: phase-conditioned Markov weights + feature modifiers
  (pressure, distance to goal, goal angle, consecutive passes)
- **Destination**: bivariate Gaussian conditioned on phase/zone, clamped to pitch
- **Outcomes**: per-event-type Bernoulli / categorical with feature-based parameters

Fit to observed data:
```python
model = TransitionModel(pitch, seed=42)
model.fit(list_of_world_states, list_of_events)
```

## Prediction

```python
predictor = EventPredictor(pitch, seed=42)

# Top-k next events
pred = predictor.predict_next(state, top_k=3)
print(pred.summary())

# Beam search over event-type sequences
beams = predictor.beam_search(state, horizon=4, beam_width=3)

# Monte Carlo rollouts
seqs = predictor.predict_sequence(state, horizon=6, samples=50)
xg   = predictor.cumulative_xg(seqs)
```

## Simulation

```python
sim    = MatchSimulator(pitch, home_team="Arsenal", away_team="Chelsea", seed=7)
result = sim.simulate_match(duration=5400.0)   # 90 min
print(result.summary())

chain, final_state = sim.simulate_possession(state, max_events=20)
```
