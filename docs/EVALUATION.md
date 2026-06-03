# Codebase Evaluation: Soccer World Model

**Scope:** assess the `soccer-world-model` codebase against the full landscape
of soccer/football modelling business cases, identify gaps, and prioritise
extensions needed for "ubiquitous relevance" — i.e. a system that can plausibly
service any analytics, simulation, scouting, broadcast, betting, or operational
workflow in the sport.

**Method:** business cases were enumerated by surveying the public product
surfaces of the major analytics vendors (StatsBomb, Hudl/Wyscout, SkillCorner,
Opta/Stats Perform, InStat, Second Spectrum, Stats Bomb 360, Genius Sports),
betting model literature (Dixon-Coles, VAEP, xT, OBV, post-shot xG), federation
needs (FIFA Connect IDs, VAR review, competition formats), and broadcast
production needs (live graphics, augmented analytics).

---

## 1. Current capability matrix

Legend: ✅ implemented · ⚠️ partial · ❌ absent

| # | Business case                                              | Status | Module(s)                                                 |
|---|------------------------------------------------------------|--------|-----------------------------------------------------------|
| 1 | Versioned event ontology + JSON Schema                     | ✅     | `ontology`, `schema`                                      |
| 2 | Validated CSV / JSON interchange                           | ✅     | `interchange`, `csv_loader`                               |
| 3 | Spatial pitch model (zones, areas, pressure, distance)     | ✅     | `pitch`                                                   |
| 4 | World state (player / ball / game)                         | ✅     | `world_model`                                             |
| 5 | Stochastic transition model + fit/predict                  | ✅     | `stochastic`, `predictor`                                 |
| 6 | Match simulation (possession chains, full match)           | ✅     | `simulation`                                              |
| 7 | Team Elo, qualifier streaming, calibration                 | ✅     | `tournament/*`                                            |
| 8 | Per-shot xG                                                | ✅     | `schema.ShotEvent.xg`                                     |
| 9 | Post-shot xG / xS (goalkeeper save quality)                | ⚠️     | shot end-z exists, no model                               |
|10 | Expected Threat (xT) grid                                  | ❌     | —                                                         |
|11 | VAEP / Offensive Ball Value (OBV)                          | ❌     | —                                                         |
|12 | Possession value surfaces (Pitch Control)                  | ❌     | —                                                         |
|13 | Player tracking ingestion (25 Hz x/y/z)                    | ❌     | only event-level coords                                   |
|14 | Pose / biomechanics                                        | ❌     | —                                                         |
|15 | Formation / lineup / substitution / cards lifecycle        | ⚠️     | enum exists; no lineup state machine                      |
|16 | Multi-match competition (groups, knockout, league tables)  | ❌     | only pair-wise EloPredictor                               |
|17 | Bracket / draw simulation                                  | ❌     | —                                                         |
|18 | Set-piece specialist model                                 | ⚠️     | event type, no model                                      |
|19 | Goalkeeper-specific model                                  | ⚠️     | event type, no model                                      |
|20 | Referee / VAR decisions                                    | ❌     | —                                                         |
|21 | Live in-play win probability                               | ⚠️     | `predict_next`; no time-decay live model                  |
|22 | Betting market integration (odds, Kelly, CLV)              | ❌     | —                                                         |
|23 | Player rating / scouting / similarity                      | ❌     | —                                                         |
|24 | Injury / fitness / availability                            | ❌     | —                                                         |
|25 | Provider adapters (Opta, StatsBomb, Wyscout, Hudl, SC)     | ❌     | only own CSV/JSON                                         |
|26 | Real-time event streaming (Kafka, NATS, WebSocket)         | ⚠️     | `EloPredictor.update_from_stream`; no transport layer     |
|27 | Persistence (Postgres, ClickHouse, S3)                     | ❌     | —                                                         |
|28 | Competition registry (leagues, cups, transfer windows)     | ⚠️     | only WC2026                                               |
|29 | Player / club / venue / referee identity (master data)     | ❌     | —                                                         |
|30 | i18n, accented identifiers                                 | ✅     | unicode-fold in `tournament.teams.get_team`               |
|31 | Image / video stream analytics (CV)                        | ❌     | **scaffolded in `services/vision/` (this PR)**            |
|32 | Camera-to-pitch homography                                 | ❌     | **scaffolded in `services/vision/` (this PR)**            |
|33 | Spatial kernel methods (KDE, density, heat maps)           | ❌     | **scaffolded in `services/vision/kernels/` (this PR)**    |
|34 | Tactical pattern recognition (pressing triggers, blocks)   | ❌     | —                                                         |
|35 | Broadcast graphics export (SVG / WebGL / video overlays)   | ⚠️     | Flask dashboard renders; no exporter                      |
|36 | Reproducibility (seed, run id, schema version)             | ⚠️     | `seed=` accepted; no run manifest                         |

### Headline assessment

- **Core stochastic / event layer is solid** (rows 1–8).
- **Spatial intelligence above the event layer is the largest single gap**
  — xT, VAEP, pitch-control, and tracking data are foundational for all
  modern analytics products.
- **Identity & competition master data** is the second-largest gap — a
  production deployment cannot exist without canonical IDs for players,
  clubs, venues, referees, and seasons.
- **Vision input** has been entirely absent and is the bridge from "library
  for cleaned event data" to "system that observes the game". The vision
  microservice (this PR) closes that gap in scaffold form.
- **Bet / market** workflows (odds ingestion, Kelly staking, CLV tracking)
  are absent. Any business case that monetises predictions ultimately
  hinges on this layer.

---

## 2. Precision concerns in existing code

Issues found during this audit that affect numerical correctness or contract
clarity, ordered by severity.

### Severity: high

1. **`stochastic` Markov weights not normalised by phase** — the README
   describes "phase-conditioned Markov weights + feature modifiers" but the
   current implementation does not guarantee the modified distribution sums
   to 1 prior to sampling. Worth verifying that `EventDistribution` re-
   normalises *after* applying modifiers. (Action: add a unit test that
   randomises feature inputs and asserts `sum(probs) == 1 ± 1e-9`.)

2. **`Pitch.pressure_index` uses unspecified opponent radius** — pressure
   in the literature is typically Gaussian-weighted by distance with a
   ~2-3 m half-life. The current implementation should expose the radius
   as a parameter and default to a literature-justified value.

3. **xG values stored on `ShotEvent` are not enforced ∈ [0, 1]** — the
   Pydantic constraint is `ge=0, le=1` but `xg` is `Optional[float]`. If a
   provider supplies `xg=None`, downstream metrics (`brier`, `log_loss`)
   silently degrade. Action: define what `None` means or coerce.

### Severity: medium

4. **Coordinate frame ambiguity on direction of play** — events store raw
   `x ∈ [0, 105]` but possession can switch which side a team attacks
   (e.g., second half). The system currently relies on caller convention.
   This is fine for single-half analysis but breaks per-team rate stats
   computed across full matches. Action: add `attacking_direction` to
   `MatchMetadata` and a `Pitch.flip_for_team()` helper.

5. **`EloPredictor` `expected_goals_*` clamp at 0.05** — fine for the
   lower bound but no upper clamp; rating diffs > 600 produce xG > 5 which
   is unrealistic. Action: add a high-end cap or replace with a saturating
   transform.

6. **`load_baseline_matches()` returns dicts, not Pydantic models** — the
   rest of the codebase has migrated to Pydantic for type safety; the
   tournament dataset is the last hold-out. Action: introduce a
   `HistoricalMatch` Pydantic model.

### Severity: low

7. **No `__version__` exposure for sub-packages** — `soccer_model.__version__`
   exists but `soccer_model.tournament.__version__` does not. Useful for
   downstream services that pin against the tournament module alone.

8. **`Confederation(str, Enum)`** should be `StrEnum` on py≥3.11 (ruff UP042).
   Cosmetic.

9. **`evaluation.py` does `from .predictor import EloPredictor` inside the
   function** — necessary today to break a cycle. Cycle could be removed
   by moving `evaluate_predictions` from `predictor.py` to `evaluation.py`.

---

## 3. Recommended roadmap

### Immediate (this PR)
- ✅ Vision microservice scaffold (`services/vision/`)
- ✅ Spatial kernel methods (Gaussian KDE, homography) — usable today
- ✅ Evaluation document (this file)

### Short term (next 1–2 PRs)
- **Identity layer**: `soccer_model.identity` — `Player`, `Club`, `Venue`,
  `Referee`, `Season`, `Competition` Pydantic models with FIFA Connect / Opta
  ID slots.
- **xT (Expected Threat) grid** in `soccer_model.values.xt` — calibrated
  from any event stream.
- **Pitch control surface** in `soccer_model.values.pitch_control` —
  Spearman model, takes tracking frames if available, falls back to event
  positions.
- **Provider adapters** (`soccer_model.adapters.{statsbomb,opta,wyscout}`)
  — read each vendor's open-data format → `MatchEventStream`.

### Medium term
- **Competition / standings**: `soccer_model.competition` — group stage,
  knockout bracket, league-table simulation; integrates with EloPredictor.
- **VAEP / Action Values**: per-action value attribution → player ratings.
- **Live match service**: WebSocket transport for `WorldState` updates +
  live win-probability stream.
- **Market layer**: odds ingestion, Kelly fraction, CLV tracking.

### Long term
- **Tracking ingestion**: 25 Hz tracking data + linkage to events.
- **Pose / biomechanics**: skeletal data integration.
- **Tactical pattern miner**: pressing triggers, defensive blocks, formations.
- **Broadcast exporter**: SVG / WebGL overlays for live graphics.

---

## 4. Microservice topology

The vision microservice introduces the first **deployable boundary** in this
project. Recommended longer-term topology:

```
                ┌────────────────────────────┐
                │  soccer-vision-service     │  ← services/vision/  (THIS PR)
                │  (FastAPI · ML · CV)       │
                │                            │
   video ──────►│  /v1/frames/analyze        │── canonical events ──┐
                │  /v1/stream  (WebSocket)   │                      │
                │  /v1/sessions              │                      │
                └────────────────────────────┘                      ▼
                                                       ┌────────────────────────┐
                                                       │  soccer-world-model    │
                                                       │  (this package, core)  │
                                                       │  • schema · interchange│
                                                       │  • stochastic · sim    │
                                                       │  • tournament · elo    │
                                                       └────────────────────────┘
                                                                    │
                ┌────────────────────────────┐                      │
                │  soccer-market-service     │  (future)            │
                │  /v1/odds /v1/kelly        │ ◄────────────────────┘
                └────────────────────────────┘
                              │
                ┌────────────────────────────┐
                │  soccer-live-service       │  (future)
                │  WebSocket fan-out         │
                └────────────────────────────┘
```

Each service:
- Has its own `pyproject.toml` (independent release cadence)
- Depends on `soccer-world-model` for the *contract* (schema types)
- Communicates via JSON/Protobuf using `MatchEventStream` as the wire format
- Can be deployed as a separate container

This keeps the core library light (no torch, no cv2, no GPU) while letting
heavyweight ML services live behind clean HTTP boundaries.
