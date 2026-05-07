"""
Stochastic transition model.

Given a WorldState, produces a probability distribution over the next
event type and samples concrete event instances from that distribution.

Design
------
* **EventDistribution** – categorical + spatial parameters for one step.
* **TransitionModel**   – maps WorldState features → EventDistribution.

The parametric model uses hand-tuned priors that can be overridden by
fitting ``TransitionModel`` to observed event sequences via
``TransitionModel.fit()``.

Spatial sampling
----------------
Ball destinations are drawn from bivariate Gaussians whose means and
covariances are conditioned on:
  - possession phase / pitch zone
  - event type
  - distance to goal
  - pressure on ball
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .events import (
    AnyEvent,
    BodyPart,
    Dribble,
    EventOutcome,
    EventType,
    Foot,
    Foul,
    GoalkeeperAction,
    Header,
    Pass,
    PassType,
    SetPiece,
    SetPieceType,
    Shot,
    Tackle,
    Touch,
)
from .world_model import PossessionPhase, WorldState

Vector2D = tuple[float, float]

# ── prior event-type transition weights ──────────────────────────────────────
#
# Rows = current PossessionPhase (by .value index, 1-based → 0-offset below)
# Cols = next EventType (same order as _EVENT_TYPES list)
#
# These are *unnormalised* weights; TransitionModel normalises them and
# applies feature-based modifiers before sampling.

_EVENT_TYPES: list[EventType] = [
    EventType.PASS,
    EventType.TOUCH,
    EventType.SHOT,
    EventType.DRIBBLE,
    EventType.TACKLE,
    EventType.HEADER,
    EventType.FOUL,
    EventType.GOALKEEPER_ACTION,
    EventType.SET_PIECE,
]

# shape: (n_phases, n_event_types) — columns match _EVENT_TYPES order
# phases: BUILD_UP(0) PROGRESSION(1) FINAL_THIRD(2) TRANS_ATT(3)
#         TRANS_DEF(4) SP_ATT(5) SP_DEF(6) OUT_OF_PLAY(7)
_BASE_WEIGHTS = np.array([
    # PASS  TOUCH  SHOT  DRIB  TACKL  HEAD  FOUL  GK     SP
    [0.55,  0.22,  0.00, 0.04, 0.07,  0.02, 0.03, 0.05,  0.02],  # BUILD_UP
    [0.52,  0.20,  0.01, 0.08, 0.09,  0.03, 0.03, 0.01,  0.03],  # PROGRESSION
    [0.42,  0.20,  0.03, 0.10, 0.12,  0.04, 0.04, 0.01,  0.04],  # FINAL_THIRD
    [0.40,  0.20,  0.02, 0.10, 0.14,  0.04, 0.04, 0.01,  0.05],  # TRANS_ATT
    [0.42,  0.18,  0.02, 0.06, 0.18,  0.04, 0.04, 0.02,  0.04],  # TRANS_DEF
    [0.30,  0.12,  0.10, 0.04, 0.06,  0.18, 0.05, 0.01,  0.14],  # SP_ATT
    [0.25,  0.12,  0.03, 0.03, 0.18,  0.14, 0.06, 0.05,  0.14],  # SP_DEF
    [0.10,  0.05,  0.01, 0.02, 0.03,  0.03, 0.02, 0.04,  0.70],  # OUT_OF_PLAY
], dtype=float)


@dataclass
class EventDistribution:
    """
    Probability distribution over next-event types plus spatial parameters.

    Attributes
    ----------
    probs       : np.ndarray        shape (n_event_types,); sums to 1.
    event_types : List[EventType]   parallel to probs.
    dest_mean   : Vector2D          Expected ball destination.
    dest_cov    : np.ndarray        2×2 covariance for ball destination.
    outcome_p   : Dict[EventType, Dict[EventOutcome, float]]
                    Per-type outcome probabilities.
    """

    probs:       np.ndarray
    event_types: list[EventType]
    dest_mean:   Vector2D
    dest_cov:    np.ndarray
    outcome_p:   dict[EventType, dict[EventOutcome, float]] = field(default_factory=dict)

    def sample_event_type(self, rng: np.random.Generator) -> EventType:
        """Draw one event type from the categorical distribution."""
        idx = rng.choice(len(self.event_types), p=self.probs)
        return self.event_types[idx]

    def sample_destination(self, rng: np.random.Generator) -> Vector2D:
        """Sample a ball destination from the bivariate Gaussian."""
        pt = rng.multivariate_normal(list(self.dest_mean), self.dest_cov)
        return (float(pt[0]), float(pt[1]))

    def top_k(self, k: int = 3) -> list[tuple[EventType, float]]:
        """Return the k most probable event types with their probabilities."""
        order = np.argsort(self.probs)[::-1]
        return [(self.event_types[i], float(self.probs[i])) for i in order[:k]]


class TransitionModel:
    """
    Stochastic world-model transition function.

        P(e_{t+1} | world_state_t)

    Usage
    -----
    >>> model = TransitionModel(pitch)
    >>> dist  = model.predict(world_state)
    >>> event = model.sample(world_state)          # one concrete event
    >>> events = model.sample_n(world_state, n=5)  # n independent samples
    """

    def __init__(self, pitch, seed: int | None = None) -> None:
        self.pitch = pitch
        self.rng   = np.random.default_rng(seed)
        self._base_weights = _BASE_WEIGHTS.copy()

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, state: WorldState) -> EventDistribution:
        """Return the full next-event distribution for the given world state."""
        feats  = state.feature_vector()
        probs  = self._event_type_probs(state, feats)
        d_mean, d_cov = self._destination_params(state, feats)
        out_p  = self._outcome_probs(state, feats)
        return EventDistribution(
            probs=probs,
            event_types=_EVENT_TYPES,
            dest_mean=d_mean,
            dest_cov=d_cov,
            outcome_p=out_p,
        )

    def sample(self, state: WorldState) -> AnyEvent:
        """Sample one concrete event from the transition distribution."""
        dist       = self.predict(state)
        event_type = dist.sample_event_type(self.rng)
        dest       = dist.sample_destination(self.rng)
        dest       = self.pitch.clamp(*dest)
        return self._build_event(state, event_type, dest, dist)

    def sample_n(self, state: WorldState, n: int = 10) -> list[AnyEvent]:
        """Sample *n* independent next-event candidates (does not chain them)."""
        return [self.sample(state) for _ in range(n)]

    def fit(
        self,
        states: Sequence[WorldState],
        events: Sequence[AnyEvent],
    ) -> None:
        """
        Maximum-likelihood update of base weights from observed (state, event) pairs.

        Each observation increments the weight for (phase, event_type) by 1.
        Weights are normalised row-wise after accumulation.
        """
        n_phases = _BASE_WEIGHTS.shape[0]
        counts   = np.ones_like(self._base_weights) * 0.01  # Laplace smoothing

        for state, event in zip(states, events):
            phase_idx = (state.possession_phase.value - 1) % n_phases
            if event.event_type in _EVENT_TYPES:
                et_idx = _EVENT_TYPES.index(event.event_type)
                counts[phase_idx, et_idx] += 1.0

        self._base_weights = counts / counts.sum(axis=1, keepdims=True)

    # ── internal: event-type probabilities ───────────────────────────────────

    def _event_type_probs(
        self,
        state: WorldState,
        feats: dict,
    ) -> np.ndarray:
        phase_idx = (state.possession_phase.value - 1) % _BASE_WEIGHTS.shape[0]
        weights   = self._base_weights[phase_idx].copy()

        # Modifier: high pressure → more tackles, fewer dribbles, more fouls
        pressure = feats["pressure"]
        if pressure > 1.0:
            scale = min(pressure / 2.0, 2.5)
            weights[_EVENT_TYPES.index(EventType.TACKLE)] *= scale
            weights[_EVENT_TYPES.index(EventType.FOUL)]   *= scale * 0.6
            weights[_EVENT_TYPES.index(EventType.DRIBBLE)] /= scale

        # Modifier: close to goal → more shots, fewer passes (capped at 2×)
        dist_goal = feats["dist_to_goal"]
        if dist_goal < 18.0:
            boost = min(2.0, max(1.0, (18.0 - dist_goal) / 9.0))
            weights[_EVENT_TYPES.index(EventType.SHOT)] *= boost
            weights[_EVENT_TYPES.index(EventType.PASS)] /= max(boost * 0.3, 1.0)

        # Modifier: large goal angle → slightly more shots (capped)
        goal_angle = feats["goal_angle"]
        if goal_angle > 0.4:
            weights[_EVENT_TYPES.index(EventType.SHOT)] *= min(1.3, 1.0 + goal_angle * 0.4)

        # Modifier: long pass chains → increase chance of turnover events
        n_passes = feats["consecutive_passes"]
        if n_passes > 5:
            extra = min((n_passes - 5) * 0.05, 0.3)
            weights[_EVENT_TYPES.index(EventType.TACKLE)] += extra
            weights[_EVENT_TYPES.index(EventType.PASS)]   = max(
                weights[_EVENT_TYPES.index(EventType.PASS)] - extra, 0.05
            )

        # Normalise
        total = weights.sum()
        return weights / total

    # ── internal: destination Gaussian ───────────────────────────────────────

    def _destination_params(
        self,
        state: WorldState,
        feats: dict,
    ) -> tuple[Vector2D, np.ndarray]:
        """
        Return (mean, cov) for the bivariate Gaussian over ball destination.

        Mean is biased toward the opponent goal proportionally to the
        attacking phase; covariance reflects spatial spread by event context.
        """
        bx, by = state.ball.x, state.ball.y
        phase  = state.possession_phase

        goal_y  = self.pitch.width / 2.0

        if phase == PossessionPhase.BUILD_UP:
            mean_x = bx + 8.0
            mean_y = goal_y + (by - goal_y) * 0.3
            spread = 18.0
        elif phase == PossessionPhase.PROGRESSION:
            mean_x = bx + 12.0
            mean_y = goal_y + (by - goal_y) * 0.5
            spread = 14.0
        elif phase == PossessionPhase.FINAL_THIRD:
            mean_x = bx + 6.0
            mean_y = goal_y + (by - goal_y) * 0.7
            spread = 10.0
        elif phase in (PossessionPhase.TRANSITION_ATT,):
            mean_x = bx + 18.0
            mean_y = goal_y
            spread = 20.0
        else:
            mean_x = bx
            mean_y = by
            spread = 12.0

        mean_x = float(np.clip(mean_x, 0.0, self.pitch.length))
        mean_y = float(np.clip(mean_y, 0.0, self.pitch.width))

        cov = np.array([
            [spread ** 2,        0.0],
            [0.0,         (spread * 0.6) ** 2],
        ])
        return (mean_x, mean_y), cov

    # ── internal: outcome probabilities ──────────────────────────────────────

    def _outcome_probs(
        self,
        state: WorldState,
        feats: dict,
    ) -> dict[EventType, dict[EventOutcome, float]]:
        dist_goal  = feats["dist_to_goal"]
        goal_angle = feats["goal_angle"]
        pressure   = feats["pressure"]

        # Pass outcome
        pass_success = max(0.3, min(0.95, 0.80 - pressure * 0.08))
        pass_outcomes = {
            EventOutcome.SUCCESS:     pass_success,
            EventOutcome.INTERCEPTED: (1 - pass_success) * 0.6,
            EventOutcome.OUT_OF_PLAY: (1 - pass_success) * 0.4,
        }

        # Shot outcome  — simple logistic on distance & angle
        raw_xg  = _logistic(-0.12 * dist_goal + 0.8 * goal_angle - 1.0)
        xg      = float(np.clip(raw_xg, 0.02, 0.40))
        shot_outcomes = {
            EventOutcome.GOAL:       xg,
            EventOutcome.SAVED:      (1 - xg) * 0.55,
            EventOutcome.OFF_TARGET: (1 - xg) * 0.30,
            EventOutcome.BLOCKED:    (1 - xg) * 0.15,
        }

        # Dribble outcome
        drib_won = max(0.35, min(0.75, 0.55 - pressure * 0.06))
        dribble_outcomes = {
            EventOutcome.WON:  drib_won,
            EventOutcome.LOST: 1 - drib_won,
        }

        # Tackle outcome
        tackle_won = max(0.30, min(0.70, 0.50 + pressure * 0.04))
        tackle_outcomes = {
            EventOutcome.WON:  tackle_won,
            EventOutcome.LOST: 1 - tackle_won,
        }

        return {
            EventType.PASS:    pass_outcomes,
            EventType.SHOT:    shot_outcomes,
            EventType.DRIBBLE: dribble_outcomes,
            EventType.TACKLE:  tackle_outcomes,
        }

    # ── internal: concrete event construction ────────────────────────────────

    def _build_event(
        self,
        state: WorldState,
        event_type: EventType,
        dest: Vector2D,
        dist: EventDistribution,
    ) -> AnyEvent:
        team   = state.possession_team or state.game_state.home_team
        player = (state.ball.possessing_player or
                  next((p.player_id for p in state.players_for_team(team)), "unknown"))
        ts     = state.game_state.timestamp
        bx, by = state.ball.x, state.ball.y
        dx, dy = dest
        foot   = Foot.LEFT if self.rng.random() < 0.35 else Foot.RIGHT

        def sample_outcome(et: EventType) -> EventOutcome:
            op = dist.outcome_p.get(et)
            if not op:
                return EventOutcome.SUCCESS
            outcomes = list(op.keys())
            probs    = np.array(list(op.values()), dtype=float)
            probs   /= probs.sum()
            return outcomes[self.rng.choice(len(outcomes), p=probs)]

        if event_type == EventType.PASS:
            teammate = _pick_random_teammate(state, team, player, self.rng)
            return Pass(
                timestamp=ts, team=team, player=player,
                pitch_x=bx, pitch_y=by,
                foot=foot, to_player=teammate,
                origin_x=bx, origin_y=by,
                dest_x=dx, dest_y=dy,
                pass_type=_classify_pass(bx, by, dx, dy),
                outcome=sample_outcome(EventType.PASS),
            )

        if event_type == EventType.TOUCH:
            return Touch(
                timestamp=ts, team=team, player=player,
                pitch_x=bx, pitch_y=by,
                foot=foot,
                outcome=EventOutcome.SUCCESS,
            )

        if event_type == EventType.SHOT:
            feats = state.feature_vector()
            raw_xg = _logistic(
                -0.08 * feats["dist_to_goal"]
                + 1.2  * feats["goal_angle"]
                - 0.5
            )
            xg = float(np.clip(raw_xg, 0.02, 0.75))
            goal_y = self.pitch.width / 2.0
            target_y = float(np.clip(
                self.rng.normal(goal_y, 2.5),
                self.pitch._home_goal.post_y_left,
                self.pitch._home_goal.post_y_right,
            ))
            return Shot(
                timestamp=ts, team=team, player=player,
                pitch_x=bx, pitch_y=by,
                body_part=BodyPart.RIGHT_FOOT if foot == Foot.RIGHT else BodyPart.LEFT_FOOT,
                origin_x=bx, origin_y=by,
                target_x=self.pitch.length,
                target_y=target_y,
                target_z=float(np.clip(self.rng.normal(1.0, 0.6), 0.0, 2.44)),
                xg=xg,
                outcome=sample_outcome(EventType.SHOT),
            )

        if event_type == EventType.DRIBBLE:
            end_x = float(np.clip(bx + self.rng.normal(3.0, 2.0), 0, self.pitch.length))
            end_y = float(np.clip(by + self.rng.normal(0.0, 2.0), 0, self.pitch.width))
            return Dribble(
                timestamp=ts, team=team, player=player,
                pitch_x=bx, pitch_y=by,
                foot=foot,
                start_x=bx, start_y=by,
                end_x=end_x, end_y=end_y,
                outcome=sample_outcome(EventType.DRIBBLE),
            )

        if event_type == EventType.TACKLE:
            tackled = _pick_nearest_opponent(state, team, bx, by)
            return Tackle(
                timestamp=ts, team=team, player=player,
                pitch_x=bx, pitch_y=by,
                tackled_player=tackled,
                tackle_x=bx, tackle_y=by,
                outcome=sample_outcome(EventType.TACKLE),
            )

        if event_type == EventType.HEADER:
            return Header(
                timestamp=ts, team=team, player=player,
                pitch_x=bx, pitch_y=by,
                action="clearance" if state.possession_phase == PossessionPhase.BUILD_UP else "pass",
                dest_x=dx, dest_y=dy,
                outcome=EventOutcome.SUCCESS,
            )

        if event_type == EventType.FOUL:
            fouled = _pick_nearest_opponent(state, team, bx, by)
            return Foul(
                timestamp=ts, team=team, player=player,
                pitch_x=bx, pitch_y=by,
                fouled_player=fouled,
                foul_x=bx, foul_y=by,
                card=None,
            )

        if event_type == EventType.GOALKEEPER_ACTION:
            return GoalkeeperAction(
                timestamp=ts, team=team, player=player,
                pitch_x=bx, pitch_y=by,
                outcome=EventOutcome.SAVED,
            )

        # SET_PIECE fallback
        return SetPiece(
            timestamp=ts, team=team, player=player,
            pitch_x=bx, pitch_y=by,
            set_piece_type=SetPieceType.FREE_KICK,
            foot=foot,
            dest_x=dx, dest_y=dy,
            outcome=EventOutcome.SUCCESS,
        )


# ── module-level helpers ──────────────────────────────────────────────────────

def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _classify_pass(ox: float, oy: float, dx: float, dy: float) -> PassType:
    dist = math.hypot(dx - ox, dy - oy)
    if dist > 32.0:
        return PassType.LONG
    angle = math.degrees(math.atan2(dy - oy, dx - ox))
    if abs(angle) > 135:
        return PassType.BACK
    if dist < 15.0:
        return PassType.SHORT
    if abs(dy - oy) > 20.0:
        return PassType.CROSS
    return PassType.THROUGH


def _pick_random_teammate(
    state: WorldState,
    team: str,
    exclude: str,
    rng: np.random.Generator,
) -> str:
    teammates = [
        p.player_id for p in state.players_for_team(team)
        if p.player_id != exclude
    ]
    if not teammates:
        return "unknown"
    return str(rng.choice(teammates))


def _pick_nearest_opponent(
    state: WorldState,
    team: str,
    x: float,
    y: float,
) -> str:
    opp_team = (
        state.game_state.away_team
        if team == state.game_state.home_team
        else state.game_state.home_team
    )
    opponents = state.players_for_team(opp_team)
    if not opponents:
        return "unknown"
    return min(opponents, key=lambda p: math.hypot(p.x - x, p.y - y)).player_id
