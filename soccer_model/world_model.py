"""
World model — mutable snapshot of full match state at any instant.

The WorldState is the input consumed by the stochastic transition model.
It is updated in-place after each observed (or sampled) event.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto

from .events import (
    AnyEvent,
    Dribble,
    EventOutcome,
    Foul,
    GoalkeeperAction,
    Header,
    Pass,
    SetPiece,
    Shot,
    Tackle,
    Touch,
)
from .pitch import Pitch, PitchArea, PitchZone

Vector2D = tuple[float, float]


# ── enumerations ─────────────────────────────────────────────────────────────

class GamePhase(Enum):
    PRE_MATCH        = auto()
    FIRST_HALF       = auto()
    HALF_TIME        = auto()
    SECOND_HALF      = auto()
    EXTRA_TIME_1     = auto()
    EXTRA_TIME_2     = auto()
    PENALTY_SHOOTOUT = auto()
    POST_MATCH       = auto()


class PossessionPhase(Enum):
    """Tactical possession state."""
    BUILD_UP         = auto()   # deep buildup from GK / CB
    PROGRESSION      = auto()   # moving ball through midfield
    FINAL_THIRD      = auto()   # attacking third possession
    TRANSITION_ATT   = auto()   # quick counter-attack
    TRANSITION_DEF   = auto()   # recovering from losing ball
    SET_PIECE_ATT    = auto()
    SET_PIECE_DEF    = auto()
    OUT_OF_PLAY      = auto()


# ── player & ball state ───────────────────────────────────────────────────────

@dataclass
class PlayerState:
    """
    Physical and tactical state of a single player at one instant.

    Parameters
    ----------
    player_id       : str
    team            : str
    x, y            : float         Current position on pitch (metres).
    vx, vy          : float         Velocity components (m/s).
    stamina         : float         Remaining stamina ∈ [0, 1].
    in_possession   : bool          True if this player currently has the ball.
    pressuring      : bool          True if actively pressing an opponent.
    """

    player_id:      str
    team:           str
    x:              float   = 0.0
    y:              float   = 0.0
    vx:             float   = 0.0
    vy:             float   = 0.0
    stamina:        float   = 1.0
    in_possession:  bool    = False
    pressuring:     bool    = False

    def position(self) -> Vector2D:
        return (self.x, self.y)

    def velocity(self) -> Vector2D:
        return (self.vx, self.vy)

    def speed(self) -> float:
        import math
        return math.hypot(self.vx, self.vy)

    def move_to(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


@dataclass
class BallState:
    """
    State of the ball.

    Parameters
    ----------
    x, y, z         : float         3-D position (z = height in metres).
    vx, vy, vz      : float         3-D velocity (m/s).
    in_play         : bool          False when out-of-bounds or dead ball.
    possessing_team : Optional[str] Team currently in possession; None if contested.
    possessing_player : Optional[str]
    """

    x:                   float           = 52.5
    y:                   float           = 34.0
    z:                   float           = 0.0
    vx:                  float           = 0.0
    vy:                  float           = 0.0
    vz:                  float           = 0.0
    in_play:             bool            = True
    possessing_team:     str | None   = None
    possessing_player:   str | None   = None

    def position_2d(self) -> Vector2D:
        return (self.x, self.y)

    def speed(self) -> float:
        import math
        return math.hypot(self.vx, self.vy)

    def is_airborne(self) -> float:
        return self.z > 0.3


@dataclass
class GameState:
    """
    Match-level bookkeeping.

    Parameters
    ----------
    home_team, away_team : str
    home_score, away_score : int
    minute              : float         Match minute (0–90+).
    second              : float         Seconds within the minute.
    phase               : GamePhase
    period_time         : float         Seconds elapsed in the current half.
    """

    home_team:       str        = "home"
    away_team:       str        = "away"
    home_score:      int        = 0
    away_score:      int        = 0
    minute:          float      = 0.0
    second:          float      = 0.0
    phase:           GamePhase  = GamePhase.FIRST_HALF
    period_time:     float      = 0.0

    @property
    def timestamp(self) -> float:
        """Total match time in seconds."""
        return self.minute * 60.0 + self.second

    def score_diff(self, team: str) -> int:
        """Positive if `team` is winning."""
        if team == self.home_team:
            return self.home_score - self.away_score
        return self.away_score - self.home_score


# ── world state ───────────────────────────────────────────────────────────────

@dataclass
class WorldState:
    """
    Complete description of the match world at a single moment.

    This is the *state* fed into the TransitionModel.

    Parameters
    ----------
    pitch               : Pitch
    game_state          : GameState
    ball                : BallState
    players             : Dict[str, PlayerState]    keyed by player_id
    possession_team     : Optional[str]
    possession_phase    : PossessionPhase
    event_history       : List[AnyEvent]            recent events (oldest first)
    consecutive_passes  : int                       passes in current possession chain
    """

    pitch:               Pitch
    game_state:          GameState                         = field(default_factory=GameState)
    ball:                BallState                         = field(default_factory=BallState)
    players:             dict[str, PlayerState]            = field(default_factory=dict)
    possession_team:     str | None                     = None
    possession_phase:    PossessionPhase                   = PossessionPhase.BUILD_UP
    event_history:       list[AnyEvent]                    = field(default_factory=list)
    consecutive_passes:  int                               = 0

    # ── helpers ───────────────────────────────────────────────────────────────

    def add_player(self, player: PlayerState) -> None:
        self.players[player.player_id] = player

    def get_player(self, player_id: str) -> PlayerState | None:
        return self.players.get(player_id)

    def players_for_team(self, team: str) -> list[PlayerState]:
        return [p for p in self.players.values() if p.team == team]

    def opponent_positions(self, team: str) -> list[Vector2D]:
        """All positions of opponents relative to `team`."""
        opp = self.game_state.away_team if team == self.game_state.home_team else self.game_state.home_team
        return [p.position() for p in self.players_for_team(opp)]

    def ball_carrier(self) -> PlayerState | None:
        return self.players.get(self.ball.possessing_player or "")

    def pitch_zone(self, attacking_direction: int = 1) -> PitchZone | None:
        if not self.ball.in_play:
            return None
        return self.pitch.zone(self.ball.x, attacking_direction)

    def pitch_area(self) -> PitchArea | None:
        if not self.ball.in_play:
            return None
        return self.pitch.area(self.ball.x, self.ball.y)

    def pressure_on_ball(self) -> float:
        if self.ball.possessing_team is None:
            return 0.0
        opp_positions = self.opponent_positions(self.ball.possessing_team)
        return self.pitch.pressure_index(self.ball.position_2d(), opp_positions)

    def snapshot(self) -> WorldState:
        """Deep copy of the current state (useful for Monte Carlo rollouts)."""
        return copy.deepcopy(self)

    # ── state transition ──────────────────────────────────────────────────────

    def apply_event(self, event: AnyEvent) -> None:
        """
        Mutate the world state to reflect the given event.

        Handles ball position, possession transfer, score updates,
        and phase transitions.
        """
        self.event_history.append(event)
        self.game_state.minute  = event.timestamp // 60
        self.game_state.second  = event.timestamp  % 60
        self.game_state.period_time = event.timestamp

        if isinstance(event, (Touch, Pass, Dribble, Shot, Header, GoalkeeperAction, SetPiece)):
            self.ball.x = event.pitch_x
            self.ball.y = event.pitch_y
            self.ball.in_play = True
            self.ball.possessing_team   = event.team
            self.ball.possessing_player = event.player
            self.possession_team        = event.team
            if p := self.get_player(event.player):
                p.in_possession = True
                p.move_to(event.pitch_x, event.pitch_y)

        if isinstance(event, Pass):
            self._apply_pass(event)
        elif isinstance(event, Shot):
            self._apply_shot(event)
        elif isinstance(event, Tackle):
            self._apply_tackle(event)
        elif isinstance(event, Foul):
            self._apply_foul(event)

        self._update_possession_phase()

    def _apply_pass(self, event: Pass) -> None:
        if event.outcome == EventOutcome.SUCCESS:
            self.consecutive_passes += 1
            if recv := self.get_player(event.to_player):
                recv.in_possession = True
                recv.move_to(event.dest_x, event.dest_y)
            self.ball.possessing_player = event.to_player
            self.ball.x = event.dest_x
            self.ball.y = event.dest_y
        else:
            self.consecutive_passes = 0
            self.ball.possessing_team   = None
            self.ball.possessing_player = None

    def _apply_shot(self, event: Shot) -> None:
        self.consecutive_passes = 0
        if event.outcome == EventOutcome.GOAL:
            if event.team == self.game_state.home_team:
                self.game_state.home_score += 1
            else:
                self.game_state.away_score += 1
            self.ball.in_play = False
            self.possession_phase = PossessionPhase.OUT_OF_PLAY
        elif event.outcome in (EventOutcome.SAVED, EventOutcome.OFF_TARGET, EventOutcome.BLOCKED):
            self.ball.possessing_team   = None
            self.ball.possessing_player = None

    def _apply_tackle(self, event: Tackle) -> None:
        if event.outcome == EventOutcome.WON:
            self.consecutive_passes = 0
            self.possession_team = event.team
            self.ball.possessing_team   = event.team
            self.ball.possessing_player = event.player
            self.ball.x = event.tackle_x
            self.ball.y = event.tackle_y

    def _apply_foul(self, event: Foul) -> None:
        self.consecutive_passes = 0
        self.ball.in_play = False
        self.possession_phase = PossessionPhase.SET_PIECE_DEF

    def _update_possession_phase(self) -> None:
        if not self.ball.in_play:
            return
        zone = self.pitch_zone()
        area = self.pitch_area()

        if area in (PitchArea.OPP_PENALTY_AREA, PitchArea.OPP_GOAL_AREA):
            self.possession_phase = PossessionPhase.FINAL_THIRD
        elif zone == PitchZone.DEFENSIVE_THIRD:
            self.possession_phase = PossessionPhase.BUILD_UP
        elif zone == PitchZone.MIDDLE_THIRD:
            self.possession_phase = PossessionPhase.PROGRESSION
        elif zone == PitchZone.ATTACKING_THIRD:
            self.possession_phase = PossessionPhase.FINAL_THIRD

    # ── feature extraction (consumed by TransitionModel) ─────────────────────

    def feature_vector(self) -> dict[str, float]:
        """
        Compact numerical feature set describing the current world state.
        Used as input to the stochastic transition model.
        """
        ball_x, ball_y = self.ball.x, self.ball.y
        dist_goal  = self.pitch.distance_to_goal(ball_x, ball_y)
        goal_angle = self.pitch.goal_angle(ball_x, ball_y)
        pressure   = self.pressure_on_ball()

        n_att = len(self.players_for_team(self.possession_team or ""))
        n_def = sum(1 for t in [self.game_state.home_team, self.game_state.away_team]
                    if t != self.possession_team
                    for _ in self.players_for_team(t))

        return {
            "ball_x":            ball_x,
            "ball_y":            ball_y,
            "ball_norm_x":       ball_x / self.pitch.length,
            "ball_norm_y":       ball_y / self.pitch.width,
            "dist_to_goal":      dist_goal,
            "goal_angle":        goal_angle,
            "pressure":          pressure,
            "consecutive_passes": float(self.consecutive_passes),
            "score_diff":        float(self.game_state.score_diff(self.possession_team or "")),
            "period_time":       self.game_state.period_time,
            "n_attackers":       float(n_att),
            "n_defenders":       float(n_def),
            "possession_phase":  float(self.possession_phase.value),
        }
