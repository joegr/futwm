"""
MatchSimulator — full-match or partial-possession Monte Carlo simulation.

Builds on TransitionModel to generate complete synthetic match event logs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .events import AnyEvent, EventType
from .pitch import Pitch
from .stochastic import TransitionModel
from .world_model import (
    BallState,
    GamePhase,
    GameState,
    PlayerState,
    PossessionPhase,
    WorldState,
)


@dataclass
class MatchResult:
    """Output of a simulated match."""

    home_team:    str
    away_team:    str
    home_score:   int
    away_score:   int
    events:       list[AnyEvent]
    total_xg:     dict[str, float]   # keyed by team name
    n_shots:      dict[str, int]

    def summary(self) -> str:
        return (
            f"{self.home_team} {self.home_score}–{self.away_score} {self.away_team}\n"
            f"Events: {len(self.events)}\n"
            f"xG  {self.home_team}: {self.total_xg.get(self.home_team, 0.0):.2f} | "
            f"{self.away_team}: {self.total_xg.get(self.away_team, 0.0):.2f}\n"
            f"Shots  {self.home_team}: {self.n_shots.get(self.home_team, 0)} | "
            f"{self.away_team}: {self.n_shots.get(self.away_team, 0)}"
        )


class MatchSimulator:
    """
    Simulate a full match (or an arbitrary duration) using the world model.

    Parameters
    ----------
    pitch       : Pitch
    home_team   : str
    away_team   : str
    home_players: List[str]     Player IDs for home team (11 expected).
    away_players: List[str]     Player IDs for away team (11 expected).
    dt          : float         Seconds per event step (default 4 s).
    seed        : Optional[int]
    """

    def __init__(
        self,
        pitch: Pitch,
        home_team:    str = "home",
        away_team:    str = "away",
        home_players: list[str] | None = None,
        away_players: list[str] | None = None,
        dt:           float = 4.0,
        seed:         int | None = None,
    ) -> None:
        self.pitch        = pitch
        self.home_team    = home_team
        self.away_team    = away_team
        self.home_players = home_players or [f"H{i}" for i in range(1, 12)]
        self.away_players = away_players or [f"A{i}" for i in range(1, 12)]
        self.dt           = dt
        self.model        = TransitionModel(pitch, seed=seed)
        self.rng          = self.model.rng

    # ── public API ────────────────────────────────────────────────────────────

    def simulate_match(
        self,
        duration: float = 5400.0,   # 90 minutes in seconds
    ) -> MatchResult:
        """
        Run a full match simulation.

        Parameters
        ----------
        duration : float   Total match time in seconds.
        """
        state  = self._initial_state()
        events: list[AnyEvent] = []

        t = 0.0
        while t < duration:
            event = self.model.sample(state)
            state.apply_event(event)
            events.append(event)

            # handle half-time
            if t >= 2700.0 and state.game_state.phase == GamePhase.FIRST_HALF:
                state.game_state.phase = GamePhase.SECOND_HALF
                state = self._restart_after_half(state)

            # handle out-of-play restarts
            if not state.ball.in_play:
                state = self._restart_dead_ball(state, t)

            # switch possession stochastically based on last event
            state = self._maybe_switch_possession(state, event)

            t += self.dt
            state.game_state.minute = t // 60
            state.game_state.second = t  % 60
            state.game_state.period_time = t

        return self._build_result(state, events)

    def simulate_possession(
        self,
        state: WorldState,
        max_events: int = 30,
    ) -> tuple[list[AnyEvent], WorldState]:
        """
        Simulate one possession chain until the ball is lost or out of play.

        Returns
        -------
        (events_in_chain, final_world_state)
        """
        events: list[AnyEvent] = []
        team   = state.possession_team

        for _ in range(max_events):
            event = self.model.sample(state)
            state.apply_event(event)
            events.append(event)

            # possession ended
            if state.possession_team != team:
                break
            if not state.ball.in_play:
                break
            if event.event_type in (EventType.SHOT, EventType.FOUL):
                break

        return events, state

    # ── state construction helpers ────────────────────────────────────────────

    def _initial_state(self) -> WorldState:
        gs = GameState(
            home_team=self.home_team,
            away_team=self.away_team,
            phase=GamePhase.FIRST_HALF,
        )
        ball = BallState(
            x=self.pitch.length / 2.0,
            y=self.pitch.width  / 2.0,
            in_play=True,
            possessing_team=self.home_team,
            possessing_player=self.home_players[9],  # striker takes kick-off
        )
        state = WorldState(
            pitch=self.pitch,
            game_state=gs,
            ball=ball,
            possession_team=self.home_team,
            possession_phase=PossessionPhase.BUILD_UP,
        )
        self._seed_players(state)
        return state

    def _seed_players(self, state: WorldState) -> None:
        """Place 11v11 players in rough formation positions."""
        # home — 4-4-2, attacking left → right
        home_positions = _formation_442(
            self.pitch.length, self.pitch.width, attacking_right=True
        )
        for pid, (x, y) in zip(self.home_players, home_positions):
            state.add_player(PlayerState(player_id=pid, team=self.home_team, x=x, y=y))

        # away — 4-4-2, attacking right → left (mirrored)
        away_positions = _formation_442(
            self.pitch.length, self.pitch.width, attacking_right=False
        )
        for pid, (x, y) in zip(self.away_players, away_positions):
            state.add_player(PlayerState(player_id=pid, team=self.away_team, x=x, y=y))

        # mark kick-off player as in possession
        carrier = state.get_player(self.home_players[9])
        if carrier:
            carrier.in_possession = True

    def _restart_after_half(self, state: WorldState) -> WorldState:
        state.ball.x = self.pitch.length / 2.0
        state.ball.y = self.pitch.width  / 2.0
        state.ball.in_play          = True
        state.possession_team       = self.away_team
        state.ball.possessing_team  = self.away_team
        state.ball.possessing_player = self.away_players[9]
        state.consecutive_passes    = 0
        state.possession_phase      = PossessionPhase.BUILD_UP
        return state

    def _restart_dead_ball(self, state: WorldState, t: float) -> WorldState:
        state.ball.in_play = True
        state.consecutive_passes = 0
        state.possession_phase = PossessionPhase.BUILD_UP

        # simple heuristic: give ball to the team that didn't just act
        last = state.event_history[-1] if state.event_history else None
        if last:
            new_team = (
                self.away_team if last.team == self.home_team else self.home_team
            )
        else:
            new_team = self.home_team

        state.possession_team        = new_team
        state.ball.possessing_team   = new_team
        players = state.players_for_team(new_team)
        if players:
            p = players[self.rng.integers(0, len(players))]
            state.ball.possessing_player = p.player_id
            state.ball.x = float(np.clip(p.x, 0, self.pitch.length))
            state.ball.y = float(np.clip(p.y, 0, self.pitch.width))
        return state

    def _maybe_switch_possession(
        self, state: WorldState, event: AnyEvent
    ) -> WorldState:
        """
        After a TACKLE (won) or failed PASS the model already switches
        possession inside apply_event. This method handles low-probability
        spontaneous turnovers during open play.
        """
        if event.event_type not in (EventType.PASS, EventType.DRIBBLE, EventType.TOUCH):
            return state
        # ~8 % random turnover per event step (mimics loose balls, interceptions)
        if self.rng.random() < 0.08:
            opp = (
                self.away_team if state.possession_team == self.home_team
                else self.home_team
            )
            opp_players = state.players_for_team(opp)
            if opp_players:
                p = opp_players[self.rng.integers(0, len(opp_players))]
                state.possession_team        = opp
                state.ball.possessing_team   = opp
                state.ball.possessing_player = p.player_id
                state.consecutive_passes     = 0
                # reset ball to the new team's defensive / middle zone
                state.ball.x = float(np.clip(p.x, 10, self.pitch.length - 10))
                state.ball.y = float(np.clip(p.y, 5, self.pitch.width - 5))
                state.possession_phase       = PossessionPhase.BUILD_UP
        return state

    # ── result aggregation ────────────────────────────────────────────────────

    @staticmethod
    def _build_result(state: WorldState, events: list[AnyEvent]) -> MatchResult:
        total_xg: dict[str, float] = {}
        n_shots:  dict[str, int]   = {}
        for e in events:
            if e.event_type == EventType.SHOT:
                xg = getattr(e, "xg", 0.0)
                total_xg[e.team] = total_xg.get(e.team, 0.0) + xg
                n_shots[e.team]  = n_shots.get(e.team, 0) + 1
        return MatchResult(
            home_team=state.game_state.home_team,
            away_team=state.game_state.away_team,
            home_score=state.game_state.home_score,
            away_score=state.game_state.away_score,
            events=events,
            total_xg=total_xg,
            n_shots=n_shots,
        )


# ── formation helper ──────────────────────────────────────────────────────────

def _formation_442(
    length: float, width: float, attacking_right: bool
) -> list[tuple[float, float]]:
    """
    Return 11 (x, y) positions for a 4-4-2 formation.
    attacking_right=True  → home side, GK at x≈0, strikers at x≈length*0.85
    attacking_right=False → away side, mirrored
    """
    width / 2.0
    positions_norm = [
        # GK
        (0.05, 0.50),
        # Defence (4)
        (0.20, 0.20), (0.20, 0.40), (0.20, 0.60), (0.20, 0.80),
        # Midfield (4)
        (0.45, 0.20), (0.45, 0.40), (0.45, 0.60), (0.45, 0.80),
        # Forwards (2)
        (0.80, 0.35), (0.80, 0.65),
    ]
    positions = [(nx * length, ny * width) for nx, ny in positions_norm]
    if not attacking_right:
        positions = [(length - x, y) for x, y in positions]
    return positions
