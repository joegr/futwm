"""
CSV data ingestion — parse match event CSV files into the world model.

Supported CSV schema (column names are case-insensitive, underscored):
─────────────────────────────────────────────────────────────────────
Required:
  timestamp   float   match seconds from kick-off
  team        str     team identifier
  player      str     player identifier
  event_type  str     one of: touch, pass, shot, dribble, tackle, header,
                      foul, goalkeeper_action, set_piece
  x           float   pitch x (metres)
  y           float   pitch y (metres)

Optional:
  end_x, end_y       float   destination coords (for pass, dribble, header, …)
  foot               str     left | right | both
  to_player          str     pass / set-piece recipient
  outcome            str     success | failure | goal | saved | …
  pass_type          str     short | long | through | cross | switch | back
  body_part          str     left_foot | right_foot | head | chest | other
  xg                 float   expected goals value
  card               str     yellow | red | (empty)
  set_piece_type     str     corner_kick | free_kick | penalty | …
  action             str     (for header: pass / shot / clearance / flick-on)
  gk_action_type     str     save | punch | claim | distribution | dive
  tackled_player     str
  fouled_player      str

Unknown columns are silently ignored.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from .events import (
    AnyEvent,
    BodyPart,
    Dribble,
    EventOutcome,
    EventType,
    Foot,
    Foul,
    GoalkeeperAction,
    GoalkeeperActionType,
    Header,
    Pass,
    PassType,
    SetPiece,
    SetPieceType,
    Shot,
    Tackle,
    Touch,
)
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

# ── column name normalisation ─────────────────────────────────────────────────

def _norm_key(k: str) -> str:
    return k.strip().lower().replace(" ", "_")


# ── enum look-ups ─────────────────────────────────────────────────────────────

_EVENT_TYPE_MAP: dict[str, EventType] = {e.value: e for e in EventType}
_OUTCOME_MAP:    dict[str, EventOutcome] = {e.value: e for e in EventOutcome}
_FOOT_MAP:       dict[str, Foot] = {e.value: e for e in Foot}
_BODY_PART_MAP:  dict[str, BodyPart] = {e.value: e for e in BodyPart}
_PASS_TYPE_MAP:  dict[str, PassType] = {e.value: e for e in PassType}
_SP_TYPE_MAP:    dict[str, SetPieceType] = {e.value: e for e in SetPieceType}
_GK_ACTION_MAP:  dict[str, GoalkeeperActionType] = {e.value: e for e in GoalkeeperActionType}


def _get(row: dict, key: str, default=""):
    return row.get(key, default) or default


def _float(row: dict, key: str, default: float = 0.0) -> float:
    v = row.get(key, "")
    if v == "" or v is None:
        return default
    return float(v)


# ── row → Event ──────────────────────────────────────────────────────────────

def _row_to_event(row: dict, pitch: Pitch, seq: int) -> AnyEvent | None:
    """Convert a single normalised CSV row dict to a concrete event object."""
    raw_type = _get(row, "event_type").strip().lower()
    etype = _EVENT_TYPE_MAP.get(raw_type)
    if etype is None:
        return None

    ts      = _float(row, "timestamp")
    team    = _get(row, "team")
    player  = _get(row, "player")
    px      = _float(row, "x")
    py      = _float(row, "y")
    ex      = _float(row, "end_x", px)
    ey      = _float(row, "end_y", py)
    foot    = _FOOT_MAP.get(_get(row, "foot").lower(), Foot.RIGHT)
    outcome = _OUTCOME_MAP.get(_get(row, "outcome").lower(), EventOutcome.SUCCESS)

    # clamp to pitch
    px, py = pitch.clamp(px, py)
    ex, ey = pitch.clamp(ex, ey)

    common = dict(
        timestamp=ts, team=team, player=player,
        pitch_x=px, pitch_y=py, sequence_id=seq,
    )

    if etype == EventType.TOUCH:
        return Touch(**common, foot=foot, outcome=outcome)

    if etype == EventType.PASS:
        to_p  = _get(row, "to_player")
        ptype = _PASS_TYPE_MAP.get(_get(row, "pass_type").lower(), PassType.SHORT)
        return Pass(
            **common, foot=foot, to_player=to_p,
            origin_x=px, origin_y=py, dest_x=ex, dest_y=ey,
            pass_type=ptype, outcome=outcome,
        )

    if etype == EventType.SHOT:
        bp = _BODY_PART_MAP.get(_get(row, "body_part").lower(), BodyPart.RIGHT_FOOT)
        xg = _float(row, "xg", 0.0)
        return Shot(
            **common, body_part=bp,
            origin_x=px, origin_y=py,
            target_x=ex, target_y=ey,
            target_z=_float(row, "target_z", 1.0),
            xg=xg, outcome=outcome,
        )

    if etype == EventType.DRIBBLE:
        return Dribble(
            **common, foot=foot,
            start_x=px, start_y=py, end_x=ex, end_y=ey,
            outcome=outcome,
        )

    if etype == EventType.TACKLE:
        return Tackle(
            **common,
            tackled_player=_get(row, "tackled_player"),
            tackle_x=px, tackle_y=py,
            outcome=outcome,
        )

    if etype == EventType.HEADER:
        return Header(
            **common,
            action=_get(row, "action", "clearance"),
            dest_x=ex, dest_y=ey,
            outcome=outcome,
            aerial_duel=_get(row, "aerial_duel").lower() in ("true", "1", "yes"),
        )

    if etype == EventType.FOUL:
        card_raw = _get(row, "card").lower()
        card = card_raw if card_raw in ("yellow", "red") else None
        return Foul(
            **common,
            fouled_player=_get(row, "fouled_player"),
            foul_x=px, foul_y=py,
            card=card,
        )

    if etype == EventType.GOALKEEPER_ACTION:
        gk_type = _GK_ACTION_MAP.get(
            _get(row, "gk_action_type").lower(), GoalkeeperActionType.SAVE
        )
        return GoalkeeperAction(
            **common, action_type=gk_type,
            foot=foot if _get(row, "foot") else None,
            dest_x=ex, dest_y=ey,
            outcome=outcome,
        )

    if etype == EventType.SET_PIECE:
        sp_type = _SP_TYPE_MAP.get(
            _get(row, "set_piece_type").lower(), SetPieceType.FREE_KICK
        )
        return SetPiece(
            **common, set_piece_type=sp_type,
            foot=foot if _get(row, "foot") else None,
            to_player=_get(row, "to_player"),
            dest_x=ex, dest_y=ey, outcome=outcome,
        )

    return None


# ── public API ────────────────────────────────────────────────────────────────

class MatchCSV:
    """
    Parsed match data from a CSV file.

    Attributes
    ----------
    events          : List[AnyEvent]
    teams           : List[str]             unique team names (usually 2)
    players         : Dict[str, str]        player_id → team
    pitch           : Pitch
    states          : List[WorldState]      world state *before* each event
    """

    def __init__(
        self,
        events: list[AnyEvent],
        teams: list[str],
        players: dict[str, str],
        pitch: Pitch,
        states: list[WorldState],
    ) -> None:
        self.events  = events
        self.teams   = teams
        self.players = players
        self.pitch   = pitch
        self.states  = states

    def to_json(self) -> list[dict]:
        """Serialise events as a list of plain dicts (JSON-friendly)."""
        out = []
        for i, ev in enumerate(self.events):
            d: dict = {
                "index":      i,
                "timestamp":  ev.timestamp,
                "team":       ev.team,
                "player":     ev.player,
                "event_type": ev.event_type.value,
                "x":          ev.pitch_x,
                "y":          ev.pitch_y,
            }
            if isinstance(ev, Pass):
                d.update(end_x=ev.dest_x, end_y=ev.dest_y,
                         to_player=ev.to_player, pass_type=ev.pass_type.value,
                         outcome=ev.outcome.value)
            elif isinstance(ev, Shot):
                d.update(end_x=ev.target_x, end_y=ev.target_y,
                         xg=ev.xg, outcome=ev.outcome.value,
                         body_part=ev.body_part.value)
            elif isinstance(ev, Dribble):
                d.update(end_x=ev.end_x, end_y=ev.end_y,
                         outcome=ev.outcome.value)
            elif isinstance(ev, Tackle):
                d.update(tackled_player=ev.tackled_player,
                         outcome=ev.outcome.value)
            elif isinstance(ev, Foul):
                d.update(fouled_player=ev.fouled_player,
                         card=ev.card, outcome=ev.outcome.value)
            elif isinstance(ev, Header):
                d.update(end_x=ev.dest_x, end_y=ev.dest_y,
                         action=ev.action, outcome=ev.outcome.value)
            elif isinstance(ev, (Touch, GoalkeeperAction, SetPiece)):
                d["outcome"] = ev.outcome.value
            out.append(d)
        return out

    def fit_model(self, model: TransitionModel) -> TransitionModel:
        """Fit the transition model on this match's (state, event) pairs."""
        model.fit(self.states, self.events)
        return model


def load_csv(
    source: str | Path | io.StringIO,
    pitch: Pitch | None = None,
    home_team: str | None = None,
    away_team: str | None = None,
) -> MatchCSV:
    """
    Parse a CSV source into a MatchCSV.

    Parameters
    ----------
    source      : file path, pathlib.Path, or StringIO of CSV text.
    pitch       : Pitch instance (defaults to standard 105 × 68).
    home_team   : override first-detected team as home.
    away_team   : override second-detected team as away.

    Returns
    -------
    MatchCSV with events, teams, players, pitch, and per-event WorldStates.
    """
    pitch = pitch or Pitch()

    # read raw rows
    if isinstance(source, io.StringIO):
        reader = csv.DictReader(source)
    else:
        with open(source, newline="", encoding="utf-8") as f:
            text = f.read()
        reader = csv.DictReader(io.StringIO(text))

    rows = [{_norm_key(k): v for k, v in row.items()} for row in reader]

    # detect teams & players
    team_set: dict[str, None] = {}
    player_map: dict[str, str] = {}
    for r in rows:
        t = _get(r, "team")
        p = _get(r, "player")
        if t:
            team_set[t] = None
        if p and t:
            player_map[p] = t

    teams = list(team_set.keys())[:2]
    if not teams:
        teams = ["home", "away"]
    ht = home_team or (teams[0] if len(teams) > 0 else "home")
    at = away_team or (teams[1] if len(teams) > 1 else "away")

    # parse events
    events: list[AnyEvent] = []
    for seq, r in enumerate(rows):
        ev = _row_to_event(r, pitch, seq)
        if ev is not None:
            events.append(ev)

    # build world-state timeline
    states: list[WorldState] = []
    state  = _initial_state(pitch, ht, at, player_map)
    for ev in events:
        states.append(state.snapshot())
        state.apply_event(ev)

    return MatchCSV(
        events=events,
        teams=[ht, at],
        players=player_map,
        pitch=pitch,
        states=states,
    )


def _initial_state(
    pitch: Pitch,
    home: str,
    away: str,
    player_map: dict[str, str],
) -> WorldState:
    gs = GameState(home_team=home, away_team=away, phase=GamePhase.FIRST_HALF)
    ball = BallState(
        x=pitch.length / 2.0, y=pitch.width / 2.0,
        in_play=True, possessing_team=home,
    )
    state = WorldState(
        pitch=pitch, game_state=gs, ball=ball,
        possession_team=home,
        possession_phase=PossessionPhase.BUILD_UP,
    )
    # place all known players at centre circle initially
    cx, cy = pitch.length / 2.0, pitch.width / 2.0
    for pid, team in player_map.items():
        state.add_player(PlayerState(player_id=pid, team=team, x=cx, y=cy))
    return state
