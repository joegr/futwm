"""
Event schema for soccer match modeling.

All coordinate fields (x, y) are in metres relative to a Pitch instance.
Events are immutable dataclasses; use ``replace()`` (dataclasses.replace) to
derive modified copies.

Event hierarchy
---------------
BaseEvent
├── Touch
├── Pass
├── Shot
├── Dribble
├── Tackle
├── Header
├── Foul
├── GoalkeeperAction
└── SetPiece
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum

Vector2D = tuple[float, float]


# ── enumerations ─────────────────────────────────────────────────────────────

class EventType(Enum):
    TOUCH              = "touch"
    PASS               = "pass"
    SHOT               = "shot"
    DRIBBLE            = "dribble"
    TACKLE             = "tackle"
    HEADER             = "header"
    FOUL               = "foul"
    GOALKEEPER_ACTION  = "goalkeeper_action"
    SET_PIECE          = "set_piece"


class Foot(Enum):
    LEFT  = "left"
    RIGHT = "right"
    BOTH  = "both"   # used for throw-ins / two-footed


class BodyPart(Enum):
    LEFT_FOOT  = "left_foot"
    RIGHT_FOOT = "right_foot"
    HEAD       = "head"
    CHEST      = "chest"
    OTHER      = "other"


class EventOutcome(Enum):
    SUCCESS          = "success"
    FAILURE          = "failure"
    BLOCKED          = "blocked"
    INTERCEPTED      = "intercepted"
    OUT_OF_PLAY      = "out_of_play"
    GOAL             = "goal"
    SAVED            = "saved"
    OFF_TARGET       = "off_target"
    FOUL_COMMITTED   = "foul_committed"
    WON              = "won"
    LOST             = "lost"


class SetPieceType(Enum):
    CORNER_KICK      = "corner_kick"
    FREE_KICK        = "free_kick"
    PENALTY          = "penalty"
    THROW_IN         = "throw_in"
    GOAL_KICK        = "goal_kick"
    KICK_OFF         = "kick_off"


class GoalkeeperActionType(Enum):
    SAVE             = "save"
    PUNCH            = "punch"
    CLAIM            = "claim"
    DISTRIBUTION     = "distribution"
    DIVE             = "dive"


class PassType(Enum):
    SHORT            = "short"
    LONG             = "long"
    THROUGH          = "through"
    CROSS            = "cross"
    SWITCH           = "switch"
    BACK             = "back"


# ── base event ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BaseEvent:
    """
    Common fields shared by every soccer event.

    Parameters
    ----------
    event_id : str
        Unique identifier (UUID4 by default).
    timestamp : float
        Match time in seconds from kick-off.
    team : str
        Team identifier for the player performing the action.
    player : str
        Player identifier (e.g. jersey number or name slug).
    pitch_x : float
        x coordinate (metres) where the event starts.
    pitch_y : float
        y coordinate (metres) where the event starts.
    event_type : EventType
        Discriminator tag; set automatically by each subclass.
    sequence_id : int
        Monotonically increasing sequence number within a possession chain.
    under_pressure : bool
        Whether the acting player was under immediate opponent pressure.
    """

    event_id:       str        = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:      float      = 0.0
    team:           str        = ""
    player:         str        = ""
    pitch_x:        float      = 0.0
    pitch_y:        float      = 0.0
    event_type:     EventType  = EventType.TOUCH
    sequence_id:    int        = 0
    under_pressure: bool       = False

    def position(self) -> Vector2D:
        """Return (pitch_x, pitch_y) as a 2-tuple."""
        return (self.pitch_x, self.pitch_y)


# ── concrete event types ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Touch(BaseEvent):
    """
    Any first touch / ball reception / carry that does not advance to a
    specific action type.

    Fields
    ------
    foot : Foot
        Foot used for the touch.
    """

    event_type: EventType       = field(default=EventType.TOUCH, init=False)
    foot:       Foot            = Foot.RIGHT
    outcome:    EventOutcome    = EventOutcome.SUCCESS


@dataclass(frozen=True)
class Pass(BaseEvent):
    """
    Deliberate kick to a team-mate.

    Fields
    ------
    foot        : Foot              Foot used.
    to_player   : str               Target player identifier.
    origin_x    : float             x at pass origin (same as pitch_x; kept explicit).
    origin_y    : float             y at pass origin.
    dest_x      : float             Intended destination x.
    dest_y      : float             Intended destination y.
    pass_type   : PassType          Tactical classification.
    outcome     : EventOutcome      SUCCESS, INTERCEPTED, OUT_OF_PLAY, etc.
    aerial      : bool              Whether the pass travelled through the air.
    """

    event_type:  EventType     = field(default=EventType.PASS, init=False)
    foot:        Foot          = Foot.RIGHT
    to_player:   str           = ""
    origin_x:    float         = 0.0
    origin_y:    float         = 0.0
    dest_x:      float         = 0.0
    dest_y:      float         = 0.0
    pass_type:   PassType      = PassType.SHORT
    outcome:     EventOutcome  = EventOutcome.SUCCESS
    aerial:      bool          = False

    def vector(self) -> Vector2D:
        """Displacement vector (Δx, Δy) from origin to destination."""
        return (self.dest_x - self.origin_x, self.dest_y - self.origin_y)


@dataclass(frozen=True)
class Shot(BaseEvent):
    """
    Attempt on goal.

    Fields
    ------
    body_part   : BodyPart          Striking body part.
    origin_x    : float             x at shot origin.
    origin_y    : float             y at shot origin.
    target_x    : float             Goal-frame target x (typically 0 or length).
    target_y    : float             Goal-frame target y (within goal width).
    target_z    : float             Height at goal line (0–2.44 m).
    xg          : float             Expected goals value [0, 1].
    outcome     : EventOutcome      GOAL, SAVED, OFF_TARGET, BLOCKED.
    first_time  : bool              Shot taken first-time (no preceding touch).
    """

    event_type:  EventType     = field(default=EventType.SHOT, init=False)
    body_part:   BodyPart      = BodyPart.RIGHT_FOOT
    origin_x:    float         = 0.0
    origin_y:    float         = 0.0
    target_x:    float         = 105.0
    target_y:    float         = 34.0
    target_z:    float         = 1.0
    xg:          float         = 0.0
    outcome:     EventOutcome  = EventOutcome.OFF_TARGET
    first_time:  bool          = False


@dataclass(frozen=True)
class Dribble(BaseEvent):
    """
    Attempted take-on / carry past an opponent.

    Fields
    ------
    foot        : Foot              Preferred foot during the dribble.
    start_x     : float             x at dribble start (== pitch_x).
    start_y     : float             y at dribble start (== pitch_y).
    end_x       : float             x at dribble end.
    end_y       : float             y at dribble end.
    outcome     : EventOutcome      WON (beat the defender) or LOST.
    """

    event_type:  EventType     = field(default=EventType.DRIBBLE, init=False)
    foot:        Foot          = Foot.RIGHT
    start_x:     float         = 0.0
    start_y:     float         = 0.0
    end_x:       float         = 0.0
    end_y:       float         = 0.0
    outcome:     EventOutcome  = EventOutcome.WON

    def displacement(self) -> float:
        """Euclidean distance covered in the dribble."""
        import math
        return math.hypot(self.end_x - self.start_x, self.end_y - self.start_y)


@dataclass(frozen=True)
class Tackle(BaseEvent):
    """
    Defensive challenge for the ball.

    Fields
    ------
    tackled_player  : str           Player being challenged.
    tackle_x        : float         x position of the tackle.
    tackle_y        : float         y position of the tackle.
    outcome         : EventOutcome  WON or LOST (possession perspective).
    """

    event_type:      EventType     = field(default=EventType.TACKLE, init=False)
    tackled_player:  str           = ""
    tackle_x:        float         = 0.0
    tackle_y:        float         = 0.0
    outcome:         EventOutcome  = EventOutcome.WON


@dataclass(frozen=True)
class Header(BaseEvent):
    """
    Contact made with the ball using the head.

    Fields
    ------
    action      : str               Intended action: "pass", "shot", "clearance", "flick-on".
    dest_x      : float             Target x (for pass/shot headers).
    dest_y      : float             Target y.
    outcome     : EventOutcome
    aerial_duel : bool              Competed for in an aerial duel with an opponent.
    """

    event_type:   EventType     = field(default=EventType.HEADER, init=False)
    action:       str           = "clearance"
    dest_x:       float         = 0.0
    dest_y:       float         = 0.0
    outcome:      EventOutcome  = EventOutcome.SUCCESS
    aerial_duel:  bool          = False


@dataclass(frozen=True)
class Foul(BaseEvent):
    """
    Illegal challenge.

    Fields
    ------
    fouled_player   : str           Player who was fouled.
    foul_x          : float         x position.
    foul_y          : float         y position.
    card            : Optional[str] None, "yellow", or "red".
    outcome         : EventOutcome  FOUL_COMMITTED always for the committing player.
    """

    event_type:    EventType        = field(default=EventType.FOUL, init=False)
    fouled_player: str              = ""
    foul_x:        float            = 0.0
    foul_y:        float            = 0.0
    card:          str | None    = None
    outcome:       EventOutcome     = EventOutcome.FOUL_COMMITTED


@dataclass(frozen=True)
class GoalkeeperAction(BaseEvent):
    """
    Goalkeeper-specific action.

    Fields
    ------
    action_type : GoalkeeperActionType
    foot        : Optional[Foot]    Set when distributing.
    dest_x      : float             Target x for distributions.
    dest_y      : float             Target y for distributions.
    outcome     : EventOutcome
    """

    event_type:   EventType                = field(default=EventType.GOALKEEPER_ACTION, init=False)
    action_type:  GoalkeeperActionType     = GoalkeeperActionType.SAVE
    foot:         Foot | None           = None
    dest_x:       float                    = 0.0
    dest_y:       float                    = 0.0
    outcome:      EventOutcome             = EventOutcome.SAVED


@dataclass(frozen=True)
class SetPiece(BaseEvent):
    """
    Dead-ball restarts.

    Fields
    ------
    set_piece_type  : SetPieceType
    foot            : Optional[Foot]    Delivery foot (not used for throw-ins).
    to_player       : str               Primary recipient.
    dest_x          : float             Target x.
    dest_y          : float             Target y.
    outcome         : EventOutcome
    """

    event_type:      EventType          = field(default=EventType.SET_PIECE, init=False)
    set_piece_type:  SetPieceType       = SetPieceType.FREE_KICK
    foot:            Foot | None     = None
    to_player:       str                = ""
    dest_x:          float              = 0.0
    dest_y:          float              = 0.0
    outcome:         EventOutcome       = EventOutcome.SUCCESS


# ── type union ────────────────────────────────────────────────────────────────

AnyEvent = (
    Touch
    | Pass
    | Shot
    | Dribble
    | Tackle
    | Header
    | Foul
    | GoalkeeperAction
    | SetPiece
)

EVENT_CLASS_MAP: dict[EventType, type] = {
    EventType.TOUCH:             Touch,
    EventType.PASS:              Pass,
    EventType.SHOT:              Shot,
    EventType.DRIBBLE:           Dribble,
    EventType.TACKLE:            Tackle,
    EventType.HEADER:            Header,
    EventType.FOUL:              Foul,
    EventType.GOALKEEPER_ACTION: GoalkeeperAction,
    EventType.SET_PIECE:         SetPiece,
}
