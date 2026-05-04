"""
Ontology — formal definitions for the soccer event model.

This module is the single source of truth for:
  - Schema version
  - Event taxonomy (types, sub-types, outcomes)
  - Field semantics and constraints
  - Coordinate system specification

Other modules import from here rather than defining their own enums.

Version history
───────────────
0.1.0   Initial ontology: 9 event types, FIFA-standard coordinate space.
"""

from __future__ import annotations

from enum import Enum
from typing import Final

# ── schema version ───────────────────────────────────────────────────────────

SCHEMA_VERSION: Final[str] = "0.1.0"

# ── coordinate system ────────────────────────────────────────────────────────
#
# All spatial fields use the FIFA-standard reference frame:
#   x-axis  : longitudinal, 0 = home goal-line, L = away goal-line (metres)
#   y-axis  : lateral,      0 = bottom touchline, W = top touchline (metres)
#   z-axis  : vertical,     0 = ground, positive upward (metres)
#
# Default pitch dimensions: L = 105 m, W = 68 m
# Coordinate (0, 0) is the bottom-left corner facing from behind the home goal.
#
# All event models enforce: 0 ≤ x ≤ L, 0 ≤ y ≤ W, 0 ≤ z (where applicable).

DEFAULT_PITCH_LENGTH: Final[float] = 105.0
DEFAULT_PITCH_WIDTH:  Final[float] = 68.0
GOAL_WIDTH:           Final[float] = 7.32
GOAL_HEIGHT:          Final[float] = 2.44


# ── event type taxonomy ──────────────────────────────────────────────────────

class EventType(str, Enum):
    """
    Top-level event classification.

    Each event type has a defined set of required and optional fields.
    See ``schema.py`` for the formal Pydantic model per type.
    """
    TOUCH             = "touch"
    PASS              = "pass"
    SHOT              = "shot"
    DRIBBLE           = "dribble"
    TACKLE            = "tackle"
    HEADER            = "header"
    FOUL              = "foul"
    GOALKEEPER_ACTION = "goalkeeper_action"
    SET_PIECE         = "set_piece"


class Foot(str, Enum):
    """Foot used for the action."""
    LEFT  = "left"
    RIGHT = "right"
    BOTH  = "both"


class BodyPart(str, Enum):
    """Body part making contact with the ball."""
    LEFT_FOOT  = "left_foot"
    RIGHT_FOOT = "right_foot"
    HEAD       = "head"
    CHEST      = "chest"
    OTHER      = "other"


class EventOutcome(str, Enum):
    """
    Outcome of the event from the perspective of the acting player.

    Semantics
    ---------
    SUCCESS        : Action achieved intended effect (pass received, touch controlled).
    FAILURE        : Generic failure not captured by a more specific code.
    BLOCKED        : Shot / pass physically blocked by an opponent.
    INTERCEPTED    : Pass read and collected by an opponent.
    OUT_OF_PLAY    : Ball left the field as a result of the action.
    GOAL           : Shot resulted in a goal.
    SAVED          : Shot stopped by the goalkeeper.
    OFF_TARGET     : Shot missed the goal frame entirely.
    WON            : Duel / tackle won possession.
    LOST           : Duel / tackle lost.
    FOUL_COMMITTED : Illegal challenge committed by the acting player.
    """
    SUCCESS        = "success"
    FAILURE        = "failure"
    BLOCKED        = "blocked"
    INTERCEPTED    = "intercepted"
    OUT_OF_PLAY    = "out_of_play"
    GOAL           = "goal"
    SAVED          = "saved"
    OFF_TARGET     = "off_target"
    WON            = "won"
    LOST           = "lost"
    FOUL_COMMITTED = "foul_committed"


class PassType(str, Enum):
    """Tactical classification of a pass."""
    SHORT   = "short"       # < 15 m
    LONG    = "long"        # > 32 m
    THROUGH = "through"     # ball played into space behind defence
    CROSS   = "cross"       # delivery into the penalty area from a wide position
    SWITCH  = "switch"      # lateral ball > 30 m to change flank
    BACK    = "back"        # backwards relative to attacking direction


class SetPieceType(str, Enum):
    """Dead-ball restart type."""
    CORNER_KICK = "corner_kick"
    FREE_KICK   = "free_kick"
    PENALTY     = "penalty"
    THROW_IN    = "throw_in"
    GOAL_KICK   = "goal_kick"
    KICK_OFF    = "kick_off"


class GoalkeeperActionType(str, Enum):
    """Goalkeeper-specific action."""
    SAVE         = "save"
    PUNCH        = "punch"
    CLAIM        = "claim"
    DISTRIBUTION = "distribution"
    DIVE         = "dive"


class HeaderAction(str, Enum):
    """Intent of a headed action."""
    PASS      = "pass"
    SHOT      = "shot"
    CLEARANCE = "clearance"
    FLICK_ON  = "flick_on"


class Card(str, Enum):
    """Disciplinary card shown."""
    YELLOW = "yellow"
    RED    = "red"


# ── valid outcome constraints per event type ─────────────────────────────────
# Used by validators to reject impossible combinations.

VALID_OUTCOMES: dict[EventType, set[EventOutcome]] = {
    EventType.TOUCH: {
        EventOutcome.SUCCESS, EventOutcome.FAILURE,
    },
    EventType.PASS: {
        EventOutcome.SUCCESS, EventOutcome.INTERCEPTED,
        EventOutcome.OUT_OF_PLAY, EventOutcome.BLOCKED,
    },
    EventType.SHOT: {
        EventOutcome.GOAL, EventOutcome.SAVED,
        EventOutcome.OFF_TARGET, EventOutcome.BLOCKED,
    },
    EventType.DRIBBLE: {
        EventOutcome.WON, EventOutcome.LOST,
    },
    EventType.TACKLE: {
        EventOutcome.WON, EventOutcome.LOST,
    },
    EventType.HEADER: {
        EventOutcome.SUCCESS, EventOutcome.FAILURE,
        EventOutcome.GOAL, EventOutcome.SAVED,
        EventOutcome.OFF_TARGET, EventOutcome.BLOCKED,
    },
    EventType.FOUL: {
        EventOutcome.FOUL_COMMITTED,
    },
    EventType.GOALKEEPER_ACTION: {
        EventOutcome.SUCCESS, EventOutcome.SAVED,
        EventOutcome.FAILURE,
    },
    EventType.SET_PIECE: {
        EventOutcome.SUCCESS, EventOutcome.FAILURE,
        EventOutcome.OUT_OF_PLAY, EventOutcome.INTERCEPTED,
    },
}

# ── field documentation (for schema export & downstream tooling) ─────────────

FIELD_DOCS: dict[str, str] = {
    "schema_version":   "Ontology version this record conforms to.",
    "event_id":         "Globally unique event identifier (UUID4).",
    "timestamp":        "Match clock in seconds from kick-off (≥ 0).",
    "team":             "Identifier of the team performing the action.",
    "player":           "Identifier of the player performing the action.",
    "event_type":       "Top-level event classification.",
    "x":                "Pitch x-coordinate at event origin (metres, 0–L).",
    "y":                "Pitch y-coordinate at event origin (metres, 0–W).",
    "end_x":            "Pitch x-coordinate at event destination (metres, 0–L).",
    "end_y":            "Pitch y-coordinate at event destination (metres, 0–W).",
    "foot":             "Foot used for the action.",
    "body_part":        "Body part making contact with the ball.",
    "outcome":          "Result of the action from the acting player's perspective.",
    "to_player":        "Identifier of the intended recipient (passes, set pieces).",
    "pass_type":        "Tactical classification of the pass.",
    "xg":               "Expected goals probability [0, 1] for shots.",
    "set_piece_type":   "Type of dead-ball restart.",
    "gk_action_type":   "Goalkeeper action sub-type.",
    "header_action":    "Intent of the headed action.",
    "tackled_player":   "Player who was dispossessed (tackles).",
    "fouled_player":    "Player who was fouled.",
    "card":             "Disciplinary card shown (yellow/red), if any.",
    "under_pressure":   "Whether the acting player was under immediate opponent pressure.",
    "sequence_id":      "Monotonic index within a possession chain.",
    "pitch_length":     "Pitch x-axis span for coordinate validation (metres).",
    "pitch_width":      "Pitch y-axis span for coordinate validation (metres).",
}
