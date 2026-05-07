"""
Formal Pydantic v2 event schema with strict validation.

This is the **canonical interchange model**. All data entering or leaving
the system is validated against these models.

Design principles
─────────────────
1. Every field has explicit type, constraints, and documentation.
2. Coordinate fields are bounded by pitch dimensions (configurable).
3. Outcome fields are constrained to valid values per event type.
4. Models export JSON Schema via ``.model_json_schema()``.
5. Serialization is deterministic (sorted keys, consistent float precision).
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .ontology import (
    DEFAULT_PITCH_LENGTH,
    DEFAULT_PITCH_WIDTH,
    SCHEMA_VERSION,
    VALID_OUTCOMES,
    BodyPart,
    Card,
    EventOutcome,
    EventType,
    Foot,
    GoalkeeperActionType,
    HeaderAction,
    PassType,
    SetPieceType,
)

# ── coordinate annotation types ──────────────────────────────────────────────

PitchX = Annotated[float, Field(ge=0, le=DEFAULT_PITCH_LENGTH, description="x-coordinate on pitch (metres, 0–105)")]
PitchY = Annotated[float, Field(ge=0, le=DEFAULT_PITCH_WIDTH,  description="y-coordinate on pitch (metres, 0–68)")]
Probability = Annotated[float, Field(ge=0.0, le=1.0, description="Probability value [0, 1]")]
Timestamp = Annotated[float, Field(ge=0.0, description="Match time in seconds from kick-off")]


# ── base event model ─────────────────────────────────────────────────────────

class EventBase(BaseModel):
    """
    Common fields shared by every event record.

    All concrete event types extend this base.
    """
    model_config = ConfigDict(
        use_enum_values=False,
        str_strip_whitespace=True,
        validate_default=True,
        extra="forbid",
    )

    schema_version: str       = Field(default=SCHEMA_VERSION, description="Ontology version")
    event_id:       str       = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique event ID (UUID4)")
    timestamp:      Timestamp
    team:           str       = Field(min_length=1, description="Team identifier")
    player:         str       = Field(min_length=1, description="Player identifier")
    event_type:     EventType
    x:              PitchX    = Field(description="Event origin x-coordinate (metres)")
    y:              PitchY    = Field(description="Event origin y-coordinate (metres)")
    sequence_id:    int       = Field(default=0, ge=0, description="Index within possession chain")
    under_pressure: bool      = Field(default=False, description="Opponent pressure on actor")

    # configurable pitch bounds (for non-standard pitches)
    pitch_length:   float     = Field(default=DEFAULT_PITCH_LENGTH, gt=0, exclude=True)
    pitch_width:    float     = Field(default=DEFAULT_PITCH_WIDTH,  gt=0, exclude=True)

    @field_validator("x")
    @classmethod
    def validate_x(cls, v: float, info) -> float:
        length = info.data.get("pitch_length", DEFAULT_PITCH_LENGTH)
        if not (0.0 <= v <= length):
            raise ValueError(f"x={v} outside pitch bounds [0, {length}]")
        return v

    @field_validator("y")
    @classmethod
    def validate_y(cls, v: float, info) -> float:
        width = info.data.get("pitch_width", DEFAULT_PITCH_WIDTH)
        if not (0.0 <= v <= width):
            raise ValueError(f"y={v} outside pitch bounds [0, {width}]")
        return v


# ── concrete event types ─────────────────────────────────────────────────────

class TouchEvent(EventBase):
    """Ball reception or first touch — no directional intent."""
    event_type: Literal[EventType.TOUCH] = EventType.TOUCH
    foot:       Foot                     = Foot.RIGHT
    outcome:    EventOutcome             = EventOutcome.SUCCESS

    @field_validator("outcome")
    @classmethod
    def _check_outcome(cls, v: EventOutcome) -> EventOutcome:
        if v not in VALID_OUTCOMES[EventType.TOUCH]:
            raise ValueError(f"Invalid outcome '{v}' for Touch. Valid: {VALID_OUTCOMES[EventType.TOUCH]}")
        return v


class PassEvent(EventBase):
    """Deliberate distribution to a team-mate."""
    event_type: Literal[EventType.PASS] = EventType.PASS
    foot:       Foot                    = Foot.RIGHT
    to_player:  str                     = Field(min_length=1, description="Intended recipient")
    end_x:      PitchX                  = Field(description="Intended destination x")
    end_y:      PitchY                  = Field(description="Intended destination y")
    pass_type:  PassType                = PassType.SHORT
    outcome:    EventOutcome            = EventOutcome.SUCCESS
    aerial:     bool                    = Field(default=False, description="Ball travelled through the air")

    @field_validator("outcome")
    @classmethod
    def _check_outcome(cls, v: EventOutcome) -> EventOutcome:
        if v not in VALID_OUTCOMES[EventType.PASS]:
            raise ValueError(f"Invalid outcome '{v}' for Pass. Valid: {VALID_OUTCOMES[EventType.PASS]}")
        return v


class ShotEvent(EventBase):
    """Attempt on goal."""
    event_type: Literal[EventType.SHOT] = EventType.SHOT
    body_part:  BodyPart                = BodyPart.RIGHT_FOOT
    end_x:      PitchX                  = Field(description="Goal-frame target x")
    end_y:      PitchY                  = Field(description="Goal-frame target y")
    end_z:      float                   = Field(default=1.0, ge=0.0, le=5.0, description="Height at goal line (metres)")
    xg:         Probability             = Field(default=0.0, description="Expected goals value")
    outcome:    EventOutcome            = EventOutcome.OFF_TARGET
    first_time: bool                    = Field(default=False, description="Shot without preceding touch")

    @field_validator("outcome")
    @classmethod
    def _check_outcome(cls, v: EventOutcome) -> EventOutcome:
        if v not in VALID_OUTCOMES[EventType.SHOT]:
            raise ValueError(f"Invalid outcome '{v}' for Shot. Valid: {VALID_OUTCOMES[EventType.SHOT]}")
        return v


class DribbleEvent(EventBase):
    """Take-on or ball carry past an opponent."""
    event_type: Literal[EventType.DRIBBLE] = EventType.DRIBBLE
    foot:       Foot                       = Foot.RIGHT
    end_x:      PitchX                     = Field(description="Dribble end x-coordinate")
    end_y:      PitchY                     = Field(description="Dribble end y-coordinate")
    outcome:    EventOutcome               = EventOutcome.WON

    @field_validator("outcome")
    @classmethod
    def _check_outcome(cls, v: EventOutcome) -> EventOutcome:
        if v not in VALID_OUTCOMES[EventType.DRIBBLE]:
            raise ValueError(f"Invalid outcome '{v}' for Dribble. Valid: {VALID_OUTCOMES[EventType.DRIBBLE]}")
        return v


class TackleEvent(EventBase):
    """Defensive challenge for the ball."""
    event_type:     Literal[EventType.TACKLE] = EventType.TACKLE
    tackled_player: str                       = Field(min_length=1, description="Player being challenged")
    outcome:        EventOutcome              = EventOutcome.WON

    @field_validator("outcome")
    @classmethod
    def _check_outcome(cls, v: EventOutcome) -> EventOutcome:
        if v not in VALID_OUTCOMES[EventType.TACKLE]:
            raise ValueError(f"Invalid outcome '{v}' for Tackle. Valid: {VALID_OUTCOMES[EventType.TACKLE]}")
        return v


class HeaderEvent(EventBase):
    """Headed ball contact."""
    event_type:    Literal[EventType.HEADER] = EventType.HEADER
    header_action: HeaderAction              = HeaderAction.CLEARANCE
    end_x:         PitchX                    = Field(description="Headed ball destination x")
    end_y:         PitchY                    = Field(description="Headed ball destination y")
    outcome:       EventOutcome              = EventOutcome.SUCCESS
    aerial_duel:   bool                      = Field(default=False, description="Contested with opponent")

    @field_validator("outcome")
    @classmethod
    def _check_outcome(cls, v: EventOutcome) -> EventOutcome:
        if v not in VALID_OUTCOMES[EventType.HEADER]:
            raise ValueError(f"Invalid outcome '{v}' for Header. Valid: {VALID_OUTCOMES[EventType.HEADER]}")
        return v


class FoulEvent(EventBase):
    """Illegal challenge."""
    event_type:    Literal[EventType.FOUL] = EventType.FOUL
    fouled_player: str                     = Field(min_length=1, description="Player who was fouled")
    card:          Card | None          = Field(default=None, description="Card shown, if any")
    outcome:       EventOutcome            = EventOutcome.FOUL_COMMITTED

    @field_validator("outcome")
    @classmethod
    def _check_outcome(cls, v: EventOutcome) -> EventOutcome:
        if v not in VALID_OUTCOMES[EventType.FOUL]:
            raise ValueError(f"Invalid outcome '{v}' for Foul. Valid: {VALID_OUTCOMES[EventType.FOUL]}")
        return v


class GoalkeeperEvent(EventBase):
    """Goalkeeper-specific action."""
    event_type:     Literal[EventType.GOALKEEPER_ACTION] = EventType.GOALKEEPER_ACTION
    gk_action_type: GoalkeeperActionType                 = GoalkeeperActionType.SAVE
    foot:           Foot | None                       = Field(default=None, description="Foot used (distributions)")
    end_x:          PitchX | None                     = Field(default=None, description="Distribution target x")
    end_y:          PitchY | None                     = Field(default=None, description="Distribution target y")
    outcome:        EventOutcome                         = EventOutcome.SAVED

    @field_validator("outcome")
    @classmethod
    def _check_outcome(cls, v: EventOutcome) -> EventOutcome:
        if v not in VALID_OUTCOMES[EventType.GOALKEEPER_ACTION]:
            valid = VALID_OUTCOMES[EventType.GOALKEEPER_ACTION]
            raise ValueError(f"Invalid outcome '{v}' for GoalkeeperAction. Valid: {valid}")
        return v


class SetPieceEvent(EventBase):
    """Dead-ball restart."""
    event_type:     Literal[EventType.SET_PIECE] = EventType.SET_PIECE
    set_piece_type: SetPieceType                 = SetPieceType.FREE_KICK
    foot:           Foot | None               = Field(default=None)
    to_player:      str | None                = Field(default=None, description="Primary recipient")
    end_x:          PitchX | None             = Field(default=None, description="Delivery target x")
    end_y:          PitchY | None             = Field(default=None, description="Delivery target y")
    outcome:        EventOutcome                 = EventOutcome.SUCCESS

    @field_validator("outcome")
    @classmethod
    def _check_outcome(cls, v: EventOutcome) -> EventOutcome:
        if v not in VALID_OUTCOMES[EventType.SET_PIECE]:
            raise ValueError(f"Invalid outcome '{v}' for SetPiece. Valid: {VALID_OUTCOMES[EventType.SET_PIECE]}")
        return v


# ── discriminated union ──────────────────────────────────────────────────────

AnyEvent = Annotated[
    TouchEvent | PassEvent | ShotEvent | DribbleEvent | TackleEvent
    | HeaderEvent | FoulEvent | GoalkeeperEvent | SetPieceEvent,
    Field(discriminator="event_type"),
]

EVENT_MODEL_MAP: dict[EventType, type[EventBase]] = {
    EventType.TOUCH:             TouchEvent,
    EventType.PASS:              PassEvent,
    EventType.SHOT:              ShotEvent,
    EventType.DRIBBLE:           DribbleEvent,
    EventType.TACKLE:            TackleEvent,
    EventType.HEADER:            HeaderEvent,
    EventType.FOUL:              FoulEvent,
    EventType.GOALKEEPER_ACTION: GoalkeeperEvent,
    EventType.SET_PIECE:         SetPieceEvent,
}


# ── match container ──────────────────────────────────────────────────────────

class MatchMetadata(BaseModel):
    """Top-level metadata for a match event stream."""
    model_config = ConfigDict(extra="forbid")

    schema_version: str       = SCHEMA_VERSION
    match_id:       str       = Field(default_factory=lambda: str(uuid.uuid4()))
    home_team:      str       = Field(min_length=1)
    away_team:      str       = Field(min_length=1)
    pitch_length:   float     = Field(default=DEFAULT_PITCH_LENGTH, gt=0)
    pitch_width:    float     = Field(default=DEFAULT_PITCH_WIDTH,  gt=0)
    description:    str       = ""


class MatchEventStream(BaseModel):
    """
    Complete validated event stream for a match.

    This is the canonical JSON interchange format.
    """
    model_config = ConfigDict(extra="forbid")

    metadata: MatchMetadata
    events:   list[AnyEvent]

    @model_validator(mode="after")
    def _check_timestamps_ordered(self) -> MatchEventStream:
        for i in range(1, len(self.events)):
            if self.events[i].timestamp < self.events[i - 1].timestamp:
                raise ValueError(
                    f"Events must be timestamp-ordered: event[{i}].timestamp "
                    f"({self.events[i].timestamp}) < event[{i-1}].timestamp "
                    f"({self.events[i-1].timestamp})"
                )
        return self


# ── JSON Schema export ───────────────────────────────────────────────────────

def export_json_schema() -> dict[str, Any]:
    """
    Export the full JSON Schema for the MatchEventStream interchange format.

    Returns a JSON-serializable dict suitable for writing to a .json file.
    """
    return MatchEventStream.model_json_schema(
        ref_template="#/$defs/{model}",
        mode="serialization",
    )


def export_event_schemas() -> dict[str, dict[str, Any]]:
    """Export individual JSON Schemas per event type."""
    return {
        et.value: model.model_json_schema(mode="serialization")
        for et, model in EVENT_MODEL_MAP.items()
    }
