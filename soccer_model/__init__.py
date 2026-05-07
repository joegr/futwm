"""
soccer_model — Stochastic world-model based prediction of soccer events.

Provides a formal event ontology, validated interchange formats (JSON/CSV),
a spatial Pitch model, and a stochastic transition engine for next-event
prediction and Monte Carlo simulation.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .csv_loader import MatchCSV
from .csv_loader import load_csv as _legacy_load_csv  # noqa: F401
from .events import (
    BaseEvent,
    Dribble,
    EventOutcome,
    EventType,
    Foot,
    Foul,
    GoalkeeperAction,
    Header,
    Pass,
    SetPiece,
    Shot,
    Tackle,
    Touch,
)
from .interchange import (
    read_csv,
    read_json,
    to_json_dict,
    validate_event_stream,
    validate_single_event,
    write_csv,
    write_json,
)
from .ontology import SCHEMA_VERSION
from .pitch import Pitch, PitchArea, PitchZone
from .predictor import EventPredictor
from .schema import (
    DribbleEvent,
    FoulEvent,
    GoalkeeperEvent,
    HeaderEvent,
    MatchEventStream,
    MatchMetadata,
    PassEvent,
    SetPieceEvent,
    ShotEvent,
    TackleEvent,
    TouchEvent,
    export_event_schemas,
    export_json_schema,
)
from .schema import (
    EventBase as SchemaEventBase,
)
from .simulation import MatchSimulator
from .stochastic import EventDistribution, TransitionModel
from .world_model import BallState, GamePhase, GameState, PlayerState, WorldState


def load_sample() -> MatchEventStream:
    """
    Load the bundled sample match (Arsenal vs Chelsea, 58 events).

    Returns a fully validated MatchEventStream ready for visualization
    or model fitting.
    """
    from pathlib import Path
    sample_path = Path(__file__).parent / "data" / "sample_match.csv"
    return read_csv(
        sample_path,
        home_team="Arsenal",
        away_team="Chelsea",
    )


__all__ = [
    # version
    "__version__", "SCHEMA_VERSION",
    # spatial
    "Pitch", "PitchZone", "PitchArea",
    # legacy event dataclasses
    "EventType", "EventOutcome", "Foot",
    "BaseEvent", "Touch", "Pass", "Shot", "Dribble", "Tackle", "Header",
    "Foul", "GoalkeeperAction", "SetPiece",
    # formal schema (Pydantic)
    "SchemaEventBase", "TouchEvent", "PassEvent", "ShotEvent",
    "DribbleEvent", "TackleEvent", "HeaderEvent", "FoulEvent",
    "GoalkeeperEvent", "SetPieceEvent",
    "MatchEventStream", "MatchMetadata",
    "export_json_schema", "export_event_schemas",
    # interchange
    "read_json", "write_json", "read_csv", "write_csv",
    "validate_event_stream", "validate_single_event", "to_json_dict",
    # world model
    "PlayerState", "BallState", "GamePhase", "GameState", "WorldState",
    "TransitionModel", "EventDistribution",
    "EventPredictor", "MatchSimulator",
    # convenience
    "load_sample", "MatchCSV",
]
