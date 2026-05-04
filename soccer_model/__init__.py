"""
soccer_model — Stochastic world-model based prediction of soccer events.

Provides a formal event ontology, validated interchange formats (JSON/CSV),
a spatial Pitch model, and a stochastic transition engine for next-event
prediction and Monte Carlo simulation.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .ontology import SCHEMA_VERSION
from .pitch import Pitch, PitchZone, PitchArea
from .events import (
    EventType, EventOutcome, Foot,
    BaseEvent, Touch, Pass, Shot, Dribble, Tackle, Header, Foul,
    GoalkeeperAction, SetPiece,
)
from .schema import (
    EventBase as SchemaEventBase,
    TouchEvent, PassEvent, ShotEvent, DribbleEvent, TackleEvent,
    HeaderEvent, FoulEvent, GoalkeeperEvent, SetPieceEvent,
    MatchEventStream, MatchMetadata,
    export_json_schema, export_event_schemas,
)
from .interchange import (
    read_json, write_json, read_csv, write_csv,
    validate_event_stream, validate_single_event,
    to_json_dict,
)
from .world_model import PlayerState, BallState, GamePhase, GameState, WorldState
from .stochastic import TransitionModel, EventDistribution
from .predictor import EventPredictor
from .simulation import MatchSimulator
from .csv_loader import load_csv as _legacy_load_csv, MatchCSV


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
