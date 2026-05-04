"""
Interchange — serialization and deserialization of event streams.

Supports two interchange formats:
  1. JSON  (canonical, lossless)
  2. CSV   (tabular, widely compatible)

Both formats validate against the formal Pydantic schema on read.
Both formats include the schema_version for forwards-compatibility checks.

Usage
─────
    from soccer_model.interchange import (
        read_json, write_json,
        read_csv, write_csv,
        validate_event_stream,
    )

    # Read & validate
    stream = read_json("match.json")
    stream = read_csv("match.csv", home_team="Arsenal", away_team="Chelsea")

    # Write
    write_json(stream, "match.json")
    write_csv(stream, "match.csv")

    # Validate an existing dict/list
    errors = validate_event_stream(raw_dict)
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from pydantic import ValidationError

from .ontology import (
    SCHEMA_VERSION,
    DEFAULT_PITCH_LENGTH,
    DEFAULT_PITCH_WIDTH,
    EventType,
    EventOutcome,
    Foot,
    BodyPart,
    PassType,
    SetPieceType,
    GoalkeeperActionType,
    HeaderAction,
    Card,
)
from .schema import (
    AnyEvent,
    EventBase,
    MatchEventStream,
    MatchMetadata,
    EVENT_MODEL_MAP,
    TouchEvent,
    PassEvent,
    ShotEvent,
    DribbleEvent,
    TackleEvent,
    HeaderEvent,
    FoulEvent,
    GoalkeeperEvent,
    SetPieceEvent,
)


# ── validation ───────────────────────────────────────────────────────────────

class ValidationResult:
    """Result of schema validation."""

    def __init__(self, valid: bool, errors: list[str], warnings: list[str]) -> None:
        self.valid    = valid
        self.errors   = errors
        self.warnings = warnings

    def __bool__(self) -> bool:
        return self.valid

    def summary(self) -> str:
        if self.valid:
            return f"Valid ({len(self.warnings)} warnings)"
        return f"Invalid: {len(self.errors)} errors, {len(self.warnings)} warnings"


def validate_event_stream(data: dict[str, Any]) -> ValidationResult:
    """
    Validate a raw dict against the MatchEventStream schema.

    Parameters
    ----------
    data : dict
        Must have ``metadata`` and ``events`` keys.

    Returns
    -------
    ValidationResult with .valid, .errors, .warnings
    """
    errors: list[str]   = []
    warnings: list[str] = []

    # version check
    meta = data.get("metadata", {})
    version = meta.get("schema_version", "")
    if version and version != SCHEMA_VERSION:
        warnings.append(
            f"Schema version mismatch: data={version}, expected={SCHEMA_VERSION}"
        )

    try:
        MatchEventStream.model_validate(data)
    except ValidationError as exc:
        for err in exc.errors():
            loc = " → ".join(str(l) for l in err["loc"])
            errors.append(f"[{loc}] {err['msg']}")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


def validate_single_event(data: dict[str, Any]) -> ValidationResult:
    """Validate a single event dict."""
    errors: list[str]   = []
    warnings: list[str] = []

    raw_type = data.get("event_type", "")
    try:
        et = EventType(raw_type)
    except ValueError:
        errors.append(f"Unknown event_type: '{raw_type}'")
        return ValidationResult(False, errors, warnings)

    model_cls = EVENT_MODEL_MAP[et]
    try:
        model_cls.model_validate(data)
    except ValidationError as exc:
        for err in exc.errors():
            loc = " → ".join(str(l) for l in err["loc"])
            errors.append(f"[{loc}] {err['msg']}")

    return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)


# ── JSON I/O ─────────────────────────────────────────────────────────────────

def write_json(
    stream: MatchEventStream,
    path: Union[str, Path],
    indent: int = 2,
) -> None:
    """
    Write a validated MatchEventStream to a JSON file.

    The output is the canonical interchange format.
    """
    data = stream.model_dump(mode="json", by_alias=True, exclude_none=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, sort_keys=False)


def read_json(
    source: Union[str, Path, io.StringIO],
    strict: bool = True,
) -> MatchEventStream:
    """
    Read and validate a JSON event stream.

    Parameters
    ----------
    source : path or StringIO
    strict : bool
        If True, raise on validation failure. If False, return partial results.
    """
    if isinstance(source, io.StringIO):
        data = json.load(source)
    else:
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)

    if strict:
        return MatchEventStream.model_validate(data)
    else:
        try:
            return MatchEventStream.model_validate(data)
        except ValidationError:
            # attempt best-effort parse
            return _best_effort_parse(data)


def to_json_dict(stream: MatchEventStream) -> dict[str, Any]:
    """Serialize to a plain dict (JSON-ready) without writing to file."""
    return stream.model_dump(mode="json", by_alias=True, exclude_none=True)


# ── CSV I/O ──────────────────────────────────────────────────────────────────

# canonical CSV column order
_CSV_COLUMNS = [
    "schema_version", "event_id", "timestamp", "team", "player",
    "event_type", "x", "y", "end_x", "end_y",
    "foot", "body_part", "to_player", "outcome",
    "pass_type", "xg", "set_piece_type", "gk_action_type",
    "header_action", "tackled_player", "fouled_player", "card",
    "under_pressure", "sequence_id", "aerial", "first_time", "aerial_duel",
]


def write_csv(
    stream: MatchEventStream,
    path: Union[str, Path],
) -> None:
    """
    Write a validated MatchEventStream to CSV.

    First row is always a comment with metadata:
        # schema_version=0.1.0,home_team=Arsenal,away_team=Chelsea,...
    """
    meta = stream.metadata
    header_comment = (
        f"# schema_version={meta.schema_version},"
        f"match_id={meta.match_id},"
        f"home_team={meta.home_team},"
        f"away_team={meta.away_team},"
        f"pitch_length={meta.pitch_length},"
        f"pitch_width={meta.pitch_width}"
    )

    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(header_comment + "\n")
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for ev in stream.events:
            row = _event_to_csv_row(ev)
            writer.writerow(row)


def read_csv(
    source: Union[str, Path, io.StringIO],
    home_team: Optional[str] = None,
    away_team: Optional[str] = None,
    pitch_length: float = DEFAULT_PITCH_LENGTH,
    pitch_width:  float = DEFAULT_PITCH_WIDTH,
    strict: bool = True,
) -> MatchEventStream:
    """
    Read and validate a CSV event stream.

    Parameters
    ----------
    source       : file path or StringIO
    home_team    : override home team (auto-detected if not set)
    away_team    : override away team
    pitch_length : pitch x-axis dimension
    pitch_width  : pitch y-axis dimension
    strict       : raise on validation errors

    Returns
    -------
    MatchEventStream — fully validated.
    """
    if isinstance(source, io.StringIO):
        text = source.read()
    else:
        with open(source, "r", encoding="utf-8") as f:
            text = f.read()

    # parse metadata from comment line if present
    meta_line = ""
    lines = text.splitlines()
    if lines and lines[0].startswith("#"):
        meta_line = lines[0][1:].strip()
        text = "\n".join(lines[1:])

    meta_kv = _parse_meta_comment(meta_line)
    detected_home = meta_kv.get("home_team", "")
    detected_away = meta_kv.get("away_team", "")

    reader = csv.DictReader(io.StringIO(text))
    rows = [{_norm(k): v for k, v in row.items()} for row in reader]

    # auto-detect teams from data if not provided
    if not home_team:
        teams_seen: dict[str, None] = {}
        for r in rows:
            t = r.get("team", "").strip()
            if t:
                teams_seen[t] = None
        team_list = list(teams_seen.keys())
        home_team = detected_home or (team_list[0] if len(team_list) > 0 else "home")
        away_team = detected_away or (team_list[1] if len(team_list) > 1 else "away")

    # parse events
    events: list[EventBase] = []
    parse_errors: list[str] = []

    for i, row in enumerate(rows):
        try:
            ev = _csv_row_to_event(row, pitch_length, pitch_width)
            events.append(ev)
        except (ValidationError, ValueError, KeyError) as exc:
            if strict:
                raise ValueError(f"Row {i+1}: {exc}") from exc
            parse_errors.append(f"Row {i+1}: {exc}")

    meta = MatchMetadata(
        schema_version=meta_kv.get("schema_version", SCHEMA_VERSION),
        match_id=meta_kv.get("match_id", ""),
        home_team=home_team,
        away_team=away_team or "away",
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )

    stream = MatchEventStream(metadata=meta, events=events)
    return stream


# ── internal helpers ─────────────────────────────────────────────────────────

def _norm(key: str) -> str:
    return key.strip().lower().replace(" ", "_")


def _parse_meta_comment(line: str) -> dict[str, str]:
    kv: dict[str, str] = {}
    for part in line.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k.strip()] = v.strip()
    return kv


def _s(v) -> str:
    """Safely coerce a CSV cell to a stripped string (handles None)."""
    if v is None:
        return ""
    return str(v).strip()


def _safe_float(v, default: float = 0.0) -> Optional[float]:
    s = _s(v)
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return default


def _event_to_csv_row(ev: EventBase) -> dict[str, str]:
    """Flatten a Pydantic event model to a CSV row dict."""
    d = ev.model_dump(mode="json", exclude_none=True)
    row: dict[str, str] = {c: "" for c in _CSV_COLUMNS}

    row["schema_version"] = d.get("schema_version", SCHEMA_VERSION)
    row["event_id"]       = d.get("event_id", "")
    row["timestamp"]      = str(d.get("timestamp", ""))
    row["team"]           = d.get("team", "")
    row["player"]         = d.get("player", "")
    row["event_type"]     = d.get("event_type", "")
    row["x"]              = str(d.get("x", ""))
    row["y"]              = str(d.get("y", ""))
    row["sequence_id"]    = str(d.get("sequence_id", "0"))
    row["under_pressure"] = str(d.get("under_pressure", False)).lower()
    row["outcome"]        = d.get("outcome", "")

    # type-specific fields
    for key in ("end_x", "end_y", "foot", "body_part", "to_player",
                "pass_type", "xg", "set_piece_type", "gk_action_type",
                "header_action", "tackled_player", "fouled_player", "card",
                "aerial", "first_time", "aerial_duel"):
        if key in d:
            val = d[key]
            row[key] = str(val) if val is not None else ""

    return row


def _csv_row_to_event(row: dict[str, str], pitch_length: float, pitch_width: float) -> EventBase:
    """Parse a normalised CSV row dict into a validated Pydantic event model."""
    raw_type = _s(row.get("event_type")).lower()
    et = EventType(raw_type)
    model_cls = EVENT_MODEL_MAP[et]

    import uuid as _uuid

    base: dict[str, Any] = {
        "schema_version": row.get("schema_version", SCHEMA_VERSION) or SCHEMA_VERSION,
        "event_id":       _s(row.get("event_id")) or str(_uuid.uuid4()),
        "timestamp":      float(_s(row.get("timestamp")) or 0),
        "team":           _s(row.get("team")),
        "player":         _s(row.get("player")),
        "event_type":     et,
        "x":              float(_s(row.get("x")) or 0),
        "y":              float(_s(row.get("y")) or 0),
        "sequence_id":    int(_s(row.get("sequence_id")) or 0),
        "under_pressure": _s(row.get("under_pressure")).lower() in ("true", "1", "yes"),
        "pitch_length":   pitch_length,
        "pitch_width":    pitch_width,
    }

    outcome_raw = _s(row.get("outcome")).lower()
    if outcome_raw:
        base["outcome"] = EventOutcome(outcome_raw)

    # type-specific fields
    if et == EventType.TOUCH:
        foot_raw = _s(row.get("foot")).lower()
        if foot_raw:
            base["foot"] = Foot(foot_raw)

    elif et == EventType.PASS:
        base["end_x"]     = float(_s(row.get("end_x")) or _s(row.get("x")) or 0)
        base["end_y"]     = float(_s(row.get("end_y")) or _s(row.get("y")) or 0)
        base["to_player"] = _s(row.get("to_player")) or "unknown"
        foot_raw = _s(row.get("foot")).lower()
        if foot_raw:
            base["foot"] = Foot(foot_raw)
        pt_raw = _s(row.get("pass_type")).lower()
        if pt_raw:
            base["pass_type"] = PassType(pt_raw)
        base["aerial"] = _s(row.get("aerial")).lower() in ("true", "1", "yes")

    elif et == EventType.SHOT:
        base["end_x"] = float(_s(row.get("end_x")) or pitch_length)
        base["end_y"] = float(_s(row.get("end_y")) or pitch_width / 2)
        bp_raw = _s(row.get("body_part")).lower()
        if bp_raw:
            base["body_part"] = BodyPart(bp_raw)
        xg_raw = _s(row.get("xg"))
        if xg_raw:
            base["xg"] = float(xg_raw)
        base["first_time"] = _s(row.get("first_time")).lower() in ("true", "1", "yes")

    elif et == EventType.DRIBBLE:
        base["end_x"] = float(_s(row.get("end_x")) or _s(row.get("x")) or 0)
        base["end_y"] = float(_s(row.get("end_y")) or _s(row.get("y")) or 0)
        foot_raw = _s(row.get("foot")).lower()
        if foot_raw:
            base["foot"] = Foot(foot_raw)

    elif et == EventType.TACKLE:
        base["tackled_player"] = _s(row.get("tackled_player")) or "unknown"

    elif et == EventType.HEADER:
        base["end_x"] = float(_s(row.get("end_x")) or _s(row.get("x")) or 0)
        base["end_y"] = float(_s(row.get("end_y")) or _s(row.get("y")) or 0)
        ha_raw = _s(row.get("header_action") or row.get("action")).lower()
        if ha_raw:
            base["header_action"] = HeaderAction(ha_raw)
        base["aerial_duel"] = _s(row.get("aerial_duel")).lower() in ("true", "1", "yes")

    elif et == EventType.FOUL:
        base["fouled_player"] = _s(row.get("fouled_player")) or "unknown"
        card_raw = _s(row.get("card")).lower()
        if card_raw in ("yellow", "red"):
            base["card"] = Card(card_raw)

    elif et == EventType.GOALKEEPER_ACTION:
        gk_raw = _s(row.get("gk_action_type")).lower()
        if gk_raw:
            base["gk_action_type"] = GoalkeeperActionType(gk_raw)
        end_x = _safe_float(row.get("end_x"))
        end_y = _safe_float(row.get("end_y"))
        if end_x is not None:
            base["end_x"] = end_x
        if end_y is not None:
            base["end_y"] = end_y
        foot_raw = _s(row.get("foot")).lower()
        if foot_raw:
            base["foot"] = Foot(foot_raw)

    elif et == EventType.SET_PIECE:
        sp_raw = _s(row.get("set_piece_type")).lower()
        if sp_raw:
            base["set_piece_type"] = SetPieceType(sp_raw)
        end_x = _safe_float(row.get("end_x"))
        end_y = _safe_float(row.get("end_y"))
        if end_x is not None:
            base["end_x"] = end_x
        if end_y is not None:
            base["end_y"] = end_y
        foot_raw = _s(row.get("foot")).lower()
        if foot_raw:
            base["foot"] = Foot(foot_raw)
        tp = _s(row.get("to_player"))
        if tp:
            base["to_player"] = tp

    return model_cls.model_validate(base)


def _best_effort_parse(data: dict[str, Any]) -> MatchEventStream:
    """Attempt to parse a stream ignoring individual event errors."""
    meta_raw = data.get("metadata", {})
    meta = MatchMetadata.model_validate(meta_raw)

    events: list[EventBase] = []
    for ev_raw in data.get("events", []):
        try:
            raw_type = ev_raw.get("event_type", "")
            et = EventType(raw_type)
            model_cls = EVENT_MODEL_MAP[et]
            events.append(model_cls.model_validate(ev_raw))
        except (ValidationError, ValueError):
            continue

    return MatchEventStream(metadata=meta, events=events)
