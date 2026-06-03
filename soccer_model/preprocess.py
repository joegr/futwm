"""
Row-level preprocessing pipeline for match event data.

Operates on **CSV-style row dicts** — the same shape the user sees in the
input form and in a spreadsheet. Every operation is a pure function from
``list[dict] -> list[dict]`` plus an audit-log entry. This is the bridge
between *human-readable* form input and the strictly-typed event objects
produced downstream by :func:`soccer_model.csv_loader.load_csv`.

The pipeline is intentionally column-oriented (not Pydantic-model-oriented)
so that:

* the same operations work both at **data-input time** (single row added
  via the UI form) and at **data-preprocessing time** (bulk operations
  over the whole match)
* the user can preview the effect of an op before the typed events are
  rebuilt — which is much faster than re-running the full event loader

Operations
----------
Every operation is documented with a JSON-schema-like spec exposed via
:func:`operation_specs` so the frontend can render its own form fields
without hard-coding them.

The supported ops are:

* ``rename_player``       — global rename, e.g. "L. Messi" → "Lionel Messi"
* ``rename_team``         — global rename
* ``swap_teams``          — swap home/away labels **and** mirror x
* ``mirror``              — reflect coordinates along an axis
* ``time_shift``          — add a constant to every timestamp
* ``time_scale``          — multiply every timestamp by a constant
* ``round_time``          — round timestamps to N decimals
* ``clamp_to_pitch``      — clip coords into the supplied pitch length/width
* ``transform_coords``    — convert coords from another frame
                            (statsbomb / opta_percent / normalised / centred_cm)
                            into FIFA metres
* ``keep_team``           — drop rows whose team != supplied team
* ``drop_team``           — drop rows whose team == supplied team
* ``drop_event_type``     — drop rows of a given event_type
* ``filter_time_range``   — keep rows whose timestamp ∈ [min, max]
* ``sort_by_time``        — stable-sort rows by timestamp ascending
* ``set_field``           — set a single column to a fixed value on every row
* ``snap_to_zone``        — snap (x, y) to the centre of the nearest pitch zone
* ``deduplicate``         — drop rows that are byte-identical to the previous one
* ``validate``            — no-op; runs the row through the typed loader and
                            collects per-row errors (does not mutate)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .pitch import Pitch

__all__ = [
    "PreprocessError",
    "OpSpec",
    "PipelineResult",
    "apply_pipeline",
    "apply_operation",
    "operation_specs",
    "validate_row",
]


# ── error types ──────────────────────────────────────────────────────────────

class PreprocessError(ValueError):
    """Raised when a preprocessing operation has invalid parameters."""


# ── op spec (for frontend introspection) ─────────────────────────────────────

@dataclass(slots=True, frozen=True)
class OpSpec:
    """
    Declarative description of a single preprocessing operation.

    The frontend reads :func:`operation_specs` to render parameter inputs
    automatically — keeping the form in sync with the backend without a
    duplicate definition.
    """
    op: str
    label: str
    description: str
    params: list[dict[str, Any]]    # [{"name": str, "type": "str|float|int|enum|bool", "enum": [...], ...}]
    mutates: bool = True


@dataclass(slots=True)
class PipelineResult:
    """Result of running an ordered list of operations."""
    rows: list[dict]
    log: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)


# ── public entry point ───────────────────────────────────────────────────────

def apply_pipeline(rows: list[dict], operations: Iterable[dict]) -> PipelineResult:
    """
    Apply *operations* sequentially to *rows* and return a :class:`PipelineResult`.

    Each operation is a dict ``{"op": "<name>", **params}`` matching the
    spec of one entry from :func:`operation_specs`. Operations are applied
    in order, and each operation sees the rows produced by the previous
    one (i.e. it's a left-fold).

    Errors raised by a single op are recorded in ``result.errors`` and
    pipeline execution **stops at the first failure** — the caller can
    inspect the partial state and the error message to recover.
    """
    out_rows = [dict(r) for r in rows]   # shallow copies, never mutate caller data
    log: list[dict] = []
    errors: list[dict] = []

    for i, op in enumerate(operations):
        if not isinstance(op, dict):
            errors.append({"index": i, "error": f"operation at position {i} must be a dict, got {type(op).__name__}"})
            break
        name = op.get("op")
        params = {k: v for k, v in op.items() if k != "op"}
        try:
            before = len(out_rows)
            out_rows = apply_operation(out_rows, name, **params)
            after = len(out_rows)
            log.append({"step": i, "op": name, "params": params, "rows_before": before, "rows_after": after})
        except PreprocessError as e:
            errors.append({"step": i, "op": name, "params": params, "error": str(e)})
            break
        except Exception as e:                # pragma: no cover - defensive
            errors.append({"step": i, "op": name, "params": params, "error": f"{type(e).__name__}: {e}"})
            break

    return PipelineResult(rows=out_rows, log=log, errors=errors)


def apply_operation(rows: list[dict], op: str | None, **params: Any) -> list[dict]:
    """Apply a single named operation to *rows*. Returns the new row list."""
    if not op:
        raise PreprocessError("operation 'op' field is required")
    handler = _HANDLERS.get(op)
    if handler is None:
        raise PreprocessError(
            f"unknown operation {op!r}; known: {sorted(_HANDLERS)}"
        )
    return handler(rows, **params)


def operation_specs() -> list[OpSpec]:
    """Return the public list of available operations for UI auto-rendering."""
    return list(_SPECS)


# ── helpers ──────────────────────────────────────────────────────────────────

def _as_float(v: Any, name: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError) as e:
        raise PreprocessError(f"{name!r} must be a number, got {v!r}") from e


def _as_nonempty_str(v: Any, name: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise PreprocessError(f"{name!r} must be a non-empty string")
    return v.strip()


def _row_float(row: dict, key: str) -> float | None:
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _set_float(row: dict, key: str, value: float) -> None:
    row[key] = value


# ── operation handlers ───────────────────────────────────────────────────────
# Each is a function (rows, **params) -> rows. Pure: never mutates input.

def _op_rename_player(rows: list[dict], **p) -> list[dict]:
    src = _as_nonempty_str(p.get("from"), "from")
    dst = _as_nonempty_str(p.get("to"), "to")
    for r in rows:
        if r.get("player") == src:
            r["player"] = dst
        if r.get("to_player") == src:
            r["to_player"] = dst
        if r.get("tackled_player") == src:
            r["tackled_player"] = dst
        if r.get("fouled_player") == src:
            r["fouled_player"] = dst
    return rows


def _op_rename_team(rows: list[dict], **p) -> list[dict]:
    src = _as_nonempty_str(p.get("from"), "from")
    dst = _as_nonempty_str(p.get("to"), "to")
    for r in rows:
        if r.get("team") == src:
            r["team"] = dst
    return rows


def _op_swap_teams(rows: list[dict], **p) -> list[dict]:
    """Swap home/away labels AND mirror coordinates so the perspective stays right."""
    pitch_length = _as_float(p.get("pitch_length", 105.0), "pitch_length")
    teams: list[str] = []
    for r in rows:
        t = r.get("team")
        if t and t not in teams:
            teams.append(t)
        if len(teams) == 2:
            break
    if len(teams) < 2:
        raise PreprocessError("need at least two distinct teams to swap")
    a, b = teams[0], teams[1]
    for r in rows:
        if r.get("team") == a:
            r["team"] = b
        elif r.get("team") == b:
            r["team"] = a
    return _op_mirror(rows, axis="x", pitch_length=pitch_length, pitch_width=68.0)


def _op_mirror(rows: list[dict], **p) -> list[dict]:
    axis = _as_nonempty_str(p.get("axis", "x"), "axis").lower()
    if axis not in ("x", "y"):
        raise PreprocessError(f"axis must be 'x' or 'y', got {axis!r}")
    if axis == "x":
        L = _as_float(p.get("pitch_length", 105.0), "pitch_length")
        for r in rows:
            for k in ("x", "end_x"):
                v = _row_float(r, k)
                if v is not None:
                    _set_float(r, k, L - v)
    else:
        W = _as_float(p.get("pitch_width", 68.0), "pitch_width")
        for r in rows:
            for k in ("y", "end_y"):
                v = _row_float(r, k)
                if v is not None:
                    _set_float(r, k, W - v)
    return rows


def _op_time_shift(rows: list[dict], **p) -> list[dict]:
    delta = _as_float(p.get("delta_s", 0.0), "delta_s")
    for r in rows:
        v = _row_float(r, "timestamp")
        if v is not None:
            r["timestamp"] = max(0.0, v + delta)
    return rows


def _op_time_scale(rows: list[dict], **p) -> list[dict]:
    factor = _as_float(p.get("factor", 1.0), "factor")
    if factor <= 0:
        raise PreprocessError(f"factor must be > 0, got {factor}")
    for r in rows:
        v = _row_float(r, "timestamp")
        if v is not None:
            r["timestamp"] = v * factor
    return rows


def _op_round_time(rows: list[dict], **p) -> list[dict]:
    ndigits = int(p.get("ndigits", 1))
    if ndigits < 0:
        raise PreprocessError(f"ndigits must be >= 0, got {ndigits}")
    for r in rows:
        v = _row_float(r, "timestamp")
        if v is not None:
            r["timestamp"] = round(v, ndigits)
    return rows


def _op_clamp_to_pitch(rows: list[dict], **p) -> list[dict]:
    L = _as_float(p.get("pitch_length", 105.0), "pitch_length")
    W = _as_float(p.get("pitch_width", 68.0), "pitch_width")
    if L <= 0 or W <= 0:
        raise PreprocessError("pitch dimensions must be positive")
    for r in rows:
        for kx, ky in (("x", "y"), ("end_x", "end_y")):
            vx = _row_float(r, kx)
            vy = _row_float(r, ky)
            if vx is not None:
                _set_float(r, kx, max(0.0, min(L, vx)))
            if vy is not None:
                _set_float(r, ky, max(0.0, min(W, vy)))
    return rows


def _op_transform_coords(rows: list[dict], **p) -> list[dict]:
    """Project coordinates from a source frame into FIFA metres.

    Uses :func:`soccer_model.adapters.coordinates.transform_xy` so that
    the same transforms used by external-data adapters are available at
    preprocessing time.
    """
    # Import lazily to keep adapters optional at module import time.
    from .adapters.coordinates import CoordinateFrame, transform_xy

    src_name = _as_nonempty_str(p.get("from_frame"), "from_frame").lower()
    try:
        src = CoordinateFrame(src_name)
    except ValueError as e:
        valid = ", ".join(c.value for c in CoordinateFrame)
        raise PreprocessError(
            f"unknown coordinate frame {src_name!r}; valid: {valid}"
        ) from e
    if src is CoordinateFrame.FIFA_METRES:
        return rows  # already there

    L = _as_float(p.get("pitch_length", 105.0), "pitch_length")
    W = _as_float(p.get("pitch_width", 68.0), "pitch_width")

    for r in rows:
        for kx, ky in (("x", "y"), ("end_x", "end_y")):
            vx, vy = _row_float(r, kx), _row_float(r, ky)
            if vx is None or vy is None:
                continue
            nx, ny = transform_xy(
                vx, vy, src=src, dst=CoordinateFrame.FIFA_METRES,
                pitch_length_m=L, pitch_width_m=W,
            )
            _set_float(r, kx, round(nx, 4))
            _set_float(r, ky, round(ny, 4))
    return rows


def _op_keep_team(rows: list[dict], **p) -> list[dict]:
    team = _as_nonempty_str(p.get("team"), "team")
    return [r for r in rows if r.get("team") == team]


def _op_drop_team(rows: list[dict], **p) -> list[dict]:
    team = _as_nonempty_str(p.get("team"), "team")
    return [r for r in rows if r.get("team") != team]


def _op_drop_event_type(rows: list[dict], **p) -> list[dict]:
    et = _as_nonempty_str(p.get("event_type"), "event_type").lower()
    return [r for r in rows if (r.get("event_type") or "").lower() != et]


def _op_filter_time_range(rows: list[dict], **p) -> list[dict]:
    lo = _as_float(p.get("min_s", 0.0), "min_s")
    hi = _as_float(p.get("max_s", 1e9), "max_s")
    if hi < lo:
        raise PreprocessError(f"max_s ({hi}) must be >= min_s ({lo})")
    out = []
    for r in rows:
        v = _row_float(r, "timestamp")
        if v is None or lo <= v <= hi:
            out.append(r)
    return out


def _op_sort_by_time(rows: list[dict], **p) -> list[dict]:
    return sorted(rows, key=lambda r: (_row_float(r, "timestamp") or 0.0))


def _op_set_field(rows: list[dict], **p) -> list[dict]:
    field_name = _as_nonempty_str(p.get("field"), "field")
    value = p.get("value", "")
    where_team = p.get("where_team")
    where_type = p.get("where_event_type")
    for r in rows:
        if where_team and r.get("team") != where_team:
            continue
        if where_type and (r.get("event_type") or "").lower() != where_type.lower():
            continue
        r[field_name] = value
    return rows


def _op_snap_to_zone(rows: list[dict], **p) -> list[dict]:
    """Snap (x, y) to the centre of its containing pitch zone (3x3 grid by default).

    Useful for noisy hand-entered data — quantises coordinates to a small
    set of canonical points (e.g. "defensive third left half-space").
    """
    nx = int(p.get("nx", 3))
    ny = int(p.get("ny", 3))
    if nx < 1 or ny < 1:
        raise PreprocessError("nx, ny must be >= 1")
    L = _as_float(p.get("pitch_length", 105.0), "pitch_length")
    W = _as_float(p.get("pitch_width", 68.0), "pitch_width")
    cell_w = L / nx
    cell_h = W / ny
    for r in rows:
        for kx, ky in (("x", "y"), ("end_x", "end_y")):
            vx, vy = _row_float(r, kx), _row_float(r, ky)
            if vx is None or vy is None:
                continue
            ix = min(nx - 1, max(0, int(vx / cell_w)))
            iy = min(ny - 1, max(0, int(vy / cell_h)))
            _set_float(r, kx, round((ix + 0.5) * cell_w, 3))
            _set_float(r, ky, round((iy + 0.5) * cell_h, 3))
    return rows


def _op_deduplicate(rows: list[dict], **p) -> list[dict]:
    out: list[dict] = []
    last_signature: tuple | None = None
    for r in rows:
        # Compare on a stable subset of fields — full dict comparison would
        # catch noise like a comment column changing.
        sig = (
            r.get("timestamp"), r.get("team"), r.get("player"),
            r.get("event_type"), r.get("x"), r.get("y"),
        )
        if sig != last_signature:
            out.append(r)
            last_signature = sig
    return out


def _op_validate(rows: list[dict], **p) -> list[dict]:
    """Validation pass — does NOT mutate. Errors surface in PipelineResult.errors."""
    pitch_length = _as_float(p.get("pitch_length", 105.0), "pitch_length")
    pitch_width = _as_float(p.get("pitch_width", 68.0), "pitch_width")
    pitch = Pitch(length=pitch_length, width=pitch_width)
    bad: list[dict] = []
    for i, r in enumerate(rows):
        err = validate_row(r, pitch=pitch)
        if err:
            bad.append({"row": i, "errors": err, "snippet": _snippet(r)})
    if bad:
        # surface as PreprocessError so pipeline reports a structured error
        raise PreprocessError(f"validation failed for {len(bad)} row(s): {bad[:10]}")
    return rows


def _snippet(r: dict) -> dict:
    return {k: r.get(k) for k in ("timestamp", "team", "player", "event_type", "x", "y") if k in r}


# ── single-row validator (also used by the /api/events endpoint) ─────────────

def validate_row(row: dict, *, pitch: Pitch | None = None) -> list[str]:
    """
    Validate a single CSV-style row dict.

    Returns the list of error messages; an empty list means the row is
    well-formed. Uses the same enum maps as the CSV loader so the rules
    stay consistent.
    """
    from .csv_loader import (
        _BODY_PART_MAP,
        _EVENT_TYPE_MAP,
        _FOOT_MAP,
        _GK_ACTION_MAP,
        _OUTCOME_MAP,
        _PASS_TYPE_MAP,
        _SP_TYPE_MAP,
    )
    errors: list[str] = []
    p = pitch or Pitch()

    # required fields
    for req in ("timestamp", "team", "player", "event_type", "x", "y"):
        if row.get(req) in (None, ""):
            errors.append(f"missing required field: {req}")

    if errors:
        return errors

    # enum membership checks
    et = (row.get("event_type") or "").lower()
    if et not in _EVENT_TYPE_MAP:
        errors.append(f"unknown event_type {et!r}; valid: {sorted(_EVENT_TYPE_MAP)}")

    if (oc := row.get("outcome")):
        if oc.lower() not in _OUTCOME_MAP:
            errors.append(f"unknown outcome {oc!r}")

    if (foot := row.get("foot")):
        if foot.lower() not in _FOOT_MAP:
            errors.append(f"unknown foot {foot!r}")

    if (bp := row.get("body_part")):
        if bp.lower() not in _BODY_PART_MAP:
            errors.append(f"unknown body_part {bp!r}")

    if (pt := row.get("pass_type")):
        if pt.lower() not in _PASS_TYPE_MAP:
            errors.append(f"unknown pass_type {pt!r}")

    if (sp := row.get("set_piece_type")):
        if sp.lower() not in _SP_TYPE_MAP:
            errors.append(f"unknown set_piece_type {sp!r}")

    if (gk := row.get("gk_action_type")):
        if gk.lower() not in _GK_ACTION_MAP:
            errors.append(f"unknown gk_action_type {gk!r}")

    # numeric / range checks
    try:
        ts = float(row["timestamp"])
        if ts < 0:
            errors.append(f"timestamp must be >= 0, got {ts}")
    except (TypeError, ValueError):
        errors.append(f"timestamp must be a number, got {row.get('timestamp')!r}")

    for kx, ky in (("x", "y"), ("end_x", "end_y")):
        vx = row.get(kx)
        vy = row.get(ky)
        if vx in (None, "") or vy in (None, ""):
            continue
        try:
            fx, fy = float(vx), float(vy)
        except (TypeError, ValueError):
            errors.append(f"{kx},{ky} must be numbers")
            continue
        if not (0.0 <= fx <= p.length):
            errors.append(f"{kx}={fx} outside pitch [0, {p.length}]")
        if not (0.0 <= fy <= p.width):
            errors.append(f"{ky}={fy} outside pitch [0, {p.width}]")

    if (card := row.get("card")):
        if card.lower() not in ("yellow", "red", ""):
            errors.append(f"unknown card {card!r}")

    return errors


# ── handler dispatch + introspection ─────────────────────────────────────────

_HANDLERS: dict[str, Callable[..., list[dict]]] = {
    "rename_player":     _op_rename_player,
    "rename_team":       _op_rename_team,
    "swap_teams":        _op_swap_teams,
    "mirror":            _op_mirror,
    "time_shift":        _op_time_shift,
    "time_scale":        _op_time_scale,
    "round_time":        _op_round_time,
    "clamp_to_pitch":    _op_clamp_to_pitch,
    "transform_coords":  _op_transform_coords,
    "keep_team":         _op_keep_team,
    "drop_team":         _op_drop_team,
    "drop_event_type":   _op_drop_event_type,
    "filter_time_range": _op_filter_time_range,
    "sort_by_time":      _op_sort_by_time,
    "set_field":         _op_set_field,
    "snap_to_zone":      _op_snap_to_zone,
    "deduplicate":       _op_deduplicate,
    "validate":          _op_validate,
}


_SPECS: list[OpSpec] = [
    OpSpec(
        op="rename_player",
        label="Rename player",
        description="Replace every occurrence of a player name (also in to_player / tackled_player / fouled_player).",
        params=[
            {"name": "from", "type": "str", "required": True, "label": "From"},
            {"name": "to",   "type": "str", "required": True, "label": "To"},
        ],
    ),
    OpSpec(
        op="rename_team",
        label="Rename team",
        description="Replace every occurrence of a team name.",
        params=[
            {"name": "from", "type": "str", "required": True, "label": "From"},
            {"name": "to",   "type": "str", "required": True, "label": "To"},
        ],
    ),
    OpSpec(
        op="swap_teams",
        label="Swap home / away",
        description="Swap the two team labels and mirror coordinates so the attacking direction stays consistent.",
        params=[
            {"name": "pitch_length", "type": "float", "default": 105.0, "label": "Pitch length (m)"},
        ],
    ),
    OpSpec(
        op="mirror",
        label="Mirror coordinates",
        description="Reflect coordinates along the chosen axis.",
        params=[
            {"name": "axis", "type": "enum", "enum": ["x", "y"], "default": "x", "label": "Axis"},
            {"name": "pitch_length", "type": "float", "default": 105.0, "label": "Pitch length (m)"},
            {"name": "pitch_width",  "type": "float", "default":  68.0, "label": "Pitch width (m)"},
        ],
    ),
    OpSpec(
        op="time_shift",
        label="Shift timestamps",
        description="Add a constant (in seconds) to every timestamp. Negative values shift earlier.",
        params=[{"name": "delta_s", "type": "float", "default": 0.0, "label": "Δ seconds"}],
    ),
    OpSpec(
        op="time_scale",
        label="Scale timestamps",
        description="Multiply every timestamp by a constant. Useful for fps mismatches.",
        params=[{"name": "factor", "type": "float", "default": 1.0, "label": "Factor"}],
    ),
    OpSpec(
        op="round_time",
        label="Round timestamps",
        description="Round timestamps to N decimal places.",
        params=[{"name": "ndigits", "type": "int", "default": 1, "label": "Decimal places"}],
    ),
    OpSpec(
        op="clamp_to_pitch",
        label="Clamp to pitch",
        description="Clip x/y/end_x/end_y into [0, length] × [0, width].",
        params=[
            {"name": "pitch_length", "type": "float", "default": 105.0, "label": "Pitch length (m)"},
            {"name": "pitch_width",  "type": "float", "default":  68.0, "label": "Pitch width (m)"},
        ],
    ),
    OpSpec(
        op="transform_coords",
        label="Transform coordinate frame",
        description="Convert coordinates from another provider frame into FIFA metres (105 × 68).",
        params=[
            {"name": "from_frame", "type": "enum",
             "enum": ["statsbomb", "opta_percent", "normalised", "centred_cm"],
             "required": True, "label": "From frame"},
            {"name": "pitch_length", "type": "float", "default": 105.0, "label": "Target pitch length (m)"},
            {"name": "pitch_width",  "type": "float", "default":  68.0, "label": "Target pitch width (m)"},
        ],
    ),
    OpSpec(
        op="keep_team",
        label="Keep only one team",
        description="Drop every row whose team is not the supplied one.",
        params=[{"name": "team", "type": "str", "required": True, "label": "Team"}],
    ),
    OpSpec(
        op="drop_team",
        label="Drop a team",
        description="Drop every row from the supplied team.",
        params=[{"name": "team", "type": "str", "required": True, "label": "Team"}],
    ),
    OpSpec(
        op="drop_event_type",
        label="Drop event type",
        description="Drop every row of a given event_type.",
        params=[{"name": "event_type", "type": "str", "required": True, "label": "Event type"}],
    ),
    OpSpec(
        op="filter_time_range",
        label="Keep time range",
        description="Keep rows whose timestamp lies in [min, max] seconds.",
        params=[
            {"name": "min_s", "type": "float", "default":   0.0, "label": "Min (s)"},
            {"name": "max_s", "type": "float", "default": 5400.0, "label": "Max (s)"},
        ],
    ),
    OpSpec(
        op="sort_by_time",
        label="Sort by timestamp",
        description="Stable-sort rows by timestamp ascending.",
        params=[],
    ),
    OpSpec(
        op="set_field",
        label="Set field value",
        description="Set a single column to a fixed value on every row (optionally filtered).",
        params=[
            {"name": "field", "type": "str", "required": True, "label": "Field"},
            {"name": "value", "type": "str", "default": "",    "label": "Value"},
            {"name": "where_team",       "type": "str", "default": "", "label": "Only on team (optional)"},
            {"name": "where_event_type", "type": "str", "default": "", "label": "Only on event_type (optional)"},
        ],
    ),
    OpSpec(
        op="snap_to_zone",
        label="Snap to pitch zone",
        description="Quantise every coordinate to the centre of its enclosing zone (nx × ny grid).",
        params=[
            {"name": "nx", "type": "int", "default": 3, "label": "Zones along length"},
            {"name": "ny", "type": "int", "default": 3, "label": "Zones along width"},
            {"name": "pitch_length", "type": "float", "default": 105.0},
            {"name": "pitch_width",  "type": "float", "default":  68.0},
        ],
    ),
    OpSpec(
        op="deduplicate",
        label="Deduplicate consecutive rows",
        description="Drop rows byte-identical to the previous row on (timestamp, team, player, type, x, y).",
        params=[],
    ),
    OpSpec(
        op="validate",
        label="Validate rows (dry-run)",
        description="Run every row through the typed loader and report errors without mutating data.",
        params=[
            {"name": "pitch_length", "type": "float", "default": 105.0},
            {"name": "pitch_width",  "type": "float", "default":  68.0},
        ],
        mutates=False,
    ),
]
