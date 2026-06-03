"""
Coordinate-frame transforms between every supported provider system.

Five frames cover all 20 surveyed providers; see ``docs/PROVIDERS.md``
§2.3. Transforms are affine and lossless modulo float precision.
"""

from __future__ import annotations

from enum import Enum


class CoordinateFrame(str, Enum):
    """Supported coordinate frames."""

    FIFA_METRES = "fifa_metres"          # canonical: 105 × 68 m, origin BL
    STATSBOMB = "statsbomb"              # 120 × 80 yards, origin TL (y down)
    OPTA_PERCENT = "opta_percent"        # 100 × 100 %, origin BL
    NORMALISED = "normalised"            # 1 × 1, origin BL
    CENTRED_CM = "centred_cm"            # ±5250 × ±3400 cm, origin centre


# StatsBomb publishes yards but their coordinate frame is documented as
# 120 × 80 with y running TOP-down (i.e. y=0 is the touchline at top).
_STATSBOMB_LENGTH = 120.0
_STATSBOMB_WIDTH = 80.0
_YARDS_TO_METRES = 0.9144


def transform_xy(
    x: float,
    y: float,
    *,
    src: CoordinateFrame,
    dst: CoordinateFrame,
    pitch_length_m: float = 105.0,
    pitch_width_m: float = 68.0,
) -> tuple[float, float]:
    """Convert ``(x, y)`` between coordinate frames."""
    if src is dst:
        return float(x), float(y)
    # Convert src → FIFA_METRES, then FIFA_METRES → dst.
    xm, ym = _to_metres(x, y, src, pitch_length_m, pitch_width_m)
    if dst is CoordinateFrame.FIFA_METRES:
        return xm, ym
    return _from_metres(xm, ym, dst, pitch_length_m, pitch_width_m)


def _to_metres(
    x: float,
    y: float,
    frame: CoordinateFrame,
    L: float,
    W: float,
) -> tuple[float, float]:
    if frame is CoordinateFrame.FIFA_METRES:
        return float(x), float(y)
    if frame is CoordinateFrame.STATSBOMB:
        # StatsBomb has y running top-down on a 120x80 yards pitch.
        # Flip y, then scale yards → metres → user pitch dims.
        x_yd = float(x)
        y_yd = _STATSBOMB_WIDTH - float(y)
        x_m_full = x_yd * _YARDS_TO_METRES
        y_m_full = y_yd * _YARDS_TO_METRES
        return (
            x_m_full * (L / (_STATSBOMB_LENGTH * _YARDS_TO_METRES)),
            y_m_full * (W / (_STATSBOMB_WIDTH * _YARDS_TO_METRES)),
        )
    if frame is CoordinateFrame.OPTA_PERCENT:
        return float(x) / 100.0 * L, float(y) / 100.0 * W
    if frame is CoordinateFrame.NORMALISED:
        return float(x) * L, float(y) * W
    if frame is CoordinateFrame.CENTRED_CM:
        # Tracab-style: centred at (0,0), units cm. Convert to metres,
        # then translate origin to bottom-left.
        return float(x) / 100.0 + L / 2.0, float(y) / 100.0 + W / 2.0
    raise ValueError(f"unsupported source frame: {frame!r}")


def _from_metres(
    x_m: float,
    y_m: float,
    frame: CoordinateFrame,
    L: float,
    W: float,
) -> tuple[float, float]:
    if frame is CoordinateFrame.FIFA_METRES:
        return x_m, y_m
    if frame is CoordinateFrame.STATSBOMB:
        x_full = x_m * (_STATSBOMB_LENGTH * _YARDS_TO_METRES) / L
        y_full = y_m * (_STATSBOMB_WIDTH * _YARDS_TO_METRES) / W
        x_yd = x_full / _YARDS_TO_METRES
        y_yd = y_full / _YARDS_TO_METRES
        return x_yd, _STATSBOMB_WIDTH - y_yd
    if frame is CoordinateFrame.OPTA_PERCENT:
        return x_m / L * 100.0, y_m / W * 100.0
    if frame is CoordinateFrame.NORMALISED:
        return x_m / L, y_m / W
    if frame is CoordinateFrame.CENTRED_CM:
        return (x_m - L / 2.0) * 100.0, (y_m - W / 2.0) * 100.0
    raise ValueError(f"unsupported destination frame: {frame!r}")
