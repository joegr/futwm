"""
Pitch — canonical spatial reference for all event coordinates.

FIFA standard dimensions (metres):
  length : 105 m  (x-axis, 0 = home goal-line, 105 = away goal-line)
  width  :  68 m  (y-axis, 0 = bottom touchline, 68 = top touchline)

All event coordinates are validated against Pitch bounds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple

Vector2D = Tuple[float, float]


class PitchZone(Enum):
    """Longitudinal thirds relative to the *attacking* direction (left→right)."""
    DEFENSIVE_THIRD   = auto()
    MIDDLE_THIRD      = auto()
    ATTACKING_THIRD   = auto()


class PitchArea(Enum):
    """Named functional regions on the pitch."""
    OPEN_PLAY            = auto()
    OWN_PENALTY_AREA     = auto()
    OWN_GOAL_AREA        = auto()
    OPP_PENALTY_AREA     = auto()
    OPP_GOAL_AREA        = auto()
    OWN_HALF             = auto()
    OPP_HALF             = auto()
    CENTRE_CIRCLE        = auto()


@dataclass(frozen=True)
class GoalMouth:
    """Posts and crossbar specification for one goal."""
    center_x: float
    post_y_left:  float
    post_y_right: float
    crossbar_z:   float = 2.44

    @property
    def width(self) -> float:
        return abs(self.post_y_right - self.post_y_left)

    def angle_from(self, x: float, y: float) -> float:
        """Solid angle (radians) subtended by the goal mouth from position (x, y)."""
        dx = abs(self.center_x - x)
        half_w = self.width / 2.0
        if dx == 0:
            return math.pi
        return 2.0 * math.atan(half_w / dx)


@dataclass
class Pitch:
    """
    Spatial authority for the match.

    Parameters
    ----------
    length : float
        x-axis span in metres (default 105).
    width : float
        y-axis span in metres (default 68).
    """

    length: float = 105.0
    width:  float = 68.0

    # ── derived geometry (computed once) ────────────────────────────────────
    _goal_width:       float = field(init=False, repr=False)
    _goal_area_depth:  float = field(init=False, repr=False)
    _penalty_depth:    float = field(init=False, repr=False)
    _centre_radius:    float = field(init=False, repr=False)
    _home_goal:        GoalMouth = field(init=False, repr=False)
    _away_goal:        GoalMouth = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._goal_width      = 7.32
        self._goal_area_depth = 5.5
        self._penalty_depth   = 16.5
        self._centre_radius   = 9.15

        half_y     = self.width / 2.0
        half_goal  = self._goal_width / 2.0

        self._home_goal = GoalMouth(
            center_x=0.0,
            post_y_left=half_y - half_goal,
            post_y_right=half_y + half_goal,
        )
        self._away_goal = GoalMouth(
            center_x=self.length,
            post_y_left=half_y - half_goal,
            post_y_right=half_y + half_goal,
        )

    # ── coordinate validation ────────────────────────────────────────────────

    def contains(self, x: float, y: float) -> bool:
        """Return True if (x, y) lies within or on the pitch boundary."""
        return 0.0 <= x <= self.length and 0.0 <= y <= self.width

    def clamp(self, x: float, y: float) -> Vector2D:
        """Clamp (x, y) to the nearest in-bounds point."""
        return (
            max(0.0, min(self.length, x)),
            max(0.0, min(self.width,  y)),
        )

    def validate(self, x: float, y: float, label: str = "coordinate") -> None:
        """Raise ValueError if (x, y) is outside the pitch."""
        if not self.contains(x, y):
            raise ValueError(
                f"{label} ({x:.2f}, {y:.2f}) is outside pitch bounds "
                f"[0–{self.length}, 0–{self.width}]"
            )

    # ── spatial classification ───────────────────────────────────────────────

    def zone(self, x: float, attacking_direction: int = 1) -> PitchZone:
        """
        Return the longitudinal third for position x.

        Parameters
        ----------
        x : float
            x coordinate on the pitch.
        attacking_direction : int
            +1 if the team attacks left→right (home), -1 if right→left (away).
        """
        third = self.length / 3.0
        if attacking_direction == 1:
            if x < third:
                return PitchZone.DEFENSIVE_THIRD
            if x < 2 * third:
                return PitchZone.MIDDLE_THIRD
            return PitchZone.ATTACKING_THIRD
        else:
            if x > 2 * third:
                return PitchZone.DEFENSIVE_THIRD
            if x > third:
                return PitchZone.MIDDLE_THIRD
            return PitchZone.ATTACKING_THIRD

    def area(self, x: float, y: float) -> PitchArea:
        """Return the most specific named PitchArea for (x, y)."""
        half_y    = self.width / 2.0
        half_goal = self._goal_width / 2.0

        # ── own penalty / goal areas (x near 0) ──────────────────────────────
        if x <= self._penalty_depth:
            pen_y_lo = half_y - (self._goal_width / 2.0 + 11.0)
            pen_y_hi = half_y + (self._goal_width / 2.0 + 11.0)
            if pen_y_lo <= y <= pen_y_hi:
                if x <= self._goal_area_depth:
                    ga_y_lo = half_y - half_goal - 5.5
                    ga_y_hi = half_y + half_goal + 5.5
                    if ga_y_lo <= y <= ga_y_hi:
                        return PitchArea.OWN_GOAL_AREA
                return PitchArea.OWN_PENALTY_AREA

        # ── opp penalty / goal areas (x near length) ─────────────────────────
        if x >= self.length - self._penalty_depth:
            pen_y_lo = half_y - (self._goal_width / 2.0 + 11.0)
            pen_y_hi = half_y + (self._goal_width / 2.0 + 11.0)
            if pen_y_lo <= y <= pen_y_hi:
                if x >= self.length - self._goal_area_depth:
                    ga_y_lo = half_y - half_goal - 5.5
                    ga_y_hi = half_y + half_goal + 5.5
                    if ga_y_lo <= y <= ga_y_hi:
                        return PitchArea.OPP_GOAL_AREA
                return PitchArea.OPP_PENALTY_AREA

        # ── centre circle ─────────────────────────────────────────────────────
        cx, cy = self.length / 2.0, self.width / 2.0
        if math.hypot(x - cx, y - cy) <= self._centre_radius:
            return PitchArea.CENTRE_CIRCLE

        # ── halves ────────────────────────────────────────────────────────────
        if x <= self.length / 2.0:
            return PitchArea.OWN_HALF
        return PitchArea.OPP_HALF

    # ── spatial measurements ─────────────────────────────────────────────────

    def distance(self, a: Vector2D, b: Vector2D) -> float:
        """Euclidean distance between two pitch positions."""
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def bearing(self, origin: Vector2D, target: Vector2D) -> float:
        """Bearing in radians from origin to target (0 = right / +x direction)."""
        return math.atan2(target[1] - origin[1], target[0] - origin[0])

    def distance_to_goal(self, x: float, y: float, goal: str = "away") -> float:
        """
        Straight-line distance from (x, y) to the centre of the specified goal.

        Parameters
        ----------
        goal : str
            ``"away"`` (default, attacking goal) or ``"home"`` (own goal).
        """
        g = self._away_goal if goal == "away" else self._home_goal
        goal_y = (g.post_y_left + g.post_y_right) / 2.0
        return math.hypot(x - g.center_x, y - goal_y)

    def goal_angle(self, x: float, y: float, goal: str = "away") -> float:
        """Solid angle (radians) of the goal mouth as seen from (x, y)."""
        g = self._away_goal if goal == "away" else self._home_goal
        return g.angle_from(x, y)

    def pressure_index(
        self,
        pos: Vector2D,
        opponent_positions: list[Vector2D],
        radius: float = 5.0,
    ) -> float:
        """
        Spatial pressure: sum of inverse distances to opponents within *radius* metres.

        Returns a non-negative float; higher ⇒ more pressure.
        """
        total = 0.0
        for opp in opponent_positions:
            d = self.distance(pos, opp)
            if 0 < d <= radius:
                total += 1.0 / d
        return total

    # ── normalisation helpers ─────────────────────────────────────────────────

    def normalise(self, x: float, y: float) -> Vector2D:
        """Map (x, y) → [0, 1]² relative to pitch dimensions."""
        return (x / self.length, y / self.width)

    def denormalise(self, nx: float, ny: float) -> Vector2D:
        """Map normalised [0, 1]² back to pitch metres."""
        return (nx * self.length, ny * self.width)

    # ── display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"Pitch(length={self.length}m, width={self.width}m)"
