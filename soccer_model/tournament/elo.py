"""
Elo rating system for international football.

Implements the World Football Elo Ratings formulation
(http://www.eloratings.net/about), which extends standard Elo with:

* Tournament-weighted K-factor
* Goal-difference multiplier
* Optional home advantage adjustment

Formulae
--------
Expected score (probability of "win-equivalent" for team A):

    E_A = 1 / (1 + 10 ** (-(R_A - R_B + H) / 400))

where H is the home advantage (0 if neutral / 100 by convention here).

Update rule:

    R_A' = R_A + K * G * (W - E_A)

where:
* K  — tournament weight (15 friendlies, 30 qualifiers, 50 WC, 60 final)
* G  — goal-difference multiplier:
        N == 0..1   → 1
        N == 2      → 1.5
        N >= 3      → (11 + N) / 8
* W  — actual result: 1 win, 0.5 draw, 0 loss
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

# ── constants ────────────────────────────────────────────────────────────────

#: Default K-factor for unspecified matches (qualifiers / continental finals).
DEFAULT_K_FACTOR: float = 40.0

#: K-factor for World Cup group / knockout matches (per eloratings.net).
K_FACTOR_WORLD_CUP: float = 50.0

#: K-factor for the World Cup final.
K_FACTOR_WORLD_CUP_FINAL: float = 60.0

#: K-factor for friendlies.
K_FACTOR_FRIENDLY: float = 15.0

#: Conventional home-advantage Elo bonus (points) for a regular international
#: match played in the home team's country. eloratings.net uses 100; empirical
#: studies of international football put the value in the 80–110 range.
DEFAULT_HOME_ADVANTAGE: float = 100.0

#: Smaller advantage applied to a World Cup *host nation* playing in its own
#: country during the tournament. Hosts benefit from familiarity, climate and
#: crowd, but international neutral-venue matches at a host's stadium are not
#: equivalent to a competitive league home leg, so the value is typically
#: lower than DEFAULT_HOME_ADVANTAGE. Empirically ≈ 50 Elo.
HOST_NATION_ADVANTAGE: float = 50.0

#: Initial / mean rating for an unrated team.
DEFAULT_INITIAL_RATING: float = 1500.0


# ── core functions ───────────────────────────────────────────────────────────

def expected_score(
    rating_a: float,
    rating_b: float,
    home_advantage: float = 0.0,
) -> float:
    """
    Probability that team A wins (or earns the W=1 result), accounting for
    home advantage applied to A. Use ``-home_advantage`` to put B at home.
    """
    diff = rating_a - rating_b + home_advantage
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def goal_difference_multiplier(goal_diff: int) -> float:
    """
    eloratings.net goal-difference multiplier *G*.

    Parameters
    ----------
    goal_diff : int
        Absolute goal difference (|home_goals - away_goals|).
    """
    n = abs(int(goal_diff))
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return (11.0 + n) / 8.0


def _validate_rating(name: str, value: float) -> float:
    """Raise if *value* is not a finite number."""
    if isinstance(value, bool):  # bool is a subclass of int; reject explicitly
        raise TypeError(f"{name} must be a real number, got bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,
    *,
    goal_diff: int = 0,
    k: float = DEFAULT_K_FACTOR,
    home_advantage: float = 0.0,
) -> tuple[float, float]:
    """
    Apply one Elo update for a match between A (with home advantage applied)
    and B.

    Parameters
    ----------
    rating_a, rating_b : float
        Pre-match ratings.
    score_a : float
        Actual result for A: 1.0 win, 0.5 draw, 0.0 loss.
    goal_diff : int
        |goals_a - goals_b|; used by the G multiplier.
    k : float
        Tournament weight (K-factor).
    home_advantage : float
        Elo bonus to add to A's effective rating (use 0 for neutral venue).

    Returns
    -------
    (new_rating_a, new_rating_b)
    """
    e_a   = expected_score(rating_a, rating_b, home_advantage)
    g     = goal_difference_multiplier(goal_diff)
    delta = k * g * (score_a - e_a)
    return rating_a + delta, rating_b - delta


# ── stateful container ───────────────────────────────────────────────────────

@dataclass
class EloRating:
    """
    Mutable Elo rating book keyed by team name.

    Provides a thin wrapper that tracks every team's current rating and applies
    bulk updates from match results.

    Thread safety
    -------------
    This class is **not** thread-safe. Wrap it in your own lock if multiple
    threads will update concurrently. Single-producer streaming (one match at
    a time, in arrival order) is the recommended usage pattern.
    """
    ratings: dict[str, float] = field(default_factory=dict)
    default: float = DEFAULT_INITIAL_RATING

    def __post_init__(self) -> None:
        if not isinstance(self.ratings, dict):
            raise TypeError(
                f"ratings must be a dict, got {type(self.ratings).__name__}"
            )
        # Validate provided ratings up-front so streaming updates don't fail
        # in the middle of a batch with a confusing math error.
        validated: dict[str, float] = {}
        for team, value in self.ratings.items():
            if not isinstance(team, str) or not team:
                raise TypeError(f"team key must be a non-empty string, got {team!r}")
            validated[team] = _validate_rating(f"ratings[{team!r}]", value)
        self.ratings = validated
        self.default = _validate_rating("default", self.default)

    def get(self, team: str) -> float:
        """Return the current rating for *team*, or the default if unrated."""
        return self.ratings.get(team, self.default)

    def has(self, team: str) -> bool:
        """Return True if *team* has an explicit rating (no default fallback)."""
        return team in self.ratings

    def set(self, team: str, rating: float) -> None:
        """Set the rating for *team*. Validates that *rating* is finite."""
        if not isinstance(team, str) or not team:
            raise TypeError("team must be a non-empty string")
        self.ratings[team] = _validate_rating(f"rating for {team!r}", rating)

    def expected(
        self,
        home: str,
        away: str,
        *,
        home_advantage: float = DEFAULT_HOME_ADVANTAGE,
        neutral: bool = False,
    ) -> float:
        """Probability of a "home win-equivalent" outcome (W=1 for home)."""
        h = 0.0 if neutral else home_advantage
        return expected_score(self.get(home), self.get(away), h)

    def record_match(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        *,
        k: float = DEFAULT_K_FACTOR,
        home_advantage: float = DEFAULT_HOME_ADVANTAGE,
        neutral: bool = False,
    ) -> tuple[float, float]:
        """
        Update both teams' ratings from a single observed result.

        Returns the (new_home_rating, new_away_rating) tuple.

        Raises
        ------
        ValueError
            If ``home == away``, goal counts are negative, or ``k`` is not
            a finite non-negative number.
        """
        if not isinstance(home, str) or not home:
            raise TypeError("home must be a non-empty string")
        if not isinstance(away, str) or not away:
            raise TypeError("away must be a non-empty string")
        if home == away:
            raise ValueError(f"home and away must be different teams (both {home!r})")
        if not isinstance(home_goals, int) or isinstance(home_goals, bool):
            raise TypeError(f"home_goals must be int, got {type(home_goals).__name__}")
        if not isinstance(away_goals, int) or isinstance(away_goals, bool):
            raise TypeError(f"away_goals must be int, got {type(away_goals).__name__}")
        if home_goals < 0 or away_goals < 0:
            raise ValueError(f"goal counts must be non-negative, got {home_goals}-{away_goals}")
        k_v = _validate_rating("k", k)
        if k_v < 0:
            raise ValueError(f"k must be non-negative, got {k}")

        # actual score from home's perspective
        if home_goals > away_goals:
            score_h = 1.0
        elif home_goals < away_goals:
            score_h = 0.0
        else:
            score_h = 0.5

        h = 0.0 if neutral else _validate_rating("home_advantage", home_advantage)
        new_h, new_a = update_elo(
            self.get(home),
            self.get(away),
            score_h,
            goal_diff=abs(home_goals - away_goals),
            k=k_v,
            home_advantage=h,
        )
        self.set(home, new_h)
        self.set(away, new_a)
        return new_h, new_a

    # ── streaming / batch ────────────────────────────────────────────────────

    def update_from_iterable(
        self,
        matches: Iterable[dict],
        *,
        k: float = DEFAULT_K_FACTOR,
        home_advantage: float = DEFAULT_HOME_ADVANTAGE,
        skip_invalid: bool = False,
    ) -> int:
        """
        Apply Elo updates from an iterable of match dicts. Supports streaming
        (single-pass generators are fine — nothing is materialised internally).

        Each match dict must contain ``home``, ``away``, ``home_goals``,
        ``away_goals``. Optional per-match keys override the defaults:
        ``k``, ``home_advantage``, ``neutral``.

        Parameters
        ----------
        matches : iterable of dict
        k, home_advantage : float
            Defaults applied when a match dict does not specify them.
        skip_invalid : bool
            If True, malformed matches are silently skipped (logged via
            ``warnings``) instead of aborting the stream. Useful for live
            data with occasional bad records.

        Returns
        -------
        int
            Number of matches successfully applied.
        """
        import warnings

        applied = 0
        for i, m in enumerate(matches):
            try:
                self.record_match(
                    m["home"], m["away"],
                    int(m["home_goals"]), int(m["away_goals"]),
                    k=float(m.get("k", k)),
                    home_advantage=float(m.get("home_advantage", home_advantage)),
                    neutral=bool(m.get("neutral", False)),
                )
                applied += 1
            except (KeyError, TypeError, ValueError) as e:
                if not skip_invalid:
                    raise
                warnings.warn(
                    f"skipping match #{i} ({m!r}): {e}",
                    stacklevel=2,
                )
        return applied

    # ── serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """JSON-friendly snapshot of the rating book."""
        return {"ratings": dict(self.ratings), "default": float(self.default)}

    @classmethod
    def from_dict(cls, data: dict) -> EloRating:
        """Reconstruct an :class:`EloRating` from :meth:`to_dict` output."""
        if not isinstance(data, dict):
            raise TypeError(f"expected dict, got {type(data).__name__}")
        return cls(
            ratings=dict(data.get("ratings", {})),
            default=float(data.get("default", DEFAULT_INITIAL_RATING)),
        )

    def copy(self) -> EloRating:
        """Return a shallow copy with an independent ratings dict."""
        return EloRating(ratings=dict(self.ratings), default=self.default)
