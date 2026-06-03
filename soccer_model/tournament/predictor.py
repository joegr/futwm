"""
Elo-based match outcome predictor.

Converts Elo expected score → 3-way (win / draw / loss) probabilities and
samples plausible scorelines via a calibrated Poisson model.

Design
------
Plain Elo only yields a *win-equivalent* expected score E ∈ [0, 1] which
folds draws into half-credit. To recover three-way probabilities we apply
a Davidson-style draw adjustment:

    p_draw      ≈ DRAW_FACTOR * exp(-(rating_diff)^2 / DRAW_SCALE^2)
    p_home_win  ≈ E - p_draw / 2
    p_away_win  ≈ (1 - E) - p_draw / 2

(clipped to [0, 1] and renormalised).

For score sampling we use independent Poisson goal counts whose means are
derived from each team's Elo above league average and a tournament-wide
expected goals-per-team baseline. This is intentionally simple — the goal
is a *baseline* to compare more sophisticated event-level models against.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field

from .elo import (
    DEFAULT_HOME_ADVANTAGE,
    DEFAULT_K_FACTOR,
    EloRating,
    expected_score,
)
from .ratings import INITIAL_RATINGS
from .teams import get_team


class UnknownTeamWarning(UserWarning):
    """Emitted when a team name is not in the rating book (strict=False)."""


def _canonical(name: str) -> tuple[str, bool]:
    """
    Resolve *name* to the canonical WC2026 team name.

    Returns
    -------
    (canonical_name, recognised)
        ``recognised`` is True if *name* matched a WC2026 team. When False,
        the original *name* is returned unchanged so callers can decide
        how to handle the unknown case.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"team identifier must be a non-empty string, got {name!r}")
    try:
        return get_team(name).name, True
    except KeyError:
        return name, False


def _resolve_advantage(
    home: str,
    away: str,
    *,
    venue: str | None,
    neutral: bool,
    home_advantage: float,
) -> float:
    """
    Resolve the effective Elo home-advantage bonus to add to the home team.

    Rules
    -----
    * ``neutral=True``                          → 0 (overrides venue)
    * ``venue is None`` or ``venue == home``    → ``+home_advantage``
      (traditional Elo convention: match in home team's country)
    * ``venue == away``                         → ``-home_advantage``
      (rare inversion — fixture's nominal "home" is actually away)
    * any other venue                           → 0 (third-party neutral)

    Choice of ``home_advantage`` value is the caller's responsibility:

    * :data:`DEFAULT_HOME_ADVANTAGE` (≈ 100) for competitive home legs
      (qualifiers, friendlies in the home country).
    * :data:`HOST_NATION_ADVANTAGE` (≈ 50) for tournament neutral-venue
      contexts where one team happens to also be the tournament host.
    """
    if neutral:
        return 0.0
    if venue is None:
        return home_advantage
    v, _ = _canonical(venue)
    if v == home:
        return home_advantage
    if v == away:
        return -home_advantage
    return 0.0

# ── tuneable constants ───────────────────────────────────────────────────────

#: Maximum draw probability (when teams are evenly matched).
DRAW_FACTOR: float = 0.30

#: Elo-difference scale at which draws become unlikely.
DRAW_SCALE:  float = 280.0

#: Tournament-average expected goals per team per match (≈ WC 2022 mean).
BASE_GOALS_PER_TEAM: float = 1.30

#: Elo points equivalent to one extra expected goal (rough calibration).
ELO_POINTS_PER_GOAL: float = 130.0


# ── prediction dataclass ─────────────────────────────────────────────────────

@dataclass
class MatchPrediction:
    """Output of ``EloPredictor.predict`` for a single fixture."""
    home:           str
    away:           str
    home_rating:    float
    away_rating:    float
    p_home_win:     float
    p_draw:         float
    p_away_win:     float
    expected_home_goals: float
    expected_away_goals: float
    neutral:        bool = False

    def most_likely_outcome(self) -> str:
        """Return ``'home_win'``, ``'draw'``, or ``'away_win'``."""
        probs = {
            "home_win": self.p_home_win,
            "draw":     self.p_draw,
            "away_win": self.p_away_win,
        }
        return max(probs, key=probs.get)

    def to_dict(self) -> dict:
        return {
            "home":         self.home,
            "away":         self.away,
            "home_rating":  self.home_rating,
            "away_rating":  self.away_rating,
            "p_home_win":   self.p_home_win,
            "p_draw":       self.p_draw,
            "p_away_win":   self.p_away_win,
            "expected_home_goals": self.expected_home_goals,
            "expected_away_goals": self.expected_away_goals,
            "neutral":      self.neutral,
        }


# ── predictor ────────────────────────────────────────────────────────────────

@dataclass
class EloPredictor:
    """
    Baseline match-outcome predictor backed by an :class:`EloRating` book.

    Parameters
    ----------
    elo : EloRating, optional
        Pre-built rating book. Defaults to a copy of :data:`INITIAL_RATINGS`.
    home_advantage : float
        Elo bonus for the home side. Choose the value to match the
        competitive context:

        * :data:`DEFAULT_HOME_ADVANTAGE` (≈ 100) — competitive home leg
          (WC qualifier, friendly in the home country, league match).
        * :data:`HOST_NATION_ADVANTAGE` (≈ 50) — tournament neutral-venue
          where one team is the tournament host (e.g. Mexico playing in
          Mexico during WC 2026).
        * ``0`` — true neutral venue.

        Per-match overrides are also supported via the ``venue`` and
        ``neutral`` arguments to :meth:`predict` and
        :meth:`update_from_match`.

    strict : bool, default False
        If True, :meth:`predict` and :meth:`update_from_match` raise
        :class:`ValueError` when a team is not in the rating book. If
        False (the default), unknown teams use the rating book's default
        rating and an :class:`UnknownTeamWarning` is emitted.
    """
    elo:            EloRating = field(
        default_factory=lambda: EloRating(ratings=dict(INITIAL_RATINGS))
    )
    home_advantage: float     = DEFAULT_HOME_ADVANTAGE
    strict:         bool      = False

    # ── internal helpers ─────────────────────────────────────────────────────

    def _resolve_team(self, name: str) -> str:
        """
        Canonicalise *name* and either raise (strict) or warn (lenient) when
        the team is unknown.
        """
        canonical, recognised = _canonical(name)
        if recognised:
            return canonical
        # canonical is the raw input (unchanged); check if it's at least in
        # the rating book (a user may have seeded custom teams).
        if self.elo.has(canonical):
            return canonical
        if self.strict:
            raise ValueError(
                f"Unknown team {name!r}: not a WC2026 qualifier and not in "
                f"the rating book. Pass strict=False to fall back to the "
                f"default rating ({self.elo.default})."
            )
        warnings.warn(
            f"Unknown team {name!r}; using default rating {self.elo.default}. "
            f"Pass strict=True to raise instead.",
            UnknownTeamWarning,
            stacklevel=3,
        )
        return canonical

    # ── core prediction ──────────────────────────────────────────────────────

    def predict(
        self,
        home: str,
        away: str,
        *,
        venue: str | None = None,
        neutral: bool = False,
    ) -> MatchPrediction:
        """
        Produce 3-way outcome probabilities and expected goal counts.

        Parameters
        ----------
        home, away : str
            Team names or 3-letter codes.
        venue : str, optional
            Country / team where the match is played. Determines which
            home-advantage rule applies:

            * ``None`` (default) — match assumed in *home*'s country, full
              ``home_advantage`` applied.
            * ``home`` — venue *is* the home team's country; if home is also
              a tournament host, ``host_advantage`` is applied (smaller than
              full league home advantage).
            * any other team — venue is a third-party country (typical for
              WC neutral-venue group matches), no advantage applied.

        neutral : bool
            Force a fully neutral venue. Overrides ``venue``.
        """
        home = self._resolve_team(home)
        away = self._resolve_team(away)
        if home == away:
            raise ValueError(
                f"home and away must be different teams (both resolve to {home!r})"
            )
        r_h = self.elo.get(home)
        r_a = self.elo.get(away)
        h   = _resolve_advantage(
            home, away,
            venue=venue, neutral=neutral,
            home_advantage=self.home_advantage,
        )

        e_h = expected_score(r_h, r_a, h)               # home win-equivalent

        # Davidson-style draw probability, peaks when teams are level
        diff      = (r_h - r_a + h)
        p_draw    = DRAW_FACTOR * math.exp(-(diff ** 2) / (DRAW_SCALE ** 2))

        p_h_raw = e_h - p_draw / 2.0
        p_a_raw = (1.0 - e_h) - p_draw / 2.0

        # Clip and renormalise (handles edge cases where draw eats too much)
        p_h_raw = max(p_h_raw, 1e-6)
        p_a_raw = max(p_a_raw, 1e-6)
        total   = p_h_raw + p_draw + p_a_raw
        p_h     = p_h_raw / total
        p_d     = p_draw  / total
        p_a     = p_a_raw / total

        # expected-goal split: scale baseline by (rating_diff / scale)
        avg_rating = (r_h + r_a) / 2.0
        x_h = max(0.05, BASE_GOALS_PER_TEAM
                  + (r_h + h - avg_rating) / ELO_POINTS_PER_GOAL)
        x_a = max(0.05, BASE_GOALS_PER_TEAM
                  + (r_a - avg_rating) / ELO_POINTS_PER_GOAL)

        return MatchPrediction(
            home=home, away=away,
            home_rating=r_h, away_rating=r_a,
            p_home_win=p_h, p_draw=p_d, p_away_win=p_a,
            expected_home_goals=x_h, expected_away_goals=x_a,
            neutral=neutral,
        )

    # ── score sampling (Poisson) ─────────────────────────────────────────────

    def sample_score(
        self,
        home: str,
        away: str,
        *,
        venue: str | None = None,
        neutral: bool = False,
        rng=None,
    ) -> tuple[int, int]:
        """
        Sample a single plausible scoreline from independent Poisson goal
        counts derived from the prediction's expected-goal values.
        """
        import numpy as np

        if rng is None:
            rng = np.random.default_rng()
        pred = self.predict(home, away, venue=venue, neutral=neutral)
        h = int(rng.poisson(pred.expected_home_goals))
        a = int(rng.poisson(pred.expected_away_goals))
        return h, a

    # ── learning from observed matches ───────────────────────────────────────

    def update_from_match(
        self,
        home: str,
        away: str,
        home_goals: int,
        away_goals: int,
        *,
        k: float = DEFAULT_K_FACTOR,
        venue: str | None = None,
        neutral: bool = False,
    ) -> None:
        """Update Elo ratings from one observed result (in place)."""
        home = self._resolve_team(home)
        away = self._resolve_team(away)
        # record_match itself rejects home == away, but we also need the
        # canonical names in place before computing the venue advantage.
        h = _resolve_advantage(
            home, away,
            venue=venue, neutral=neutral,
            home_advantage=self.home_advantage,
        )
        # The underlying record_match expects an explicit advantage value,
        # passed via home_advantage with neutral=False so it is applied as-is.
        self.elo.record_match(
            home, away, home_goals, away_goals,
            k=k,
            home_advantage=h,
            neutral=False,
        )

    # ── streaming ingestion ──────────────────────────────────────────────────

    def update_from_stream(
        self,
        matches: Iterable[dict],
        *,
        k: float = DEFAULT_K_FACTOR,
        sort_by_date: bool = False,
        skip_invalid: bool = False,
    ) -> int:
        """
        Replay a sequence of match results, updating ratings in place.

        Designed for streaming scenarios — accepts any iterable, including
        single-pass generators. Each match dict must have ``home``, ``away``,
        ``home_goals``, ``away_goals`` and may optionally provide ``date``,
        ``venue``, ``neutral``, and ``k`` (per-match K-factor override).

        Parameters
        ----------
        matches : iterable of dict
        k : float
            Default K-factor when a match does not specify one.
        sort_by_date : bool
            If True, materialise the iterable and sort by the ``date`` key
            before applying. **Required** when historical replay arrives
            out of order, because Elo updates are path-dependent.
            Leave False for true live streams where matches arrive in
            chronological order.
        skip_invalid : bool
            If True, malformed records emit a warning and are skipped
            instead of aborting the stream.

        Returns
        -------
        int
            Number of matches successfully applied.
        """
        seq: Iterable[dict]
        if sort_by_date:
            seq = sorted(matches, key=lambda m: m.get("date") or "")
        else:
            seq = matches

        applied = 0
        for i, m in enumerate(seq):
            try:
                self.update_from_match(
                    m["home"], m["away"],
                    int(m["home_goals"]), int(m["away_goals"]),
                    k=float(m.get("k", k)),
                    venue=m.get("venue"),
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

    # ── batch evaluation ─────────────────────────────────────────────────────

    def evaluate(self, matches: list[dict]) -> dict:
        """
        Score predictions against a list of observed matches.

        Parameters
        ----------
        matches : list[dict]
            Each dict must contain ``home``, ``away``, ``home_goals``,
            ``away_goals`` and optionally ``neutral`` (bool).

        Returns
        -------
        dict
            ``{'brier': ..., 'log_loss': ..., 'accuracy': ..., 'n': N}``
        """
        from .evaluation import evaluate_predictions

        preds: list[MatchPrediction] = []
        for m in matches:
            preds.append(self.predict(
                m["home"], m["away"],
                venue=m.get("venue"),
                neutral=bool(m.get("neutral", False)),
            ))
        return evaluate_predictions(preds, matches)
