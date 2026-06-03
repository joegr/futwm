"""
Evaluation metrics for 3-way match outcome predictions.

Implements the standard scoring rules:

* **Brier score** (multi-class, lower is better, perfect = 0)
* **Log loss** (cross-entropy, lower is better, perfect = 0)
* **Accuracy** (top-1 hit rate)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .predictor import MatchPrediction


@dataclass
class EvaluationResult:
    """Container for batch evaluation outputs."""
    brier:    float
    log_loss: float
    accuracy: float
    n:        int

    def to_dict(self) -> dict:
        return {
            "brier":    self.brier,
            "log_loss": self.log_loss,
            "accuracy": self.accuracy,
            "n":        self.n,
        }


# ── outcome encoding ─────────────────────────────────────────────────────────

def _observed_outcome_vec(home_goals: int, away_goals: int) -> tuple[int, int, int]:
    """One-hot (home_win, draw, away_win) of the observed result."""
    if home_goals > away_goals:
        return (1, 0, 0)
    if home_goals < away_goals:
        return (0, 0, 1)
    return (0, 1, 0)


def _predicted_vec(p: MatchPrediction) -> tuple[float, float, float]:
    return (p.p_home_win, p.p_draw, p.p_away_win)


# ── metrics ──────────────────────────────────────────────────────────────────

def brier_score(
    predictions: list[MatchPrediction],
    matches:     list[dict],
) -> float:
    """
    Multi-class Brier score: mean squared error between predicted probability
    vector and one-hot observed outcome.
    """
    if len(predictions) != len(matches):
        raise ValueError("predictions and matches must be the same length")
    if not matches:
        return 0.0

    total = 0.0
    for pred, m in zip(predictions, matches):
        obs = _observed_outcome_vec(int(m["home_goals"]), int(m["away_goals"]))
        prb = _predicted_vec(pred)
        total += sum((p - o) ** 2 for p, o in zip(prb, obs))
    return total / len(matches)


def log_loss(
    predictions: list[MatchPrediction],
    matches:     list[dict],
    *,
    eps: float = 1e-12,
) -> float:
    """
    Categorical cross-entropy of predicted probabilities against the observed
    one-hot outcome.
    """
    if len(predictions) != len(matches):
        raise ValueError("predictions and matches must be the same length")
    if not matches:
        return 0.0

    total = 0.0
    for pred, m in zip(predictions, matches):
        obs = _observed_outcome_vec(int(m["home_goals"]), int(m["away_goals"]))
        prb = _predicted_vec(pred)
        for p, o in zip(prb, obs):
            if o:
                total -= math.log(max(p, eps))
    return total / len(matches)


def accuracy(
    predictions: list[MatchPrediction],
    matches:     list[dict],
) -> float:
    """Fraction of matches where the most-probable outcome matches reality."""
    if len(predictions) != len(matches):
        raise ValueError("predictions and matches must be the same length")
    if not matches:
        return 0.0

    hits = 0
    for pred, m in zip(predictions, matches):
        actual = (
            "home_win" if m["home_goals"] > m["away_goals"]
            else "away_win" if m["home_goals"] < m["away_goals"]
            else "draw"
        )
        if pred.most_likely_outcome() == actual:
            hits += 1
    return hits / len(matches)


def evaluate_predictions(
    predictions: list[MatchPrediction],
    matches:     list[dict],
) -> dict:
    """Compute all three metrics in one pass."""
    return EvaluationResult(
        brier=brier_score(predictions, matches),
        log_loss=log_loss(predictions, matches),
        accuracy=accuracy(predictions, matches),
        n=len(matches),
    ).to_dict()


# ── home-advantage calibration ───────────────────────────────────────────────

def calibrate_home_advantage(
    matches:    list[dict],
    *,
    elo=None,
    grid:       tuple[float, ...] = (0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0),
    metric:     str = "log_loss",
) -> dict:
    """
    Grid-search the optimal home-advantage value on a labelled match dataset.

    Only matches with explicit ``venue`` set to one of the two teams (or
    omitted with ``neutral=False``) actually exercise the home-advantage
    parameter. Pure neutral-venue fixtures are evaluated unchanged.

    Parameters
    ----------
    matches : list[dict]
        Each dict needs ``home``, ``away``, ``home_goals``, ``away_goals``;
        plus optional ``venue`` and ``neutral``.
    elo : EloRating, optional
        Rating book. Defaults to the bundled :data:`INITIAL_RATINGS`.
    grid : tuple[float, ...]
        Candidate home-advantage values to evaluate.
    metric : {"brier", "log_loss", "accuracy"}
        Metric to minimise (or maximise for accuracy).

    Returns
    -------
    dict
        ``{"best": float, "by_value": {h: result_dict, ...}, "metric": metric}``
    """
    from .predictor import EloPredictor   # local import to avoid cycle

    if metric not in ("brier", "log_loss", "accuracy"):
        raise ValueError(f"Unknown metric: {metric!r}")
    if not grid:
        raise ValueError("grid must be non-empty")
    if not matches:
        raise ValueError("matches must be non-empty")
    # Validate grid entries up-front
    grid_f: list[float] = []
    for h in grid:
        h_f = float(h)
        if not (h_f == h_f and abs(h_f) != float("inf")):  # finite check w/o math import
            raise ValueError(f"grid value not finite: {h!r}")
        grid_f.append(h_f)

    by_value: dict[float, dict] = {}
    for h_f in grid_f:
        kwargs = {"home_advantage": h_f}
        predictor = EloPredictor(elo=elo.copy(), **kwargs) if elo else EloPredictor(**kwargs)
        by_value[h_f] = predictor.evaluate(matches)

    if metric == "accuracy":
        best = max(by_value, key=lambda v: by_value[v]["accuracy"])
    else:
        best = min(by_value, key=lambda v: by_value[v][metric])

    return {"best": best, "by_value": by_value, "metric": metric}
