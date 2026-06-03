"""
soccer_model.tournament — FIFA World Cup 2026 baseline prediction module.

Provides:
- Canonical 48-team list (`TEAMS`, `Team`, `Confederation`)
- Approximate baseline Elo ratings as of late 2025 (`INITIAL_RATINGS`)
- Elo rating system (eloratings.net formula with WC K-factor)
- Match outcome predictor (win/draw/loss probabilities + score sampling)
- Evaluation metrics (Brier score, log-loss, accuracy) for baseline comparison
- Bundled baseline match dataset for cross-checking predictions

Quick example
-------------
>>> from soccer_model.tournament import EloPredictor, load_baseline_matches
>>> p = EloPredictor()
>>> p.predict("Argentina", "Brazil")
MatchPrediction(home='Argentina', away='Brazil', p_home_win=0.45, p_draw=0.27, p_away_win=0.28, ...)
>>> matches = load_baseline_matches()  # n < 50 historical matches
>>> p.evaluate(matches)
{'brier': 0.198, 'log_loss': 0.95, 'accuracy': 0.55, 'n': 30}
"""

from __future__ import annotations

from .elo import (
    DEFAULT_HOME_ADVANTAGE,
    DEFAULT_K_FACTOR,
    HOST_NATION_ADVANTAGE,
    K_FACTOR_WORLD_CUP,
    K_FACTOR_WORLD_CUP_FINAL,
    EloRating,
    expected_score,
    goal_difference_multiplier,
    update_elo,
)
from .evaluation import (
    EvaluationResult,
    accuracy,
    brier_score,
    calibrate_home_advantage,
    evaluate_predictions,
    log_loss,
)
from .predictor import EloPredictor, MatchPrediction, UnknownTeamWarning
from .ratings import INITIAL_RATINGS, INITIAL_RATINGS_AS_OF, get_rating
from .teams import (
    TEAMS,
    TEAMS_BY_CONFEDERATION,
    Confederation,
    Team,
    get_team,
    is_qualified,
    load_baseline_matches,
)

__all__ = [
    "DEFAULT_HOME_ADVANTAGE",
    "DEFAULT_K_FACTOR",
    "EvaluationResult",
    "HOST_NATION_ADVANTAGE",
    "INITIAL_RATINGS",
    "INITIAL_RATINGS_AS_OF",
    "K_FACTOR_WORLD_CUP",
    "K_FACTOR_WORLD_CUP_FINAL",
    "TEAMS",
    "TEAMS_BY_CONFEDERATION",
    "Confederation",
    "EloPredictor",
    "EloRating",
    "MatchPrediction",
    "Team",
    "UnknownTeamWarning",
    "accuracy",
    "brier_score",
    "calibrate_home_advantage",
    "evaluate_predictions",
    "expected_score",
    "get_rating",
    "get_team",
    "goal_difference_multiplier",
    "is_qualified",
    "load_baseline_matches",
    "log_loss",
    "update_elo",
]
