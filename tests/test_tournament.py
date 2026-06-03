"""Tests for the WC2026 tournament module: teams, Elo, predictions, evaluation."""

from __future__ import annotations

import math
import warnings

import pytest

from soccer_model.tournament import (
    DEFAULT_HOME_ADVANTAGE,
    HOST_NATION_ADVANTAGE,
    INITIAL_RATINGS,
    K_FACTOR_WORLD_CUP,
    TEAMS,
    TEAMS_BY_CONFEDERATION,
    Confederation,
    EloPredictor,
    EloRating,
    Team,
    UnknownTeamWarning,
    accuracy,
    brier_score,
    calibrate_home_advantage,
    expected_score,
    get_rating,
    get_team,
    goal_difference_multiplier,
    is_qualified,
    load_baseline_matches,
    log_loss,
    update_elo,
)


@pytest.fixture
def quiet_predictor():
    """EloPredictor with UnknownTeamWarning suppressed for noisy test cases."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnknownTeamWarning)
        yield EloPredictor()


# ── teams registry ───────────────────────────────────────────────────────────

def test_48_teams_registered():
    assert len(TEAMS) == 48
    assert all(isinstance(t, Team) for t in TEAMS)


def test_three_hosts():
    hosts = [t for t in TEAMS if t.host]
    assert len(hosts) == 3
    assert {t.name for t in hosts} == {"Canada", "Mexico", "United States"}


def test_confederation_breakdown():
    counts = {c: len(ts) for c, ts in TEAMS_BY_CONFEDERATION.items()}
    # AFC=8 (incl Iraq via play-off), CAF=9 (incl DR Congo), CONCACAF=6 (3 hosts+3),
    # CONMEBOL=6, OFC=1, UEFA=16
    assert counts[Confederation.AFC]      == 9    # 8 + Iraq
    assert counts[Confederation.CAF]      == 10   # 9 + DR Congo
    assert counts[Confederation.CONCACAF] == 6
    assert counts[Confederation.CONMEBOL] == 6
    assert counts[Confederation.OFC]      == 1
    assert counts[Confederation.UEFA]     == 16
    assert sum(counts.values()) == 48


def test_unique_codes_and_names():
    assert len({t.code for t in TEAMS}) == 48
    assert len({t.name for t in TEAMS}) == 48


def test_get_team_lookup():
    assert get_team("Argentina").code == "ARG"
    assert get_team("ARG").name      == "Argentina"
    assert get_team("argentina").code == "ARG"   # case-insensitive
    with pytest.raises(KeyError):
        get_team("Atlantis")


def test_is_qualified():
    assert is_qualified("Brazil")
    assert is_qualified("BRA")
    assert not is_qualified("Italy")          # missed WC 2026
    assert not is_qualified("Atlantis")


# ── Elo math ─────────────────────────────────────────────────────────────────

def test_expected_score_symmetric():
    e_ab = expected_score(1800, 1800)
    assert e_ab == pytest.approx(0.5, abs=1e-9)


def test_expected_score_higher_rating_wins():
    assert expected_score(2000, 1500) > 0.9
    assert expected_score(1500, 2000) < 0.1


def test_expected_score_home_advantage():
    e_neutral = expected_score(1800, 1800, 0.0)
    e_home    = expected_score(1800, 1800, 100.0)
    assert e_home > e_neutral


def test_goal_difference_multiplier():
    assert goal_difference_multiplier(0) == 1.0
    assert goal_difference_multiplier(1) == 1.0
    assert goal_difference_multiplier(2) == 1.5
    assert goal_difference_multiplier(3) == pytest.approx(14 / 8)
    assert goal_difference_multiplier(5) == pytest.approx(16 / 8)


def test_update_elo_zero_sum():
    new_a, new_b = update_elo(1800, 1800, 1.0, goal_diff=1, k=40)
    delta_a = new_a - 1800
    delta_b = new_b - 1800
    assert delta_a == pytest.approx(-delta_b, abs=1e-9)
    assert delta_a > 0  # winner gains


def test_update_elo_draw_levels_close_teams():
    # equal ratings, draw → no change
    new_a, new_b = update_elo(1800, 1800, 0.5, goal_diff=0, k=40)
    assert new_a == pytest.approx(1800.0)
    assert new_b == pytest.approx(1800.0)


# ── EloRating book ───────────────────────────────────────────────────────────

def test_elo_rating_get_default():
    book = EloRating(ratings={"A": 1700.0}, default=1500.0)
    assert book.get("A") == 1700.0
    assert book.get("B") == 1500.0


def test_elo_rating_record_match_zero_sum():
    book = EloRating(ratings={"A": 1800.0, "B": 1800.0})
    new_a, new_b = book.record_match(
        "A", "B", 2, 0, k=40.0, home_advantage=0.0, neutral=True,
    )
    assert (new_a - 1800.0) == pytest.approx(-(new_b - 1800.0))
    assert new_a > 1800.0


# ── ratings snapshot ─────────────────────────────────────────────────────────

def test_initial_ratings_cover_all_teams():
    assert len(INITIAL_RATINGS) == 48
    for t in TEAMS:
        assert t.name in INITIAL_RATINGS, f"{t.name} missing from INITIAL_RATINGS"


def test_initial_ratings_plausible_range():
    for name, r in INITIAL_RATINGS.items():
        assert 1300.0 < r < 2300.0, f"{name} rating {r} outside plausible range"


def test_top_teams_outrank_minnows():
    assert get_rating("Argentina") > get_rating("Curaçao") + 400


# ── EloPredictor ─────────────────────────────────────────────────────────────

def test_predictor_probabilities_sum_to_one():
    p = EloPredictor()
    pred = p.predict("Argentina", "Brazil")
    s = pred.p_home_win + pred.p_draw + pred.p_away_win
    assert s == pytest.approx(1.0, abs=1e-9)


def test_predictor_favourite_has_higher_win_prob():
    p = EloPredictor()
    pred = p.predict("Argentina", "Curaçao", neutral=True)
    assert pred.p_home_win > pred.p_away_win
    assert pred.p_home_win > 0.7


def test_predictor_neutral_vs_home_advantage():
    p = EloPredictor()
    pred_home    = p.predict("Mexico", "Argentina", neutral=False)
    pred_neutral = p.predict("Mexico", "Argentina", neutral=True)
    assert pred_home.p_home_win > pred_neutral.p_home_win


def test_predictor_expected_goals_positive():
    p = EloPredictor()
    pred = p.predict("Spain", "England", neutral=True)
    assert pred.expected_home_goals > 0
    assert pred.expected_away_goals > 0


def test_predictor_most_likely_outcome():
    p = EloPredictor()
    pred = p.predict("Brazil", "Haiti", neutral=True)
    assert pred.most_likely_outcome() == "home_win"


def test_predictor_update_changes_rating():
    p = EloPredictor()
    r_before = p.elo.get("Argentina")
    p.update_from_match(
        "Argentina", "Saudi Arabia", 1, 2,
        k=K_FACTOR_WORLD_CUP, neutral=True,
    )
    r_after = p.elo.get("Argentina")
    assert r_after < r_before  # lost, should drop


# ── baseline dataset ─────────────────────────────────────────────────────────

def test_load_baseline_matches():
    matches = load_baseline_matches()
    assert 20 <= len(matches) <= 50
    for m in matches:
        assert {"date", "home", "away", "home_goals", "away_goals", "neutral"} <= m.keys()
        assert isinstance(m["home_goals"], int)
        assert isinstance(m["away_goals"], int)


def test_baseline_dataset_well_formed():
    """All match dicts must have well-typed core fields."""
    matches = load_baseline_matches()
    for m in matches:
        assert isinstance(m["home"], str) and m["home"]
        assert isinstance(m["away"], str) and m["away"]
        assert m["home_goals"] >= 0
        assert m["away_goals"] >= 0
        assert isinstance(m["neutral"], bool)
        # venue is optional; when set it must be a non-empty string
        if m.get("venue") is not None:
            assert isinstance(m["venue"], str) and m["venue"]


def test_baseline_majority_wc2026():
    """At least one of every match's two teams is a WC2026 qualifier."""
    matches = load_baseline_matches()
    for m in matches:
        assert is_qualified(m["home"]) or is_qualified(m["away"]), (
            f"Neither team qualified: {m}"
        )


def test_baseline_has_qualifier_subset():
    """Some matches should have venue set (qualifiers) for HFA calibration."""
    matches = load_baseline_matches()
    with_venue = [m for m in matches if m.get("venue")]
    assert len(with_venue) >= 5


# ── metrics ──────────────────────────────────────────────────────────────────

def test_brier_score_perfect_zero():
    from soccer_model.tournament.predictor import MatchPrediction

    pred = MatchPrediction(
        home="A", away="B", home_rating=1800, away_rating=1800,
        p_home_win=1.0, p_draw=0.0, p_away_win=0.0,
        expected_home_goals=1.5, expected_away_goals=0.5,
    )
    match = {"home": "A", "away": "B", "home_goals": 2, "away_goals": 0}
    assert brier_score([pred], [match]) == pytest.approx(0.0)


def test_brier_score_worst_two():
    from soccer_model.tournament.predictor import MatchPrediction

    pred = MatchPrediction(
        home="A", away="B", home_rating=1800, away_rating=1800,
        p_home_win=0.0, p_draw=0.0, p_away_win=1.0,
        expected_home_goals=1.0, expected_away_goals=1.0,
    )
    match = {"home": "A", "away": "B", "home_goals": 2, "away_goals": 0}
    # observed (1,0,0) vs predicted (0,0,1) → squared err = 1+0+1 = 2
    assert brier_score([pred], [match]) == pytest.approx(2.0)


@pytest.mark.filterwarnings("ignore::soccer_model.tournament.UnknownTeamWarning")
def test_log_loss_finite_for_all_predictions():
    p = EloPredictor()
    matches = load_baseline_matches()
    preds = [p.predict(m["home"], m["away"], neutral=m["neutral"]) for m in matches]
    ll = log_loss(preds, matches)
    assert math.isfinite(ll)
    assert ll > 0.0


@pytest.mark.filterwarnings("ignore::soccer_model.tournament.UnknownTeamWarning")
def test_accuracy_in_range():
    p = EloPredictor()
    matches = load_baseline_matches()
    preds = [p.predict(m["home"], m["away"], neutral=m["neutral"]) for m in matches]
    acc = accuracy(preds, matches)
    assert 0.0 <= acc <= 1.0


@pytest.mark.filterwarnings("ignore::soccer_model.tournament.UnknownTeamWarning")
def test_evaluate_predictions_full_pipeline():
    p = EloPredictor()
    matches = load_baseline_matches()
    result = p.evaluate(matches)
    assert "brier" in result and "log_loss" in result and "accuracy" in result
    assert result["n"] == len(matches)
    # Sanity: baseline Elo should beat uniform-random (1/3) on real matches
    assert result["accuracy"] >= 0.3
    # uniform predictor Brier = 2/3; well-calibrated Elo should beat that
    assert result["brier"] < 0.7


# ── home / host advantage constants ──────────────────────────────────────────

def test_home_advantage_constants_positive_and_ordered():
    assert DEFAULT_HOME_ADVANTAGE > 0
    assert HOST_NATION_ADVANTAGE > 0
    # Hosting at a WC tournament should be a *smaller* boost than a true
    # competitive home leg (no away-fans suppression, neutral-feel stadiums,
    # less travel disparity vs an Argentina home qualifier).
    assert HOST_NATION_ADVANTAGE < DEFAULT_HOME_ADVANTAGE


# ── venue-aware predictor ────────────────────────────────────────────────────

def test_venue_none_applies_full_home_advantage():
    p = EloPredictor()
    pred_default = p.predict("Mexico", "Canada")              # venue=None
    pred_neutral = p.predict("Mexico", "Canada", neutral=True)
    assert pred_default.p_home_win > pred_neutral.p_home_win


def test_venue_at_home_team_equivalent_to_default():
    """venue == home is equivalent to omitting venue (full home advantage)."""
    p = EloPredictor()
    pred_default = p.predict("Mexico", "Canada")                   # venue=None
    pred_at_home = p.predict("Mexico", "Canada", venue="Mexico")   # explicit
    assert pred_default.p_home_win == pytest.approx(pred_at_home.p_home_win)


def test_host_advantage_via_constructor():
    """For tournament-host scenarios, callers pass HOST_NATION_ADVANTAGE explicitly."""
    p_full = EloPredictor(home_advantage=DEFAULT_HOME_ADVANTAGE)
    p_host = EloPredictor(home_advantage=HOST_NATION_ADVANTAGE)
    pred_full = p_full.predict("Mexico", "Canada", venue="Mexico")
    pred_host = p_host.predict("Mexico", "Canada", venue="Mexico")
    pred_neut = p_full.predict("Mexico", "Canada", neutral=True)
    # ordering: neutral < host (50) < full league home (100)
    assert pred_neut.p_home_win < pred_host.p_home_win < pred_full.p_home_win


def test_venue_third_party_is_neutral():
    p = EloPredictor()
    # Argentina playing Brazil in USA → no advantage to either
    pred_third = p.predict("Argentina", "Brazil", venue="United States")
    pred_neut  = p.predict("Argentina", "Brazil", neutral=True)
    assert pred_third.p_home_win == pytest.approx(pred_neut.p_home_win, abs=1e-9)
    assert pred_third.p_draw     == pytest.approx(pred_neut.p_draw, abs=1e-9)


def test_venue_at_away_team_inverts_advantage():
    p = EloPredictor()
    # Argentina (home) plays Mexico but match is in Mexico → Mexico gets host bonus
    pred_at_mex = p.predict("Argentina", "Mexico", venue="Mexico")
    pred_neut   = p.predict("Argentina", "Mexico", neutral=True)
    # Argentina's win prob should drop below the neutral baseline
    assert pred_at_mex.p_home_win < pred_neut.p_home_win


def test_venue_accepts_3letter_code():
    p = EloPredictor()
    pred_code = p.predict("Mexico", "Canada", venue="MEX")
    pred_name = p.predict("Mexico", "Canada", venue="Mexico")
    assert pred_code.p_home_win == pytest.approx(pred_name.p_home_win, abs=1e-9)


def test_neutral_overrides_venue():
    p = EloPredictor()
    pred_v = p.predict("Mexico", "Canada", venue="Mexico", neutral=True)
    pred_n = p.predict("Mexico", "Canada", neutral=True)
    assert pred_v.p_home_win == pytest.approx(pred_n.p_home_win, abs=1e-9)


def test_update_from_match_uses_venue():
    """Beating a stronger team at home should yield Elo gain consistent with HFA."""
    p_home    = EloPredictor()
    p_neutral = EloPredictor()
    r0 = p_home.elo.get("Argentina")
    # Argentina beats Brazil 1-0 at home (Argentina) vs at neutral
    p_home.update_from_match("Argentina", "Brazil", 1, 0, venue="Argentina")
    p_neutral.update_from_match("Argentina", "Brazil", 1, 0, neutral=True)
    gain_home    = p_home.elo.get("Argentina") - r0
    gain_neutral = p_neutral.elo.get("Argentina") - r0
    # winning at neutral is *more* valuable than winning at home (less expected)
    assert gain_neutral > gain_home > 0


# ── calibration ──────────────────────────────────────────────────────────────

@pytest.mark.filterwarnings("ignore::soccer_model.tournament.UnknownTeamWarning")
def test_calibrate_home_advantage_returns_grid_minimum():
    matches = load_baseline_matches()
    result = calibrate_home_advantage(
        matches,
        grid=(0.0, 50.0, 100.0, 150.0),
        metric="log_loss",
    )
    assert "best" in result and "by_value" in result
    assert result["best"] in (0.0, 50.0, 100.0, 150.0)
    # The "best" entry must actually have the lowest log-loss in the grid
    best_ll = result["by_value"][result["best"]]["log_loss"]
    for h, r in result["by_value"].items():
        assert r["log_loss"] >= best_ll - 1e-9, f"H={h} beats reported best"


@pytest.mark.filterwarnings("ignore::soccer_model.tournament.UnknownTeamWarning")
def test_calibrate_supports_accuracy_objective():
    matches = load_baseline_matches()
    result = calibrate_home_advantage(matches, grid=(0.0, 100.0), metric="accuracy")
    assert result["metric"] == "accuracy"
    best_acc = result["by_value"][result["best"]]["accuracy"]
    for r in result["by_value"].values():
        assert r["accuracy"] <= best_acc + 1e-9


def test_calibrate_rejects_unknown_metric():
    matches = [{"home": "Argentina", "away": "Brazil", "home_goals": 1, "away_goals": 0}]
    with pytest.raises(ValueError, match="metric"):
        calibrate_home_advantage(matches, metric="bogus")


def test_calibrate_rejects_empty_inputs():
    with pytest.raises(ValueError, match="grid"):
        calibrate_home_advantage(
            [{"home": "A", "away": "B", "home_goals": 1, "away_goals": 0}],
            grid=(),
        )
    with pytest.raises(ValueError, match="matches"):
        calibrate_home_advantage([], grid=(0.0, 100.0))


def test_calibrate_rejects_non_finite_grid():
    matches = [{"home": "Argentina", "away": "Brazil", "home_goals": 1, "away_goals": 0}]
    with pytest.raises(ValueError, match="finite"):
        calibrate_home_advantage(matches, grid=(0.0, float("inf")))


# ── edge cases: team lookup ──────────────────────────────────────────────────

def test_get_team_unicode_fold():
    """ASCII transliteration of accented names should resolve."""
    assert get_team("curacao").name == "Curaçao"
    assert get_team("CURACAO").name == "Curaçao"
    assert get_team("  Curaçao  ").name == "Curaçao"


def test_get_team_rejects_none():
    with pytest.raises(KeyError):
        get_team(None)


def test_get_team_rejects_non_string():
    with pytest.raises(TypeError):
        get_team(123)


def test_get_team_rejects_empty_string():
    with pytest.raises(KeyError):
        get_team("")
    with pytest.raises(KeyError):
        get_team("   ")


def test_is_qualified_never_raises():
    """is_qualified must be total — returns False for any garbage input."""
    assert is_qualified(None) is False
    assert is_qualified("") is False
    assert is_qualified(123) is False  # type: ignore[arg-type]
    assert is_qualified("Atlantis") is False
    assert is_qualified("Argentina") is True


# ── edge cases: EloRating validation ─────────────────────────────────────────

def test_elo_rating_rejects_nan():
    with pytest.raises(ValueError, match="finite"):
        EloRating(ratings={"A": float("nan")})


def test_elo_rating_rejects_inf():
    with pytest.raises(ValueError, match="finite"):
        EloRating(ratings={"A": float("inf")})


def test_elo_rating_rejects_non_numeric_rating():
    with pytest.raises(TypeError):
        EloRating(ratings={"A": "not a number"})  # type: ignore[dict-item]


def test_elo_rating_rejects_bool_rating():
    """bool subclasses int; explicit rejection prevents silent True→1.0 coercion."""
    with pytest.raises(TypeError):
        EloRating(ratings={"A": True})  # type: ignore[dict-item]


def test_elo_rating_rejects_empty_team_key():
    with pytest.raises(TypeError):
        EloRating(ratings={"": 1500.0})


def test_elo_rating_set_validates():
    book = EloRating(ratings={"A": 1500.0})
    with pytest.raises(ValueError):
        book.set("A", float("nan"))
    with pytest.raises(TypeError):
        book.set("", 1500.0)


def test_record_match_rejects_same_team():
    book = EloRating(ratings={"A": 1500.0})
    with pytest.raises(ValueError, match="different teams"):
        book.record_match("A", "A", 1, 0)


def test_record_match_rejects_negative_goals():
    book = EloRating(ratings={"A": 1500.0, "B": 1500.0})
    with pytest.raises(ValueError, match="non-negative"):
        book.record_match("A", "B", -1, 0)
    with pytest.raises(ValueError, match="non-negative"):
        book.record_match("A", "B", 0, -1)


def test_record_match_rejects_non_int_goals():
    book = EloRating(ratings={"A": 1500.0, "B": 1500.0})
    with pytest.raises(TypeError):
        book.record_match("A", "B", 1.5, 0)  # type: ignore[arg-type]
    # bool is a subtype of int — explicitly rejected to avoid True→1
    with pytest.raises(TypeError):
        book.record_match("A", "B", True, False)  # type: ignore[arg-type]


def test_record_match_rejects_non_finite_k():
    book = EloRating(ratings={"A": 1500.0, "B": 1500.0})
    with pytest.raises(ValueError, match="finite"):
        book.record_match("A", "B", 1, 0, k=float("inf"))


def test_record_match_rejects_negative_k():
    book = EloRating(ratings={"A": 1500.0, "B": 1500.0})
    with pytest.raises(ValueError, match="non-negative"):
        book.record_match("A", "B", 1, 0, k=-10.0)


def test_has_method():
    book = EloRating(ratings={"A": 1500.0})
    assert book.has("A")
    assert not book.has("B")
    # get() returns default but has() does not
    assert book.get("B") == 1500.0
    assert not book.has("B")


# ── streaming: EloRating.update_from_iterable ────────────────────────────────

def test_update_from_iterable_basic():
    book = EloRating(ratings={"A": 1800.0, "B": 1800.0})
    n = book.update_from_iterable([
        {"home": "A", "away": "B", "home_goals": 1, "away_goals": 0},
        {"home": "B", "away": "A", "home_goals": 2, "away_goals": 2},
    ], home_advantage=0.0)
    assert n == 2
    # A won then drew → A should be above 1800
    assert book.get("A") > 1800.0


def test_update_from_iterable_accepts_generator():
    """Streaming a single-pass generator must work without materialising."""
    book = EloRating(ratings={"A": 1500.0, "B": 1500.0})

    def stream():
        for i in range(5):
            yield {"home": "A", "away": "B", "home_goals": 1, "away_goals": 0}

    n = book.update_from_iterable(stream(), home_advantage=0.0)
    assert n == 5
    assert book.get("A") > 1500.0


def test_update_from_iterable_per_match_k_override():
    book_default = EloRating(ratings={"A": 1500.0, "B": 1500.0})
    book_high    = EloRating(ratings={"A": 1500.0, "B": 1500.0})
    match = {"home": "A", "away": "B", "home_goals": 1, "away_goals": 0}
    book_default.update_from_iterable([match], k=10.0, home_advantage=0.0)
    book_high.update_from_iterable([{**match, "k": 100.0}], k=10.0, home_advantage=0.0)
    # higher k → bigger rating shift
    assert book_high.get("A") - 1500.0 > book_default.get("A") - 1500.0


def test_update_from_iterable_raises_on_invalid_by_default():
    book = EloRating(ratings={"A": 1500.0, "B": 1500.0})
    bad = [{"home": "A", "away": "A", "home_goals": 1, "away_goals": 0}]  # same team
    with pytest.raises(ValueError):
        book.update_from_iterable(bad)


def test_update_from_iterable_skip_invalid():
    """skip_invalid=True allows resilient processing of dirty live feeds."""
    book = EloRating(ratings={"A": 1500.0, "B": 1500.0})
    stream = [
        {"home": "A", "away": "B", "home_goals": 1, "away_goals": 0},   # OK
        {"home": "A", "away": "A", "home_goals": 1, "away_goals": 0},   # same team
        {"home": "A", "away": "B", "home_goals": -1, "away_goals": 0},  # negative goals
        {"home": "A"},                                                  # missing keys
        {"home": "A", "away": "B", "home_goals": 2, "away_goals": 1},   # OK
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n = book.update_from_iterable(stream, skip_invalid=True)
    assert n == 2


# ── serialization: EloRating.to_dict / from_dict ─────────────────────────────

def test_elo_rating_roundtrip():
    book = EloRating(ratings={"Argentina": 2143.0, "Brazil": 2030.0}, default=1500.0)
    snap = book.to_dict()
    restored = EloRating.from_dict(snap)
    assert restored.ratings == book.ratings
    assert restored.default == book.default


def test_elo_rating_from_dict_rejects_bad_input():
    with pytest.raises(TypeError):
        EloRating.from_dict("not a dict")  # type: ignore[arg-type]


def test_elo_rating_to_dict_json_serializable():
    import json
    book = EloRating(ratings=dict(INITIAL_RATINGS))
    # to_dict output must be json-serializable for checkpointing
    s = json.dumps(book.to_dict())
    restored = EloRating.from_dict(json.loads(s))
    assert restored.get("Argentina") == book.get("Argentina")


# ── edge cases: predictor ────────────────────────────────────────────────────

def test_predictor_rejects_same_team():
    p = EloPredictor()
    with pytest.raises(ValueError, match="different teams"):
        p.predict("Argentina", "Argentina")
    with pytest.raises(ValueError, match="different teams"):
        p.predict("ARG", "Argentina")        # different spellings, same canonical team


def test_predictor_rejects_empty_team():
    p = EloPredictor()
    with pytest.raises(ValueError):
        p.predict("", "Brazil")
    with pytest.raises(ValueError):
        p.predict("Argentina", "")


def test_predictor_strict_mode_raises_on_unknown():
    p = EloPredictor(strict=True)
    with pytest.raises(ValueError, match="Unknown team"):
        p.predict("Atlantis", "Brazil")


def test_predictor_lenient_mode_warns_on_unknown():
    p = EloPredictor(strict=False)
    with pytest.warns(UnknownTeamWarning, match="Atlantis"):
        pred = p.predict("Atlantis", "Brazil")
    # Atlantis got the default rating, so Brazil is heavy favourite
    assert pred.p_away_win > pred.p_home_win


def test_predictor_known_team_no_warning():
    """Predicting between two qualified teams must not warn."""
    p = EloPredictor()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnknownTeamWarning)
        p.predict("Argentina", "Brazil")  # would raise if warning fired


def test_predictor_seeded_team_no_warning():
    """A user-seeded team in the rating book should not trigger UnknownTeamWarning."""
    book = EloRating(ratings={**INITIAL_RATINGS, "Atlantis": 1700.0})
    p = EloPredictor(elo=book)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnknownTeamWarning)
        pred = p.predict("Atlantis", "Brazil")
    assert pred.home_rating == 1700.0


# ── streaming: EloPredictor.update_from_stream ───────────────────────────────

def test_update_from_stream_basic():
    p = EloPredictor()
    r0 = p.elo.get("Argentina")
    n = p.update_from_stream([
        {"home": "Argentina", "away": "Brazil", "home_goals": 1, "away_goals": 0,
         "neutral": True},
    ])
    assert n == 1
    assert p.elo.get("Argentina") > r0


def test_update_from_stream_generator():
    p = EloPredictor()

    def feed():
        yield {"home": "Argentina", "away": "Brazil", "home_goals": 1, "away_goals": 0,
               "neutral": True}
        yield {"home": "Spain", "away": "France", "home_goals": 2, "away_goals": 1,
               "neutral": True}

    assert p.update_from_stream(feed()) == 2


def test_update_from_stream_sort_by_date():
    """Out-of-order matches must be replayable in chronological order."""
    p_in_order  = EloPredictor()
    p_out_order = EloPredictor()

    matches = [
        {"date": "2024-01-01", "home": "Argentina", "away": "Brazil",
         "home_goals": 1, "away_goals": 0, "neutral": True},
        {"date": "2024-02-01", "home": "Brazil",   "away": "Argentina",
         "home_goals": 2, "away_goals": 0, "neutral": True},
        {"date": "2024-03-01", "home": "Argentina", "away": "Brazil",
         "home_goals": 0, "away_goals": 1, "neutral": True},
    ]
    shuffled = [matches[2], matches[0], matches[1]]

    p_in_order.update_from_stream(matches)
    p_out_order.update_from_stream(shuffled, sort_by_date=True)

    # Same final ratings regardless of input order when sorted
    assert p_in_order.elo.get("Argentina") == pytest.approx(p_out_order.elo.get("Argentina"))
    assert p_in_order.elo.get("Brazil")    == pytest.approx(p_out_order.elo.get("Brazil"))


def test_update_from_stream_skip_invalid():
    p = EloPredictor()
    feed = [
        {"home": "Argentina", "away": "Brazil", "home_goals": 1, "away_goals": 0,
         "neutral": True},
        {"home": "Argentina", "away": "Argentina", "home_goals": 1, "away_goals": 0},  # bad
        {"home": "Spain", "away": "France", "home_goals": 2, "away_goals": 1,
         "neutral": True},
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n = p.update_from_stream(feed, skip_invalid=True)
    assert n == 2


def test_update_from_stream_raises_by_default():
    p = EloPredictor()
    bad = [{"home": "Argentina", "away": "Argentina", "home_goals": 1, "away_goals": 0}]
    with pytest.raises(ValueError):
        p.update_from_stream(bad)


def test_update_from_stream_per_match_k():
    """Per-match K-factor override (e.g., WC final K=60 vs qualifier K=40)."""
    p_low  = EloPredictor()
    p_high = EloPredictor()
    r0 = p_low.elo.get("Argentina")
    p_low.update_from_stream([
        {"home": "Argentina", "away": "Brazil", "home_goals": 1, "away_goals": 0,
         "neutral": True, "k": 10},
    ])
    p_high.update_from_stream([
        {"home": "Argentina", "away": "Brazil", "home_goals": 1, "away_goals": 0,
         "neutral": True, "k": 60},
    ])
    assert (p_high.elo.get("Argentina") - r0) > (p_low.elo.get("Argentina") - r0)


# ── streaming: checkpoint + restore mid-stream ───────────────────────────────

def test_checkpoint_restore_mid_stream():
    """A live stream can be paused, serialised, and resumed without drift."""
    feed = [
        {"home": "Argentina", "away": "Brazil",      "home_goals": 1, "away_goals": 0,
         "neutral": True},
        {"home": "Spain",     "away": "France",      "home_goals": 2, "away_goals": 1,
         "neutral": True},
        {"home": "Germany",   "away": "Netherlands", "home_goals": 0, "away_goals": 1,
         "neutral": True},
    ]
    # Reference run: apply all three in one go
    p_ref = EloPredictor()
    p_ref.update_from_stream(feed)

    # Resume run: apply first match, checkpoint, restore, apply remainder
    p_a = EloPredictor()
    p_a.update_from_stream(feed[:1])
    snap = p_a.elo.to_dict()
    p_b = EloPredictor(elo=EloRating.from_dict(snap))
    p_b.update_from_stream(feed[1:])

    for team in ("Argentina", "Brazil", "Spain", "France", "Germany", "Netherlands"):
        assert p_ref.elo.get(team) == pytest.approx(p_b.elo.get(team))


# ── edge cases: corrupt CSV behaviour ────────────────────────────────────────

def test_load_baseline_matches_dataset_consistent():
    """The bundled dataset must load and round-trip without error."""
    matches = load_baseline_matches()
    assert len(matches) > 0
    for m in matches:
        # All matches should have non-negative integer goals
        assert isinstance(m["home_goals"], int) and m["home_goals"] >= 0
        assert isinstance(m["away_goals"], int) and m["away_goals"] >= 0
        # Date column must be ISO-ish format
        assert len(m["date"]) >= 8
