"""
Baseline Elo ratings for WC 2026 qualified teams.

Values approximate the World Football Elo Ratings (eloratings.net) snapshot
for each team as of late 2025 / early 2026. They are intended as a *starting
point* — the recommended workflow is:

    from soccer_model.tournament import EloRating, INITIAL_RATINGS
    elo = EloRating(ratings=dict(INITIAL_RATINGS))
    elo.record_match("Argentina", "Brazil", 1, 0, k=K_FACTOR_WORLD_CUP)

Then iterate over recent results to refine the ratings before tournament play.

Disclaimer
----------
These values are public-domain numerical estimates, not licensed data from
eloratings.net. They should be refreshed against an authoritative source
(eloratings.net, FIFA rankings) before any competitive use.
"""

from __future__ import annotations

#: Approximate snapshot date for the values below (YYYY-MM-DD).
INITIAL_RATINGS_AS_OF: str = "2025-11-01"

#: Mapping of team name → approximate Elo rating.
INITIAL_RATINGS: dict[str, float] = {
    # Top tier (>2000)
    "Argentina":              2143.0,
    "Spain":                  2115.0,
    "France":                 2096.0,
    "Brazil":                 2030.0,
    "Portugal":               2025.0,
    "England":                2020.0,
    "Netherlands":            2010.0,

    # Strong (1900–2000)
    "Germany":                1985.0,
    "Croatia":                1925.0,
    "Belgium":                1910.0,
    "Colombia":               1890.0,
    "Uruguay":                1880.0,
    "Morocco":                1860.0,
    "Ecuador":                1850.0,

    # Solid (1800–1900)
    "Switzerland":            1840.0,
    "Austria":                1840.0,
    "Iran":                   1830.0,
    "Mexico":                 1825.0,
    "Senegal":                1820.0,
    "Japan":                  1820.0,
    "Norway":                 1810.0,
    "Ivory Coast":            1810.0,
    "South Korea":            1810.0,
    "United States":          1810.0,

    # Mid (1700–1800)
    "Turkey":                 1780.0,
    "Egypt":                  1770.0,
    "Scotland":               1760.0,
    "Sweden":                 1750.0,
    "Australia":              1750.0,
    "Algeria":                1750.0,
    "Czech Republic":         1740.0,
    "Canada":                 1740.0,
    "Paraguay":               1735.0,
    "Bosnia and Herzegovina": 1720.0,
    "Ghana":                  1720.0,
    "Panama":                 1700.0,
    "Tunisia":                1700.0,
    "Uzbekistan":             1700.0,

    # Lower (1500–1700)
    "South Africa":           1680.0,
    "DR Congo":               1680.0,
    "Saudi Arabia":           1650.0,
    "Qatar":                  1630.0,
    "Cape Verde":             1620.0,
    "Iraq":                   1620.0,
    "Jordan":                 1620.0,
    "New Zealand":            1620.0,
    "Haiti":                  1500.0,
    "Curaçao":                1480.0,
}

assert len(INITIAL_RATINGS) == 48, (
    f"Expected ratings for all 48 teams, got {len(INITIAL_RATINGS)}"
)


def get_rating(team: str, default: float = 1500.0) -> float:
    """
    Return the baseline Elo rating for *team*, or *default* if not in the
    snapshot.
    """
    return INITIAL_RATINGS.get(team, default)
