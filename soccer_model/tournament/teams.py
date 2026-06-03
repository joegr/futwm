"""
WC2026 team registry — the 48 qualified national teams.

Source: 2026 FIFA World Cup qualification (Wikipedia, accessed 2026-05).

Confederation breakdown:
    AFC      8   (+1 inter-conf play-off slot)
    CAF      9   (+1 inter-conf play-off slot)
    CONCACAF 6   (3 hosts + 3 qualifiers)
    CONMEBOL 6
    OFC      1   (+1 inter-conf play-off slot, lost)
    UEFA     16
    Total    48
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Confederation(str, Enum):
    """FIFA confederations."""
    AFC      = "AFC"        # Asia
    CAF      = "CAF"        # Africa
    CONCACAF = "CONCACAF"   # North/Central America & Caribbean
    CONMEBOL = "CONMEBOL"   # South America
    OFC      = "OFC"        # Oceania
    UEFA     = "UEFA"       # Europe


@dataclass(frozen=True)
class Team:
    """A national team participating in WC 2026."""
    name:          str
    code:          str            # FIFA 3-letter code
    confederation: Confederation
    host:          bool = False   # True for Canada, Mexico, USA

    def __str__(self) -> str:
        return self.name


# ── 48-team registry ─────────────────────────────────────────────────────────

TEAMS: tuple[Team, ...] = (
    # Hosts (CONCACAF)
    Team("Canada",         "CAN", Confederation.CONCACAF, host=True),
    Team("Mexico",         "MEX", Confederation.CONCACAF, host=True),
    Team("United States",  "USA", Confederation.CONCACAF, host=True),

    # CONCACAF qualifiers
    Team("Panama",         "PAN", Confederation.CONCACAF),
    Team("Curaçao",        "CUW", Confederation.CONCACAF),
    Team("Haiti",          "HAI", Confederation.CONCACAF),

    # CONMEBOL
    Team("Argentina",      "ARG", Confederation.CONMEBOL),
    Team("Brazil",         "BRA", Confederation.CONMEBOL),
    Team("Ecuador",        "ECU", Confederation.CONMEBOL),
    Team("Uruguay",        "URU", Confederation.CONMEBOL),
    Team("Colombia",       "COL", Confederation.CONMEBOL),
    Team("Paraguay",       "PAR", Confederation.CONMEBOL),

    # UEFA
    Team("Spain",          "ESP", Confederation.UEFA),
    Team("France",         "FRA", Confederation.UEFA),
    Team("England",        "ENG", Confederation.UEFA),
    Team("Portugal",       "POR", Confederation.UEFA),
    Team("Netherlands",    "NED", Confederation.UEFA),
    Team("Germany",        "GER", Confederation.UEFA),
    Team("Belgium",        "BEL", Confederation.UEFA),
    Team("Croatia",        "CRO", Confederation.UEFA),
    Team("Switzerland",    "SUI", Confederation.UEFA),
    Team("Austria",        "AUT", Confederation.UEFA),
    Team("Norway",         "NOR", Confederation.UEFA),
    Team("Scotland",       "SCO", Confederation.UEFA),
    Team("Turkey",         "TUR", Confederation.UEFA),
    Team("Sweden",         "SWE", Confederation.UEFA),
    Team("Czech Republic", "CZE", Confederation.UEFA),
    Team("Bosnia and Herzegovina", "BIH", Confederation.UEFA),

    # AFC
    Team("Japan",          "JPN", Confederation.AFC),
    Team("Iran",           "IRN", Confederation.AFC),
    Team("South Korea",    "KOR", Confederation.AFC),
    Team("Australia",      "AUS", Confederation.AFC),
    Team("Saudi Arabia",   "KSA", Confederation.AFC),
    Team("Qatar",          "QAT", Confederation.AFC),
    Team("Uzbekistan",     "UZB", Confederation.AFC),
    Team("Jordan",         "JOR", Confederation.AFC),

    # CAF
    Team("Morocco",        "MAR", Confederation.CAF),
    Team("Senegal",        "SEN", Confederation.CAF),
    Team("Ivory Coast",    "CIV", Confederation.CAF),
    Team("Egypt",          "EGY", Confederation.CAF),
    Team("Algeria",        "ALG", Confederation.CAF),
    Team("Tunisia",        "TUN", Confederation.CAF),
    Team("Ghana",          "GHA", Confederation.CAF),
    Team("South Africa",   "RSA", Confederation.CAF),
    Team("Cape Verde",     "CPV", Confederation.CAF),

    # OFC
    Team("New Zealand",    "NZL", Confederation.OFC),

    # Inter-confederation play-offs (final 2 slots)
    Team("DR Congo",       "COD", Confederation.CAF),
    Team("Iraq",           "IRQ", Confederation.AFC),
)

assert len(TEAMS) == 48, f"WC2026 must have 48 teams, got {len(TEAMS)}"


# ── lookup helpers ───────────────────────────────────────────────────────────

_BY_NAME: dict[str, Team] = {t.name: t for t in TEAMS}
_BY_CODE: dict[str, Team] = {t.code: t for t in TEAMS}


def _fold(s: str) -> str:
    """Lowercase + strip combining marks (so 'curacao' matches 'Curaçao')."""
    nfd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn").lower().strip()


_BY_FOLD: dict[str, Team] = {}
for _t in TEAMS:
    _BY_FOLD[_fold(_t.name)] = _t
    _BY_FOLD[_fold(_t.code)] = _t


TEAMS_BY_CONFEDERATION: dict[Confederation, tuple[Team, ...]] = {
    conf: tuple(t for t in TEAMS if t.confederation == conf)
    for conf in Confederation
}


def get_team(identifier: str) -> Team:
    """
    Look up a team by name or 3-letter code.

    Accepts case-insensitive input, surrounding whitespace, and ASCII
    transliterations of accented characters (e.g. ``"curacao"`` →
    ``Curaçao``).

    Raises
    ------
    KeyError
        If *identifier* is ``None``, empty, or does not match any team.
    TypeError
        If *identifier* is not a string.
    """
    if identifier is None:
        raise KeyError("Unknown team: None")
    if not isinstance(identifier, str):
        raise TypeError(
            f"Team identifier must be a string, got {type(identifier).__name__}"
        )
    s = identifier.strip()
    if not s:
        raise KeyError("Unknown team: ''")
    # 1. exact name match (preserves caller's spelling)
    if s in _BY_NAME:
        return _BY_NAME[s]
    # 2. exact uppercase code match
    if s.upper() in _BY_CODE:
        return _BY_CODE[s.upper()]
    # 3. accent-folded / case-insensitive fallback
    folded = _fold(s)
    if folded in _BY_FOLD:
        return _BY_FOLD[folded]
    raise KeyError(f"Unknown team: {identifier!r}")


def is_qualified(identifier: str) -> bool:
    """
    Return True if *identifier* matches a WC 2026 qualified team.

    Returns False for ``None``, empty strings, unknown teams, and
    non-string inputs — never raises.
    """
    try:
        get_team(identifier)
        return True
    except (KeyError, TypeError):
        return False


# ── baseline match dataset loader ────────────────────────────────────────────

def load_baseline_matches() -> list[dict]:
    """
    Load the bundled baseline match dataset (recent international matches
    used to validate the Elo predictor against real outcomes).

    Returns
    -------
    list[dict]
        Each dict has keys: ``date``, ``home``, ``away``, ``home_goals``,
        ``away_goals``, ``competition``, ``neutral`` (bool).
    """
    import csv

    path = Path(__file__).parent / "data" / "baseline_matches.csv"
    rows: list[dict] = []
    # utf-8-sig transparently handles an optional BOM in user-supplied data.
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"date", "home", "away", "home_goals", "away_goals"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = required.difference(reader.fieldnames or ())
            raise ValueError(f"baseline_matches.csv missing required columns: {sorted(missing)}")

        for line_no, r in enumerate(reader, start=2):  # header is line 1
            home = (r.get("home") or "").strip()
            away = (r.get("away") or "").strip()
            if not home or not away:
                raise ValueError(f"line {line_no}: empty home/away team")
            try:
                hg = int(r["home_goals"])
                ag = int(r["away_goals"])
            except (TypeError, ValueError) as e:
                raise ValueError(f"line {line_no}: non-integer goal count") from e
            if hg < 0 or ag < 0:
                raise ValueError(f"line {line_no}: negative goal count ({hg}-{ag})")

            venue_raw = (r.get("venue") or "").strip()
            rows.append({
                "date":         (r.get("date") or "").strip(),
                "home":         home,
                "away":         away,
                "home_goals":   hg,
                "away_goals":   ag,
                "competition":  (r.get("competition") or "").strip(),
                "neutral":      (r.get("neutral") or "").strip().lower() in ("1", "true", "yes"),
                "venue":        venue_raw or None,
            })
    return rows
