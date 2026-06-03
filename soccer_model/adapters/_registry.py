"""
First-party adapter registrations.

This module imports the StatsBomb reference adapter and declares 19
*scaffold* adapters for the remaining providers identified in
``docs/PROVIDERS.md``. Each scaffold:

* registers under a stable ``provider_id``
* declares a complete, documented :class:`ProviderInfo`
* raises :class:`NotImplementedError` from :meth:`load_match` with a
  link to the docs

This way ``list_adapters()`` returns the full intended surface from day
one and the contract for each provider is locked in writing — future
PRs only have to fill in ``load_match``.
"""

from __future__ import annotations

from typing import Any

from ..ontology import EventType
from ..schema import MatchEventStream
from .base import (
    CoordinateFrame,
    DirectionConvention,
    Encoding,
    IdentityScheme,
    ProviderAdapter,
    ProviderInfo,
    TimeBasis,
    Transport,
)
from .registry import register_adapter

# Importing this triggers @register_adapter("statsbomb") via side-effect.
from . import statsbomb  # noqa: F401


# ── helper to keep the 19 scaffolds tiny ─────────────────────────────────────

def _scaffold(provider_id: str, info: ProviderInfo) -> type[ProviderAdapter]:
    """Build a scaffold adapter class with the given metadata."""

    class _Scaffold(ProviderAdapter):
        info_cls = info  # exposed for tests
        EVENT_TYPE_MAP: dict[str, EventType] = {}

        def load_match(self, source: Any, **kwargs: Any) -> MatchEventStream:
            raise NotImplementedError(
                f"{self.info.provider_id}: scaffold adapter — load_match not yet "
                f"implemented. See docs/PROVIDERS.md for the field mapping. "
                f"Source URL: {self.info.url or '(not public)'}"
            )

    _Scaffold.__name__ = f"{_camel(provider_id)}Adapter"
    _Scaffold.info = info  # type: ignore[attr-defined]
    return register_adapter(provider_id)(_Scaffold)


def _camel(s: str) -> str:
    return "".join(p.capitalize() for p in s.replace("-", "_").split("_"))


# ── 19 scaffolds (event + tracking + REST + standards) ───────────────────────

# 2. Wyscout v3 — Hudl
_scaffold(
    "wyscout_v3",
    ProviderInfo(
        provider_id="wyscout_v3",
        display_name="Wyscout (Hudl) v3",
        transport=Transport.REST,
        encoding=Encoding.JSON,
        coordinate_frame=CoordinateFrame.OPTA_PERCENT,
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_FIRST_HALF_ONLY,
        time_basis=TimeBasis.PERIOD_SECONDS,
        identity_scheme=IdentityScheme.VENDOR_INTERNAL,
        public=False,
        url="https://apidocs.wyscout.com/",
        notes="Public dataset mirror at github.com/koenvo/wyscout-soccer-match-event-dataset",
    ),
)

# 3. Opta F24 (Stats Perform)
_scaffold(
    "opta_f24",
    ProviderInfo(
        provider_id="opta_f24",
        display_name="Opta F24 (Stats Perform)",
        transport=Transport.FILE,
        encoding=Encoding.XML,
        coordinate_frame=CoordinateFrame.OPTA_PERCENT,
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_FIRST_HALF_ONLY,
        time_basis=TimeBasis.PERIOD_SECONDS,
        identity_scheme=IdentityScheme.VENDOR_INTERNAL,
        public=False,
        url="https://www.statsperform.com/opta/",
        notes="80+ type_id codes; spec is the F24 PDF distributed under license.",
    ),
)

# 4. Sportec Solutions / DFL
_scaffold(
    "sportec",
    ProviderInfo(
        provider_id="sportec",
        display_name="Sportec Solutions (DFL/Bundesliga)",
        transport=Transport.FILE,
        encoding=Encoding.XML,
        coordinate_frame=CoordinateFrame.FIFA_METRES,
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_FIRST_HALF_ONLY,
        time_basis=TimeBasis.MATCH_SECONDS,
        identity_scheme=IdentityScheme.VENDOR_INTERNAL,
        public=True,
        url="https://www.nature.com/articles/s41597-025-04505-y",
        notes="DFL position-data v2; full season 2022/23 in Sci-Data 2025.",
    ),
)

# 5. Impect
_scaffold(
    "impect",
    ProviderInfo(
        provider_id="impect",
        display_name="Impect",
        transport=Transport.REST,
        encoding=Encoding.JSON,
        coordinate_frame=CoordinateFrame.FIFA_METRES,
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_FIRST_HALF_ONLY,
        time_basis=TimeBasis.MATCH_SECONDS,
        identity_scheme=IdentityScheme.VENDOR_INTERNAL,
        public=True,
        url="https://github.com/ImpectAPI/open-data",
        notes="Packing / pressure-defence metrics layered on event data.",
    ),
)

# 6. PFF FC
_scaffold(
    "pff_fc",
    ProviderInfo(
        provider_id="pff_fc",
        display_name="PFF FC",
        transport=Transport.FILE,
        encoding=Encoding.JSON,
        coordinate_frame=CoordinateFrame.FIFA_METRES,
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_FIRST_HALF_ONLY,
        time_basis=TimeBasis.MATCH_SECONDS,
        identity_scheme=IdentityScheme.VENDOR_INTERNAL,
        public=True,
        url="https://fc.pff.com/",
        notes="Free WC2022 dataset includes events + broadcast tracking + PFF grades.",
    ),
)

# 7. Metrica Sports sample-data
_scaffold(
    "metrica",
    ProviderInfo(
        provider_id="metrica",
        display_name="Metrica Sports (open sample-data)",
        transport=Transport.FILE,
        encoding=Encoding.JSON,        # events; tracking is CSV/EPTS XML
        coordinate_frame=CoordinateFrame.NORMALISED,
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_FIRST_HALF_ONLY,
        time_basis=TimeBasis.FRAME_AT_HZ,
        identity_scheme=IdentityScheme.ANONYMISED,
        public=True,
        url="https://github.com/metrica-sports/sample-data",
        frame_rate_hz=25.0,
    ),
)

# 8. SkillCorner (broadcast-CV tracking)
_scaffold(
    "skillcorner",
    ProviderInfo(
        provider_id="skillcorner",
        display_name="SkillCorner Broadcast Tracking (open data)",
        transport=Transport.FILE,
        encoding=Encoding.JSON,
        coordinate_frame=CoordinateFrame.FIFA_METRES,
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_FIRST_HALF_ONLY,
        time_basis=TimeBasis.FRAME_AT_HZ,
        identity_scheme=IdentityScheme.VENDOR_INTERNAL,
        public=True,
        url="https://github.com/SkillCorner/opendata",
        frame_rate_hz=10.0,
    ),
)

# 9. Second Spectrum
_scaffold(
    "second_spectrum",
    ProviderInfo(
        provider_id="second_spectrum",
        display_name="Second Spectrum (Genius Sports)",
        transport=Transport.FILE,
        encoding=Encoding.BINARY,
        coordinate_frame=CoordinateFrame.FIFA_METRES,
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_FIRST_HALF_ONLY,
        time_basis=TimeBasis.FRAME_AT_HZ,
        identity_scheme=IdentityScheme.VENDOR_INTERNAL,
        public=False,
        url="https://www.geniussports.com/",
        frame_rate_hz=25.0,
    ),
)

# 10. Tracab (ChyronHego)
_scaffold(
    "tracab",
    ProviderInfo(
        provider_id="tracab",
        display_name="Tracab (ChyronHego)",
        transport=Transport.FILE,
        encoding=Encoding.BINARY,
        coordinate_frame=CoordinateFrame.CENTRED_CM,
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_FIRST_HALF_ONLY,
        time_basis=TimeBasis.FRAME_AT_HZ,
        identity_scheme=IdentityScheme.VENDOR_INTERNAL,
        public=False,
        url="https://tracab.com/",
        frame_rate_hz=25.0,
    ),
)

# 11. Signality
_scaffold(
    "signality",
    ProviderInfo(
        provider_id="signality",
        display_name="Signality (Spiideo)",
        transport=Transport.FILE,
        encoding=Encoding.JSON,
        coordinate_frame=CoordinateFrame.FIFA_METRES,
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_FIRST_HALF_ONLY,
        time_basis=TimeBasis.FRAME_AT_HZ,
        identity_scheme=IdentityScheme.VENDOR_INTERNAL,
        public=False,
        url="https://www.spiideo.com/",
        frame_rate_hz=25.0,
    ),
)

# 12. Hawk-Eye 2D
_scaffold(
    "hawkeye_2d",
    ProviderInfo(
        provider_id="hawkeye_2d",
        display_name="Hawk-Eye Innovations 2D Tracking",
        transport=Transport.FILE,
        encoding=Encoding.JSON,
        coordinate_frame=CoordinateFrame.FIFA_METRES,
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_FIRST_HALF_ONLY,
        time_basis=TimeBasis.FRAME_AT_HZ,
        identity_scheme=IdentityScheme.VENDOR_INTERNAL,
        public=False,
        url="https://www.hawkeyeinnovations.com/data",
        frame_rate_hz=50.0,
    ),
)

# 13. Sportradar
_scaffold(
    "sportradar",
    ProviderInfo(
        provider_id="sportradar",
        display_name="Sportradar Soccer API",
        transport=Transport.PUSH,
        encoding=Encoding.JSON,
        coordinate_frame=CoordinateFrame.OPTA_PERCENT,
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_FIRST_HALF_ONLY,
        time_basis=TimeBasis.UTC,
        identity_scheme=IdentityScheme.VENDOR_INTERNAL,
        public=False,
        url="https://developer.sportradar.com/soccer/docs/soccer-ig-api-basics",
    ),
)

# 14. API-Football (api-sports.io)
_scaffold(
    "api_football",
    ProviderInfo(
        provider_id="api_football",
        display_name="API-Football (api-sports.io)",
        transport=Transport.REST,
        encoding=Encoding.JSON,
        coordinate_frame=CoordinateFrame.OPTA_PERCENT,
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_FIRST_HALF_ONLY,
        time_basis=TimeBasis.PERIOD_SECONDS,
        identity_scheme=IdentityScheme.VENDOR_INTERNAL,
        public=False,
        url="https://www.api-football.com/documentation-v3",
        notes="Coarse event types only (Goal/Card/subst); no x/y on most events.",
    ),
)

# 15. football-data.org
_scaffold(
    "football_data_org",
    ProviderInfo(
        provider_id="football_data_org",
        display_name="football-data.org",
        transport=Transport.REST,
        encoding=Encoding.JSON,
        coordinate_frame=CoordinateFrame.FIFA_METRES,   # not used; meta only
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_FIRST_HALF_ONLY,
        time_basis=TimeBasis.UTC,
        identity_scheme=IdentityScheme.VENDOR_INTERNAL,
        public=True,
        url="https://www.football-data.org/documentation/api",
        notes="Match metadata + scores + lineups; no event x/y. Useful for IDs and standings.",
    ),
)

# 16. SportMonks v3
_scaffold(
    "sportmonks",
    ProviderInfo(
        provider_id="sportmonks",
        display_name="SportMonks Football API v3",
        transport=Transport.REST,
        encoding=Encoding.JSON,
        coordinate_frame=CoordinateFrame.OPTA_PERCENT,
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_FIRST_HALF_ONLY,
        time_basis=TimeBasis.PERIOD_SECONDS,
        identity_scheme=IdentityScheme.VENDOR_INTERNAL,
        public=False,
        url="https://docs.sportmonks.com/football/",
        notes="Has ballCoordinates includes for some plans; events sub-resource.",
    ),
)

# 17. Understat
_scaffold(
    "understat",
    ProviderInfo(
        provider_id="understat",
        display_name="Understat (xG / shot data)",
        transport=Transport.SCRAPE,
        encoding=Encoding.HTML,                          # JSON inside <script>
        coordinate_frame=CoordinateFrame.NORMALISED,
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_ALWAYS,
        time_basis=TimeBasis.PERIOD_SECONDS,
        identity_scheme=IdentityScheme.FREE_TEXT,
        public=True,
        url="https://understat.com/",
        notes="Shots only (with xG); no possession context.",
    ),
)

# 18. EPTS / FIFA standard
_scaffold(
    "epts_fifa",
    ProviderInfo(
        provider_id="epts_fifa",
        display_name="EPTS / FIFA Standard Tracking Format",
        transport=Transport.FILE,
        encoding=Encoding.XML,                           # XML metadata + raw file
        coordinate_frame=CoordinateFrame.FIFA_METRES,
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_FIRST_HALF_ONLY,
        time_basis=TimeBasis.FRAME_AT_HZ,
        identity_scheme=IdentityScheme.VENDOR_INTERNAL,
        public=True,
        url="https://inside.fifa.com/innovation/standards/epts",
        frame_rate_hz=25.0,
        notes="Standard, not a single provider — but ingestion semantics are shared.",
    ),
)

# 19. FIFA Connect Data Standard 3.3
_scaffold(
    "fifa_connect",
    ProviderInfo(
        provider_id="fifa_connect",
        display_name="FIFA Connect Data Standard 3.3",
        transport=Transport.REST,
        encoding=Encoding.XML,
        coordinate_frame=CoordinateFrame.FIFA_METRES,    # identity-only; coords unused
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_ALWAYS,
        time_basis=TimeBasis.UTC,
        identity_scheme=IdentityScheme.FIFA_CONNECT,
        public=True,
        url="https://data.fifaconnect.org/",
        notes="Identity layer (FIFA IDs) — used to enrich, not to produce events.",
    ),
)

# 20. SPADL / Atomic-SPADL
_scaffold(
    "spadl",
    ProviderInfo(
        provider_id="spadl",
        display_name="SPADL / Atomic-SPADL (socceraction)",
        transport=Transport.FILE,
        encoding=Encoding.JSON,                          # works for parquet/csv too
        coordinate_frame=CoordinateFrame.FIFA_METRES,
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_ALWAYS,
        time_basis=TimeBasis.PERIOD_SECONDS,
        identity_scheme=IdentityScheme.VENDOR_INTERNAL,
        public=True,
        url="https://socceraction.readthedocs.io/",
        notes="22 action types; the academic reference target schema.",
    ),
)
