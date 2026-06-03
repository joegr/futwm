# Soccer Data Providers — Research Synthesis & Generalization

This document is a survey of **20 soccer event/tracking data schemas and
ingestion APIs** in active production use, plus the cross-cutting
abstractions that let us treat them uniformly inside `soccer_model`.

It is the authoritative reference for the `soccer_model.adapters` package:
every entry below has a corresponding `ProviderAdapter` registration.

---

## 1. The 20 schemas

Legend for **Transport**: `JSON` (REST/file), `XML` (file/feed), `CSV`,
`Push` (server-pushed JSON/XML stream). **Coords** lists the native
coordinate frame.

### Event-data providers

| # | Provider                       | Transport       | Coords           | Event taxonomy       | Public? | Source                                                                 |
|---|--------------------------------|-----------------|------------------|----------------------|---------|------------------------------------------------------------------------|
| 1 | **StatsBomb** Open Data        | JSON files      | 120 × 80 yards   | ~40 types + sub-types| Yes     | <https://github.com/statsbomb/open-data>                               |
| 2 | **Wyscout v3** (Hudl)          | REST JSON       | 100 × 100 %      | ~40 type+subType     | Partial | <https://apidocs.wyscout.com/>                                         |
| 3 | **Opta F24** (Stats Perform)   | XML feed        | 100 × 100 %      | 80+ `type_id` codes  | No      | F24 spec PDF                                                           |
| 4 | **Sportec Solutions** (DFL/BL) | XML files       | 105 × 68 m       | DFL position-data v2 | Yes (papers) | Sci-Data 2025                                                      |
| 5 | **Impect**                     | JSON (open API) | 105 × 68 m       | Packing / pressure metrics | Yes | <https://github.com/ImpectAPI/open-data>                              |
| 6 | **PFF FC**                     | JSON            | 105 × 68 m       | Events + grades      | Free WC22 | <https://fc.pff.com/>                                                 |

### Tracking-data providers

| # | Provider                       | Transport            | Hz       | Coords             | Source                                                          |
|---|--------------------------------|----------------------|----------|--------------------|-----------------------------------------------------------------|
| 7 | **Metrica Sports** sample-data | CSV + EPTS XML, JSON | 25 Hz    | 1 × 1 normalized   | <https://github.com/metrica-sports/sample-data>                 |
| 8 | **SkillCorner** broadcast-CV   | JSON                 | 10 Hz    | 105 × 68 m         | <https://github.com/SkillCorner/opendata>                       |
| 9 | **Second Spectrum**            | proprietary          | 25 Hz    | feet (court frame) | Genius Sports / MLS                                             |
|10 | **Tracab** (ChyronHego)        | binary `.dat` + XML  | 25 Hz    | cm (centred)       | <https://tracab.com/>                                           |
|11 | **Signality** (Spiideo)        | proprietary          | 25 Hz    | broadcast frame    | <https://www.spiideo.com/>                                      |
|12 | **Hawk-Eye 2D**                | proprietary          | 50 Hz    | mm                 | <https://www.hawkeyeinnovations.com/data>                       |

### REST APIs (broadcast / consumer)

| # | Provider                  | Transport            | Source                                                                 |
|---|---------------------------|----------------------|------------------------------------------------------------------------|
|13 | **Sportradar Soccer**     | REST + Push (JSON/XML) | <https://developer.sportradar.com/soccer/docs/soccer-ig-api-basics> |
|14 | **API-Football** (api-sports.io) | REST JSON            | <https://www.api-football.com/documentation-v3>                  |
|15 | **football-data.org**     | REST JSON            | <https://www.football-data.org/documentation/api>                      |
|16 | **SportMonks** v3         | REST JSON            | <https://docs.sportmonks.com/football/>                                |
|17 | **Understat**             | JSON-in-`<script>`   | <https://understat.com/>                                               |

### Standards / academic

| # | Provider / standard       | Transport            | Source                                                                 |
|---|---------------------------|----------------------|------------------------------------------------------------------------|
|18 | **EPTS / FIFA standard**  | XML metadata + raw file | <https://inside.fifa.com/innovation/standards/epts>                |
|19 | **FIFA Connect Data Standard 3.3** | XML / SDK   | <https://data.fifaconnect.org/>                                        |
|20 | **SPADL / Atomic-SPADL**  | tabular (12-tuple)   | <https://socceraction.readthedocs.io/>                                 |

> Bonus references that informed the abstraction:
> **kloppy** — already a successful generalization across most of the above
> (<https://github.com/PySport/kloppy>); **SoccerNet v3** — video action labels
> (<https://www.soccer-net.org/>); **FBref** — StatsBomb-derived advanced stats.

---

## 2. Cross-cutting dimensions

Reading across the 20 schemas reveals **seven dimensions** of variation.
The `soccer_model.adapters.base.ProviderInfo` record captures each one,
and every adapter must declare its values:

### 2.1 Transport
File / REST / push / scraper. Determines whether the adapter is a
*loader* (read once) or a *streamer* (consume indefinitely). Maps to
`ProviderInfo.transport ∈ {FILE, REST, PUSH, WEBSOCKET, SCRAPE}`.

### 2.2 Encoding
JSON, XML, CSV, binary, HTML. Maps to `ProviderInfo.encoding`.

### 2.3 Coordinate frame
Five distinct conventions appear:

| Frame label    | Range          | Used by                                |
|----------------|----------------|----------------------------------------|
| `FIFA_METRES`  | 105 × 68 m     | `soccer_model` canonical, Sportec, PFF |
| `STATSBOMB`    | 120 × 80 yards | StatsBomb Open Data                    |
| `OPTA_PERCENT` | 100 × 100 %    | Opta F24, Wyscout, SportMonks          |
| `NORMALISED`   | 1 × 1          | Metrica sample data                    |
| `CENTRED_CM`   | ±5250 × ±3400 cm | Tracab                              |

`soccer_model.adapters.coordinates` provides the affine transforms
between every pair (lossless modulo float precision).

### 2.4 Direction of play
Three conventions:

* **`HOME_LEFT_TO_RIGHT_ALWAYS`** — coordinates flipped between halves
  so home always attacks right (StatsBomb, SPADL).
* **`HOME_LEFT_TO_RIGHT_FIRST_HALF_ONLY`** — physical pitch
  convention; coords remain raw (Opta, Tracab).
* **`ACTION_EXECUTING_TEAM`** — every action's coords expressed from
  the perspective of the team performing it (kloppy default option).

The adapter declares the source convention; `soccer_model` re-orients
to its canonical `HOME_LEFT_TO_RIGHT_ALWAYS`.

### 2.5 Event taxonomy
Vendor taxonomies range from ~22 (SPADL) to ~80 (Opta) action types.
We resolve each native type to one of the **9 canonical
`soccer_model.EventType` values** (`touch`, `pass`, `shot`, `dribble`,
`tackle`, `header`, `foul`, `goalkeeper_action`, `set_piece`) and store
the original sub-type in `attributes["native_type"]`.

The adapter therefore exposes a `EVENT_TYPE_MAP: dict[str, EventType]`
attribute. Where a vendor type splits across multiple canonical types
(e.g. Opta `1` = `Pass` and Wyscout `5` = `Free Kick` which becomes
either `pass` or `set_piece` depending on `subType`), the mapping is a
function `(native_type, sub_type) -> EventType`.

### 2.6 Identity scheme
Five identity layers:

1. **Internal IDs** (StatsBomb UUID, Opta numeric IDs, Wyscout `wyId`)
2. **Vendor cross-ref** (FBref → StatsBomb, Hudl → Wyscout)
3. **FIFA Connect ID** — the only globally-unique standard
4. **Free-text names** (Understat, scrapers)
5. **No identity** (some open datasets anonymise teams/players)

The adapter populates `MatchMetadata.{home_team,away_team}`,
`EventBase.{team,player}` from whichever IDs the source supplies and
records the chosen scheme in `ProviderInfo.identity_scheme`.

### 2.7 Time representation
* **Match seconds from kick-off** (StatsBomb, SPADL — what we use).
* **Period seconds** (Opta `min:sec` per period) — needs `period × 45 × 60`
  added.
* **UTC timestamps** (REST APIs / live feeds) — needs subtraction of
  match start.
* **Frame index @ Hz** (tracking data) — needs division by frame rate.

`ProviderInfo.time_basis ∈ {MATCH_SECONDS, PERIOD_SECONDS, UTC, FRAME_AT_HZ}`
together with optional `frame_rate_hz` covers all four.

---

## 3. Generalized adapter contract

Captured in `soccer_model.adapters.base`:

```python
class ProviderAdapter(ABC):
    """One adapter per source. Declares ProviderInfo, implements load."""

    info: ClassVar[ProviderInfo]                # static metadata
    EVENT_TYPE_MAP: ClassVar[dict | Callable]   # native → canonical

    @abstractmethod
    def load_match(self, source: Any, **kw) -> MatchEventStream: ...

    # default impls provided:
    def normalise_coords(self, x, y) -> tuple[float, float]: ...
    def normalise_time(self, raw, period: int = 1) -> float: ...
    def map_event_type(self, native_type, sub_type=None) -> EventType: ...
```

The contract is intentionally minimal:

* every adapter takes "some source" (path, URL, dict, file-like) and
  returns a fully-validated `MatchEventStream`
* coordinate / time / type normalisation are baked into the base class
  — adapters only have to map fields, not re-derive arithmetic
* the base class enforces `validate_event_stream(...)` after build, so
  no malformed stream can leak out of an adapter

Registration is via decorator:

```python
@register_adapter("statsbomb")
class StatsBombAdapter(ProviderAdapter):
    info = ProviderInfo(
        provider_id="statsbomb",
        display_name="StatsBomb Open Data",
        transport=Transport.FILE,
        encoding=Encoding.JSON,
        coordinate_frame=CoordinateFrame.STATSBOMB,
        direction_convention=DirectionConvention.HOME_LEFT_TO_RIGHT_ALWAYS,
        time_basis=TimeBasis.MATCH_SECONDS,
        identity_scheme=IdentityScheme.VENDOR_INTERNAL,
        public=True,
        url="https://github.com/statsbomb/open-data",
    )
    ...
```

Lookup at runtime:

```python
from soccer_model.adapters import get_adapter, list_adapters

print(list_adapters())          # all 20 registered ids
adapter = get_adapter("statsbomb")
stream = adapter.load_match("path/to/event_file.json")
```

---

## 4. Status matrix

✅ = fully implemented · 🧱 = scaffold (registered, declared mapping, raises `NotImplementedError`)

| #  | provider_id            | Status |
|----|------------------------|--------|
| 1  | `statsbomb`            | ✅     |
| 2  | `wyscout_v3`           | 🧱     |
| 3  | `opta_f24`             | 🧱     |
| 4  | `sportec`              | 🧱     |
| 5  | `impect`               | 🧱     |
| 6  | `pff_fc`               | 🧱     |
| 7  | `metrica`              | 🧱     |
| 8  | `skillcorner`          | 🧱     |
| 9  | `second_spectrum`      | 🧱     |
| 10 | `tracab`               | 🧱     |
| 11 | `signality`            | 🧱     |
| 12 | `hawkeye_2d`           | 🧱     |
| 13 | `sportradar`           | 🧱     |
| 14 | `api_football`         | 🧱     |
| 15 | `football_data_org`    | 🧱     |
| 16 | `sportmonks`           | 🧱     |
| 17 | `understat`            | 🧱     |
| 18 | `epts_fifa`            | 🧱     |
| 19 | `fifa_connect`         | 🧱     |
| 20 | `spadl`                | 🧱     |

Each scaffold module documents the source's shape and the **target field
mapping** so future contributors can fill in `load_match` in a single
focused PR per provider.

---

## 5. Why not just use `kloppy`?

`kloppy` is excellent and we explicitly cite it. The reasons we ship our
own adapter layer:

1. **Schema fidelity.** Our `MatchEventStream` is Pydantic v2 with
   strict validation; kloppy's data model is dataclass-based. Direct
   ingestion lets us preserve the strictness.
2. **Sub-package boundary.** `soccer_model.adapters` is opt-in — the
   core library has zero adapter dependencies. Importing the
   StatsBomb adapter doesn't pull in `lxml` or any other heavy XML
   parser.
3. **Bridge available.** `soccer_model.adapters.kloppy_bridge`
   (future PR) will re-emit any `kloppy.EventDataset` as a
   `MatchEventStream`, giving us 14 vendors for free.

---

## 6. Updating this list

Every new adapter PR must:

1. Add a row to §1 above with transport, coords, source.
2. Mark the entry in §4 as `✅`.
3. Register via `@register_adapter("...")`.
4. Add at least one round-trip test under
   `tests/test_adapters_<provider>.py`.
