"""
Provider-adapter ABC and declarative metadata.

The seven cross-cutting dimensions identified in the provider survey
(``docs/PROVIDERS.md``) are encoded as enum-typed fields on
:class:`ProviderInfo`. Every adapter declares one of these and inherits
default implementations of coordinate / time / event-type normalisation
from :class:`ProviderAdapter`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from ..ontology import EventType
from ..schema import MatchEventStream
from .coordinates import CoordinateFrame, transform_xy

__all__ = [
    "ProviderAdapter",
    "ProviderInfo",
    "Transport",
    "Encoding",
    "DirectionConvention",
    "IdentityScheme",
    "TimeBasis",
    "CoordinateFrame",
]


# ── enums for the seven cross-cutting dimensions ─────────────────────────────

class Transport(str, Enum):
    """How the data reaches us."""
    FILE = "file"
    REST = "rest"
    PUSH = "push"
    WEBSOCKET = "websocket"
    SCRAPE = "scrape"


class Encoding(str, Enum):
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    HTML = "html"
    BINARY = "binary"


class DirectionConvention(str, Enum):
    HOME_LEFT_TO_RIGHT_ALWAYS = "home_ltr_always"
    HOME_LEFT_TO_RIGHT_FIRST_HALF_ONLY = "home_ltr_first_half"
    ACTION_EXECUTING_TEAM = "action_executing_team"


class IdentityScheme(str, Enum):
    VENDOR_INTERNAL = "vendor_internal"
    VENDOR_CROSSREF = "vendor_crossref"
    FIFA_CONNECT = "fifa_connect"
    FREE_TEXT = "free_text"
    ANONYMISED = "anonymised"


class TimeBasis(str, Enum):
    MATCH_SECONDS = "match_seconds"
    PERIOD_SECONDS = "period_seconds"
    UTC = "utc"
    FRAME_AT_HZ = "frame_at_hz"


# ── declarative provider metadata ────────────────────────────────────────────

@dataclass(slots=True, frozen=True)
class ProviderInfo:
    """Static metadata for one provider — populated on the adapter class."""

    provider_id: str
    display_name: str
    transport: Transport
    encoding: Encoding
    coordinate_frame: CoordinateFrame
    direction_convention: DirectionConvention
    time_basis: TimeBasis
    identity_scheme: IdentityScheme
    public: bool = False
    url: str = ""
    notes: str = ""
    frame_rate_hz: float | None = None  # only for tracking providers

    def __post_init__(self) -> None:
        if not self.provider_id:
            raise ValueError("provider_id must be non-empty")
        if " " in self.provider_id:
            raise ValueError(f"provider_id may not contain spaces: {self.provider_id!r}")
        if self.frame_rate_hz is not None and self.frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be > 0 if provided")


# ── adapter ABC ──────────────────────────────────────────────────────────────

EventTypeMap = dict[str, EventType] | Callable[[str, str | None], EventType]


@dataclass(slots=True)
class _AdapterRuntime:
    """Per-instance state used during a single load_match() call."""
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0
    extra: dict[str, Any] = field(default_factory=dict)


class ProviderAdapter(ABC):
    """
    Common contract for every soccer data provider adapter.

    Subclasses must:

    * set the class-level :attr:`info` attribute (a :class:`ProviderInfo`)
    * declare :attr:`EVENT_TYPE_MAP` — either a static dict mapping
      ``native_type -> EventType`` or a callable
      ``(native_type, sub_type) -> EventType``
    * implement :meth:`load_match`, returning a fully-validated
      :class:`MatchEventStream`

    The base class provides default implementations of:

    * :meth:`map_event_type` — looks up :attr:`EVENT_TYPE_MAP`
    * :meth:`normalise_coords` — transform from native frame to FIFA m
    * :meth:`normalise_time` — convert ``(period, raw_time)`` to seconds
      from kick-off
    """

    info: ClassVar[ProviderInfo]
    EVENT_TYPE_MAP: ClassVar[EventTypeMap] = {}

    def __init__(
        self,
        *,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
    ) -> None:
        if not hasattr(self, "info"):
            raise TypeError(
                f"{type(self).__name__} must declare a class-level `info` attribute"
            )
        self._runtime = _AdapterRuntime(pitch_length_m=pitch_length_m, pitch_width_m=pitch_width_m)

    # ── primary API ──────────────────────────────────────────────────────────

    @abstractmethod
    def load_match(self, source: Any, **kwargs: Any) -> MatchEventStream:
        """
        Load one match from *source* and return a validated
        :class:`MatchEventStream`.

        Implementations should call :meth:`_finalise` before returning so
        the canonical validator is enforced.
        """

    # ── shared helpers ───────────────────────────────────────────────────────

    def map_event_type(self, native_type: str, sub_type: str | None = None) -> EventType:
        """Resolve a vendor type code to a canonical :class:`EventType`."""
        m = self.EVENT_TYPE_MAP
        if callable(m):
            return m(native_type, sub_type)  # type: ignore[arg-type]
        try:
            return m[native_type]
        except KeyError as e:
            raise KeyError(
                f"{self.info.provider_id}: no canonical mapping for native event type "
                f"{native_type!r} (sub_type={sub_type!r})"
            ) from e

    def normalise_coords(self, x: float, y: float) -> tuple[float, float]:
        """Project from the source coordinate frame to FIFA metres."""
        return transform_xy(
            x, y,
            src=self.info.coordinate_frame,
            dst=CoordinateFrame.FIFA_METRES,
            pitch_length_m=self._runtime.pitch_length_m,
            pitch_width_m=self._runtime.pitch_width_m,
        )

    def normalise_time(
        self,
        raw: float,
        *,
        period: int = 1,
        period_length_s: float = 45 * 60,
    ) -> float:
        """
        Convert a vendor-supplied time to seconds from kick-off.

        Behaviour depends on :attr:`ProviderInfo.time_basis`:

        * ``MATCH_SECONDS`` — pass-through
        * ``PERIOD_SECONDS`` — add ``(period - 1) * period_length_s``
        * ``UTC`` — caller must subtract kick-off; we just pass-through
          the relative offset they supplied
        * ``FRAME_AT_HZ`` — divide frame index by ``frame_rate_hz``
        """
        basis = self.info.time_basis
        if basis is TimeBasis.MATCH_SECONDS:
            return float(raw)
        if basis is TimeBasis.PERIOD_SECONDS:
            if period < 1:
                raise ValueError(f"period must be >= 1, got {period}")
            return float(raw) + (period - 1) * period_length_s
        if basis is TimeBasis.UTC:
            return float(raw)  # caller-relative
        if basis is TimeBasis.FRAME_AT_HZ:
            hz = self.info.frame_rate_hz
            if hz is None or hz <= 0:
                raise ValueError(
                    f"{self.info.provider_id}: frame_rate_hz must be set on ProviderInfo"
                )
            return float(raw) / float(hz)
        raise ValueError(f"unknown time_basis: {basis!r}")

    # ── final-step validation ────────────────────────────────────────────────

    def _finalise(self, stream: MatchEventStream) -> MatchEventStream:
        """
        Run a defensive validation pass on the produced stream.

        Adapters call this before returning; it round-trips the stream
        through ``validate_event_stream`` to guarantee canonical conformance.
        """
        # Importing inside the method to avoid a top-level cycle with
        # soccer_model.interchange (which imports adapters' coords helper).
        from ..interchange import to_json_dict, validate_event_stream

        result = validate_event_stream(to_json_dict(stream))
        if not result.valid:
            raise ValueError(
                f"{self.info.provider_id}: produced an invalid MatchEventStream — "
                f"errors={result.errors}"
            )
        return stream
