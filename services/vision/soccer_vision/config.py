"""
Runtime configuration for the vision service, sourced from environment
variables with the ``SOCCER_VISION_`` prefix.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service-wide runtime configuration."""

    model_config = SettingsConfigDict(
        env_prefix="SOCCER_VISION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── network ──────────────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8088
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── pipeline defaults ────────────────────────────────────────────────────
    pitch_length_m: float = 105.0
    pitch_width_m: float = 68.0

    # If True, the readiness probe requires the ML stack (opencv + torch +
    # ultralytics) to be importable. If False (default), readiness only
    # checks the pure-numpy kernels.
    require_ml_for_ready: bool = Field(default=False)

    # Maximum frame size accepted on /v1/frames/analyze, in bytes. Defaults
    # to 10 MB which covers a 4K JPEG with margin.
    max_frame_bytes: int = 10 * 1024 * 1024


# Singleton accessor — settings are immutable for the life of the process.
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached :class:`Settings` instance (constructed on first call)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
