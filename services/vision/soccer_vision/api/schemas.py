"""Request / response pydantic models for the vision API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(description="'ok' or 'degraded'")
    service: str = "soccer-vision-service"
    version: str
    ml_available: bool = Field(
        description="True when the optional ML stack (cv2/torch/ultralytics) is importable"
    )


# ── sessions ─────────────────────────────────────────────────────────────────

class HomographyPoints(BaseModel):
    """
    The four-or-more pixel↔pitch correspondences needed to calibrate a
    session's camera-to-pitch transform.
    """

    src_points: list[tuple[float, float]] = Field(
        min_length=4, description="Pixel coordinates"
    )
    dst_points: list[tuple[float, float]] = Field(
        min_length=4, description="Pitch coordinates in metres"
    )

    @field_validator("dst_points")
    @classmethod
    def _same_length(cls, v: list[tuple[float, float]], info: Any) -> list[tuple[float, float]]:
        src = info.data.get("src_points")
        if src is not None and len(src) != len(v):
            raise ValueError("src_points and dst_points must have equal length")
        return v


class SessionCreate(BaseModel):
    home_team: str = Field(min_length=1)
    away_team: str = Field(min_length=1)
    homography: HomographyPoints
    pitch_length_m: float = Field(default=105.0, gt=0)
    pitch_width_m: float = Field(default=68.0, gt=0)
    contact_radius_m: float = Field(default=1.5, gt=0)


class SessionInfo(BaseModel):
    session_id: str
    home_team: str
    away_team: str
    pitch_length_m: float
    pitch_width_m: float
    contact_radius_m: float
    homography_rms_px: float = Field(
        description="Reprojection RMS error in pixels — sanity check"
    )
    frames_processed: int = 0
    events_emitted: int = 0


# ── frame analysis ───────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """
    Analyse a single frame, supplied as a base64-encoded PNG / JPEG.

    Using a base64 body keeps the contract pure JSON (no multipart) at
    the cost of ~33% payload inflation — acceptable for sub-MB frames.
    For full HD streams use the WebSocket endpoint instead.
    """

    session_id: str
    timestamp_s: float = Field(ge=0.0)
    frame_b64: str = Field(min_length=1, description="base64-encoded image bytes")
    image_format: str = Field(default="png", pattern=r"^(png|jpe?g|webp)$")


class AnalyzeResponse(BaseModel):
    session_id: str
    timestamp_s: float
    frame_index: int
    n_detections: int
    n_tracks: int
    # Events serialised as dicts to keep the contract independent of the
    # exact ``soccer_model`` event subclass. They are guaranteed to
    # validate against ``soccer_model.validate_event_stream``.
    events: list[dict] = Field(default_factory=list)
