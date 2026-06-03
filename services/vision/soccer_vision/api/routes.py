"""
HTTP + WebSocket route handlers.

Routes are split out from ``server.py`` so the FastAPI app object can be
imported by tests without triggering ``uvicorn.run``.
"""

from __future__ import annotations

import base64
import io
import json
import uuid
from dataclasses import dataclass, field

import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status

from .. import __version__
from ..config import get_settings
from ..detection.opencv_stub import OpenCVBallStub
from ..events.emitter import EmitterConfig
from ..kernels.homography import HomographyError, PlanarHomography
from ..pipeline import VisionPipeline
from ..streaming.source import Frame
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    SessionCreate,
    SessionInfo,
)

router = APIRouter(prefix="/v1")


# ── in-memory session store ──────────────────────────────────────────────────
#
# Production deployments should swap this for Redis / Postgres. The
# contract (create, lookup, increment counters) is intentionally trivial
# so the storage layer is easy to replace.


@dataclass(slots=True)
class _SessionState:
    info: SessionInfo
    pipeline: VisionPipeline
    frames_processed: int = 0
    events_emitted: int = 0
    next_frame_index: int = 0
    pixel_correspondences: list[tuple[float, float]] = field(default_factory=list)
    pitch_correspondences: list[tuple[float, float]] = field(default_factory=list)


_SESSIONS: dict[str, _SessionState] = {}


def _build_pipeline(cfg: SessionCreate, homography: PlanarHomography) -> VisionPipeline:
    """Construct the default pipeline for a new session.

    The default detector is the deterministic :class:`OpenCVBallStub` so
    sessions can be exercised end-to-end without the ML extras. Production
    deployments override this by passing a real :class:`Detector`
    implementation when constructing :class:`VisionPipeline` directly.
    """
    return VisionPipeline(
        detector=OpenCVBallStub(),
        homography=homography,
        emitter_config=EmitterConfig(
            home_team=cfg.home_team,
            away_team=cfg.away_team,
            contact_radius_m=cfg.contact_radius_m,
            pitch_length_m=cfg.pitch_length_m,
            pitch_width_m=cfg.pitch_width_m,
        ),
    )


# ── health probes ────────────────────────────────────────────────────────────

@router.get("/healthz", response_model=HealthResponse, tags=["health"])
def healthz() -> HealthResponse:
    """Liveness probe — service can answer HTTP requests."""
    return HealthResponse(status="ok", version=__version__, ml_available=_ml_available())


@router.get("/readyz", response_model=HealthResponse, tags=["health"])
def readyz() -> HealthResponse:
    """
    Readiness probe — service can do real work.

    By default this only checks pure-numpy kernels. Set
    ``SOCCER_VISION_REQUIRE_ML_FOR_READY=true`` to also gate on the ML
    stack being importable.
    """
    settings = get_settings()
    ml_ok = _ml_available()
    ok = (not settings.require_ml_for_ready) or ml_ok
    status_str = "ok" if ok else "degraded"
    return HealthResponse(status=status_str, version=__version__, ml_available=ml_ok)


def _ml_available() -> bool:
    """True when OpenCV + PyTorch + Ultralytics can be imported."""
    try:
        import cv2  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ── sessions ─────────────────────────────────────────────────────────────────

@router.post(
    "/sessions",
    response_model=SessionInfo,
    status_code=status.HTTP_201_CREATED,
    tags=["sessions"],
)
def create_session(body: SessionCreate) -> SessionInfo:
    try:
        h = PlanarHomography.fit(body.homography.src_points, body.homography.dst_points)
    except HomographyError as e:
        raise HTTPException(status_code=400, detail=f"homography fit failed: {e}") from e
    rms = h.reprojection_error(body.homography.src_points, body.homography.dst_points)

    sid = uuid.uuid4().hex
    info = SessionInfo(
        session_id=sid,
        home_team=body.home_team,
        away_team=body.away_team,
        pitch_length_m=body.pitch_length_m,
        pitch_width_m=body.pitch_width_m,
        contact_radius_m=body.contact_radius_m,
        homography_rms_px=rms,
    )
    pipeline = _build_pipeline(body, h)
    _SESSIONS[sid] = _SessionState(
        info=info,
        pipeline=pipeline,
        pixel_correspondences=list(body.homography.src_points),
        pitch_correspondences=list(body.homography.dst_points),
    )
    return info


@router.get("/sessions/{session_id}", response_model=SessionInfo, tags=["sessions"])
def get_session(session_id: str) -> SessionInfo:
    state = _SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session not found")
    # Refresh counters
    state.info.frames_processed = state.frames_processed
    state.info.events_emitted = state.events_emitted
    return state.info


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["sessions"])
def delete_session(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)


# ── frame analysis ───────────────────────────────────────────────────────────

@router.post("/frames/analyze", response_model=AnalyzeResponse, tags=["frames"])
def analyze_frame(body: AnalyzeRequest) -> AnalyzeResponse:
    state = _SESSIONS.get(body.session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="session not found")

    pixels = _decode_image(body.frame_b64, body.image_format)
    frame = Frame(index=state.next_frame_index, timestamp=body.timestamp_s, pixels=pixels)
    state.next_frame_index += 1
    result = state.pipeline.process_frame(frame)

    state.frames_processed += 1
    state.events_emitted += len(result.events)

    return AnalyzeResponse(
        session_id=body.session_id,
        timestamp_s=body.timestamp_s,
        frame_index=result.frame_index,
        n_detections=result.n_detections,
        n_tracks=result.n_tracks,
        events=[e.model_dump(mode="json") for e in result.events],
    )


@router.websocket("/stream/{session_id}")
async def stream(websocket: WebSocket, session_id: str) -> None:
    """
    Bidirectional WebSocket stream.

    Client → server messages are JSON of shape ``AnalyzeRequest`` (minus
    ``session_id``, which is taken from the path).  Server → client
    messages are JSON of shape ``AnalyzeResponse``.  Closing the socket
    leaves the session intact so a new client can reconnect.
    """
    state = _SESSIONS.get(session_id)
    if state is None:
        await websocket.close(code=4404, reason="session not found")
        return

    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_text()
            try:
                req_dict = json.loads(payload)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "invalid json"}))
                continue
            req_dict.setdefault("session_id", session_id)
            try:
                req = AnalyzeRequest.model_validate(req_dict)
            except Exception as e:                    # pydantic ValidationError
                await websocket.send_text(json.dumps({"error": str(e)}))
                continue

            try:
                resp = analyze_frame(req)
                await websocket.send_text(resp.model_dump_json())
            except HTTPException as e:
                await websocket.send_text(json.dumps({"error": e.detail}))
    except WebSocketDisconnect:
        return


# ── helpers ──────────────────────────────────────────────────────────────────

def _decode_image(b64: str, fmt: str) -> np.ndarray:
    """
    Decode a base64-encoded image into an HxWx3 RGB ndarray.

    Uses Pillow as a pure-Python decoder. If the body exceeds
    ``settings.max_frame_bytes`` after decoding, raises HTTP 413.
    """
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid base64: {e}") from e
    settings = get_settings()
    if len(raw) > settings.max_frame_bytes:
        raise HTTPException(status_code=413, detail="frame too large")

    try:
        from PIL import Image  # lazy import — keeps base install slim
    except ImportError as e:                          # pragma: no cover
        raise HTTPException(
            status_code=500,
            detail="Pillow is required to decode frames; install pillow",
        ) from e
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not decode {fmt}: {e}") from e
    return np.asarray(im.convert("RGB"))
