"""
ASGI server entry point.

`app` is the FastAPI application used by uvicorn and by tests via the
``httpx.AsyncClient``. `cli()` is the console-script entry registered in
``pyproject.toml`` (`soccer-vision`) that boots uvicorn from the
environment-driven :class:`Settings`.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from .. import __version__
from ..config import get_settings
from .routes import router


def _create_app() -> FastAPI:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)

    app = FastAPI(
        title="soccer-vision-service",
        version=__version__,
        summary=(
            "Computer-vision microservice for soccer/football. Ingests video "
            "frames, runs CV + spatial kernels, and emits canonical "
            "soccer_model.MatchEventStream events."
        ),
        docs_url="/v1/docs",
        redoc_url="/v1/redoc",
        openapi_url="/v1/openapi.json",
    )
    app.include_router(router)
    return app


app = _create_app()


def cli() -> None:                                # pragma: no cover - boots a server
    """Entry point for the ``soccer-vision`` console script."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "soccer_vision.api.server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )
