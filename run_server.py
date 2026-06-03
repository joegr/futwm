#!/usr/bin/env python3
"""Launch the soccer world-model dashboard server.

Honours environment variables so the same entry point works on the host
(loopback, debug on) and inside a container (bind all interfaces).

    HOST   default 127.0.0.1   set to 0.0.0.0 in Docker
    PORT   default 5050
    DEBUG  default 1           set to 0 in production
"""
from __future__ import annotations

import os

from soccer_model.server import run


def _envbool(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").lower() in ("1", "true", "yes", "on")


if __name__ == "__main__":
    run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5050")),
        debug=_envbool("DEBUG", True),
    )
