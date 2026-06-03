# syntax=docker/dockerfile:1.7
#
# soccer-world-model dashboard (Flask + static frontend).
#
# Two-stage build:
#   1. builder  — compiles deps into a venv, includes the [server] extras
#   2. runtime  — slim image with just the venv + app code
#
# Build:
#   docker build -t soccer-world-model .
#
# Run:
#   docker run --rm -p 5050:5050 soccer-world-model
# then open http://localhost:5050

ARG PYTHON_VERSION=3.12

# ── builder ──────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Build essentials for any wheels that need to compile (numpy/scipy on aarch64).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Self-contained venv keeps the runtime image lean.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Copy only what hatchling needs to build the wheel. `frontend/` is a
# force-include in pyproject.toml so it must be present at build time.
COPY pyproject.toml README.md ./
COPY soccer_model ./soccer_model
COPY frontend     ./frontend

RUN pip install ".[server]"

# ── runtime ──────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOST=0.0.0.0 \
    PORT=5050 \
    DEBUG=0

# Non-root user for the running process.
RUN useradd --create-home --shell /bin/bash app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Copy only what the server actually needs at runtime — the package, the
# static frontend, the example match, and the launch script.
COPY --chown=app:app soccer_model ./soccer_model
COPY --chown=app:app frontend     ./frontend
COPY --chown=app:app examples     ./examples
COPY --chown=app:app run_server.py ./run_server.py

USER app

EXPOSE 5050

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5050/api/pitch', timeout=2).status == 200 else 1)"

CMD ["python", "run_server.py"]
