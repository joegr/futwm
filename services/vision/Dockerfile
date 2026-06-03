# syntax=docker/dockerfile:1.7
#
# soccer-vision-service container.
#
# Two-stage build:
#   1. builder  — compiles deps into a venv (with optional ML extras)
#   2. runtime  — slim image with just the venv + service code
#
# Build:
#   docker build -t soccer-vision-service services/vision
#
# Build with ML/streaming extras:
#   docker build --build-arg EXTRAS="ml,streaming" -t soccer-vision-service:ml services/vision

ARG PYTHON_VERSION=3.12

# ── builder ──────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

ARG EXTRAS=""

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# OpenCV/ffmpeg shared libs only when ML/streaming extras are requested
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY services/vision/pyproject.toml services/vision/README.md ./
COPY services/vision/soccer_vision ./soccer_vision

# Build a venv at /opt/venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install the core soccer-world-model from the local checkout if available,
# else from PyPI. We pin to the local path for monorepo dev convenience.
# The root pyproject.toml requires README.md and force-includes frontend/,
# so both must be present at build time even though the vision service
# doesn't use the frontend at runtime.
COPY pyproject.toml /tmp/core/pyproject.toml
COPY README.md      /tmp/core/README.md
COPY soccer_model   /tmp/core/soccer_model
COPY frontend       /tmp/core/frontend
RUN pip install /tmp/core && \
    if [ -n "$EXTRAS" ]; then \
        pip install ".[${EXTRAS}]" ; \
    else \
        pip install "." ; \
    fi

# ── runtime ──────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Runtime libs only (no -dev packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# non-root user
RUN useradd --create-home --shell /bin/bash svc
USER svc

COPY --from=builder /opt/venv /opt/venv

EXPOSE 8088
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8088/v1/healthz', timeout=2).status==200 else 1)"

ENTRYPOINT ["soccer-vision"]
