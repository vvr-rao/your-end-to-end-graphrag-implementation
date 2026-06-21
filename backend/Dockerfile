# =============================================================================
# Backend image — used by both the Render web (uvicorn) and worker (arq) services.
# =============================================================================
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/root/.local/bin:${PATH}"

# Astral's install script uses curl to fetch the uv binary; python:3.12-slim
# doesn't ship curl by default, so install it (+ ca-certificates for TLS).
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install uv (single static binary)
ADD https://astral.sh/uv/install.sh /tmp/uv-install.sh
RUN sh /tmp/uv-install.sh && rm /tmp/uv-install.sh

WORKDIR /app

# Resolve and install dependencies first for better layer caching.
# README.md is required because pyproject.toml's [project].readme field
# points at it -- hatchling validates the file exists when building the
# project wheel in the final install step below.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

# Now copy app code
COPY backend ./backend
COPY config ./config

# Final install with the project itself
RUN uv sync --frozen --no-dev

# Render injects $PORT at runtime
ENV PORT=8000
EXPOSE 8000

# Default command runs the web service. The worker service overrides via render.yaml.
CMD ["sh", "-c", "uv run uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}"]
