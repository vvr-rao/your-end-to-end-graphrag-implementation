# =============================================================================
# Backend image — used by both the Render web (uvicorn) and worker (arq) services.
# =============================================================================
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/root/.local/bin:${PATH}"

# Install uv (single static binary)
ADD https://astral.sh/uv/install.sh /tmp/uv-install.sh
RUN sh /tmp/uv-install.sh && rm /tmp/uv-install.sh

WORKDIR /app

# Resolve and install dependencies first for better layer caching
COPY pyproject.toml uv.lock ./
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
