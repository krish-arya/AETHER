# Backend / worker image (FastAPI API, Celery worker, migrations, data scripts).
#
#   docker build -t aether-backend .
#
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first so this layer is cached until the lockfile changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY backend ./backend
COPY ml ./ml
COPY configs ./configs
COPY scripts ./scripts
RUN uv sync --frozen --no-dev

# Run as an unprivileged user; artifacts/ is the only writable project path.
RUN useradd --create-home --uid 10001 aether \
    && mkdir -p /app/artifacts \
    && chown -R aether:aether /app/artifacts
USER aether

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
