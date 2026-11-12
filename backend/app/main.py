"""FastAPI application factory.

Run locally with::

    uv run uvicorn app.main:app --reload
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import APP_VERSION, get_settings
from app.core.logging import configure_logging
from app.core.redis import close_redis
from app.database.session import dispose_engine

API_V1_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    await close_redis()
    await dispose_engine()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=APP_VERSION,
        description=(
            "Quantitative research and decision-support API. Not investment advice; "
            "historical and simulated performance does not guarantee future results."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix=API_V1_PREFIX)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "service": settings.app_name,
            "version": APP_VERSION,
            "docs": "/docs",
            "health": f"{API_V1_PREFIX}/health",
        }

    return app


app = create_app()
