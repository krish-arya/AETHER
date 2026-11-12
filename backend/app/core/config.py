"""Application settings.

All configuration comes from environment variables prefixed ``AETHER_`` (or a local
``.env`` file). Secrets are never hardcoded; see ``.env.example``.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_VERSION = "0.1.0"

DISCLAIMERS: tuple[str, ...] = (
    "Project AETHER is a research and decision-support platform. It is not a "
    "SEBI-registered investment adviser or research analyst and does not provide "
    "investment advice.",
    "Historical and simulated performance does not guarantee future results.",
    "Paper trading is simulated. No orders are ever sent to a real broker.",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AETHER_", env_file=".env", extra="ignore")

    app_name: str = "Project AETHER"
    environment: Literal["local", "test", "staging", "production"] = "local"
    log_level: str = "INFO"

    # PostgreSQL is the supported database. A ``sqlite+aiosqlite://`` URL is accepted
    # for local development and tests when Docker / PostgreSQL is unavailable.
    database_url: str = "postgresql+asyncpg://aether:aether@localhost:5432/aether"
    redis_url: str = "redis://localhost:6379/0"

    # Celery defaults to the Redis instance above when these are unset.
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None

    # JSON list in the environment, e.g. '["http://localhost:3000"]'
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    artifacts_dir: Path = Path("artifacts")
    # Default backtest configuration; API requests override individual fields.
    backtest_config: Path = Path("configs/backtesting.yaml")
    # Simple back-pressure on an expensive endpoint until real rate limiting (Phase 8).
    max_active_backtests: int = 4
    health_check_timeout_seconds: float = 2.0

    # LLM engine (Phase 3): any OpenAI-compatible endpoint. Unset => POST /news/analyze
    # answers 503; nothing is simulated. The key is a secret and never logged.
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    # Label recorded with every analysis; defaults to the host of llm_base_url.
    llm_provider: str | None = None
    # Thresholds, aliases and aggregation settings of the engine.
    llm_config: Path = Path("configs/llm.yaml")

    @property
    def broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url

    @property
    def database_backend(self) -> str:
        """Dialect name without driver, e.g. ``postgresql`` or ``sqlite``."""
        return self.database_url.split(":", 1)[0].split("+", 1)[0]


@lru_cache
def get_settings() -> Settings:
    return Settings()
