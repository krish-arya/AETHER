"""News engine wiring for the API: configuration, securities master and the LLM client.

The client is built only from explicit settings (``AETHER_LLM_BASE_URL``,
``AETHER_LLM_MODEL``, optional ``AETHER_LLM_API_KEY``). When they are missing the
engine has no client and the analyze endpoint answers 503 -- it never falls back to a
canned or simulated model.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from fastapi import Depends, HTTPException, status

from app.core.config import Settings, get_settings
from ml.llm.config import LLMConfig
from ml.llm.providers import LLMClient, OpenAICompatibleClient
from ml.llm.securities import SecuritiesMaster

NOT_CONFIGURED = (
    "No LLM is configured. Set AETHER_LLM_BASE_URL and AETHER_LLM_MODEL "
    "(and AETHER_LLM_API_KEY for hosted providers); see .env.example."
)


@dataclass
class NewsEngine:
    config: LLMConfig
    master: SecuritiesMaster
    # None when no provider is configured.
    client: LLMClient | None


@lru_cache(maxsize=4)
def _build(
    config_path: Path,
    base_url: str | None,
    model: str | None,
    provider: str | None,
    api_key: str | None,
) -> NewsEngine:
    config = LLMConfig.from_yaml(config_path)
    master = SecuritiesMaster.from_config(config)
    client = None
    if base_url and model:
        label = provider or urlparse(base_url).hostname or "openai_compatible"
        config = config.with_provider(name=label, base_url=base_url, model=model)
        client = OpenAICompatibleClient.from_config(config.provider, api_key)
    return NewsEngine(config=config, master=master, client=client)


def get_news_engine(settings: Settings = Depends(get_settings)) -> NewsEngine:
    """FastAPI dependency. 503 when the engine configuration cannot be loaded."""
    key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    try:
        return _build(
            settings.llm_config,
            settings.llm_base_url or None,
            settings.llm_model or None,
            settings.llm_provider or None,
            key or None,
        )
    except (OSError, ValueError) as error:
        message = " ".join(str(error).split())[:300]
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"The LLM engine configuration ({settings.llm_config}) could not be loaded: {message}",
        ) from error
