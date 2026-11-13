"""LLM provider abstraction: one protocol, one OpenAI-compatible HTTP implementation.

``OpenAICompatibleClient`` talks to any server that implements OpenAI's
``/chat/completions`` and ``/embeddings`` routes -- OpenAI itself, or a local Ollama
server (``http://127.0.0.1:11434/v1``). No vendor SDK is used.

Structured output
    Requests ask for ``response_format = {"type": "json_schema", ...}``. If the server
    rejects that (HTTP 400/404/422), the client retries once with ``json_object`` and
    the schema spelled out in the system prompt, and remembers the downgrade. In both
    modes the reply is only *parsed* here; it is validated by the caller
    (``ml.llm.extraction.validation``), which never trusts it.

Reliability
    Timeouts, HTTP 429 and 5xx are retried with exponential back-off (``Retry-After``
    is honoured up to a cap). Anything else fails immediately with a clear error.

Secrets
    The API key is held privately, sent only in the ``Authorization`` header, and never
    written to logs, ``repr`` or error messages.

Verification status
    Verified live against Ollama and OpenAI (``gpt-6-luna``, 2026-11-12). Parameter
    names accepted by a hosted model (``max_tokens`` vs ``max_completion_tokens``,
    ``temperature``) differ, so they are configurable rather than assumed;
    ``gpt-6-luna`` needs ``max_completion_tokens`` and its default temperature.
"""

import contextlib
import json
import logging
import threading
import time
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import httpx

from ml.llm.config import ProviderConfig

logger = logging.getLogger(__name__)

RETRY_STATUS = {408, 429, 500, 502, 503, 504}
SCHEMA_REJECTED_STATUS = {400, 404, 422}
MAX_RETRY_AFTER_SECONDS = 30.0


class LLMError(RuntimeError):
    """Base class for every provider failure."""


class LLMUnavailableError(LLMError):
    """The provider could not be reached or kept failing (network, 429, 5xx, auth)."""


class LLMResponseError(LLMError):
    """The provider answered, but not with a usable JSON object."""


@runtime_checkable
class LLMClient(Protocol):
    """What the engine needs from a language model."""

    provider: str
    model: str

    def complete_json(self, system: str, user: str, json_schema: dict) -> dict:
        """Return the model's reply to (system, user) parsed as a JSON object.

        ``json_schema`` describes the expected object. The result is *not* guaranteed
        to satisfy it. Raises ``LLMError``.
        """
        ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One embedding vector per text. Raises ``LLMError`` when unsupported."""
        ...


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


class OpenAICompatibleClient:
    """Chat-completions + embeddings client for OpenAI-compatible servers.

    Inputs:  base URL (ending in ``/v1``), API key (may be empty for local servers),
             model id. Generation settings come from ``ProviderConfig``-style keyword
             arguments; ``client`` lets tests inject an ``httpx.MockTransport``.
    Outputs: parsed JSON objects (``complete_json``) and embedding vectors (``embed``).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        *,
        provider: str = "openai_compatible",
        embedding_model: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int = 900,
        max_tokens_param: str = "max_tokens",
        seed: int | None = None,
        timeout_seconds: float = 180.0,
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        structured_output: str = "json_schema",
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.provider = provider
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_tokens_param = max_tokens_param
        self.seed = seed
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.structured_output = structured_output
        self.__api_key = api_key or ""
        self._client = client
        self.calls = 0
        # Token usage reported by the server, summed over calls (for cost estimates).
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self._usage_lock = threading.Lock()

    @classmethod
    def from_config(
        cls, config: ProviderConfig, api_key: str | None, client: httpx.Client | None = None
    ) -> "OpenAICompatibleClient":
        return cls(
            config.base_url,
            api_key,
            config.model,
            provider=config.name,
            embedding_model=config.embedding_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            max_tokens_param=config.max_tokens_param,
            seed=config.seed,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            backoff_seconds=config.backoff_seconds,
            structured_output=config.structured_output,
            client=client,
        )

    def __repr__(self) -> str:
        key = "set" if self.__api_key else "not set"
        return (
            f"OpenAICompatibleClient(provider={self.provider!r}, base_url={self.base_url!r}, "
            f"model={self.model!r}, api_key=<{key}>)"
        )

    # ------------------------------------------------------------------ transport

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.__api_key:
            headers["Authorization"] = f"Bearer {self.__api_key}"
        return headers

    @staticmethod
    def _error_text(response: httpx.Response) -> str:
        """The server's error message, shortened. Never includes request headers."""
        try:
            body = response.json()
            error = body.get("error", body) if isinstance(body, dict) else body
            message = error.get("message", error) if isinstance(error, dict) else error
        except (ValueError, AttributeError):
            message = response.text
        return " ".join(str(message).split())[:300]

    def _retry_delay(self, attempt: int, response: httpx.Response | None) -> float:
        delay = self.backoff_seconds * 2**attempt
        if response is not None:
            with contextlib.suppress(ValueError):  # Retry-After may also be an HTTP date
                delay = max(delay, float(response.headers.get("Retry-After", 0)))
        return min(delay, MAX_RETRY_AFTER_SECONDS)

    def _post(self, route: str, payload: dict) -> httpx.Response:
        """POST with retries. Returns any non-retryable response (the caller decides)."""
        url = f"{self.base_url}/{route}"
        client = self._client or httpx.Client(timeout=self.timeout_seconds)
        failure = "no attempt made"
        try:
            for attempt in range(self.max_retries + 1):
                response = None
                try:
                    response = client.post(url, json=payload, headers=self._headers())
                except httpx.TransportError as error:
                    failure = f"{type(error).__name__}: {error}"
                else:
                    if response.status_code not in RETRY_STATUS:
                        return response
                    failure = f"HTTP {response.status_code}: {self._error_text(response)}"
                if attempt < self.max_retries:
                    delay = self._retry_delay(attempt, response)
                    logger.warning(
                        "llm %s %s: %s; retrying in %.1fs", self.provider, route, failure, delay
                    )
                    time.sleep(delay)
        finally:
            if self._client is None:
                client.close()
        raise LLMUnavailableError(
            f"{self.provider} at {self.base_url} is unavailable after "
            f"{self.max_retries + 1} attempt(s): {failure}"
        )

    def _fail(self, response: httpx.Response, route: str) -> LLMError:
        text = f"{self.provider} {route} failed: HTTP {response.status_code}: "
        text += self._error_text(response)
        if response.status_code in {401, 403}:
            return LLMUnavailableError(text + " (check AETHER_LLM_API_KEY)")
        return LLMError(text)

    # ----------------------------------------------------------------- completion

    def _chat_payload(self, system: str, user: str, json_schema: dict, mode: str) -> dict:
        if mode == "json_object":
            system = (
                f"{system}\n\nReply with a single JSON object and nothing else. It must "
                f"conform to this JSON Schema:\n{json.dumps(json_schema, sort_keys=True)}"
            )
            response_format: dict = {"type": "json_object"}
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": json_schema},
            }
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": response_format,
            self.max_tokens_param: self.max_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.seed is not None:
            payload["seed"] = self.seed
        return payload

    def complete_json(self, system: str, user: str, json_schema: dict) -> dict:
        self.calls += 1
        mode = self.structured_output
        response = self._post(
            "chat/completions", self._chat_payload(system, user, json_schema, mode)
        )
        if mode == "json_schema" and response.status_code in SCHEMA_REJECTED_STATUS:
            logger.warning(
                "llm %s rejected json_schema output (HTTP %s: %s); falling back to json_object",
                self.provider,
                response.status_code,
                self._error_text(response),
            )
            mode = "json_object"
            response = self._post(
                "chat/completions", self._chat_payload(system, user, json_schema, mode)
            )
            if response.status_code == 200:
                self.structured_output = mode  # do not ask again for every article
        if response.status_code != 200:
            raise self._fail(response, "chat/completions")
        return self._parse_completion(response)

    def _parse_completion(self, response: httpx.Response) -> dict:
        try:
            body = response.json()
            choice = body["choices"][0]
            message = choice["message"]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise LLMResponseError(f"{self.provider} returned a malformed completion") from error
        usage = body.get("usage") if isinstance(body, dict) else None
        if isinstance(usage, dict):
            with self._usage_lock:
                for key in self.usage:
                    value = usage.get(key)
                    if isinstance(value, int):
                        self.usage[key] += value
        if message.get("refusal"):
            raise LLMResponseError(f"{self.provider} refused the request")
        if choice.get("finish_reason") == "length":
            raise LLMResponseError(
                f"{self.provider} reply was cut off at max_tokens={self.max_tokens}"
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError(f"{self.provider} returned an empty reply")
        try:
            parsed = json.loads(_strip_code_fence(content))
        except ValueError as error:
            raise LLMResponseError(f"{self.provider} reply is not valid JSON") from error
        if not isinstance(parsed, dict):
            raise LLMResponseError(f"{self.provider} reply is not a JSON object")
        return parsed

    # ------------------------------------------------------------------ embedding

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.embedding_model:
            raise LLMError(f"{self.provider}: no embedding model is configured")
        if not texts:
            return []
        response = self._post("embeddings", {"model": self.embedding_model, "input": list(texts)})
        if response.status_code != 200:
            raise self._fail(response, "embeddings")
        try:
            rows = sorted(response.json()["data"], key=lambda row: row.get("index", 0))
            vectors = [[float(x) for x in row["embedding"]] for row in rows]
        except (ValueError, KeyError, TypeError) as error:
            raise LLMResponseError(f"{self.provider} returned malformed embeddings") from error
        if len(vectors) != len(texts) or len({len(v) for v in vectors}) != 1:
            raise LLMResponseError(f"{self.provider} returned an unexpected number of embeddings")
        return vectors
