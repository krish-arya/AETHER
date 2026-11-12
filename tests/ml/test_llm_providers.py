"""OpenAI-compatible client, tested offline with ``httpx.MockTransport``."""

import json
import logging

import httpx
import pytest

from ml.llm.config import LLMConfig, LLMEnvironment, ProviderConfig
from ml.llm.providers import (
    LLMClient,
    LLMError,
    LLMResponseError,
    LLMUnavailableError,
    OpenAICompatibleClient,
)

SECRET = "sk-test-SECRET-1234567890"
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


def _completion(content: str, finish_reason: str = "stop", **message) -> httpx.Response:
    body = {
        "choices": [
            {"message": {"role": "assistant", "content": content, **message},
             "finish_reason": finish_reason}
        ]
    }  # fmt: skip
    return httpx.Response(200, json=body)


class FakeServer:
    """Serves queued responses and records every request."""

    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests]


def _client(server: FakeServer, api_key: str | None = SECRET, **kwargs) -> OpenAICompatibleClient:
    kwargs.setdefault("backoff_seconds", 0.0)
    return OpenAICompatibleClient(
        "http://llm.test/v1/",
        api_key,
        "test-model",
        provider="test",
        client=httpx.Client(transport=httpx.MockTransport(server)),
        **kwargs,
    )


def test_request_uses_json_schema_mode_and_bearer_auth():
    server = FakeServer([_completion('{"ok": true}')])
    client = _client(server, seed=7)
    assert isinstance(client, LLMClient)
    assert client.complete_json("SYS", "USER", SCHEMA) == {"ok": True}

    request, body = server.requests[0], server.bodies()[0]
    assert str(request.url) == "http://llm.test/v1/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {SECRET}"
    assert body["model"] == "test-model" and body["temperature"] == 0.0 and body["seed"] == 7
    assert body["max_tokens"] == 900
    assert body["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USER"},
    ]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert body["response_format"]["json_schema"]["strict"] is True


def test_optional_parameters_follow_the_configuration():
    server = FakeServer([_completion('{"ok": true}')])
    client = _client(
        server, api_key=None, temperature=None, max_tokens=64,
        max_tokens_param="max_completion_tokens",
    )  # fmt: skip
    client.complete_json("s", "u", SCHEMA)
    body = server.bodies()[0]
    assert "temperature" not in body and "seed" not in body and "max_tokens" not in body
    assert body["max_completion_tokens"] == 64
    assert "Authorization" not in server.requests[0].headers  # local servers need no key


def test_falls_back_to_json_object_when_json_schema_is_rejected():
    rejected = httpx.Response(400, json={"error": {"message": "json_schema is not supported"}})
    server = FakeServer([rejected, _completion('{"ok": true}'), _completion('{"ok": false}')])
    client = _client(server)

    assert client.complete_json("SYS", "USER", SCHEMA) == {"ok": True}
    first, second = server.bodies()
    assert first["response_format"]["type"] == "json_schema"
    assert second["response_format"] == {"type": "json_object"}
    # The schema moves into the instruction channel; the user message is untouched.
    assert json.dumps(SCHEMA, sort_keys=True) in second["messages"][0]["content"]
    assert second["messages"][0]["content"].startswith("SYS")
    assert second["messages"][1] == {"role": "user", "content": "USER"}

    # The downgrade is remembered: no second rejected request per article.
    assert client.complete_json("SYS", "USER", SCHEMA) == {"ok": False}
    assert len(server.requests) == 3
    assert server.bodies()[2]["response_format"] == {"type": "json_object"}


def test_fallback_that_also_fails_reports_the_server_error():
    missing = httpx.Response(404, json={"error": {"message": "model 'test-model' not found"}})
    server = FakeServer([missing, missing])
    with pytest.raises(LLMError, match="model 'test-model' not found"):
        _client(server).complete_json("s", "u", SCHEMA)
    assert len(server.requests) == 2


def test_transient_failures_are_retried_with_backoff():
    server = FakeServer(
        [
            httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "slow down"}),
            httpx.ConnectError("connection refused"),
            httpx.Response(503),
            _completion('{"ok": true}'),
        ]
    )
    assert _client(server).complete_json("s", "u", SCHEMA) == {"ok": True}
    assert len(server.requests) == 4


def test_persistent_failure_raises_unavailable_after_the_configured_attempts():
    server = FakeServer([httpx.Response(503)] * 5)
    with pytest.raises(LLMUnavailableError, match="unavailable after 3 attempt"):
        _client(server, max_retries=2).complete_json("s", "u", SCHEMA)
    assert len(server.requests) == 3

    down = FakeServer([httpx.ConnectError("refused")] * 2)
    with pytest.raises(LLMUnavailableError, match="ConnectError"):
        _client(down, max_retries=1).complete_json("s", "u", SCHEMA)


def test_authentication_failure_is_not_retried_and_names_the_setting():
    server = FakeServer([httpx.Response(401, json={"error": {"message": "invalid api key"}})])
    with pytest.raises(LLMUnavailableError, match="AETHER_LLM_API_KEY") as caught:
        _client(server, structured_output="json_object").complete_json("s", "u", SCHEMA)
    assert len(server.requests) == 1
    assert SECRET not in str(caught.value)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (_completion("Sure! Here is the answer."), "not valid JSON"),
        (_completion("[1, 2]"), "not a JSON object"),
        (_completion(""), "empty reply"),
        (_completion('{"ok": tr', finish_reason="length"), "cut off"),
        (_completion("", refusal="I cannot help"), "refused"),
        (httpx.Response(200, json={"unexpected": True}), "malformed"),
    ],
)
def test_unusable_replies_raise_response_errors(response, message):
    with pytest.raises(LLMResponseError, match=message):
        _client(FakeServer([response])).complete_json("s", "u", SCHEMA)


def test_code_fenced_json_is_accepted():
    server = FakeServer([_completion('```json\n{"ok": true}\n```')])
    assert _client(server).complete_json("s", "u", SCHEMA) == {"ok": True}


def test_api_key_never_appears_in_repr_logs_or_errors(caplog):
    server = FakeServer(
        [
            httpx.Response(500, text="boom"),
            httpx.Response(400, json={"error": {"message": "no json_schema"}}),
            _completion('{"ok": true}'),
            httpx.Response(403, json={"error": {"message": "forbidden"}}),
        ]
    )
    client = _client(server)
    with caplog.at_level(logging.DEBUG):
        client.complete_json("s", "u", SCHEMA)
        with pytest.raises(LLMError) as caught:
            client.complete_json("s", "u", SCHEMA)
    assert len(caplog.records) >= 2  # the retry and the fallback were logged
    for text in (repr(client), str(client), caplog.text, str(caught.value)):
        assert SECRET not in text
    assert "api_key=<set>" in repr(client)
    assert "api_key=<not set>" in repr(_client(FakeServer([]), api_key=""))

    env = LLMEnvironment(_env_file=None, api_key=SECRET, model="m")
    assert SECRET not in repr(env) and SECRET not in str(env.model_dump())
    assert env.key == SECRET


def test_embeddings():
    data = {"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]}
    server = FakeServer([httpx.Response(200, json=data)])
    client = _client(server, embedding_model="embed-model")
    assert client.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]  # re-ordered by index
    assert server.bodies()[0] == {"model": "embed-model", "input": ["a", "b"]}
    assert str(server.requests[0].url) == "http://llm.test/v1/embeddings"
    assert client.embed([]) == []

    with pytest.raises(LLMError, match="no embedding model"):
        _client(FakeServer([])).embed(["a"])
    short = FakeServer([httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})])
    with pytest.raises(LLMResponseError):
        _client(short, embedding_model="e").embed(["a", "b"])


def test_config_is_strict_and_environment_overrides_the_provider():
    config = LLMConfig.from_yaml("configs/llm.yaml")
    assert "api_key" not in json.dumps(config.model_dump(mode="json"))  # never in the file
    with pytest.raises(ValueError):
        ProviderConfig(name="x", base_url="http://x", model="m", api_key="nope")
    with pytest.raises(ValueError):
        LLMConfig.model_validate({"provider": config.provider.model_dump(), "surprise": 1})

    env = LLMEnvironment(_env_file=None, base_url="https://api.example/v1", model="model-x")
    applied = env.apply(config)
    assert applied.provider.base_url == "https://api.example/v1"
    assert applied.provider.model == "model-x"
    assert applied.provider.name == config.provider.name  # unset values keep the file's
    assert applied.config_hash() != config.config_hash()
    moved = config.model_copy(
        update={"yahoo_news": config.yahoo_news.model_copy(update={"refresh": True})}
    )
    assert moved.config_hash() == config.config_hash()

    client = OpenAICompatibleClient.from_config(applied.provider, None)
    assert (client.provider, client.model) == (config.provider.name, "model-x")
