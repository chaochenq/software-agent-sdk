"""Tests for LLM router."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from openhands.agent_server.api import create_app
from openhands.agent_server.config import Config
from openhands.agent_server.llm_router import (
    list_models,
    list_providers,
    list_verified_models,
)
from openhands.sdk.llm import LLM
from openhands.sdk.llm.exceptions import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)
from openhands.sdk.llm.utils.verified_models import VERIFIED_MODELS


@pytest.fixture
def client():
    """Create a test client."""
    config = Config(session_api_keys=[])  # Disable authentication for tests
    app = create_app(config)
    return TestClient(app)


@pytest.mark.asyncio
async def test_list_providers():
    """Test listing providers directly."""
    response = await list_providers()
    assert len(response.providers) > 0
    assert "openai" in response.providers
    assert "anthropic" in response.providers
    assert response.providers == sorted(response.providers)


@pytest.mark.asyncio
async def test_list_models():
    """Test listing models directly."""
    response = await list_models(provider=None)
    assert len(response.models) > 0
    assert response.models == sorted(set(response.models))


@pytest.mark.asyncio
async def test_list_models_filtered_by_provider():
    """Test listing models filtered by provider."""
    response = await list_models(provider="openai")
    assert len(response.models) > 0
    # Verify filtering works - there should be fewer models than unfiltered
    all_models_response = await list_models(provider=None)
    assert len(response.models) < len(all_models_response.models)


@pytest.mark.asyncio
async def test_list_models_unknown_provider():
    """Test listing models with an unknown provider returns empty list."""
    response = await list_models(provider="unknown_provider_xyz")
    assert response.models == []


@pytest.mark.asyncio
async def test_list_verified_models():
    """Test listing verified models directly."""
    response = await list_verified_models()
    assert response.models == VERIFIED_MODELS
    assert "openai" in response.models
    assert "anthropic" in response.models


def test_providers_endpoint_integration(client):
    """Test providers endpoint through the API."""
    response = client.get("/api/llm/providers")
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    assert len(data["providers"]) > 0
    assert "openai" in data["providers"]


def test_models_endpoint_integration(client):
    """Test models endpoint through the API."""
    response = client.get("/api/llm/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) > 0


def test_models_endpoint_with_provider_filter(client):
    """Test models endpoint with provider query parameter."""
    response = client.get("/api/llm/models?provider=openai")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) > 0


def test_models_endpoint_with_unknown_provider(client):
    """Test models endpoint with unknown provider returns empty list."""
    response = client.get("/api/llm/models?provider=unknown_provider_xyz")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert data["models"] == []


def test_verified_models_endpoint_integration(client):
    """Test verified models endpoint through the API."""
    response = client.get("/api/llm/models/verified")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "openai" in data["models"]
    assert "anthropic" in data["models"]


# ─────────────────────────────────────────────────────────────────────────────
# POST /llm/verify
#
# The endpoint always returns HTTP 200; clients branch on the ``status`` field.
# ``LLM.averify`` is patched per-test to make the underlying probe deterministic
# without touching the network.
# ─────────────────────────────────────────────────────────────────────────────


def _verify_payload(**overrides):
    """Minimal verify request body. Override per-test as needed."""
    base = {"model": "openai/gpt-4o", "api_key": "sk-test"}
    base.update(overrides)
    return base


def test_verify_endpoint_success(client):
    """Successful probe returns status='success' and the inferred provider."""
    with patch.object(LLM, "averify", new_callable=AsyncMock) as averify:
        averify.return_value = None
        response = client.post("/api/llm/verify", json=_verify_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["provider"] == "openai"
    assert data["message"] is None


@pytest.mark.parametrize(
    ("raised", "expected_status"),
    [
        pytest.param(
            LLMAuthenticationError("invalid api key"),
            "auth_error",
            id="auth_error",
        ),
        pytest.param(
            LLMRateLimitError("rate limit exceeded"),
            "rate_limited",
            id="rate_limited",
        ),
        pytest.param(
            LLMTimeoutError("deadline exceeded"),
            "timeout",
            id="timeout",
        ),
        pytest.param(
            LLMServiceUnavailableError("network unreachable"),
            "unreachable",
            id="unreachable",
        ),
        pytest.param(
            LLMBadRequestError("model 'fake-1' not found"),
            "bad_request",
            id="bad_request",
        ),
    ],
)
def test_verify_endpoint_maps_typed_errors(client, raised, expected_status):
    """Each typed SDK exception maps to its corresponding ``status`` value."""
    with patch.object(LLM, "averify", new_callable=AsyncMock) as averify:
        averify.side_effect = raised
        response = client.post("/api/llm/verify", json=_verify_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == expected_status
    assert data["message"] == str(raised)
    # Provider should still be reported on failure so the UI can show
    # "Anthropic auth failed" instead of an unattributed error.
    assert data["provider"] == "openai"


def test_verify_endpoint_unknown_exception_returns_unknown_error(client):
    """An unmapped exception type collapses to ``status='unknown_error'``."""
    with patch.object(LLM, "averify", new_callable=AsyncMock) as averify:
        averify.side_effect = RuntimeError("bespoke failure")
        response = client.post("/api/llm/verify", json=_verify_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unknown_error"
    assert "bespoke failure" in (data["message"] or "")


def test_verify_endpoint_missing_model_returns_422(client):
    """Missing ``model`` is a contract violation, not a verify outcome; let
    FastAPI surface it as the usual 422 unprocessable entity response so the
    bug is obvious during client development."""
    response = client.post("/api/llm/verify", json={})
    assert response.status_code == 422


def test_verify_endpoint_ignores_extra_fields(client):
    """``LLM`` is declared with ``extra='ignore'``, so unknown fields in the
    request body are silently dropped rather than rejected. Keeps the
    contract forward-compatible if a client sends keys the SDK doesn't know
    about yet."""
    with patch.object(LLM, "averify", new_callable=AsyncMock) as averify:
        averify.return_value = None
        response = client.post(
            "/api/llm/verify",
            json=_verify_payload(unknown_field="ignored"),
        )

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_verify_endpoint_forwards_api_key_unmasked(client):
    """The API key sent in the request body must reach ``LLM.api_key``
    unmasked — otherwise the probe would fail with a false auth error. This
    is the regression test for the ``model_dump`` pitfall noted in the
    router."""
    captured: dict[str, str] = {}

    async def _capture(self):
        # ``self`` is the constructed LLM. Pull the secret value out so we can
        # assert on it after the request returns.
        assert isinstance(self.api_key, SecretStr)
        captured["api_key"] = self.api_key.get_secret_value()
        return None

    with patch.object(LLM, "averify", _capture):
        response = client.post(
            "/api/llm/verify", json=_verify_payload(api_key="sk-secret-123")
        )

    assert response.status_code == 200
    assert captured["api_key"] == "sk-secret-123"


@pytest.mark.parametrize(
    "field",
    [
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
    ],
)
def test_verify_endpoint_forwards_aws_secret_fields_unmasked(client, field):
    """AWS credential fields are ``SecretStr``-typed on ``LLM`` and share the
    same masking risk as ``api_key``. Asserts each one reaches the LLM
    instance with its raw value intact so a Bedrock verify probe doesn't
    fail with a misleading auth error."""
    captured: dict[str, str] = {}

    async def _capture(self):
        value = getattr(self, field)
        assert isinstance(value, SecretStr)
        captured[field] = value.get_secret_value()
        return None

    with patch.object(LLM, "averify", _capture):
        response = client.post(
            "/api/llm/verify",
            json={
                "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                "aws_region_name": "us-east-1",
                field: "aws-secret-value",
            },
        )

    assert response.status_code == 200
    assert captured[field] == "aws-secret-value"


def test_verify_endpoint_keyless_local_server(client):
    """``api_key`` is optional — a local OpenAI-compatible server like Ollama
    can be verified with no key, just a ``base_url``."""
    with patch.object(LLM, "averify", new_callable=AsyncMock) as averify:
        averify.return_value = None
        response = client.post(
            "/api/llm/verify",
            json={
                "model": "openai/llama3",
                "base_url": "http://localhost:11434/v1",
            },
        )

    assert response.status_code == 200
    assert response.json()["status"] == "success"


# ─────────────────────────────────────────────────────────────────────────────
# Hardening behaviors (timeout cap, error message sanitization)
# ─────────────────────────────────────────────────────────────────────────────


def test_verify_endpoint_caps_hanging_probe_with_timeout(client):
    """If ``averify`` hangs past the verify endpoint's hard cap, the request
    must complete with ``status='timeout'`` rather than parking the UI on
    the SDK's 300 s default ``LLM.timeout``."""

    async def _hang(self):
        await asyncio.sleep(5)  # well past the patched 0.05 s cap

    with (
        patch("openhands.agent_server.llm_router._VERIFY_TIMEOUT_S", 0.05),
        patch.object(LLM, "averify", _hang),
    ):
        response = client.post("/api/llm/verify", json=_verify_payload())

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "timeout"
    # Provider should still be attributed on timeout for the UI banner.
    assert data["provider"] == "openai"


def test_verify_endpoint_honors_smaller_caller_timeout(client):
    """If the caller passes a ``timeout`` smaller than the endpoint cap, the
    smaller value wins — callers can opt in to a tighter probe."""
    captured: dict[str, float] = {}

    async def _spy_wait_for(coro, timeout):
        captured["timeout"] = timeout
        coro.close()  # don't actually run the probe
        raise TimeoutError

    with (
        patch("openhands.agent_server.llm_router._VERIFY_TIMEOUT_S", 30.0),
        patch("openhands.agent_server.llm_router.asyncio.wait_for", _spy_wait_for),
        patch.object(LLM, "averify", new_callable=AsyncMock),
    ):
        response = client.post("/api/llm/verify", json=_verify_payload(timeout=1))

    assert response.status_code == 200
    assert response.json()["status"] == "timeout"
    assert captured["timeout"] == 1


def test_verify_endpoint_scrubs_api_key_from_error_message(client):
    """If a provider echoes the API key back in its error body, the value
    must be scrubbed before it crosses the API boundary."""
    secret = "sk-leaky-1234567890"
    leak = f"401 Unauthorized: invalid key '{secret}' for endpoint"

    with patch.object(LLM, "averify", new_callable=AsyncMock) as averify:
        averify.side_effect = LLMAuthenticationError(leak)
        response = client.post("/api/llm/verify", json=_verify_payload(api_key=secret))

    data = response.json()
    assert data["status"] == "auth_error"
    assert secret not in data["message"]
    assert "***" in data["message"]


def test_verify_endpoint_truncates_pathological_error_messages(client):
    """Provider error bodies (truncated HTML pages, oversized JSON blobs)
    must be capped so the verify response can't balloon."""
    huge_message = "x" * 10_000

    with patch.object(LLM, "averify", new_callable=AsyncMock) as averify:
        averify.side_effect = LLMBadRequestError(huge_message)
        response = client.post("/api/llm/verify", json=_verify_payload())

    data = response.json()
    assert data["status"] == "bad_request"
    # 512-char cap including the ``…`` ellipsis sentinel.
    assert len(data["message"]) == 512
    assert data["message"].endswith("…")


@pytest.mark.parametrize(
    "field",
    ["aws_access_key_id", "aws_secret_access_key", "aws_session_token"],
)
def test_verify_endpoint_scrubs_aws_secret_from_error_message(client, field):
    """If a provider echoes an AWS credential back in its error body, the value
    must be scrubbed before it crosses the API boundary — same risk as api_key."""
    secret = "AKIA-LEAKY-SECRET-VALUE"
    leak = f"Could not sign request: invalid {field} '{secret}' for endpoint"

    with patch.object(LLM, "averify", new_callable=AsyncMock) as averify:
        averify.side_effect = LLMAuthenticationError(leak)
        response = client.post(
            "/api/llm/verify",
            json={
                "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                "aws_region_name": "us-east-1",
                field: secret,
            },
        )

    data = response.json()
    assert data["status"] == "auth_error"
    assert secret not in data["message"]
    assert "***" in data["message"]
