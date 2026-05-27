"""Tests for LLM router."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

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
    """``model_config = ConfigDict(extra='ignore')`` on ``VerifyLLMRequest``
    means extra fields in the body don't 422 — they're silently dropped.
    Keeps the contract forward-compatible as the UI sends additional LLM
    fields over time."""
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
        assert self.api_key is not None
        from pydantic import SecretStr

        if isinstance(self.api_key, SecretStr):
            captured["api_key"] = self.api_key.get_secret_value()
        else:
            captured["api_key"] = str(self.api_key)
        return None

    with patch.object(LLM, "averify", _capture):
        response = client.post(
            "/api/llm/verify", json=_verify_payload(api_key="sk-secret-123")
        )

    assert response.status_code == 200
    assert captured["api_key"] == "sk-secret-123"


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
