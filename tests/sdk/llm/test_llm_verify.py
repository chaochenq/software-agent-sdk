"""Tests for ``LLM.verify`` / ``LLM.averify``.

These exercise the credentials/connectivity probe used by
``POST /llm/verify`` on the agent server. The contract under test:

- Success: returns ``None``.
- Provider exceptions are mapped via :func:`map_provider_exception` to the
  appropriate typed SDK exception.
- The retry decorator and fallback strategy are deliberately bypassed —
  verify should fail fast on the first error and never substitute a
  different model.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout as LiteLLMTimeout,
)
from litellm.types.utils import (
    Choices,
    Message as LiteLLMMessage,
    ModelResponse,
    Usage,
)
from pydantic import SecretStr

from openhands.sdk.llm import LLM
from openhands.sdk.llm.exceptions import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)
from openhands.sdk.llm.fallback_strategy import FallbackStrategy


MODEL = "gpt-4o"
PROVIDER = "openai"


def _ok_response() -> ModelResponse:
    """Build a minimal valid LiteLLM completion response."""
    return ModelResponse(
        id="verify-ok",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=LiteLLMMessage(content="hi", role="assistant"),
            )
        ],
        created=0,
        model=MODEL,
        object="chat.completion",
        system_fingerprint="test",
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _make_llm(**overrides: Any) -> LLM:
    """Build an LLM with retry disabled so the test fails fast on regressions
    that accidentally re-enable retry for verify."""
    defaults: dict[str, Any] = {
        "usage_id": "test-verify",
        "model": MODEL,
        "api_key": SecretStr("sk-test"),
        "num_retries": 5,
        "retry_min_wait": 0,
        "retry_max_wait": 0,
        "retry_multiplier": 0,
    }
    defaults.update(overrides)
    return LLM(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Sync verify()
# ─────────────────────────────────────────────────────────────────────────────


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_verify_success_returns_none(mock_completion: MagicMock) -> None:
    mock_completion.return_value = _ok_response()
    llm = _make_llm()

    assert llm.verify() is None
    mock_completion.assert_called_once()


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_verify_sends_minimal_one_token_probe(mock_completion: MagicMock) -> None:
    """Verify probe sends a single 'hi' user message capped at 1 token."""
    mock_completion.return_value = _ok_response()
    llm = _make_llm()

    llm.verify()

    _, kwargs = mock_completion.call_args
    messages = kwargs["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    # Tools must not be sent — verify is a credentials probe, not an agent step.
    assert not kwargs.get("tools")
    # Output is capped to 1 token. ``select_chat_options`` normalises this
    # to ``max_completion_tokens`` on OpenAI-style providers.
    assert kwargs.get("max_completion_tokens") == 1 or kwargs.get("max_tokens") == 1


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        pytest.param(
            AuthenticationError("invalid key", PROVIDER, MODEL),
            LLMAuthenticationError,
            id="auth_error",
        ),
        pytest.param(
            RateLimitError("slow down", PROVIDER, MODEL),
            LLMRateLimitError,
            id="rate_limited",
        ),
        pytest.param(
            LiteLLMTimeout("deadline exceeded", MODEL, PROVIDER),
            LLMTimeoutError,
            id="timeout",
        ),
        pytest.param(
            APIConnectionError("network down", PROVIDER, MODEL),
            LLMServiceUnavailableError,
            id="api_connection_error",
        ),
        pytest.param(
            ServiceUnavailableError("provider 503", PROVIDER, MODEL),
            LLMServiceUnavailableError,
            id="service_unavailable",
        ),
        pytest.param(
            BadRequestError("unknown model 'fake-1'", MODEL, PROVIDER),
            LLMBadRequestError,
            id="bad_request_unknown_model",
        ),
    ],
)
@patch("openhands.sdk.llm.llm.litellm_completion")
def test_verify_maps_provider_exceptions(
    mock_completion: MagicMock,
    raised: Exception,
    expected: type[Exception],
) -> None:
    mock_completion.side_effect = raised
    llm = _make_llm()

    with pytest.raises(expected):
        llm.verify()


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_verify_reraises_unmapped_exception_unchanged(
    mock_completion: MagicMock,
) -> None:
    """If ``map_provider_exception`` does not recognise the error, verify
    re-raises the original exception unchanged so callers can still see it."""

    class _BespokeError(Exception):
        pass

    mock_completion.side_effect = _BespokeError("something else entirely")
    llm = _make_llm()

    with pytest.raises(_BespokeError, match="something else entirely"):
        llm.verify()


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_verify_bypasses_retry_on_retriable_errors(
    mock_completion: MagicMock,
) -> None:
    """The retry decorator wraps ``completion``; ``verify`` must NOT use it.

    A bare ``APIConnectionError`` is in ``LLM_RETRY_EXCEPTIONS`` and would
    trigger ``num_retries`` attempts in normal completion. Verify must fail
    on the first attempt.
    """
    mock_completion.side_effect = APIConnectionError("flaky network", PROVIDER, MODEL)
    llm = _make_llm(num_retries=5)

    with pytest.raises(LLMServiceUnavailableError):
        llm.verify()

    assert mock_completion.call_count == 1


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_verify_bypasses_fallback_strategy(mock_completion: MagicMock) -> None:
    """A configured fallback would silently substitute a different model
    on verify failure, which defeats the entire point of 'verify before
    save'. ``_handle_error`` is the only place the fallback strategy is
    consulted, so verify must not invoke it.
    """
    mock_completion.side_effect = AuthenticationError("bad key", PROVIDER, MODEL)
    fallback = FallbackStrategy(fallback_llms=["other-profile"])
    llm = _make_llm(fallback_strategy=fallback)

    with patch.object(LLM, "_handle_error", autospec=True) as handle_error:
        with pytest.raises(LLMAuthenticationError):
            llm.verify()

    handle_error.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Async averify()
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("openhands.sdk.llm.llm.litellm_acompletion", new_callable=AsyncMock)
async def test_averify_success_returns_none(mock_acompletion: AsyncMock) -> None:
    mock_acompletion.return_value = _ok_response()
    llm = _make_llm()

    assert await llm.averify() is None
    mock_acompletion.assert_awaited_once()


@pytest.mark.asyncio
@patch("openhands.sdk.llm.llm.litellm_acompletion", new_callable=AsyncMock)
async def test_averify_maps_auth_error(mock_acompletion: AsyncMock) -> None:
    mock_acompletion.side_effect = AuthenticationError("nope", PROVIDER, MODEL)
    llm = _make_llm()

    with pytest.raises(LLMAuthenticationError):
        await llm.averify()


@pytest.mark.asyncio
@patch("openhands.sdk.llm.llm.litellm_acompletion", new_callable=AsyncMock)
async def test_averify_bypasses_retry(mock_acompletion: AsyncMock) -> None:
    mock_acompletion.side_effect = APIConnectionError("flaky", PROVIDER, MODEL)
    llm = _make_llm(num_retries=5)

    with pytest.raises(LLMServiceUnavailableError):
        await llm.averify()

    assert mock_acompletion.await_count == 1
