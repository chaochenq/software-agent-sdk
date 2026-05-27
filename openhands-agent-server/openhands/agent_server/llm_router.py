"""Router for LLM model and provider information endpoints."""

from enum import Enum

from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from openhands.sdk.llm import LLM
from openhands.sdk.llm.exceptions import (
    LLMAuthenticationError,
    LLMBadRequestError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)
from openhands.sdk.llm.utils.litellm_provider import infer_litellm_provider
from openhands.sdk.llm.utils.unverified_models import (
    _extract_model_and_provider,
    _get_litellm_provider_names,
    get_supported_llm_models,
)
from openhands.sdk.llm.utils.verified_models import VERIFIED_MODELS
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

llm_router = APIRouter(prefix="/llm", tags=["LLM"])


class ProvidersResponse(BaseModel):
    """Response containing the list of available LLM providers."""

    providers: list[str]


class ModelsResponse(BaseModel):
    """Response containing the list of available LLM models."""

    models: list[str]


class VerifiedModelsResponse(BaseModel):
    """Response containing verified models organized by provider."""

    models: dict[str, list[str]]


@llm_router.get("/providers", response_model=ProvidersResponse)
async def list_providers() -> ProvidersResponse:
    """List all available LLM providers supported by LiteLLM."""
    providers = sorted(_get_litellm_provider_names())
    return ProvidersResponse(providers=providers)


@llm_router.get("/models", response_model=ModelsResponse)
async def list_models(
    provider: str | None = Query(
        default=None,
        description="Filter models by provider (e.g., 'openai', 'anthropic')",
    ),
) -> ModelsResponse:
    """List all available LLM models supported by LiteLLM.

    Args:
        provider: Optional provider name to filter models by.

    Note: Bedrock models are excluded unless AWS credentials are configured.
    """
    all_models = get_supported_llm_models()

    if provider is None:
        models = sorted(set(all_models))
    else:
        filtered_models = []
        for model in all_models:
            model_provider, model_id, separator = _extract_model_and_provider(model)
            if model_provider == provider:
                filtered_models.append(model)
        models = sorted(set(filtered_models))

    return ModelsResponse(models=models)


@llm_router.get("/models/verified", response_model=VerifiedModelsResponse)
async def list_verified_models() -> VerifiedModelsResponse:
    """List all verified LLM models organized by provider.

    Verified models are those that have been tested and confirmed to work well
    with OpenHands.
    """
    return VerifiedModelsResponse(models=VERIFIED_MODELS)


# ─────────────────────────────────────────────────────────────────────────────
# Verify endpoint
# ─────────────────────────────────────────────────────────────────────────────


class VerifyLLMStatus(str, Enum):
    """Outcome categories surfaced by ``POST /llm/verify``.

    All non-SUCCESS values are returned with HTTP 200 — clients should branch
    on ``status``, not on transport errors. ``RATE_LIMITED`` is reported
    separately from ``SUCCESS`` so the UI can show a soft-success banner, but
    callers may treat both as "credentials are valid".
    """

    SUCCESS = "success"
    AUTH_ERROR = "auth_error"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    UNREACHABLE = "unreachable"
    BAD_REQUEST = "bad_request"
    UNKNOWN_ERROR = "unknown_error"


class VerifyLLMRequest(BaseModel):
    """Subset of :class:`LLM` settings needed to verify credentials.

    Mirrors the LLM fields the settings UI exposes. Additional fields are
    accepted but ignored at the API boundary; if you need to verify a more
    exotic configuration, persist it via ``PATCH /settings`` and then verify
    the saved state separately.
    """

    model_config = ConfigDict(extra="ignore")

    model: str = Field(
        description="LiteLLM-style 'provider/model' identifier (or bare model).",
    )
    api_key: SecretStr | None = Field(
        default=None,
        description="API key for the provider. Optional for keyless local servers.",
    )
    base_url: str | None = Field(
        default=None,
        description="Custom base URL (e.g. local OpenAI-compatible server).",
    )
    api_version: str | None = Field(
        default=None,
        description="API version. Required for Azure OpenAI.",
    )
    aws_region_name: str | None = Field(
        default=None,
        description="AWS region. Required for Bedrock.",
    )
    aws_access_key_id: SecretStr | None = Field(default=None)
    aws_secret_access_key: SecretStr | None = Field(default=None)
    aws_session_token: SecretStr | None = Field(default=None)
    aws_profile_name: str | None = Field(default=None)
    aws_bedrock_runtime_endpoint: str | None = Field(default=None)


class VerifyLLMResponse(BaseModel):
    """Result of a verify probe.

    A successful probe returns ``status=SUCCESS`` and the inferred LiteLLM
    provider name. All failure modes are reported with HTTP 200 and a
    discriminated ``status`` so clients have a single decision tree.
    """

    status: VerifyLLMStatus
    message: str | None = Field(
        default=None,
        description="Human-readable detail from the provider, if available.",
    )
    provider: str | None = Field(
        default=None,
        description="LiteLLM provider name inferred from model + base_url.",
    )


def _verify_response_for_exception(exc: Exception) -> VerifyLLMResponse:
    """Map a verify-time exception to the appropriate response.

    Handled error classes correspond to the typed exceptions raised by
    :meth:`LLM.verify`; anything else collapses to ``UNKNOWN_ERROR`` so the
    endpoint never raises and the frontend always has a structured result to
    branch on.
    """
    if isinstance(exc, LLMAuthenticationError):
        return VerifyLLMResponse(status=VerifyLLMStatus.AUTH_ERROR, message=str(exc))
    if isinstance(exc, LLMRateLimitError):
        return VerifyLLMResponse(status=VerifyLLMStatus.RATE_LIMITED, message=str(exc))
    if isinstance(exc, LLMTimeoutError):
        return VerifyLLMResponse(status=VerifyLLMStatus.TIMEOUT, message=str(exc))
    if isinstance(exc, LLMServiceUnavailableError):
        return VerifyLLMResponse(status=VerifyLLMStatus.UNREACHABLE, message=str(exc))
    if isinstance(exc, LLMBadRequestError):
        return VerifyLLMResponse(status=VerifyLLMStatus.BAD_REQUEST, message=str(exc))
    logger.exception("llm.verify failed with an unmapped exception")
    return VerifyLLMResponse(status=VerifyLLMStatus.UNKNOWN_ERROR, message=str(exc))


@llm_router.post("/verify", response_model=VerifyLLMResponse)
async def verify_llm_config(request: VerifyLLMRequest) -> VerifyLLMResponse:
    """Verify that the provided LLM credentials can reach the provider.

    Sends a single one-token probe through :meth:`LLM.averify` and reports
    the outcome as a structured ``VerifyLLMResponse``. The probe always
    completes with HTTP 200; failure modes are encoded in ``status``.

    Verifying from the agent server (rather than the browser) means:

    - Every LiteLLM-supported provider is reachable, including Bedrock with
      SigV4 / IAM and Azure with ``api_version``.
    - No CORS restrictions, no provider-specific request shape to maintain
      on the client.
    - The verify call path is the same code path used at conversation time,
      so "verified" really does mean "the agent will be able to use this".
    """
    # NOTE: pass request fields through directly. ``request.model_dump()``
    # would mask SecretStr values to ``**********`` and the probe would
    # fail with a misleading auth error.
    try:
        llm = LLM(
            model=request.model,
            api_key=request.api_key,
            base_url=request.base_url,
            api_version=request.api_version,
            aws_region_name=request.aws_region_name,
            aws_access_key_id=request.aws_access_key_id,
            aws_secret_access_key=request.aws_secret_access_key,
            aws_session_token=request.aws_session_token,
            aws_profile_name=request.aws_profile_name,
            aws_bedrock_runtime_endpoint=request.aws_bedrock_runtime_endpoint,
        )
    except ValidationError as exc:
        # Invalid input (e.g. malformed model string) is surfaced as a bad
        # request rather than HTTP 422 so the client can render the same
        # status banner it uses for provider-side bad-request responses.
        return VerifyLLMResponse(status=VerifyLLMStatus.BAD_REQUEST, message=str(exc))

    provider = infer_litellm_provider(model=llm.model, api_base=llm.base_url)
    try:
        await llm.averify()
    except Exception as exc:  # noqa: BLE001 — verify must never raise
        result = _verify_response_for_exception(exc)
        if result.provider is None:
            result = result.model_copy(update={"provider": provider})
        return result
    return VerifyLLMResponse(status=VerifyLLMStatus.SUCCESS, provider=provider)
