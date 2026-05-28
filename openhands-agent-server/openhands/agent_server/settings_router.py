from functools import lru_cache
from typing import cast

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import ValidationError

from openhands.agent_server._secrets_exposure import (
    build_expose_context,
    get_config,
    parse_expose_secrets_header,
    translate_missing_cipher,
)
from openhands.agent_server.persistence import (
    SECRET_NAME_PATTERN,
    PersistedSettings,
    get_secrets_store,
    get_settings_store,
)
from openhands.agent_server.persistence.models import SettingsUpdatePayload
from openhands.sdk.logger import get_logger
from openhands.sdk.settings import (
    AcpEnvVarItem,
    AcpEnvVarsListResponse,
    AcpEnvVarUpsertRequest,
    ConversationSettings,
    SecretCreateRequest,
    SecretItemResponse,
    SecretsListResponse,
    SettingsResponse,
    SettingsSchema,
    SettingsUpdateRequest,
    export_agent_settings_schema,
)
from openhands.sdk.settings.model import ACPAgentSettings


logger = get_logger(__name__)

# ── Route Path Constants ─────────────────────────────────────────────────
# These are relative to the router prefix (/settings).
# When mounted on /api, full paths become /api/settings, /api/settings/secrets, etc.
# Note: RemoteWorkspace (client) uses absolute paths (e.g., "/api/settings")
# while this router uses relative paths. The paths are intentionally separate
# to match their respective contexts (router prefix vs full URL path).
SETTINGS_PATH = ""  # -> /api/settings
SECRETS_PATH = "/secrets"  # -> /api/settings/secrets
SECRET_VALUE_PATH = "/secrets/{name}"  # -> /api/settings/secrets/{name}
ACP_ENV_PATH = "/agent-env"  # -> /api/settings/agent-env
ACP_ENV_VALUE_PATH = "/agent-env/{name}"  # -> /api/settings/agent-env/{name}

settings_router = APIRouter(prefix="/settings", tags=["Settings"])


# ── Schema Endpoints ─────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _get_agent_settings_schema() -> SettingsSchema:
    # ``AgentSettings`` is now a discriminated union over
    # ``OpenHandsAgentSettings`` and ``ACPAgentSettings``; the combined
    # schema tags sections with a ``variant`` so the frontend can
    # show LLM-only or ACP-only sections based on the active
    # ``agent_kind`` value.
    return export_agent_settings_schema()


@lru_cache(maxsize=1)
def _get_conversation_settings_schema() -> SettingsSchema:
    return ConversationSettings.export_schema()


@settings_router.get("/agent-schema", response_model=SettingsSchema)
async def get_agent_settings_schema() -> SettingsSchema:
    """Return the schema used to render AgentSettings-based settings forms."""
    return _get_agent_settings_schema()


@settings_router.get("/conversation-schema", response_model=SettingsSchema)
async def get_conversation_settings_schema() -> SettingsSchema:
    """Return the schema used to render ConversationSettings-based forms."""
    return _get_conversation_settings_schema()


# ── Settings CRUD Endpoints ──────────────────────────────────────────────


def _validate_secret_name(name: str) -> None:
    """Validate secret name format.

    Secret names must:
    - Start with a letter
    - Contain only letters, numbers, and underscores
    - Be 1-64 characters long

    Raises:
        HTTPException: 422 if name format is invalid.
    """
    if not SECRET_NAME_PATTERN.match(name):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Invalid secret name format. Must start with a letter, "
                "contain only letters, numbers, and underscores, "
                "and be 1-64 characters long."
            ),
        )


@settings_router.get(SETTINGS_PATH, response_model=SettingsResponse)
async def get_settings(request: Request) -> SettingsResponse:
    """Get current settings.

    Returns the persisted settings including agent configuration,
    conversation settings, and whether an LLM API key is configured.

    Use the ``X-Expose-Secrets`` header to control secret exposure:
    - ``encrypted``: Returns cipher-encrypted values (safe for frontend clients)
    - ``plaintext``: Returns raw secret values (backend clients only!)
    - (absent): Returns redacted values ("**********")

    Security:
        When the server is configured with ``session_api_keys``, all endpoints
        under ``/api`` (including this one) require the ``X-Session-API-Key``
        header. When no session API keys are configured, endpoints are open.

        **Trust model:** All authenticated clients are treated as equally
        trusted. There is no role-based authorization for ``X-Expose-Secrets``
        modes—any authenticated client can request ``plaintext`` or
        ``encrypted`` exposure. This design assumes:

        - All clients sharing session API keys operate in the same trust domain
        - Network-level controls (firewalls, VPCs) restrict access to trusted
          clients only
        - Production deployments use session API keys to prevent anonymous access

        The ``plaintext`` mode exists for backend-to-backend communication
        (e.g., RemoteWorkspace). Frontend clients should prefer ``encrypted``
        mode for round-tripping secrets, or omit the header to receive redacted
        values.
    """
    expose_mode = parse_expose_secrets_header(request)
    config = get_config(request)
    store = get_settings_store(config)
    settings = store.load() or PersistedSettings()

    # Audit log all settings access for security visibility
    # Use WARNING level for plaintext mode to highlight security-sensitive operations
    client_host = request.client.host if request.client else "unknown"
    log_extra = {
        "client_host": client_host,
        "expose_mode": expose_mode or "redacted",
        "has_llm_api_key": settings.llm_api_key_is_set,
    }
    if expose_mode == "plaintext":
        logger.warning("Settings accessed with PLAINTEXT secrets", extra=log_extra)
    else:
        logger.info("Settings accessed", extra=log_extra)

    context = build_expose_context(expose_mode, config.cipher)
    with translate_missing_cipher():
        return SettingsResponse(
            agent_settings=settings.agent_settings.model_dump(
                mode="json", context=context
            ),
            conversation_settings=settings.conversation_settings.model_dump(
                mode="json"
            ),
            llm_api_key_is_set=settings.llm_api_key_is_set,
        )


@settings_router.patch(SETTINGS_PATH, response_model=SettingsResponse)
async def update_settings(
    request: Request, payload: SettingsUpdateRequest
) -> SettingsResponse:
    """Update settings with partial changes.

    Accepts ``agent_settings_diff`` and/or ``conversation_settings_diff``
    for incremental updates. Values are deep-merged with existing settings.

    Uses file locking to prevent concurrent updates from overwriting each other.

    Raises:
        HTTPException: 400 if the update payload contains invalid values.
    """
    config = get_config(request)
    store = get_settings_store(config)

    update_data = payload.model_dump(exclude_none=True)
    if not update_data:
        # No updates provided - this is a client error
        raise HTTPException(
            status_code=400,
            detail=(
                "At least one of agent_settings_diff or "
                "conversation_settings_diff must be provided"
            ),
        )

    # Apply updates atomically with file locking
    def apply_update(settings: PersistedSettings) -> PersistedSettings:
        settings.update(cast(SettingsUpdatePayload, update_data))
        return settings

    client_host = request.client.host if request.client else "unknown"
    try:
        settings = store.update(apply_update)
        # Audit log: settings modified
        logger.info(
            "Settings updated",
            extra={
                "client_host": client_host,
                "agent_settings_modified": "agent_settings_diff" in update_data,
                "conversation_settings_modified": (
                    "conversation_settings_diff" in update_data
                ),
            },
        )
    except (ValueError, ValidationError):
        # Audit log: validation failed
        # Note: PersistedSettings.update() raises ValueError (sanitized message)
        # while Pydantic validation raises ValidationError
        logger.warning(
            "Settings update validation failed",
            extra={"client_host": client_host},
        )
        # 422 Unprocessable Entity - semantic validation failure
        # Don't expose error details - could contain secrets in tracebacks
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Settings validation failed",
        )
    except RuntimeError as e:
        # Data corruption protection triggered (file exists but unreadable)
        logger.error(f"Settings update blocked: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Settings file is corrupted or encrypted with a different key",
        )
    except (OSError, PermissionError):
        # Note: exc_info omitted to prevent secrets in scope from leaking in tracebacks
        logger.error("Settings update failed - file I/O error")
        raise HTTPException(status_code=500, detail="Failed to update settings")

    # Don't expose secrets in PATCH response (consistent with GET behavior)
    return SettingsResponse(
        agent_settings=settings.agent_settings.model_dump(mode="json"),
        conversation_settings=settings.conversation_settings.model_dump(mode="json"),
        llm_api_key_is_set=settings.llm_api_key_is_set,
    )


# ── Secrets CRUD Endpoints ───────────────────────────────────────────────


@settings_router.get(SECRETS_PATH, response_model=SecretsListResponse)
async def list_secrets(request: Request) -> SecretsListResponse:
    """List all available secrets (names and descriptions only, no values)."""
    config = get_config(request)
    store = get_secrets_store(config)
    secrets = store.load()

    client_host = request.client.host if request.client else "unknown"
    secret_count = len(secrets.custom_secrets) if secrets else 0
    logger.info(
        "Secrets list accessed",
        extra={"client_host": client_host, "secret_count": secret_count},
    )

    if secrets is None:
        return SecretsListResponse(secrets=[])

    return SecretsListResponse(
        secrets=[
            SecretItemResponse(name=name, description=secret.description)
            for name, secret in secrets.custom_secrets.items()
        ]
    )


@settings_router.get(SECRET_VALUE_PATH)
async def get_secret_value(request: Request, name: str) -> Response:
    """Get a single secret value by name.

    Returns the raw secret value as plain text. This endpoint is designed
    to be used with LookupSecret for lazy secret resolution.

    Raises:
        HTTPException: 400 if name format is invalid, 404 if secret not found.
    """
    _validate_secret_name(name)

    config = get_config(request)
    store = get_secrets_store(config)
    value = store.get_secret(name)

    client_host = request.client.host if request.client else "unknown"
    if value is None:
        # Log failed access attempts to detect enumeration attacks
        logger.warning(
            "Secret access failed - not found",
            extra={"secret_name": name, "client_host": client_host},
        )
        # Use generic message to prevent secret name enumeration attacks
        raise HTTPException(status_code=404, detail="Secret not found")

    logger.info(
        "Secret accessed",
        extra={"secret_name": name, "client_host": client_host},
    )
    return Response(content=value, media_type="text/plain")


@settings_router.put(SECRETS_PATH, response_model=SecretItemResponse)
async def create_secret(
    request: Request, secret: SecretCreateRequest
) -> SecretItemResponse:
    """Create or update a custom secret (upsert).

    Raises:
        HTTPException: 400 if secret name format is invalid, 500 if file is corrupted.
    """
    _validate_secret_name(secret.name)

    config = get_config(request)
    store = get_secrets_store(config)

    try:
        store.set_secret(
            name=secret.name,
            value=secret.value.get_secret_value(),
            description=secret.description,
        )
    except RuntimeError as e:
        # Data corruption protection triggered (file exists but unreadable)
        logger.error(f"Secret create blocked: {e}")
        raise HTTPException(
            status_code=500,
            detail="Secrets file is corrupted or encrypted with a different key",
        )
    except (OSError, PermissionError):
        # Note: exc_info omitted to prevent secret values from leaking in tracebacks
        logger.error("Failed to save secret - file I/O error")
        raise HTTPException(status_code=500, detail="Failed to save secret")

    logger.info(
        "Secret created/updated",
        extra={
            "secret_name": secret.name,
            "client_host": request.client.host if request.client else "unknown",
        },
    )
    return SecretItemResponse(name=secret.name, description=secret.description)


@settings_router.delete(SECRET_VALUE_PATH)
async def delete_secret(request: Request, name: str) -> dict[str, bool]:
    """Delete a custom secret by name.

    Raises:
        HTTPException: 400 if name format is invalid, 404 if secret not found,
        500 if file is corrupted.
    """
    _validate_secret_name(name)

    config = get_config(request)
    store = get_secrets_store(config)

    client_host = request.client.host if request.client else "unknown"
    try:
        deleted = store.delete_secret(name)
    except RuntimeError as e:
        # Data corruption protection triggered (file exists but unreadable)
        logger.error(f"Secret delete blocked: {e}")
        raise HTTPException(
            status_code=500,
            detail="Secrets file is corrupted or encrypted with a different key",
        )

    if not deleted:
        # Log failed deletion attempts to detect enumeration attacks
        logger.warning(
            "Secret deletion failed - not found",
            extra={"secret_name": name, "client_host": client_host},
        )
        # Use generic message to prevent secret name enumeration attacks
        raise HTTPException(status_code=404, detail="Secret not found")

    logger.info(
        "Secret deleted",
        extra={"secret_name": name, "client_host": client_host},
    )
    return {"deleted": True}


# ── ACP env-var CRUD Endpoints ───────────────────────────────────────────
#
# ``acp_env`` is a ``dict[str, str]`` field on ``ACPAgentSettings`` that
# injects environment variables into the ACP subprocess at spawn time.
# Mirrors the secrets API so single-key adds and deletes don't have to
# thread through ``PATCH /api/settings``'s deep-merge (which has no
# "unset" primitive). Storage stays inside ``agent_settings`` — only the
# wire-level mutation shape is different.


def _require_acp_agent_settings(
    settings: PersistedSettings,
) -> ACPAgentSettings:
    """Return ``settings.agent_settings`` if it's an ACP variant, else 409.

    The ``acp_env`` field only exists on the ACP discriminated-union
    variant. Mutating it when the active agent is OpenHands would either
    be silently dropped on save (the union picks the OpenHands schema) or
    fail validation — the dedicated 409 is clearer.
    """
    agent_settings = settings.agent_settings
    if not isinstance(agent_settings, ACPAgentSettings):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "ACP env-vars are only available when ``agent_settings.agent_kind`` "
                "is ``'acp'``. Switch to an ACP agent before managing env-vars."
            ),
        )
    return agent_settings


@settings_router.get(ACP_ENV_PATH, response_model=AcpEnvVarsListResponse)
async def list_acp_env_vars(request: Request) -> AcpEnvVarsListResponse:
    """List the names of every ACP env-var configured for this agent.

    Values are intentionally not surfaced — callers that need to round-trip
    a plaintext value should use ``GET /api/settings`` with an
    ``X-Expose-Secrets`` header, since that path already understands the
    cipher and serialization context.

    Returns an empty list when the active agent isn't ACP; the GUI treats
    the absence of variables and the absence of an ACP agent the same way
    (the env-vars panel is only rendered on the ACP branch).
    """
    config = get_config(request)
    store = get_settings_store(config)
    settings = store.load() or PersistedSettings()

    client_host = request.client.host if request.client else "unknown"
    agent_settings = settings.agent_settings
    if not isinstance(agent_settings, ACPAgentSettings):
        logger.info(
            "ACP env-vars list accessed (non-ACP agent)",
            extra={"client_host": client_host, "env_var_count": 0},
        )
        return AcpEnvVarsListResponse(env_vars=[])

    names = sorted(agent_settings.acp_env.keys())
    logger.info(
        "ACP env-vars list accessed",
        extra={"client_host": client_host, "env_var_count": len(names)},
    )
    return AcpEnvVarsListResponse(
        env_vars=[AcpEnvVarItem(name=name) for name in names]
    )


@settings_router.put(ACP_ENV_PATH, response_model=AcpEnvVarItem)
async def upsert_acp_env_var(
    request: Request, body: AcpEnvVarUpsertRequest
) -> AcpEnvVarItem:
    """Create or replace an ACP env-var.

    Same shape is used for first-time adds and value rotations — the
    server-side mutation is an unconditional dict assignment under the
    store's file lock.

    Raises:
        HTTPException: 422 if name format is invalid, 409 if the active
        agent isn't ACP, 500 on file I/O failures.
    """
    _validate_secret_name(body.name)

    config = get_config(request)
    store = get_settings_store(config)
    client_host = request.client.host if request.client else "unknown"

    def apply(settings: PersistedSettings) -> PersistedSettings:
        agent_settings = _require_acp_agent_settings(settings)
        # ``acp_env: dict[str, str]`` — assign a fresh dict so Pydantic
        # treats this as an explicit field set (no in-place mutation
        # subtleties around model validation on re-save).
        agent_settings.acp_env = {
            **agent_settings.acp_env,
            body.name: body.value.get_secret_value(),
        }
        return settings

    try:
        store.update(apply)
    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error(f"ACP env-var upsert blocked: {e}")
        raise HTTPException(
            status_code=500,
            detail="Settings file is corrupted or encrypted with a different key",
        )
    except (OSError, PermissionError):
        # exc_info omitted to keep secret values out of tracebacks.
        logger.error("Failed to upsert ACP env-var - file I/O error")
        raise HTTPException(status_code=500, detail="Failed to save settings")

    logger.info(
        "ACP env-var upserted",
        extra={"env_var_name": body.name, "client_host": client_host},
    )
    return AcpEnvVarItem(name=body.name)


@settings_router.delete(ACP_ENV_VALUE_PATH)
async def delete_acp_env_var(request: Request, name: str) -> dict[str, bool]:
    """Delete a single ACP env-var by name.

    Raises:
        HTTPException: 422 if name format is invalid, 404 if the var
        doesn't exist, 409 if the active agent isn't ACP, 500 on file
        I/O failures.
    """
    _validate_secret_name(name)

    config = get_config(request)
    store = get_settings_store(config)
    client_host = request.client.host if request.client else "unknown"

    def apply(settings: PersistedSettings) -> PersistedSettings:
        agent_settings = _require_acp_agent_settings(settings)
        if name not in agent_settings.acp_env:
            # Generic 404 message — symmetric with the secrets endpoints,
            # which avoid leaking which exact names exist.
            raise HTTPException(status_code=404, detail="ACP env-var not found")
        agent_settings.acp_env = {
            k: v for k, v in agent_settings.acp_env.items() if k != name
        }
        return settings

    try:
        store.update(apply)
    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error(f"ACP env-var delete blocked: {e}")
        raise HTTPException(
            status_code=500,
            detail="Settings file is corrupted or encrypted with a different key",
        )
    except (OSError, PermissionError):
        logger.error("Failed to delete ACP env-var - file I/O error")
        raise HTTPException(status_code=500, detail="Failed to save settings")

    logger.info(
        "ACP env-var deleted",
        extra={"env_var_name": name, "client_host": client_host},
    )
    return {"deleted": True}
