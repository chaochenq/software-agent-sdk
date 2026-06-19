"""Deferred-init router for warm-pool agent servers.

When ``Config.deferred_init`` is True the server starts in *dormant* mode:
stateless services (VSCode, desktop, tool preload) come up as usual, but
the conversation, event, and bash routers return 503 until ``POST /api/init``
delivers the runtime configuration. This is intended for warm-pool
deployments where instances are pre-warmed before a user is matched and the
per-user workspace + credentials are attached later.

``POST /api/init`` supports two authentication mechanisms:

* **Symmetric** (default): the caller presents the dormant server's
  ``secret_key`` in the ``X-Init-API-Key`` header. The orchestrator already
  holds this key for encryption purposes.
* **Asymmetric** (when ``Config.init_public_key_file`` is set): the caller
  presents a short-lived ES256 JWT signed by a private key whose public half
  the server trusts, as ``Authorization: Bearer <jwt>``. The server only holds
  the non-secret public key, so a compromised server instance cannot forge init
  calls. The trusted keys are read once from the key file at server startup
  (``load_init_public_keys``, called from the lifespan) and cached on
  ``app.state``; a bad key file fails boot fast. See ``_verify_init_jwt``.

See: https://github.com/OpenHands/software-agent-sdk/issues/2523
"""

from __future__ import annotations

import asyncio
import os
import re
import secrets
from pathlib import Path
from typing import Any, ClassVar, Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPBearer
from fastapi.security.http import HTTPAuthorizationCredentials
from joserfc import jwk, jwt
from joserfc.errors import JoseError
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from openhands.agent_server.config import Config, WebhookSpec
from openhands.agent_server.conversation_service import ConversationService
from openhands.agent_server.server_details_router import mark_initialization_complete
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


# Symmetric init auth uses its own header (distinct from X-Session-API-Key)
# because session keys aren't known to the pool at warm-up time — they
# arrive *inside* the /api/init body. The value is checked against the dormant
# server's ``secret_key``, which the orchestrator already holds for encryption
# purposes and which will be overwritten by the init payload.
_INIT_API_KEY_HEADER = APIKeyHeader(name="X-Init-API-Key", auto_error=False)
# Asymmetric init auth carries the signed JWT as a standard bearer token.
# ``auto_error=False`` so a missing Authorization header yields a clean 401
# from ``check_init_api_key`` rather than a 403.
_INIT_BEARER = HTTPBearer(auto_error=False)


InitState = Literal["dormant", "initializing", "ready"]


class InitRequest(BaseModel):
    """Runtime configuration delivered at /api/init time.

    Each field is optional and overrides the equivalent field on the dormant
    ``Config``. Fields not provided keep the value the server was constructed
    with (typically from env vars at instance startup). The set of overridable
    fields is intentionally narrow — it covers the values that today are
    "env-var shaped" and must change per-user, not image-build-time
    configuration (Python deps, plugin set, etc.) which stays bound to the
    warm-pool flavor.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    session_api_keys: list[str] | None = Field(
        default=None,
        description=(
            "Per-user session API keys. If provided, all subsequent /api/* "
            "requests must authenticate with one of these keys via the "
            "X-Session-API-Key header."
        ),
    )
    secret_key: SecretStr | None = Field(
        default=None,
        description=(
            "Symmetric secret used to encrypt persisted secrets. If not "
            "provided, falls back to the first session_api_key (matching the "
            "default Config behavior)."
        ),
    )
    conversations_path: Path | None = Field(
        default=None,
        description=(
            "Directory where conversations are persisted. Override this to "
            "point at the mounted user workspace."
        ),
    )
    bash_events_dir: Path | None = Field(
        default=None,
        description=(
            "Directory where bash events are persisted. Typically located "
            "inside the mounted user workspace."
        ),
    )
    webhooks: list[WebhookSpec] | None = Field(
        default=None,
        description="Per-user webhooks (e.g. for streaming events back).",
    )
    web_url: str | None = Field(
        default=None,
        description=(
            "External URL where this server is reachable, used for root-path "
            "calculation. Only honored when not already set in dormant config."
        ),
    )
    allow_cors_origins: list[str] | None = Field(
        default=None,
        description="CORS origins to add to the existing localhost allowlist.",
    )
    max_concurrent_runs: int | None = Field(
        default=None,
        ge=1,
        description="Override the conversation-step concurrency limit.",
    )
    env: dict[str, str] | None = Field(
        default=None,
        description=(
            "Process environment variables to set before conversation services "
            "start. Useful for credentials consumed by tools (e.g. GITHUB_TOKEN). "
            "These are applied with ``os.environ.update``; existing values are "
            "overwritten."
        ),
    )


class InitStatus(BaseModel):
    state: InitState = Field(
        description=(
            "``dormant`` — server is up but waiting for /api/init. "
            "``initializing`` — /api/init has been received and services are "
            "starting. "
            "``ready`` — initialization complete; all /api/* routes are live."
        )
    )
    error: str | None = Field(
        default=None,
        description=(
            "If a previous /api/init attempt failed, the error message. The state "
            "rolls back to ``dormant`` so /api/init can be retried."
        ),
    )


def _build_initialized_config(base: Config, req: InitRequest) -> Config:
    """Merge dormant ``base`` config with ``req`` and clear ``deferred_init``."""
    updates: dict[str, Any] = {"deferred_init": False}
    if req.session_api_keys is not None:
        updates["session_api_keys"] = req.session_api_keys
    if req.secret_key is not None:
        updates["secret_key"] = req.secret_key
    elif req.session_api_keys and base.secret_key is None:
        # Match the Config default: fall back to first session key when no
        # secret_key was provided.
        updates["secret_key"] = SecretStr(req.session_api_keys[0])
    if req.conversations_path is not None:
        updates["conversations_path"] = req.conversations_path
    if req.bash_events_dir is not None:
        updates["bash_events_dir"] = req.bash_events_dir
    if req.webhooks is not None:
        updates["webhooks"] = req.webhooks
    if req.web_url is not None:
        updates["web_url"] = req.web_url
    if req.allow_cors_origins is not None:
        updates["allow_cors_origins"] = req.allow_cors_origins
    if req.max_concurrent_runs is not None:
        updates["max_concurrent_runs"] = req.max_concurrent_runs
    return base.model_copy(update=updates)


class InitService:
    """Tracks dormant→ready transition and serialises /api/init calls.

    A single ``asyncio.Lock`` makes concurrent /api/init posts safe; the second
    one sees ``state != "dormant"`` and gets a 400. On failure mid-init the
    state rolls back to ``dormant`` so the orchestrator can retry.
    """

    def __init__(self, app: FastAPI, base_config: Config) -> None:
        self._app = app
        self._base_config = base_config
        self._state: InitState = "dormant"
        self._error: str | None = None
        self._lock = asyncio.Lock()
        self._entered_service: ConversationService | None = None

    @property
    def state(self) -> InitState:
        return self._state

    def snapshot(self) -> InitStatus:
        return InitStatus(state=self._state, error=self._error)

    async def initialize(self, req: InitRequest) -> InitStatus:
        async with self._lock:
            if self._state != "dormant":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"server already in state: {self._state}",
                )
            self._state = "initializing"
            self._error = None
        try:
            new_config = _build_initialized_config(self._base_config, req)
            if req.env:
                # Setting env vars before services boot lets things like
                # the cipher pick up OH_SECRET_KEY-style overrides, and
                # tools pick up credentials.
                for key, value in req.env.items():
                    os.environ[key] = value

            # Reset the module-level singleton so other call sites that go
            # through ``get_default_conversation_service`` see the new
            # instance built from the merged config.
            from openhands.agent_server import conversation_service as cs_mod

            service = ConversationService.get_instance(new_config)
            cs_mod._conversation_service = service

            await service.__aenter__()
            self._entered_service = service
            self._app.state.config = new_config
            self._app.state.conversation_service = service
            mark_initialization_complete()
            self._state = "ready"
            logger.info("deferred_init: server transitioned to ready")
            return self.snapshot()
        except Exception as exc:  # pragma: no cover - logged + re-raised
            logger.exception("deferred_init: /api/init failed; rolling back to dormant")
            self._error = f"{type(exc).__name__}: {exc}"
            self._state = "dormant"
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=self._error,
            ) from exc

    async def teardown(self) -> None:
        """Tear down the conversation service if /api/init succeeded.

        Called from the FastAPI lifespan's finally clause so dormant instances
        that were never initialized don't need any cleanup.
        """
        if self._entered_service is not None:
            await self._entered_service.__aexit__(None, None, None)
            self._entered_service = None


def get_init_service(request: Request) -> InitService:
    init_service = getattr(request.app.state, "init_service", None)
    if init_service is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "server is not running with deferred_init=True; the /api/init "
                "endpoint is not available"
            ),
        )
    return init_service


# Asymmetric init tokens are ES256 (EC P-256) only — a single fixed algorithm,
# always pinned at verify time, which is what rejects the ``alg:none`` and
# public-key-as-HMAC confusion attacks. The orchestrator signs and this server
# verifies, so there is nothing to negotiate. ``_INIT_KEY_TYPE`` is the joserfc
# key type; passing it explicitly to ``jwk.import_key`` avoids the implicit-type
# warning.
_INIT_ALGORITHMS = ["ES256"]
_INIT_KEY_TYPE: Literal["EC"] = "EC"

# PEM files may concatenate multiple public-key blocks (to support key
# rotation); joserfc imports a single key per call, so we split the file into
# blocks and import each independently. Block labels are uppercase letters,
# digits, and spaces (e.g. "PUBLIC KEY").
_PEM_BLOCK_RE = re.compile(
    rb"-----BEGIN [A-Z0-9 ]+-----.+?-----END [A-Z0-9 ]+-----",
    re.DOTALL,
)


def resolve_init_token_audience(config: Config) -> str | None:
    """Resolve the expected ``aud`` claim for asymmetric init tokens.

    Precedence:
      1. ``init_token_audience`` — a literal value, if set.
      2. ``init_token_audience_env`` — the *name* of an environment variable to
         read the value from (defaults to ``AGENT_SERVER_NAME``). This lets an
         identical deployment spec give each instance a distinct audience: the
         orchestrator sets that one variable per instance while the spec stays
         constant.

    Returns ``None`` when neither yields a value.
    """
    if config.init_token_audience:
        return config.init_token_audience
    if config.init_token_audience_env:
        return os.getenv(config.init_token_audience_env) or None
    return None


def load_init_public_keys(config: Config) -> list[jwk.Key]:
    """Read and import the trusted init public keys at server startup.

    Called once at boot (see ``api_lifespan``) so a misconfigured asymmetric
    setup is a fast, loud startup failure rather than a silent per-request
    rejection. The file (``init_public_key_file``) may contain one or more
    concatenated ES256 (EC) PEM blocks, which supports key rotation: trust the
    old and new keys simultaneously during the overlap window. Individual blocks
    that fail to import are skipped with a warning so one bad block doesn't
    disable the rest.

    Returns an empty list when no key file is configured. Raises ``RuntimeError``
    when a file IS configured but (a) no audience resolves — an audience is
    required to bind tokens to this server instance — or (b) the file cannot be
    read or yields no usable keys.
    """
    path = config.init_public_key_file
    if path is None:
        return []
    if not resolve_init_token_audience(config):
        raise RuntimeError(
            "deferred_init: init_public_key_file is set but no init token "
            "audience resolved. An audience is required to bind init tokens to "
            "this server instance (cross-instance replay protection). Set the "
            "AGENT_SERVER_NAME environment variable (the default source), or set "
            "OH_INIT_TOKEN_AUDIENCE to a literal value, or point "
            "OH_INIT_TOKEN_AUDIENCE_ENV at a different environment variable."
        )
    try:
        pem_data = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            f"deferred_init: cannot read init public key file {path}: {exc}"
        ) from exc
    keys: list[jwk.Key] = []
    for block in _PEM_BLOCK_RE.findall(pem_data):
        try:
            keys.append(jwk.import_key(block, _INIT_KEY_TYPE))
        except Exception:  # JoseError / ValueError on malformed or non-EC block
            logger.warning("deferred_init: skipping unparseable init public key block")
    if not keys:
        raise RuntimeError(
            f"deferred_init: init public key file {path} contains no usable public keys"
        )
    return keys


def _verify_init_jwt(token: str, keys: list[jwk.Key], config: Config) -> None:
    """Verify an asymmetric init JWT, raising HTTP 401 on any failure.

    ``keys`` are the trusted public keys loaded once at boot by
    ``load_init_public_keys``. Verification:
      1. The signature must validate against one of ``keys`` using ONLY ES256 —
         pinning the algorithm is what rejects the ``alg:none`` and
         public-key-as-HMAC confusion attacks. Each key is tried in turn
         (supports rotation without ``kid`` coordination).
      2. ``exp`` is required, which bounds the replay window.
      3. ``aud`` must equal the resolved audience (always configured when
         asymmetric auth is enabled — enforced at boot), binding the token to
         this server instance.
    """
    decoded: jwt.Token | None = None
    for key in keys:
        try:
            decoded = jwt.decode(token, key, algorithms=_INIT_ALGORITHMS)
            break
        except JoseError:
            continue
    if decoded is None:
        # No trusted key verified the signature.
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    claims_options: dict[str, Any] = {
        "exp": {"essential": True},
        "aud": {"essential": True, "value": resolve_init_token_audience(config)},
    }
    try:
        jwt.JWTClaimsRegistry(
            leeway=config.init_token_leeway_seconds, **claims_options
        ).validate(decoded.claims)
    except JoseError as exc:
        logger.info("deferred_init: init JWT claim validation failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED) from exc


def check_init_api_key(
    request: Request,
    init_api_key: str | None = Depends(_INIT_API_KEY_HEADER),
    bearer: HTTPAuthorizationCredentials | None = Depends(_INIT_BEARER),
) -> None:
    """Auth gate for /api/init.

    Selects the mechanism from the dormant server's config:

    * ``init_public_key_file`` set → **asymmetric**: require and verify a signed
      JWT from the ``Authorization: Bearer`` header against the public keys
      loaded at boot (``app.state.init_public_keys``). The symmetric
      ``secret_key`` is NOT accepted here (fail closed) — it keeps its separate
      cipher role.
    * else ``secret_key`` set → **symmetric**: constant-time compare the
      ``X-Init-API-Key`` header against the bootstrap secret.
    * else → open (acceptable for dev).
    """
    config: Config | None = getattr(request.app.state, "config", None)
    if config is None:
        # No config at all → endpoint is open. Acceptable for dev.
        return

    if config.init_public_key_file is not None:
        # Trusted keys were read once at boot (api_lifespan) and cached here.
        keys = getattr(request.app.state, "init_public_keys", None) or []
        if bearer is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        _verify_init_jwt(bearer.credentials, keys, config)
        return

    if config.secret_key is not None:
        expected = config.secret_key.get_secret_value()
        if init_api_key is None or not secrets.compare_digest(init_api_key, expected):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        return

    # No credential configured → endpoint is open. Acceptable for dev.


def require_initialized(request: Request) -> None:
    """Dependency that 503s every /api/* route while the server is dormant.

    Returns immediately when ``deferred_init`` is False (the normal path) so
    this has zero cost for non-deferred deployments.
    """
    init_service: InitService | None = getattr(request.app.state, "init_service", None)
    if init_service is None or init_service.state == "ready":
        return
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=(
            f"server is in deferred-init state '{init_service.state}'; "
            "call POST /api/init first"
        ),
    )


init_router = APIRouter(prefix="/init", tags=["Init"])


@init_router.get("", response_model=InitStatus)
async def get_init_status(
    init_service: InitService = Depends(get_init_service),
) -> InitStatus:
    """Report the current init state.

    Authentication is intentionally not required on this endpoint so a warm
    pool controller can poll it without holding the init key. The payload
    contains no sensitive data.
    """
    return init_service.snapshot()


@init_router.post(
    "",
    response_model=InitStatus,
    dependencies=[Depends(check_init_api_key)],
)
async def initialize_server(
    req: InitRequest,
    init_service: InitService = Depends(get_init_service),
) -> InitStatus:
    """Initialize a dormant server with runtime configuration.

    Returns 400 if the server has already been initialized (state != dormant).
    Returns 500 if initialization fails; in that case the state rolls back to
    ``dormant`` so the orchestrator can retry.
    """
    return await init_service.initialize(req)
