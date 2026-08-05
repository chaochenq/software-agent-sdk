"""CORS middleware for the agent server.

``CORSDispatcher`` routes requests to one of two CORS configurations
based on path:

* Workspace cookie endpoints (``/api/auth/workspace-session`` and
  ``/api/conversations/{id}/workspace/*``) — credentialed CORS restricted
  to ``allowed_workspace_origins`` / ``workspace_cors_origin_regex``, and
  denying cross-origin requests when neither is set. These are the only
  routes that authenticate via an ambient (cookie) credential, so a
  permissive origin here is readable by any page the victim visits.
* Everything else — ``LocalhostCORSMiddleware``, which honors the
  operator's ``allow_cors_origins`` / ``allow_cors_origin_regex`` and always
  allows localhost and ``DOCKER_HOST_ADDR`` (matches OpenHands/OpenHands#4624
  intent).
"""

import os
import re
from urllib.parse import urlparse

from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send


_WORKSPACE_SESSION_PATH = "/api/auth/workspace-session"
_WORKSPACE_STATIC_RE = re.compile(r"^/api/conversations/[^/]+/workspace(/|$)")


def _is_workspace_cookie_path(path: str) -> bool:
    if path == _WORKSPACE_SESSION_PATH:
        return True
    return bool(_WORKSPACE_STATIC_RE.match(path))


class LocalhostCORSMiddleware(CORSMiddleware):
    """``CORSMiddleware`` that always allows localhost and ``DOCKER_HOST_ADDR``."""

    def __init__(
        self,
        app: ASGIApp,
        allow_origins: list[str],
        allow_origin_regex: str | None = None,
    ) -> None:
        super().__init__(
            app,
            allow_origins=allow_origins,
            allow_origin_regex=allow_origin_regex,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def is_allowed_origin(self, origin: str) -> bool:
        if origin:
            hostname = urlparse(origin).hostname or ""
            if hostname in ("localhost", "127.0.0.1"):
                return True
            docker_host_addr = os.environ.get("DOCKER_HOST_ADDR")
            if docker_host_addr and hostname == docker_host_addr:
                return True
        return bool(super().is_allowed_origin(origin))


class CORSDispatcher:
    """Dispatches each request to the workspace or default CORS middleware.

    The workspace branch is configured from ``allowed_workspace_origins`` /
    ``workspace_cors_origin_regex`` and denies cross-origin requests by
    default. These routes send credentials, so any origin allowed here can
    read a victim's workspace files, conversations and credentials from
    their ambient session.

    It deliberately still uses an explicit origin list / regex rather than
    ``allow_origins=["*"]``, for two reasons that outlive the wildcard:

    1. Starlette emits a literal ``*`` on simple responses when
       ``allow_all_origins`` is set and the request has no ``Cookie``
       header — which browsers reject together with
       ``Access-Control-Allow-Credentials: true``. The explicit path always
       echoes the request Origin (with ``Vary: Origin``).
    2. Requiring an ``http(s)://`` origin excludes ``Origin: null``
       (sandboxed iframes, ``data:`` / ``blob:`` URLs), which have no
       defined CHIPS partition key and are not legitimate clients. Anchor
       any regex you configure — an unanchored pattern matches every origin
       that merely contains the intended one.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        allow_origins: list[str],
        allow_origin_regex: str | None = None,
        allowed_workspace_origins: list[str] | None = None,
        workspace_origin_regex: str | None = None,
    ) -> None:
        self._default_cors = LocalhostCORSMiddleware(
            app,
            allow_origins=list(allow_origins),
            allow_origin_regex=allow_origin_regex,
        )
        # The workspace cookie routes send credentials, so a permissive origin
        # here lets any page a victim visits read their workspace files,
        # conversations and credentials from the ambient session. The previous
        # r"https?://.+" matched every origin on either scheme, which is
        # indistinguishable from allow_origins=["*"] with credentials enabled —
        # a combination browsers refuse precisely because it is unsafe.
        #
        # Deny by default: with nothing configured, no cross-origin request is
        # accepted on these routes and same-origin use is unaffected.
        workspace_origins = list(allowed_workspace_origins or [])
        self._workspace_cors = CORSMiddleware(
            app,
            allow_origins=workspace_origins,
            allow_origin_regex=workspace_origin_regex,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") == "http":
            # Strip FastAPI ``root_path`` so dispatch works behind
            # reverse proxies mounted under a sub-path.
            root_path = scope.get("root_path", "")
            path = scope.get("path", "/")
            route_path = path.removeprefix(root_path) if root_path else path
            if _is_workspace_cookie_path(route_path or "/"):
                await self._workspace_cors(scope, receive, send)
                return
        await self._default_cors(scope, receive, send)
