"""FastAPI routers used when ``Config.conversation_runtime == "docker"``.

These replace ``conversation_router``, ``event_router``, ``workspace_router``
and the conversation half of ``sockets_router`` from the local-mode app.
Settings, profiles, workspaces, auth, the cloud proxy, the static frontend
and ``/server_info`` all continue to be served by the outer server unchanged
— they're not conversation-scoped.

The flow on each request is:

1. Extract the ``conversation_id`` from the path (or, for ``POST
   /api/conversations``, generate one and remember it).
2. Look up — or, on creation, spawn — the matching Docker container in the
   :class:`ContainerManager`.
3. Forward the request body and headers via
   :func:`openhands.agent_server.docker_runtime.proxy.proxy_http` (or, for
   WebSockets, :func:`bridge_websocket`).
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated
from uuid import UUID, uuid4

import httpx
from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    Request,
    WebSocket,
    status,
)
from starlette.responses import JSONResponse, StreamingResponse

from openhands.agent_server.docker_runtime.container_manager import (
    ContainerManager,
    ContainerStartupError,
    DockerUnavailableError,
    RunningContainer,
)
from openhands.agent_server.docker_runtime.proxy import (
    bridge_websocket,
    proxy_http,
)
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


def get_container_manager(request: Request) -> ContainerManager:
    manager = getattr(request.app.state, "container_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Container manager is not available",
        )
    return manager


def _build_upstream_path(request: Request, path: str) -> str:
    """Reconstruct the inner-container path from the outer request.

    The inner agent-server exposes the same API surface, so we forward the
    same path verbatim. Only difference: the outer path is rooted at
    ``/api/conversations/...`` and so is the inner one, so we just pass it
    through.
    """
    query = request.url.query
    return f"{path}?{query}" if query else path


def _container_or_404(
    manager: ContainerManager, conversation_id: UUID
) -> RunningContainer:
    running = manager.get(conversation_id)
    if running is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation not found: {conversation_id}",
        )
    return running


# ---------------------------------------------------------------------------
# HTTP: /api/conversations
# ---------------------------------------------------------------------------

docker_conversation_router = APIRouter(
    prefix="/conversations", tags=["Docker Conversations"]
)


@docker_conversation_router.post("")
async def docker_start_conversation(
    request: Request,
    include_skills: Annotated[bool, Query()] = False,
) -> JSONResponse:
    """Spawn a fresh per-conversation container, then forward the request.

    The container is registered against the *resolved* conversation id (either
    the one the client supplied or a fresh UUID4 minted here). The body is
    rewritten to:

    * pin ``conversation_id`` so the inner agent-server agrees on the id,
    * rewrite ``workspace.working_dir`` to ``/workspace`` — the inner
      container's filesystem is the canonical one, not the outer host's.
    """
    manager = get_container_manager(request)

    try:
        body_bytes = await request.body()
        body = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON body: {exc}",
        ) from exc

    raw_cid = body.get("conversation_id")
    try:
        conversation_id = UUID(raw_cid) if raw_cid else uuid4()
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid conversation_id: {raw_cid!r}",
        ) from exc

    body["conversation_id"] = str(conversation_id)

    # Inside the container, the working dir is always /workspace. Whatever
    # the caller passed in points to a host path we can't reach from the
    # outer server's vantage point.
    workspace = body.get("workspace") or {}
    workspace["working_dir"] = "/workspace"
    body["workspace"] = workspace

    try:
        running = await manager.start(conversation_id)
    except DockerUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except ContainerStartupError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc

    upstream_path = (
        f"/api/conversations?include_skills={'true' if include_skills else 'false'}"
    )
    headers = {
        "content-type": request.headers.get("content-type", "application/json"),
        "X-Session-API-Key": running.session_api_key,
        "accept": request.headers.get("accept", "application/json"),
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                running.base_url + upstream_path,
                headers=headers,
                content=json.dumps(body).encode("utf-8"),
            )
    except httpx.HTTPError as exc:
        # If we managed to start the container but the very first request
        # failed, that's a startup race. Treat as 502 and tear down so we
        # don't leak a half-initialized container.
        logger.warning(
            "Initial request to fresh container %s failed: %s",
            running.container_id[:12],
            exc,
        )
        await manager.stop(conversation_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Conversation container could not accept the request: {exc}",
        ) from exc

    if response.status_code >= 400:
        # The inner server rejected the create. Don't leave the container
        # behind in that case — it'd be orphaned, since no client will know
        # to send DELETE.
        await manager.stop(conversation_id)

    return JSONResponse(
        content=response.json() if response.content else None,
        status_code=response.status_code,
    )


@docker_conversation_router.delete("/{conversation_id}")
async def docker_delete_conversation(
    conversation_id: UUID,
    request: Request,
) -> JSONResponse:
    manager = get_container_manager(request)
    running = manager.get(conversation_id)
    if running is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation not found: {conversation_id}",
        )

    # Best-effort: ask the inner server to delete its own state first, then
    # always tear the container down so we don't leak it even if the inner
    # delete failed.
    delete_status = 200
    delete_body: bytes = b""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            upstream = await client.delete(
                f"{running.base_url}/api/conversations/{conversation_id}",
                headers={"X-Session-API-Key": running.session_api_key},
            )
        delete_status = upstream.status_code
        delete_body = upstream.content
    except httpx.HTTPError as exc:
        logger.warning("Inner DELETE failed for %s: %s", conversation_id, exc)
    finally:
        await manager.stop(conversation_id)

    return JSONResponse(
        content=json.loads(delete_body) if delete_body else None,
        status_code=delete_status,
    )


_FANOUT_LIST_PATHS = {"", "/", "/search", "/count"}


@docker_conversation_router.get("/search")
async def docker_search_conversations(request: Request) -> JSONResponse:
    return await _fanout_get_aggregate(request, "/api/conversations/search")


@docker_conversation_router.get("/count")
async def docker_count_conversations(request: Request) -> JSONResponse:
    return await _fanout_count(request)


@docker_conversation_router.get("")
async def docker_list_conversations(request: Request) -> JSONResponse:
    return await _fanout_get_aggregate(request, "/api/conversations")


@docker_conversation_router.api_route(
    "/{conversation_id}",
    methods=["GET", "PATCH"],
)
async def docker_proxy_conversation_root(
    conversation_id: UUID, request: Request
) -> StreamingResponse:
    manager = get_container_manager(request)
    running = _container_or_404(manager, conversation_id)
    return await proxy_http(
        request,
        running,
        upstream_path=_build_upstream_path(
            request, f"/api/conversations/{conversation_id}"
        ),
    )


@docker_conversation_router.api_route(
    "/{conversation_id}/{tail:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def docker_proxy_conversation_subpath(
    conversation_id: UUID, tail: str, request: Request
) -> StreamingResponse:
    """Catch-all that proxies every conversation-scoped HTTP route.

    Covers ``/run``, ``/pause``, ``/interrupt``, ``/secrets``,
    ``/confirmation_policy``, ``/switch_profile``, ``/switch_llm``,
    ``/condense``, ``/fork``, ``/agent_final_response``, all of
    ``/events/...`` (from ``event_router``), and all of ``/workspace/...``
    (from ``workspace_router``).
    """
    manager = get_container_manager(request)
    running = _container_or_404(manager, conversation_id)
    upstream_path = _build_upstream_path(
        request, f"/api/conversations/{conversation_id}/{tail}"
    )
    return await proxy_http(request, running, upstream_path=upstream_path)


# ---------------------------------------------------------------------------
# HTTP fan-out helpers (list / search / count)
# ---------------------------------------------------------------------------


async def _fanout_get_aggregate(request: Request, path: str) -> JSONResponse:
    """Aggregate ``items`` lists from every container.

    Each inner agent-server has at most one conversation, so the inner
    response always looks like ``{"items": [...], "next_page_id": null}``.
    We concatenate the items in deterministic order (by container start
    time, which is the registry insertion order).

    Pagination is intentionally not implemented in this first cut: ``GET
    /api/conversations`` returns at most ``len(containers)`` rows, which
    is bounded by however many conversations the operator allows.
    """
    manager = get_container_manager(request)
    query = request.url.query
    upstream_path = f"{path}?{query}" if query else path

    async def _fetch(running: RunningContainer):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    running.base_url + upstream_path,
                    headers={"X-Session-API-Key": running.session_api_key},
                )
        except httpx.HTTPError as exc:
            logger.warning(
                "Fan-out GET %s failed for %s: %s",
                upstream_path,
                running.container_id[:12],
                exc,
            )
            return None
        if resp.status_code != 200:
            return None
        try:
            return resp.json()
        except json.JSONDecodeError:
            return None

    results = await asyncio.gather(
        *[_fetch(c) for c in manager.list()], return_exceptions=False
    )

    aggregated_items: list = []
    for result in results:
        if not result:
            continue
        items = result.get("items")
        if isinstance(items, list):
            aggregated_items.extend(items)

    return JSONResponse(
        content={"items": aggregated_items, "next_page_id": None},
        status_code=200,
    )


async def _fanout_count(request: Request) -> JSONResponse:
    manager = get_container_manager(request)
    return JSONResponse(content={"count": len(manager.list())}, status_code=200)


# ---------------------------------------------------------------------------
# WebSockets: /sockets/events/{cid}
# ---------------------------------------------------------------------------

docker_sockets_router = APIRouter(prefix="/sockets", tags=["Docker WebSockets"])


@docker_sockets_router.websocket("/events/{conversation_id}")
async def docker_events_websocket(websocket: WebSocket, conversation_id: UUID) -> None:
    manager = getattr(websocket.app.state, "container_manager", None)
    if manager is None:
        await websocket.close(code=1011)
        return
    running = manager.get(conversation_id)
    if running is None:
        # 1008 == policy violation; closest standard code for "no such conv".
        await websocket.close(code=1008)
        return
    upstream_path = f"/sockets/events/{conversation_id}"
    if websocket.url.query:
        upstream_path = f"{upstream_path}?{websocket.url.query}"
    await bridge_websocket(websocket, running, upstream_path=upstream_path)
