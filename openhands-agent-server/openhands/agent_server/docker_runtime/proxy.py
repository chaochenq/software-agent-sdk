"""HTTP and WebSocket reverse-proxy helpers used in docker runtime mode.

Both helpers are deliberately dumb: they stream bytes between the outer
agent-server and an inner per-conversation container, without inspecting
request bodies or response shapes. The auth header for the inner container
is injected here (see ``X-Session-API-Key``) so callers don't have to know.

These helpers are *only* used by routes in
:mod:`openhands.agent_server.docker_runtime.routers`; nothing outside the
docker runtime needs to know about them.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
import websockets
from fastapi import HTTPException, status
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.websockets import WebSocket, WebSocketDisconnect

from openhands.agent_server.docker_runtime.container_manager import RunningContainer
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# Hop-by-hop headers (RFC 7230) — must not be forwarded by a proxy.
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        # ``host`` and ``content-length`` are recomputed by httpx; forwarding
        # the original values causes spurious 400s when bodies are re-chunked.
        "host",
        "content-length",
    }
)

# Stream chunk size for request/response bodies. 64 KiB is the same default
# httpx uses internally; we pin it so behavior is stable across versions.
_CHUNK_SIZE = 64 * 1024


def _filter_headers(headers) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in _HOP_BY_HOP_HEADERS}


async def proxy_http(
    request: Request,
    running: RunningContainer,
    *,
    upstream_path: str,
    timeout: float | None = None,
) -> StreamingResponse:
    """Forward ``request`` to the per-conversation container.

    Args:
        request: Incoming Starlette request on the outer agent-server.
        running: Bookkeeping for the target container.
        upstream_path: Path (including any query string) on the inner
            agent-server to forward to. Typically the same path the outer
            server received, since the inner agent-server exposes the same
            API surface.
        timeout: Per-request timeout in seconds. ``None`` (the default) means
            no read timeout — conversation event streams can be long-lived.

    Returns:
        A :class:`starlette.responses.StreamingResponse` that streams the
        inner container's response body back to the original caller.

    Notes:
        A fresh :class:`httpx.AsyncClient` is created per request. We avoid a
        long-lived pool because the outer server can serve many concurrent
        conversations and each one talks to a different upstream port — and
        because making the client per-request keeps the lifespan/teardown
        story trivial. If profiling later shows per-request client setup is a
        bottleneck we can revisit.
    """
    url = running.base_url + upstream_path
    headers = _filter_headers(request.headers)
    # Inject the per-container session API key so the inner server accepts
    # us. We deliberately replace any X-Session-API-Key the *client* sent —
    # the outer server has already validated the user's key by the time
    # this helper runs (via FastAPI's session_api_key dependency).
    headers["X-Session-API-Key"] = running.session_api_key

    async def _request_body() -> AsyncIterator[bytes]:
        async for chunk in request.stream():
            if chunk:
                yield chunk

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=timeout, write=30.0, pool=10.0)
    )
    req = client.build_request(
        request.method,
        url,
        headers=headers,
        params=None,  # query string is already part of upstream_path
        content=_request_body(),
    )

    try:
        upstream = await client.send(req, stream=True)
    except (httpx.ConnectError, httpx.ReadError) as exc:
        await client.aclose()
        logger.warning(
            "Upstream connection error to %s: %s", running.container_id[:12], exc
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Conversation container unreachable: {exc}",
        ) from exc

    async def _response_body() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream.aiter_raw(chunk_size=_CHUNK_SIZE):
                yield chunk
        finally:
            await upstream.aclose()
            await client.aclose()

    return StreamingResponse(
        _response_body(),
        status_code=upstream.status_code,
        headers=_filter_headers(upstream.headers),
        media_type=upstream.headers.get("content-type"),
    )


async def bridge_websocket(
    client_ws: WebSocket,
    running: RunningContainer,
    *,
    upstream_path: str,
) -> None:
    """Bridge a WebSocket session between the browser and an inner container.

    The bridge speaks both text and binary frames. Auth for the inner server
    is injected via the ``X-Session-API-Key`` header on the connect handshake
    (the agent-server's WebSocket auth also accepts that header — see
    :mod:`openhands.agent_server.sockets`).

    Precondition: ``client_ws`` MUST already be accepted by the caller. The
    bridge does not call ``accept()`` itself because the outer server's
    WebSocket-auth helper (which accepts on success) needs to run first.
    Calling ``accept()`` a second time would raise.

    Closure semantics: when either side closes (or errors), we close the
    other side and return. We do not attempt to reconnect.
    """
    upstream_url = (
        running.base_url.replace("http://", "ws://").replace("https://", "wss://")
        + upstream_path
    )

    extra_headers = {"X-Session-API-Key": running.session_api_key}

    try:
        async with websockets.connect(
            upstream_url, additional_headers=extra_headers
        ) as upstream_ws:
            await _bridge_websocket_loop(client_ws, upstream_ws)
    except websockets.exceptions.InvalidStatus as exc:
        logger.warning(
            "Upstream WebSocket rejected (%s) to %s", exc, running.container_id[:12]
        )
        # 1011 == "internal error"; closest match for an upstream HTTP failure
        # since browsers can't see HTTP status codes from a failed upgrade.
        await client_ws.close(code=1011)
    except (OSError, websockets.exceptions.WebSocketException) as exc:
        logger.warning(
            "Upstream WebSocket connect failed to %s: %s",
            running.container_id[:12],
            exc,
        )
        await client_ws.close(code=1011)


async def _bridge_websocket_loop(client_ws: WebSocket, upstream_ws) -> None:
    async def _client_to_upstream() -> None:
        try:
            while True:
                message = await client_ws.receive()
                if message.get("type") == "websocket.disconnect":
                    return
                if "bytes" in message and message["bytes"] is not None:
                    await upstream_ws.send(message["bytes"])
                elif "text" in message and message["text"] is not None:
                    await upstream_ws.send(message["text"])
        except WebSocketDisconnect:
            return

    async def _upstream_to_client() -> None:
        try:
            async for message in upstream_ws:
                if isinstance(message, (bytes, bytearray)):
                    await client_ws.send_bytes(bytes(message))
                else:
                    await client_ws.send_text(message)
        except websockets.exceptions.ConnectionClosed:
            return

    task_a = asyncio.create_task(_client_to_upstream())
    task_b = asyncio.create_task(_upstream_to_client())
    done, pending = await asyncio.wait(
        {task_a, task_b}, return_when=asyncio.FIRST_COMPLETED
    )
    for task in pending:
        task.cancel()
    for task in pending:
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    # Whichever side closed first dictates the close. Ensure the other side
    # also closes cleanly so neither leaks file descriptors.
    try:
        await upstream_ws.close()
    except Exception:
        pass
    try:
        await client_ws.close()
    except Exception:
        pass
