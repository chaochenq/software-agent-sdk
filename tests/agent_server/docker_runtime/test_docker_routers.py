"""End-to-end tests for the docker-runtime FastAPI routers.

The "inner" container is replaced by a real FastAPI app running on an
ephemeral localhost port, plumbed in via a stub :class:`ContainerManager`.
We exercise the public HTTP and WebSocket surface of the outer agent-server
the same way a real client would (via :class:`TestClient`), so any
shape-of-the-wire bug in the proxy layer would show up here.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from uuid import UUID, uuid4

import pytest
import uvicorn
from fastapi import APIRouter, FastAPI, Header, WebSocket
from fastapi.testclient import TestClient

from openhands.agent_server.api import create_app
from openhands.agent_server.config import Config
from openhands.agent_server.docker_runtime.container_manager import RunningContainer


# ---------------------------------------------------------------------------
# Fake inner agent-server (FastAPI) bound to a real port
# ---------------------------------------------------------------------------


def _build_inner_app(session_key: str) -> FastAPI:
    """A minimal FastAPI app shaped like the per-conversation agent-server."""
    app = FastAPI()

    def _check(authorization: str | None) -> bool:
        return authorization == session_key

    api = APIRouter(prefix="/api")

    @api.post("/conversations")
    async def create_conversation(
        payload: dict,
        x_session_api_key: str = Header(default=""),
    ):
        if not _check(x_session_api_key):
            return {"detail": "unauthorized"}, 401
        # Magic flag used by ``test_post_retry_does_not_stop_existing_container``
        # to drive the inner-server-rejects-the-create branch deterministically.
        if payload.get("_force_400"):
            from fastapi import HTTPException

            raise HTTPException(status_code=400, detail="forced")
        return {"id": payload.get("conversation_id"), "echoed": payload}

    @api.delete("/conversations/{cid}")
    async def delete_conversation(
        cid: str, x_session_api_key: str = Header(default="")
    ):
        if not _check(x_session_api_key):
            return {"detail": "unauthorized"}, 401
        return {"deleted": cid}

    @api.get("/conversations/{cid}/run")
    async def get_run(cid: str, x_session_api_key: str = Header(default="")):
        if not _check(x_session_api_key):
            return {"detail": "unauthorized"}, 401
        return {"cid": cid, "status": "running"}

    @api.get("/conversations/{cid}/workspace/{file_path:path}")
    async def serve_workspace(
        cid: str,
        file_path: str,
        x_session_api_key: str = Header(default=""),
    ):
        if not _check(x_session_api_key):
            return {"detail": "unauthorized"}, 401
        return {"file": file_path, "cid": cid}

    # NB: more specific paths must be registered before ``/{cid}`` so
    # FastAPI's first-match wins doesn't treat "search" / "count" as a cid.
    @api.get("/conversations/search")
    async def search_conversations(x_session_api_key: str = Header(default="")):
        return {
            "items": [{"id": "inner-1", "workspace": {"working_dir": "/workspace"}}],
            "next_page_id": None,
        }

    # Inner agent-server's /count returns a BARE JSON integer (not an object).
    @api.get("/conversations/count")
    async def inner_count(x_session_api_key: str = Header(default="")):
        return 1

    @api.get("/conversations/{cid}")
    async def get_conversation(cid: str, x_session_api_key: str = Header(default="")):
        if not _check(x_session_api_key):
            return {"detail": "unauthorized"}, 401
        return {"id": cid, "workspace": {"working_dir": "/workspace"}}

    app.include_router(api)

    @app.websocket("/sockets/events/{cid}")
    async def events_ws(websocket: WebSocket, cid: str):
        # Auth check via header on the upgrade request.
        if websocket.headers.get("x-session-api-key") != session_key:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        # Echo back whatever the client sends, plus a server-initiated frame.
        await websocket.send_text(f"hello {cid}")
        try:
            while True:
                msg = await websocket.receive_text()
                await websocket.send_text(f"echo:{msg}")
        except Exception:
            pass

    return app


@contextmanager
def _run_inner_app(session_key: str):
    """Run the fake inner app on a real localhost port."""
    app = _build_inner_app(session_key)
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # uvicorn assigns the real port lazily; wait until it's bound.
    import time

    deadline = time.time() + 10
    port: int | None = None
    while time.time() < deadline:
        if server.started and server.servers:
            # ``servers[0].sockets[0].getsockname()`` gives us the bound port.
            sockets = list(server.servers)[0].sockets
            if sockets:
                port = sockets[0].getsockname()[1]
                break
        time.sleep(0.05)
    if port is None:
        raise RuntimeError("inner app failed to bind")
    try:
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Stub ContainerManager wired into the outer app
# ---------------------------------------------------------------------------


class _StubContainerManager:
    """A stand-in ContainerManager that points every conversation at a
    pre-existing real HTTP server (the fake inner app)."""

    def __init__(self, port: int, session_key: str) -> None:
        self._port = port
        self._session_key = session_key
        self._containers: dict[UUID, RunningContainer] = {}

    def _make(self, cid: UUID) -> RunningContainer:
        return RunningContainer(
            conversation_id=cid,
            container_id=f"fake-{cid.hex[:8]}",
            host_port=self._port,
            session_api_key=self._session_key,
            image="fake",
        )

    def preregister(self, cid: UUID) -> RunningContainer:
        """Test helper: seed the registry with a pre-existing container so
        the next ``manager.start(cid)`` call hits the ``is_new=False`` path.
        """
        self._containers[cid] = self._make(cid)
        return self._containers[cid]

    def get(self, cid: UUID) -> RunningContainer | None:
        return self._containers.get(cid)

    def list(self):
        return list(self._containers.values())

    async def start(self, cid: UUID) -> tuple[RunningContainer, bool]:
        if cid not in self._containers:
            self._containers[cid] = self._make(cid)
            return self._containers[cid], True
        return self._containers[cid], False

    async def stop(self, cid: UUID) -> bool:
        return self._containers.pop(cid, None) is not None

    async def shutdown(self) -> None:
        self._containers.clear()


@pytest.fixture
def docker_app():
    """Spin up the docker-mode outer FastAPI app + a fake inner server.

    We deliberately do NOT enter the lifespan context (no ``with TestClient(...)
    as client``): the lifespan starts a tmux/vscode/desktop service that we
    don't want to drag into these tests. Instead we set ``container_manager``
    and ``proxy_client`` directly on ``app.state``, which is what the lifespan
    would do in docker mode.
    """
    session_key = "inner-secret"
    with _run_inner_app(session_key) as port:
        app = create_app(Config(conversation_runtime="docker"))
        app.state.container_manager = _StubContainerManager(port, session_key)
        client = TestClient(app)
        try:
            yield client, app
        finally:
            client.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_post_conversations_spawns_and_forwards(docker_app):
    client, app = docker_app
    body = {
        "workspace": {"working_dir": "/host/will-be-rewritten"},
        "agent": {"kind": "Agent"},
    }
    resp = client.post("/api/conversations", json=body)
    assert resp.status_code == 200
    payload = resp.json()

    # Inner app sees a freshly-minted conversation_id and the rewritten
    # workspace path. We don't care what the id is, only that it's
    # consistent.
    inner_payload = payload["echoed"]
    assert inner_payload["workspace"]["working_dir"] == "/workspace"
    cid = UUID(inner_payload["conversation_id"])
    assert app.state.container_manager.get(cid) is not None


def test_subpath_proxied_to_inner_server(docker_app):
    client, app = docker_app
    create = client.post(
        "/api/conversations",
        json={"workspace": {"working_dir": "/x"}, "agent": {}},
    )
    cid = UUID(create.json()["echoed"]["conversation_id"])

    # Generic catch-all routes
    run = client.get(f"/api/conversations/{cid}/run")
    assert run.status_code == 200
    assert run.json() == {"cid": str(cid), "status": "running"}

    workspace = client.get(f"/api/conversations/{cid}/workspace/foo/bar.txt")
    assert workspace.status_code == 200
    assert workspace.json() == {"file": "foo/bar.txt", "cid": str(cid)}


def test_subpath_returns_404_when_no_container(docker_app):
    client, _ = docker_app
    cid = uuid4()
    resp = client.get(f"/api/conversations/{cid}/run")
    assert resp.status_code == 404


def test_delete_proxies_then_stops_container(docker_app):
    client, app = docker_app
    create = client.post(
        "/api/conversations",
        json={"workspace": {"working_dir": "/x"}, "agent": {}},
    )
    cid = UUID(create.json()["echoed"]["conversation_id"])
    assert app.state.container_manager.get(cid) is not None

    delete = client.delete(f"/api/conversations/{cid}")
    assert delete.status_code == 200
    assert delete.json() == {"deleted": str(cid)}
    # Container has been deregistered.
    assert app.state.container_manager.get(cid) is None


def test_search_aggregates_across_containers(docker_app):
    """``GET /api/conversations/search`` fans out and concatenates items.

    The wire shape must match the local ``ConversationPage``
    (``{"items": [...], "next_page_id": null}``).
    """
    client, _ = docker_app
    # Spawn two conversations
    client.post("/api/conversations", json={"workspace": {}, "agent": {}})
    client.post("/api/conversations", json={"workspace": {}, "agent": {}})

    resp = client.get("/api/conversations/search")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["next_page_id"] is None
    # Each inner server returns one item -> we should see two.
    assert len(payload["items"]) == 2


def test_batch_get_preserves_local_contract(docker_app):
    """``GET /api/conversations?ids=...`` must keep the local contract:

    * ``ids`` is required (no ``ids`` -> 422),
    * the response is a JSON list (NOT a page object),
    * missing ids slot in as ``null``.
    """
    client, _ = docker_app

    # No ids -> 422 (FastAPI validation), matching local behaviour.
    no_ids = client.get("/api/conversations")
    assert no_ids.status_code == 422

    # Spawn one conversation; look up alongside a fake id.
    created = client.post("/api/conversations", json={"workspace": {}, "agent": {}})
    cid = UUID(created.json()["echoed"]["conversation_id"])
    missing = uuid4()

    resp = client.get(f"/api/conversations?ids={cid}&ids={missing}")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 2
    assert body[0] is not None and body[0]["id"] == str(cid)
    assert body[1] is None


def test_count_returns_bare_integer(docker_app):
    """``GET /api/conversations/count`` must return a bare integer
    (matching the local-mode wire contract), not ``{"count": N}``.
    """
    client, _ = docker_app

    # Empty registry: sum across zero containers is 0.
    zero = client.get("/api/conversations/count")
    assert zero.status_code == 200
    assert zero.json() == 0  # bare int, not {"count": 0}

    client.post("/api/conversations", json={"workspace": {}, "agent": {}})
    one = client.get("/api/conversations/count")
    assert one.json() == 1
    assert one.headers["content-type"].startswith("application/json")


def test_websocket_bridges_to_inner_server(docker_app):
    client, _ = docker_app
    create = client.post(
        "/api/conversations",
        json={"workspace": {"working_dir": "/x"}, "agent": {}},
    )
    cid = UUID(create.json()["echoed"]["conversation_id"])

    with client.websocket_connect(f"/sockets/events/{cid}") as ws:
        greeting = ws.receive_text()
        assert greeting == f"hello {cid}"
        ws.send_text("ping")
        assert ws.receive_text() == "echo:ping"


def test_websocket_closes_when_conversation_unknown(docker_app):
    """When no container exists for the requested conversation, the bridge
    must close the (already-accepted) socket. Auth-on-accept happens FIRST,
    so the close fires after the handshake, not instead of it.
    """
    from starlette.websockets import WebSocketDisconnect

    client, _ = docker_app
    cid = uuid4()
    with client.websocket_connect(f"/sockets/events/{cid}") as ws:
        # Server accepts then immediately closes with 1008. Reading any
        # frame raises ``WebSocketDisconnect``.
        with pytest.raises(WebSocketDisconnect) as exc_info:
            ws.receive_text()
        assert exc_info.value.code == 1008


def test_local_mode_routes_are_unchanged():
    """Sanity check: enabling docker mode must not have leaked into local mode."""
    app = create_app(Config(conversation_runtime="local"))
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    # Local conversation router exposes the canonical POST /api/conversations
    # plus the lifecycle endpoints. Docker mode catch-all lives at
    # /api/conversations/{conversation_id}/{tail:path}; that path must NOT
    # appear in local mode.
    assert "/api/conversations" in paths
    assert "/api/conversations/{conversation_id}/{tail:path}" not in paths


# ---------------------------------------------------------------------------
# Authentication: WebSocket bridge MUST enforce the outer server's session
# keys before opening a connection to the inner container. This was a
# critical review finding.
# ---------------------------------------------------------------------------


@pytest.fixture
def docker_app_with_auth():
    """Docker-mode app with ``session_api_keys`` configured, for auth tests.

    Like ``docker_app`` we deliberately skip entering the lifespan (no
    ``with TestClient(app) as ...``): the agent-server lifespan starts a
    tmux/vscode/desktop bundle we don't want to drag into these tests.
    """
    session_key = "inner-secret"
    outer_key = "outer-secret"
    with _run_inner_app(session_key) as port:
        app = create_app(
            Config(
                conversation_runtime="docker",
                session_api_keys=[outer_key],
            )
        )
        app.state.container_manager = _StubContainerManager(port, session_key)
        client = TestClient(app)
        try:
            yield client, app, outer_key
        finally:
            client.close()


def test_websocket_rejects_wrong_session_key(docker_app_with_auth):
    """With ``session_api_keys`` set, a WS upgrade carrying a wrong key in
    the query string must be rejected BEFORE the connection is accepted.

    The auth helper's "key provided but invalid" branch rejects pre-accept,
    so TestClient surfaces this as an exception on ``websocket_connect``.

    Regression guard for review finding R3311480598.
    """
    client, app, _outer_key = docker_app_with_auth
    cid = uuid4()
    app.state.container_manager.preregister(cid)

    with pytest.raises(Exception):
        with client.websocket_connect(
            f"/sockets/events/{cid}?session_api_key=wrong",
        ):
            pass


def test_websocket_rejects_missing_first_message_auth(docker_app_with_auth):
    """With ``session_api_keys`` set and no key supplied at upgrade time,
    the auth helper falls through to first-message-auth: it accepts the
    socket, then waits for ``{"type": "auth", ...}``. A client that
    instead disconnects or sends a non-auth frame must be cut off with
    a 4001 close BEFORE the bridge to the inner server is created.

    Regression guard for review finding R3311480598.
    """
    from starlette.websockets import WebSocketDisconnect

    client, app, _outer_key = docker_app_with_auth
    cid = uuid4()
    app.state.container_manager.preregister(cid)

    # Connect with no key; the server accepts (so first-message-auth can
    # read a frame) but will close 4001 once we send a non-auth frame.
    with client.websocket_connect(f"/sockets/events/{cid}") as ws:
        ws.send_text("not an auth frame")
        with pytest.raises(WebSocketDisconnect) as exc_info:
            ws.receive_text()
        assert exc_info.value.code == 4001


def test_websocket_accepts_with_valid_outer_key(docker_app_with_auth):
    """A WS with the correct outer session key must bridge to the inner server."""
    client, app, outer_key = docker_app_with_auth
    cid = uuid4()
    app.state.container_manager.preregister(cid)

    with client.websocket_connect(
        f"/sockets/events/{cid}?session_api_key={outer_key}",
    ) as ws:
        # Inner app's echo loop is reached, so we end up with a standard
        # hello + echo round-trip.
        assert ws.receive_text() == f"hello {cid}"
        ws.send_text("ping")
        assert ws.receive_text() == "echo:ping"


# ---------------------------------------------------------------------------
# Idempotent POST: a retried-create against an existing conversation must
# NOT tear down the live container if the inner server returns 4xx.
# ---------------------------------------------------------------------------


def test_post_retry_does_not_stop_existing_container_on_inner_4xx(docker_app):
    """If ``manager.start()`` returns an existing container (``is_new=False``)
    and the inner server then returns a 4xx, we MUST leave the container
    running. Regression guard for review finding R3311480570.
    """
    client, app = docker_app

    # Seed a container so the next POST hits the "already running" branch.
    cid = uuid4()
    app.state.container_manager.preregister(cid)
    assert app.state.container_manager.get(cid) is not None

    # Force the inner server to reject the create with 400. Because the
    # container already existed (``is_new=False``), the outer route must
    # NOT tear it down.
    resp = client.post(
        "/api/conversations",
        json={
            "conversation_id": str(cid),
            "workspace": {},
            "agent": {},
            "_force_400": True,
        },
    )
    assert resp.status_code == 400
    # The live container survived the failed retry.
    assert app.state.container_manager.get(cid) is not None


def test_post_first_create_tears_down_on_inner_4xx(docker_app):
    """When ``manager.start()`` actually spawned a fresh container
    (``is_new=True``) and the inner server then rejects the create, we
    DO tear it down — otherwise an orphan container would leak.
    """
    client, app = docker_app

    cid = uuid4()
    assert app.state.container_manager.get(cid) is None

    resp = client.post(
        "/api/conversations",
        json={
            "conversation_id": str(cid),
            "workspace": {},
            "agent": {},
            "_force_400": True,
        },
    )
    assert resp.status_code == 400
    # Fresh container that the inner server rejected got cleaned up.
    assert app.state.container_manager.get(cid) is None


# ---------------------------------------------------------------------------
# Workspace static-file proxy is registered under the cookie-auth group
# so iframe/<img> embeds can authenticate via the workspace cookie.
# ---------------------------------------------------------------------------


def test_workspace_router_registered_under_cookie_auth_in_docker_mode():
    """In docker mode the workspace path must be routed via the
    workspace-cookie auth group, not via the header-only catch-all.

    Regression guard for review finding R3311480555.
    """
    app = create_app(Config(conversation_runtime="docker"))

    workspace_path = "/api/conversations/{conversation_id}/workspace/{file_path:path}"
    catchall_path = "/api/conversations/{conversation_id}/{tail:path}"

    workspace_route_index = next(
        i
        for i, route in enumerate(app.routes)
        if getattr(route, "path", None) == workspace_path
    )
    catchall_route_index = next(
        i
        for i, route in enumerate(app.routes)
        if getattr(route, "path", None) == catchall_path
    )
    # The more specific workspace route MUST be registered before the
    # catch-all so starlette's first-match wins picks it.
    assert workspace_route_index < catchall_route_index
