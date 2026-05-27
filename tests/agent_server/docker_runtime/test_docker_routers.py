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

    @api.get("/conversations")
    async def list_conversations(x_session_api_key: str = Header(default="")):
        if not _check(x_session_api_key):
            return {"detail": "unauthorized"}, 401
        return {"items": [{"id": "inner-1"}], "next_page_id": None}

    @api.get("/conversations/search")
    async def search_conversations(x_session_api_key: str = Header(default="")):
        return {"items": [{"id": "inner-1"}], "next_page_id": None}

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

    def get(self, cid: UUID) -> RunningContainer | None:
        return self._containers.get(cid)

    def list(self):
        return list(self._containers.values())

    async def start(self, cid: UUID) -> RunningContainer:
        if cid not in self._containers:
            self._containers[cid] = self._make(cid)
        return self._containers[cid]

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


def test_list_aggregates_across_containers(docker_app):
    client, _ = docker_app
    # Spawn two conversations
    client.post("/api/conversations", json={"workspace": {}, "agent": {}})
    client.post("/api/conversations", json={"workspace": {}, "agent": {}})

    resp = client.get("/api/conversations")
    assert resp.status_code == 200
    payload = resp.json()
    # Each inner server returns one item -> we should see two.
    assert len(payload["items"]) == 2


def test_count_uses_local_registry(docker_app):
    client, _ = docker_app
    assert client.get("/api/conversations/count").json() == {"count": 0}
    client.post("/api/conversations", json={"workspace": {}, "agent": {}})
    assert client.get("/api/conversations/count").json() == {"count": 1}


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
    client, _ = docker_app
    cid = uuid4()
    with pytest.raises(Exception):
        with client.websocket_connect(f"/sockets/events/{cid}"):
            pass


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
