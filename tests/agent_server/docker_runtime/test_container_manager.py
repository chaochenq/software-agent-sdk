"""Tests for :mod:`openhands.agent_server.docker_runtime.container_manager`.

We never actually run Docker here — instead we inject a fake ``run_command``
that records the arguments and returns whatever the test wants. Health
checks talk to a real :class:`http.server.HTTPServer` bound to localhost,
so the production code path (``urlopen``) is exercised end-to-end.
"""

from __future__ import annotations

import contextlib
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from uuid import uuid4

import pytest

from openhands.agent_server.config import Config
from openhands.agent_server.docker_runtime.container_manager import (
    ContainerManager,
    ContainerStartupError,
    DockerUnavailableError,
)


@dataclass
class FakeCompleted:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class _RecordingRun:
    """Stand-in for ``subprocess.run`` that just records every call.

    Returns ``responses`` in order. Each response is matched against the
    first token of the argv list (``docker``, then the verb) so tests stay
    readable.
    """

    def __init__(self, responses: dict[str, list[FakeCompleted]]) -> None:
        self.responses = {k: list(v) for k, v in responses.items()}
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        # Key off the docker subcommand (run, stop, inspect, ...).
        verb = argv[1] if len(argv) > 1 else ""
        queue = self.responses.get(verb)
        if not queue:
            return FakeCompleted(returncode=0)
        return queue.pop(0)


def _docker_config(**overrides) -> Config:
    base = {
        "conversation_runtime": "docker",
        "conversation_image": "ghcr.io/openhands/agent-server:test",
        "conversation_container_startup_timeout": 5.0,
    }
    base.update(overrides)
    return Config(**base)


@contextlib.contextmanager
def _healthy_server():
    """Run a tiny HTTP server that responds 200 to /health."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server API
            if self.path == "/health":
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args, **kwargs):  # silence noisy stderr
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


async def test_start_invokes_docker_run_and_waits_for_health(monkeypatch):
    with _healthy_server() as port:
        # Force the port allocator to hand back the healthy server's port
        # so the manager's health check actually succeeds.
        monkeypatch.setattr(
            "openhands.agent_server.docker_runtime.container_manager"
            "._find_available_tcp_port",
            lambda: port,
        )

        run = _RecordingRun(
            {
                "version": [FakeCompleted(returncode=0, stdout="Docker version")],
                "run": [FakeCompleted(returncode=0, stdout="abcdef1234567890\n")],
            }
        )
        manager = ContainerManager(
            _docker_config(), run_command=run, sleep=lambda _s: None
        )

        cid = uuid4()
        running, is_new = await manager.start(cid)

        assert is_new is True
        assert running.conversation_id == cid
        assert running.container_id == "abcdef1234567890"
        assert running.host_port == port
        assert running.session_api_key  # auto-minted

        # docker run was invoked with the expected port mapping and image.
        run_argv = next(call for call in run.calls if call[1] == "run")
        assert "-p" in run_argv
        port_arg_index = run_argv.index("-p") + 1
        # Inner container ports MUST be bound only to loopback so the only
        # path to them is via the outer agent-server's authenticated proxy.
        assert run_argv[port_arg_index] == f"127.0.0.1:{port}:8000"
        assert "ghcr.io/openhands/agent-server:test" in run_argv
        # Session key got injected via env so the inner server requires it.
        env_args = [run_argv[i + 1] for i, arg in enumerate(run_argv) if arg == "-e"]
        assert any(
            e.startswith("OH_SESSION_API_KEYS_0=") and running.session_api_key in e
            for e in env_args
        )


async def test_start_is_idempotent_per_conversation_id(monkeypatch):
    with _healthy_server() as port:
        monkeypatch.setattr(
            "openhands.agent_server.docker_runtime.container_manager"
            "._find_available_tcp_port",
            lambda: port,
        )
        run = _RecordingRun(
            {
                "version": [FakeCompleted(returncode=0, stdout="ok")],
                "run": [FakeCompleted(returncode=0, stdout="container-1\n")],
            }
        )
        manager = ContainerManager(
            _docker_config(), run_command=run, sleep=lambda _s: None
        )
        cid = uuid4()
        first, first_new = await manager.start(cid)
        second, second_new = await manager.start(cid)
        assert first is second
        assert first_new is True
        # Second call reused the existing container; callers depend on this
        # ``False`` to skip teardown after a retried-create failure.
        assert second_new is False
        # Only one ``docker run`` was issued.
        assert sum(1 for c in run.calls if c[1] == "run") == 1


async def test_start_raises_when_docker_is_missing():
    run = _RecordingRun(
        {"version": [FakeCompleted(returncode=1, stderr="docker: not found")]}
    )
    manager = ContainerManager(_docker_config(), run_command=run, sleep=lambda _s: None)
    with pytest.raises(DockerUnavailableError):
        await manager.start(uuid4())


async def test_start_cleans_up_on_container_exit(monkeypatch):
    """If the container dies during startup we must call ``docker stop``."""
    # We deliberately do not start a healthy server, so /health never
    # responds. Instead we make ``docker inspect`` report the container is
    # not running, which trips the early-exit branch.
    monkeypatch.setattr(
        "openhands.agent_server.docker_runtime.container_manager"
        "._find_available_tcp_port",
        lambda: 39999,  # unused port - urlopen will refuse
    )
    run = _RecordingRun(
        {
            "version": [FakeCompleted(returncode=0)],
            "run": [FakeCompleted(returncode=0, stdout="dead-container\n")],
            "inspect": [FakeCompleted(returncode=0, stdout="false\n")],
            "logs": [FakeCompleted(returncode=0, stdout="boom")],
            "stop": [FakeCompleted(returncode=0)],
        }
    )
    manager = ContainerManager(_docker_config(), run_command=run, sleep=lambda _s: None)
    with pytest.raises(ContainerStartupError):
        await manager.start(uuid4())
    # The teardown must have stopped the orphaned container.
    assert any(call[1] == "stop" for call in run.calls)


async def test_stop_removes_from_registry(monkeypatch):
    with _healthy_server() as port:
        monkeypatch.setattr(
            "openhands.agent_server.docker_runtime.container_manager"
            "._find_available_tcp_port",
            lambda: port,
        )
        run = _RecordingRun(
            {
                "version": [FakeCompleted(returncode=0)],
                "run": [FakeCompleted(returncode=0, stdout="some-container\n")],
                "stop": [FakeCompleted(returncode=0)],
            }
        )
        manager = ContainerManager(
            _docker_config(), run_command=run, sleep=lambda _s: None
        )
        cid = uuid4()
        await manager.start(cid)
        assert manager.get(cid) is not None
        stopped = await manager.stop(cid)
        assert stopped is True
        assert manager.get(cid) is None
        assert await manager.stop(cid) is False


async def test_shutdown_stops_all_tracked_containers(monkeypatch):
    with _healthy_server() as port:
        monkeypatch.setattr(
            "openhands.agent_server.docker_runtime.container_manager"
            "._find_available_tcp_port",
            lambda: port,
        )
        run = _RecordingRun(
            {
                "version": [FakeCompleted(returncode=0)] * 4,
                "run": [
                    FakeCompleted(returncode=0, stdout="c1\n"),
                    FakeCompleted(returncode=0, stdout="c2\n"),
                ],
                "stop": [FakeCompleted(returncode=0), FakeCompleted(returncode=0)],
            }
        )
        manager = ContainerManager(
            _docker_config(), run_command=run, sleep=lambda _s: None
        )
        await manager.start(uuid4())
        await manager.start(uuid4())
        assert len(manager.list()) == 2
        await manager.shutdown()
        assert manager.list() == []
        # We issued two ``docker stop`` calls.
        assert sum(1 for c in run.calls if c[1] == "stop") == 2
