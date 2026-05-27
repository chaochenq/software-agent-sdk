"""Spawn and track per-conversation Docker containers.

This is a deliberately small module: it owns the lifecycle of the inner
agent-server containers (one per conversation), the in-memory map from
``conversation_id`` to the container's local URL + session key, and the
``docker`` CLI calls needed to make that work. Everything else (HTTP
proxying, WebSocket bridging, request validation) lives in sibling modules.

The shape of ``docker run`` invoked here mirrors what
:class:`openhands.workspace.docker.workspace.DockerWorkspace` does in the SDK,
just adapted to live inside the agent-server.
"""

from __future__ import annotations

import asyncio
import random
import secrets
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from urllib.request import urlopen
from uuid import UUID

from openhands.agent_server.config import Config
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

# Port range to allocate host ports from for inner-container forwards.
# Mirrors DockerWorkspace's range so the two implementations don't fight.
_PORT_MIN = 30000
_PORT_MAX = 39999
_PORT_MAX_ATTEMPTS = 50


def _check_port_available(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _find_available_tcp_port() -> int:
    rng = random.SystemRandom()
    ports = list(range(_PORT_MIN, _PORT_MAX + 1))
    rng.shuffle(ports)
    for port in ports[:_PORT_MAX_ATTEMPTS]:
        if _check_port_available(port):
            return port
    raise RuntimeError(
        f"No available TCP port found in [{_PORT_MIN},{_PORT_MAX}] after "
        f"{_PORT_MAX_ATTEMPTS} attempts"
    )


@dataclass
class RunningContainer:
    """Bookkeeping for one running per-conversation agent-server container.

    Attributes:
        conversation_id: The conversation that owns this container.
        container_id: Docker container id (long form) returned by ``docker run``.
        host_port: Host port the container's ``:8000`` is mapped to.
        session_api_key: The session API key the inner agent-server was
            configured with. The outer server injects this on every proxied
            request so the inner container cannot be reached without going
            through us (assuming the host port is not exposed externally).
    """

    conversation_id: UUID
    container_id: str
    host_port: int
    session_api_key: str
    image: str = ""
    # Locking primitive so concurrent requests for the same conversation don't
    # race during shutdown.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.host_port}"


class DockerUnavailableError(RuntimeError):
    """Raised when ``docker`` is not reachable from this process."""


class ContainerStartupError(RuntimeError):
    """Raised when a freshly-started container fails to become healthy."""


class ContainerManager:
    """Tracks per-conversation Docker containers.

    The manager is in-memory only — restarting the outer agent-server forgets
    every running container. That's intentional for a first cut: durable
    container <-> conversation tracking is a follow-up concern.
    """

    def __init__(
        self,
        config: Config,
        *,
        run_command=subprocess.run,  # injectable for tests
        sleep=time.sleep,
    ) -> None:
        self._config = config
        self._containers: dict[UUID, RunningContainer] = {}
        self._lock = asyncio.Lock()
        # Injected so tests can stub out ``docker`` invocations cleanly.
        self._run_command = run_command
        self._sleep = sleep

    # -- public API --------------------------------------------------------

    @property
    def config(self) -> Config:
        return self._config

    def list(self) -> list[RunningContainer]:
        return list(self._containers.values())

    def get(self, conversation_id: UUID) -> RunningContainer | None:
        return self._containers.get(conversation_id)

    async def start(self, conversation_id: UUID) -> RunningContainer:
        """Spawn a fresh container for ``conversation_id``.

        Idempotent: if a container is already registered for this conversation
        the existing :class:`RunningContainer` is returned and no new
        container is started.
        """
        async with self._lock:
            existing = self._containers.get(conversation_id)
            if existing is not None:
                return existing

            running = await asyncio.to_thread(self._start_blocking, conversation_id)
            self._containers[conversation_id] = running
            return running

    async def stop(self, conversation_id: UUID) -> bool:
        """Stop and forget the container for ``conversation_id``.

        Returns ``True`` if a container was running, ``False`` otherwise.
        """
        async with self._lock:
            running = self._containers.pop(conversation_id, None)
        if running is None:
            return False
        await asyncio.to_thread(self._stop_blocking, running)
        return True

    async def shutdown(self) -> None:
        """Stop every tracked container. Best-effort; logs errors and
        continues so a single broken container doesn't block the rest."""
        async with self._lock:
            containers = list(self._containers.values())
            self._containers.clear()
        for running in containers:
            try:
                await asyncio.to_thread(self._stop_blocking, running)
            except Exception:
                logger.exception(
                    "Failed to stop container %s during shutdown",
                    running.container_id,
                )

    # -- internals ---------------------------------------------------------

    def _start_blocking(self, conversation_id: UUID) -> RunningContainer:
        self._ensure_docker_available()

        host_port = _find_available_tcp_port()
        session_api_key = secrets.token_urlsafe(32)
        container_name = f"oh-conv-{conversation_id.hex}-{uuid.uuid4().hex[:8]}"
        image = self._config.conversation_image

        flags: list[str] = []
        for env_name in self._config.conversation_container_forward_env:
            value = self._env_for_forward(env_name)
            if value is not None:
                flags += ["-e", f"{env_name}={value}"]
        # Always tell the inner agent-server to require this key. We inject
        # it on every proxied request from the outer server.
        flags += ["-e", f"OH_SESSION_API_KEYS_0={session_api_key}"]

        for volume in self._config.conversation_container_volumes:
            flags += ["-v", volume]

        if self._config.conversation_container_network:
            flags += ["--network", self._config.conversation_container_network]

        run_cmd = [
            "docker",
            "run",
            "-d",
            "--platform",
            self._config.conversation_container_platform,
            "--rm",
            "--ulimit",
            "nofile=65536:65536",
            "--name",
            container_name,
            "-p",
            f"{host_port}:8000",
            *flags,
            image,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
        logger.info(
            "Starting conversation container for %s on host port %d",
            conversation_id,
            host_port,
        )
        proc = self._run_command(run_cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise ContainerStartupError(
                f"docker run failed: {proc.stderr.strip() or proc.stdout.strip()}"
            )
        container_id = (proc.stdout or "").strip()
        if not container_id:
            raise ContainerStartupError("docker run returned no container id")

        running = RunningContainer(
            conversation_id=conversation_id,
            container_id=container_id,
            host_port=host_port,
            session_api_key=session_api_key,
            image=image,
        )

        try:
            self._wait_for_health(
                running, timeout=self._config.conversation_container_startup_timeout
            )
        except Exception:
            # Don't leave a stuck container behind.
            self._stop_blocking(running)
            raise

        logger.info(
            "Conversation container ready: id=%s port=%d cid=%s",
            container_id[:12],
            host_port,
            conversation_id,
        )
        return running

    def _stop_blocking(self, running: RunningContainer) -> None:
        logger.info("Stopping conversation container %s", running.container_id[:12])
        self._run_command(
            ["docker", "stop", running.container_id],
            capture_output=True,
            text=True,
            check=False,
        )

    def _ensure_docker_available(self) -> None:
        proc = self._run_command(
            ["docker", "version"], capture_output=True, text=True, check=False
        )
        if proc.returncode != 0:
            raise DockerUnavailableError(
                "Docker is not available; cannot start conversation containers"
            )

    def _wait_for_health(self, running: RunningContainer, *, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        health_url = f"{running.base_url}/health"
        while time.monotonic() < deadline:
            try:
                with urlopen(health_url, timeout=1.0) as resp:
                    if 200 <= getattr(resp, "status", 200) < 300:
                        return
            except Exception:
                pass
            # Bail out early if the container has already died: avoids
            # ticking down the entire timeout when ``docker run`` accepted
            # the command but the process inside exited immediately.
            inspect = self._run_command(
                [
                    "docker",
                    "inspect",
                    "-f",
                    "{{.State.Running}}",
                    running.container_id,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if (inspect.stdout or "").strip() != "true":
                logs = self._run_command(
                    ["docker", "logs", running.container_id],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                raise ContainerStartupError(
                    "Container stopped during startup. Logs:\n"
                    f"{(logs.stdout or '')}\n{(logs.stderr or '')}"
                )
            self._sleep(1)
        raise ContainerStartupError(
            f"Container {running.container_id[:12]} did not become healthy "
            f"within {timeout}s"
        )

    def _env_for_forward(self, name: str) -> str | None:
        """Look up an env var value to forward into a container.

        Pulled out so tests can override without monkeypatching ``os.environ``.
        """
        import os

        return os.environ.get(name)
