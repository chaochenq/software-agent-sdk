"""Docker-runtime mode for the agent-server.

When ``Config.conversation_runtime == "docker"`` the outer agent-server stops
running conversations in-process and instead spawns a Docker container per
conversation. Each container hosts its own agent-server (configured in
``local`` mode), and this outer server acts as a thin reverse proxy in front
of those containers.

Submodules:

* :mod:`.container_manager` — spawns / tracks / stops per-conversation
  containers. Wraps ``docker run`` via subprocess.
* :mod:`.proxy` — low-level HTTP and WebSocket forwarding helpers that
  stream bytes between the outer server and the appropriate container.
* :mod:`.routers` — FastAPI routers that replace the in-process
  ``conversation_router``/``event_router``/``workspace_router``/``sockets_router``
  when docker mode is active.
"""

from openhands.agent_server.docker_runtime.container_manager import (
    ContainerManager,
    RunningContainer,
)


__all__ = ["ContainerManager", "RunningContainer"]
