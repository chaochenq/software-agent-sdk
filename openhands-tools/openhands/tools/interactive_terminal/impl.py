"""Process manager for exec_command / write_stdin interactive terminal tools."""

import threading
import time

from openhands.tools.terminal.definition import TerminalAction
from openhands.tools.terminal.terminal import TerminalSession, create_terminal_session


# Yield-time bounds that match Codex's unified-exec constants.
_MIN_YIELD_SECONDS: float = 0.25  # 250 ms
_MAX_YIELD_SECONDS: float = 30.0  # 30 s
# Minimum yield for an empty-stdin poll (no characters written).
_MIN_EMPTY_POLL_YIELD_SECONDS: float = 5.0  # 5 s
# Approximate chars per LLM token — used for max_output_tokens truncation.
_CHARS_PER_TOKEN: int = 4

# Common ANSI/control-byte sequences → OpenHands special-key names.
_CONTROL_CHAR_MAP: dict[str, str] = {
    "\x03": "C-c",  # ETX / Ctrl+C
    "\x04": "C-d",  # EOT / Ctrl+D
    "\x1a": "C-z",  # SUB / Ctrl+Z
}


def _clamp_yield(yield_time_ms: int, *, is_empty_poll: bool = False) -> float:
    seconds = yield_time_ms / 1000.0
    if is_empty_poll:
        seconds = max(seconds, _MIN_EMPTY_POLL_YIELD_SECONDS)
    return max(_MIN_YIELD_SECONDS, min(seconds, _MAX_YIELD_SECONDS))


class InteractiveTerminalManager:
    """Manages multiple concurrent terminal sessions for background processes.

    Sessions are created on ``exec_command`` and referenced by integer
    ``session_id`` in subsequent ``write_stdin`` calls.  A session is removed
    automatically once the underlying process exits.
    """

    def __init__(self, work_dir: str) -> None:
        self._work_dir = work_dir
        self._sessions: dict[int, TerminalSession] = {}
        self._lock = threading.Lock()
        self._next_id = 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def exec_command(
        self,
        cmd: str,
        *,
        workdir: str | None = None,
        yield_time_ms: int = 10_000,
        max_output_tokens: int | None = None,
    ) -> tuple[str, float, int | None, int | None]:
        """Start *cmd* in a new session and return after *yield_time_ms*.

        Returns:
            ``(output, wall_time_seconds, session_id, exit_code)``

            Exactly one of ``session_id`` / ``exit_code`` is not ``None``:

            * ``session_id`` is set when the process is **still running** —
              pass it to ``write_stdin`` to poll or send input.
            * ``exit_code`` is set when the process has **completed**.
        """
        session = create_terminal_session(work_dir=workdir or self._work_dir)
        session.initialize()
        session_id = self._allocate_id()
        with self._lock:
            self._sessions[session_id] = session

        yield_s = _clamp_yield(yield_time_ms)
        action = TerminalAction(command=cmd, timeout=yield_s)

        t0 = time.monotonic()
        obs = session.execute(action)
        wall = time.monotonic() - t0

        return self._build_result(obs, wall, session, session_id, max_output_tokens)

    def write_stdin(
        self,
        session_id: int,
        *,
        chars: str = "",
        yield_time_ms: int = 5_000,
        max_output_tokens: int | None = None,
    ) -> tuple[str, float, int | None, int | None]:
        """Send *chars* to session *session_id* and return after *yield_time_ms*.

        Pass ``chars=""`` to poll for new output without writing anything.

        Returns the same ``(output, wall_time_seconds, session_id, exit_code)``
        tuple as :meth:`exec_command`.
        """
        with self._lock:
            session = self._sessions.get(session_id)

        if session is None:
            msg = (
                f"No running session with session_id={session_id}. "
                "The process may have already completed or was never started."
            )
            return msg, 0.0, None, None

        is_empty_poll = chars == ""
        yield_s = _clamp_yield(yield_time_ms, is_empty_poll=is_empty_poll)

        if is_empty_poll:
            action = TerminalAction(command="", is_input=False, timeout=yield_s)
        else:
            # Map common control characters to OpenHands special-key names.
            mapped = _CONTROL_CHAR_MAP.get(chars)
            if mapped is not None:
                action = TerminalAction(command=mapped, is_input=True, timeout=yield_s)
            else:
                # Strip a single trailing newline; the terminal adds Enter for us.
                text = chars.rstrip("\n") if chars.endswith("\n") else chars
                action = TerminalAction(command=text, is_input=True, timeout=yield_s)

        t0 = time.monotonic()
        obs = session.execute(action)
        wall = time.monotonic() - t0

        return self._build_result(obs, wall, session, session_id, max_output_tokens)

    def close(self) -> None:
        """Close all managed sessions."""
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            try:
                session.close()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _allocate_id(self) -> int:
        with self._lock:
            sid = self._next_id
            self._next_id += 1
            return sid

    def _build_result(
        self,
        obs: object,  # TerminalObservation — avoid circular import at module level
        wall: float,
        session: TerminalSession,
        session_id: int,
        max_output_tokens: int | None,
    ) -> tuple[str, float, int | None, int | None]:
        from openhands.tools.terminal.definition import TerminalObservation

        assert isinstance(obs, TerminalObservation)
        output = obs.text or ""
        if max_output_tokens is not None:
            output = output[: max_output_tokens * _CHARS_PER_TOKEN]

        if session.is_running():
            return output, wall, session_id, None

        # Process finished — remove from the sessions map.
        with self._lock:
            self._sessions.pop(session_id, None)

        # metadata.exit_code is the real exit code when PS1 appeared, or -1
        # if the process was interrupted before the prompt could be captured.
        ec = obs.metadata.exit_code
        return output, wall, None, ec
