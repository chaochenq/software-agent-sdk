"""ToolExecutor shims that connect ExecCommandTool / WriteStdinTool to the manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openhands.sdk.tool import ToolExecutor
from openhands.tools.interactive_terminal.definition import (
    ExecCommandAction,
    InteractiveTerminalObservation,
    WriteStdinAction,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.tools.interactive_terminal.impl import InteractiveTerminalManager


def _mask_output(output: str, conversation: LocalConversation | None) -> str:
    """Apply registered-secret masking to *output* if a conversation is available."""
    if not output or conversation is None:
        return output
    try:
        masked = conversation.state.secret_registry.mask_secrets_in_output(output)
        return masked or output
    except Exception:  # noqa: BLE001
        # Masking must never break tool execution — return raw output on any error
        # (e.g. malformed registry state, missing attribute on mock objects in tests).
        return output


class ExecCommandExecutor(
    ToolExecutor[ExecCommandAction, InteractiveTerminalObservation]
):
    def __init__(self, manager: InteractiveTerminalManager) -> None:
        self._manager = manager

    def __call__(
        self,
        action: ExecCommandAction,
        conversation: LocalConversation | None = None,
    ) -> InteractiveTerminalObservation:
        # TODO(issue): pre-export secrets here as TerminalExecutor._export_envs() does.
        output, wall, session_id, exit_code, original_token_count = (
            self._manager.exec_command(
                action.cmd,
                workdir=action.workdir,
                yield_time_ms=action.yield_time_ms,
                max_output_tokens=action.max_output_tokens,
            )
        )
        output = _mask_output(output, conversation)
        return InteractiveTerminalObservation.create(
            output, wall, session_id, exit_code, original_token_count
        )

    def close(self) -> None:
        self._manager.close()

    def interrupt(self) -> None:
        self._manager.interrupt()


class WriteStdinExecutor(
    ToolExecutor[WriteStdinAction, InteractiveTerminalObservation]
):
    def __init__(self, manager: InteractiveTerminalManager) -> None:
        self._manager = manager

    def __call__(
        self,
        action: WriteStdinAction,
        conversation: LocalConversation | None = None,
    ) -> InteractiveTerminalObservation:
        output, wall, session_id, exit_code, original_token_count = (
            self._manager.write_stdin(
                action.session_id,
                chars=action.chars,
                yield_time_ms=action.yield_time_ms,
                max_output_tokens=action.max_output_tokens,
            )
        )
        output = _mask_output(output, conversation)
        return InteractiveTerminalObservation.create(
            output, wall, session_id, exit_code, original_token_count
        )

    def close(self) -> None:
        self._manager.close()

    def interrupt(self) -> None:
        self._manager.interrupt()
