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


class ExecCommandExecutor(
    ToolExecutor[ExecCommandAction, InteractiveTerminalObservation]
):
    def __init__(self, manager: InteractiveTerminalManager) -> None:
        self._manager = manager

    def __call__(
        self,
        action: ExecCommandAction,
        conversation: LocalConversation | None = None,  # noqa: ARG002
    ) -> InteractiveTerminalObservation:
        output, wall, session_id, exit_code = self._manager.exec_command(
            action.cmd,
            workdir=action.workdir,
            yield_time_ms=action.yield_time_ms,
            max_output_tokens=action.max_output_tokens,
        )
        return InteractiveTerminalObservation.create(
            output, wall, session_id, exit_code
        )


class WriteStdinExecutor(
    ToolExecutor[WriteStdinAction, InteractiveTerminalObservation]
):
    def __init__(self, manager: InteractiveTerminalManager) -> None:
        self._manager = manager

    def __call__(
        self,
        action: WriteStdinAction,
        conversation: LocalConversation | None = None,  # noqa: ARG002
    ) -> InteractiveTerminalObservation:
        output, wall, session_id, exit_code = self._manager.write_stdin(
            action.session_id,
            chars=action.chars,
            yield_time_ms=action.yield_time_ms,
            max_output_tokens=action.max_output_tokens,
        )
        return InteractiveTerminalObservation.create(
            output, wall, session_id, exit_code
        )
