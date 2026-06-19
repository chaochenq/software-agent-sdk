"""Interactive terminal toolset — exec_command + write_stdin with yield_time_ms.

This package exposes two tools that closely mirror the Codex ``exec_command`` /
``write_stdin`` API and allow agents to monitor long-running background processes
without blocking the main loop:

* ``exec_command`` — start a command, yield after ``yield_time_ms``.
* ``write_stdin`` — poll or send input to a running session.

Typical usage::

    from openhands.tools.interactive_terminal import InteractiveTerminalToolSet
    from openhands.sdk import Agent, Tool

    agent = Agent(
        llm=llm,
        tools=[Tool(name=InteractiveTerminalToolSet.name)],
    )
"""

from openhands.tools.interactive_terminal.definition import (
    ExecCommandAction,
    ExecCommandTool,
    InteractiveTerminalObservation,
    InteractiveTerminalToolSet,
    WriteStdinAction,
    WriteStdinTool,
)
from openhands.tools.interactive_terminal.impl import InteractiveTerminalManager


__all__ = [
    "ExecCommandAction",
    "ExecCommandTool",
    "InteractiveTerminalManager",
    "InteractiveTerminalObservation",
    "InteractiveTerminalToolSet",
    "WriteStdinAction",
    "WriteStdinTool",
]
