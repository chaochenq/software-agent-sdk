"""Grep tool implementation for fast content search."""

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import Field

from openhands.sdk.tool import (
    Action,
    DeclaredResources,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    register_tool,
)


from openhands.tools.utils.workspace_scope import (
    ToolExecutionError,
    resolve_within_workspace,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState


class GrepAction(Action):
    """Schema for grep content search operations."""

    pattern: str = Field(description="The regex pattern to search for in file contents")
    path: str | None = Field(
        default=None,
        description=(
            "The directory (absolute path) to search in. "
            "Defaults to the current working directory."
        ),
    )
    include: str | None = Field(
        default=None,
        description=(
            "Optional file pattern to filter which files to search "
            '(e.g., "*.js", "*.{ts,tsx}")'
        ),
    )


class GrepObservation(Observation):
    """Observation from grep content search operations."""

    matches: list[str] = Field(description="List of file paths containing the pattern")
    pattern: str = Field(description="The regex pattern that was used")
    search_path: str = Field(description="The directory that was searched")
    include_pattern: str | None = Field(
        default=None, description="The file pattern filter that was used"
    )
    truncated: bool = Field(
        default=False, description="Whether results were truncated to 100 files"
    )


TOOL_DESCRIPTION = """Fast content search tool.
* Searches file contents using regular expressions
* Supports full regex syntax (eg. "log.*Error", "function\\s+\\w+", etc.)
* Filter files by pattern with the include parameter (eg. "*.js", "*.{ts,tsx}")
* Returns matching file paths sorted by modification time.
* Only the first 100 results are returned. Consider narrowing your search with stricter regex patterns or provide path parameter if you need more results.
* Use this tool when you need to find files containing specific patterns.
"""  # noqa


class GrepTool(ToolDefinition[GrepAction, GrepObservation]):
    """A ToolDefinition subclass that automatically initializes a GrepExecutor."""

    def declared_resources(self, action: Action) -> DeclaredResources:
        """Declare resource usage for parallel execution.

        All grep backends are stateless and safe to run lock-free in parallel:
        ripgrep and system grep spawn independent subprocesses, and the Python
        fallback only performs local file reads.
        """
        if not isinstance(action, GrepAction):
            raise TypeError(f"Expected GrepAction, got {type(action).__name__}")
        return DeclaredResources(keys=(), declared=True)

    def __call__(
        self,
        action: GrepAction,
        conversation: "LocalConversation | None" = None,
    ) -> Observation:
        """Scope the search root to the workspace, then search (MT-PA-003).

        Grep is read-only, which makes it the *most* attractive of the three to
        an exfiltration prompt rather than the least: pointed at `/etc` or a home
        directory it returns matching paths and lines directly into the
        conversation. A `None` search path falls back to the executor's own
        working directory, which is in scope by construction.
        """
        working_dir = getattr(self.executor, "working_dir", None)
        if not working_dir:
            raise ToolExecutionError(
                f"{self.name}: no workspace root is configured; refusing to "
                f"search unscoped."
            )
        if action.path is not None:
            resolve_within_workspace(
                action.path,
                working_dir,
                tool_name=self.name,
                parameter="path",
            )
        return super().__call__(action, conversation)

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
    ) -> Sequence["GrepTool"]:
        """Initialize GrepTool with a GrepExecutor.

        Args:
            conv_state: Conversation state to get working directory from.
                         If provided, working_dir will be taken from
                         conv_state.workspace
        """
        # Import here to avoid circular imports
        from openhands.tools.grep.impl import GrepExecutor

        working_dir = conv_state.workspace.working_dir
        if not os.path.isdir(working_dir):
            raise ValueError(f"working_dir '{working_dir}' is not a valid directory")

        # Initialize the executor
        executor = GrepExecutor(working_dir=working_dir)

        # Add working directory information to the tool description
        enhanced_description = (
            f"{TOOL_DESCRIPTION}\n\n"
            f"Your current working directory is: {working_dir}\n"
            f"When searching for content, searches are performed in this directory."
        )

        # Initialize the parent ToolDefinition with the executor
        return [
            cls(
                description=enhanced_description,
                action_type=GrepAction,
                observation_type=GrepObservation,
                annotations=ToolAnnotations(
                    title="grep",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


# Automatically register the tool when this module is imported
register_tool(GrepTool.name, GrepTool)
