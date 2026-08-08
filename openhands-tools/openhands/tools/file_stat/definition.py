"""File metadata tool: size, mtime and type for a workspace path."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import Field

from openhands.sdk.tool import (
    Action,
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


class FileStatAction(Action):
    """Schema for a file metadata lookup."""

    path: str = Field(description="Absolute path of the file to describe.")


class FileStatObservation(Observation):
    """Metadata for the requested path."""

    path: str = Field(description="The path that was described")
    size_bytes: int = Field(description="Size of the file in bytes")
    modified_at: float = Field(description="Last modification time, epoch seconds")
    is_directory: bool = Field(description="Whether the path is a directory")


TOOL_DESCRIPTION = """Report size, modification time and type for a path.
* Use this to decide whether a file is worth reading before reading it
* Returns metadata only; it never returns file contents
"""


class FileStatTool(ToolDefinition[FileStatAction, FileStatObservation]):
    """A ToolDefinition subclass that describes a single workspace path."""

    def __call__(
        self,
        action: FileStatAction,
        conversation: "LocalConversation | None" = None,
    ) -> Observation:
        """Confine the lookup to the workspace before describing the path.

        Metadata reads look harmless and are not: an unscoped stat confirms
        whether `/etc/shadow` or a home-directory key file exists, and existence
        plus size is enough to steer whatever read the attacker asks for next.
        The path is resolved against the workspace root for the same reason the
        content tools are.
        """
        working_dir = getattr(self.executor, "working_dir", None)
        if not working_dir:
            raise ToolExecutionError(
                f"{self.name}: no workspace root is configured; refusing to "
                f"describe an unscoped path."
            )
        resolved = resolve_within_workspace(
            action.path,
            working_dir,
            tool_name=self.name,
            parameter="path",
        )
        action = action.model_copy(update={"path": str(resolved)})
        return super().__call__(action, conversation)

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
    ) -> Sequence["FileStatTool"]:
        """Initialize FileStatTool with a FileStatExecutor."""
        from openhands.tools.file_stat.impl import FileStatExecutor

        working_dir = conv_state.workspace.working_dir
        executor = FileStatExecutor(working_dir=working_dir)
        return [
            cls(
                description=TOOL_DESCRIPTION,
                action_type=FileStatAction,
                observation_type=FileStatObservation,
                annotations=ToolAnnotations(
                    title="file_stat",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=executor,
            )
        ]


register_tool(FileStatTool.name, FileStatTool)
