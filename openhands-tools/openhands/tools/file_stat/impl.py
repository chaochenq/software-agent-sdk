"""Executor backing the file_stat tool."""

import os

from openhands.sdk.tool import ToolExecutor

from openhands.tools.file_stat.definition import FileStatAction, FileStatObservation


class FileStatExecutor(ToolExecutor[FileStatAction, FileStatObservation]):
    """Return metadata for a path already confined to the workspace."""

    def __init__(self, working_dir: str) -> None:
        self.working_dir = working_dir

    def __call__(self, action: FileStatAction) -> FileStatObservation:
        stat = os.stat(action.path)
        return FileStatObservation(
            path=action.path,
            size_bytes=stat.st_size,
            modified_at=stat.st_mtime,
            is_directory=os.path.isdir(action.path),
        )
