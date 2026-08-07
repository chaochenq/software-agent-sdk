"""Workspace-root path scoping for file-access tools (MT-PA-003).

The file tools take a path straight from the model's tool call. That path is
attacker-influenced whenever the conversation contains untrusted content — a
repository file, a fetched page, a prior tool observation — so an adversarial
prompt can steer any of them at `/etc/shadow`, `~/.aws/credentials`, or a
symlink planted inside the workspace that points anywhere on the host. Nothing
downstream re-checks it: the editor happily reads an absolute path, and grep
happily searches one.

Scoping is therefore enforced here, at the tool boundary, before any file I/O
happens. Three properties matter and each is easy to get subtly wrong:

* Resolution must be **canonical**. Comparing the raw string prefix accepts
  `/workspace/../etc/shadow`; comparing after `os.path.normpath` still accepts a
  symlink. Only `Path.resolve()`, which walks symlinks to their real target,
  answers the question actually being asked — *which file will the OS open?*
* The comparison must be **path-component-wise**. A string `startswith` check
  accepts `/workspace-backup` for a root of `/workspace`, because the prefix
  matches without the boundary being a directory separator.
* The root itself must be resolved, or a symlinked root (common when the
  workspace is a bind mount or a temp dir on macOS, where `/tmp` is a link to
  `/private/tmp`) makes every legitimate path look like an escape.
"""

from __future__ import annotations

import re
from pathlib import Path

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)


class ToolExecutionError(Exception):
    """A tool refused to run because its arguments violated a safety boundary."""


def _resolved_root(workspace_root: str | Path) -> Path:
    return Path(workspace_root).expanduser().resolve()


def _is_within(candidate: Path, root: Path) -> bool:
    """Whether ``candidate`` is ``root`` or lives beneath it.

    ``Path.is_relative_to`` compares path components, so it does not suffer the
    `/workspace-backup` false accept that a string prefix check does.
    """
    return candidate == root or candidate.is_relative_to(root)


def resolve_within_workspace(
    path: str | Path,
    workspace_root: str | Path,
    *,
    tool_name: str,
    parameter: str = "path",
) -> Path:
    """Canonicalise ``path`` and assert it lies inside ``workspace_root``.

    Returns the resolved path so callers can use the canonical form rather than
    re-deriving it (and re-introducing a TOCTOU gap between check and use).

    Raises:
        ToolExecutionError: if the path escapes the workspace root, including
            via `..` segments or a symlink whose target is outside.
    """
    root = _resolved_root(workspace_root)

    # `strict=False`: a `create` command legitimately names a file that does not
    # exist yet. Resolution still walks every existing parent component, so a
    # symlinked directory in the middle of the path is followed and caught.
    resolved = Path(path).expanduser().resolve()

    if not _is_within(resolved, root):
        logger.warning(
            "%s: rejected %s outside the workspace root (root=%s)",
            tool_name,
            parameter,
            root,
        )
        # The message names the boundary but not the resolved target: echoing
        # where the path landed turns a rejection into a filesystem oracle the
        # caller can probe.
        raise ToolExecutionError(
            f"{tool_name}: '{parameter}' resolves outside the workspace root "
            f"{root}. File access is confined to the workspace."
        )

    # A path may resolve inside the root and still be a symlink whose own target
    # is outside it — `resolve()` returns the target, so this is caught above for
    # the path itself, but the final component is re-checked explicitly because a
    # dangling symlink resolves to its (non-existent) target name rather than
    # raising.
    raw = Path(path).expanduser()
    if raw.is_symlink():
        target = raw.readlink()
        absolute_target = target if target.is_absolute() else raw.parent / target
        if not _is_within(absolute_target.resolve(), root):
            logger.warning(
                "%s: rejected %s — symlink points outside the workspace root",
                tool_name,
                parameter,
            )
            raise ToolExecutionError(
                f"{tool_name}: '{parameter}' is a symlink pointing outside the "
                f"workspace root {root}. File access is confined to the workspace."
            )

    return resolved


# Absolute paths and parent-directory hops are the two ways a shell command
# names a file outside the workspace without the tool layer ever seeing a `path`
# argument. This is a heuristic and is documented as one: a shell command can
# reach outside through indirection this cannot see (`$HOME`, a variable, `eval`,
# a base64-decoded string). It is a speed bump for the obvious attempt, not the
# isolation boundary — that remains the sandbox the terminal runs inside.
_TRAVERSAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("parent-directory traversal", re.compile(r"(?:^|[\s\"'=(:;|&])\.\./")),
    (
        "absolute path outside the workspace",
        re.compile(
            r"(?:^|[\s\"'=(:;|&])/(?:etc|root|proc|sys|boot|var/run|var/log)(?:/|\b)"
        ),
    ),
    (
        "home-directory access",
        re.compile(r"(?:^|[\s\"'=(:;|&])(?:~|\$HOME)(?:/|\b)"),
    ),
)


def screen_command_for_traversal(
    command: str,
    workspace_root: str | Path,
    *,
    tool_name: str,
) -> None:
    """Reject a shell command that obviously reaches outside the workspace.

    Raises:
        ToolExecutionError: on the first matching traversal pattern.
    """
    root = _resolved_root(workspace_root)

    # A command may legitimately mention the workspace root by absolute path;
    # blanking those occurrences first keeps the absolute-path rule from firing
    # on a path that is in fact in scope.
    scrubbed = command.replace(str(root), "")

    for label, pattern in _TRAVERSAL_PATTERNS:
        if pattern.search(scrubbed):
            logger.warning(
                "%s: rejected command — %s (root=%s)", tool_name, label, root
            )
            raise ToolExecutionError(
                f"{tool_name}: command rejected — {label}. Commands are confined "
                f"to the workspace root {root}."
            )
