"""Allowlist enforcement for TerminalTool command execution (MT-PA-006).

The terminal tool hands a model-authored string to a real shell, so an
untrusted instruction that reaches the model — via a fetched page, a file in
the workspace, or a tool result — becomes arbitrary command execution. This
module holds the deny-by-default gate applied before any command is
dispatched: the base binary must be on the allowlist, shell metacharacters and
substitution syntax are refused, and per-command constraints reject the
destructive flag combinations that an allowlisted binary would otherwise still
permit.

Policy lives in ``command_allowlist.yaml`` beside this module so it can be
reviewed and changed without touching the enforcement logic.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

_POLICY_PATH = Path(__file__).with_name("command_allowlist.yaml")

# Shell syntax that lets one approved base command carry an unapproved one:
# pipelines, chaining, redirection, globbing and subshells. Refused outright
# rather than parsed, because matching the shell's own parsing well enough to
# be a security boundary is not tractable here.
_METACHARACTERS = ("|", "&", ";", ">", "<", "*", "?", "[", "]", "$(", "`", "\n")

# Substitution and dynamic-execution forms. `$(...)` and backticks are covered
# above; these catch variable expansion and interpreter escapes that would let
# the real command be assembled at runtime.
_SUBSTITUTION_PATTERN = re.compile(r"\$\{?\w+\}?|\$\(|`")
_DYNAMIC_EXEC = ("eval", "exec", "source", ".")


@dataclass(frozen=True)
class CommandRejection:
    """Why a command was refused, and what to tell the model."""

    reason: str
    base_command: str | None

    def as_message(self) -> str:
        return (
            f"Command rejected by the terminal allowlist: {self.reason}\n\n"
            "Only approved base commands may run, and shell metacharacters "
            "(pipes, chaining, redirection, globbing, command substitution) "
            "are not permitted. Run one approved command at a time."
        )


@lru_cache(maxsize=1)
def _load_policy() -> dict[str, Any]:
    """Read the YAML policy once per process.

    A missing or malformed policy file is fatal by design: silently falling
    back to "allow everything" would turn a deploy mistake into an open shell.
    """
    try:
        with _POLICY_PATH.open("r", encoding="utf-8") as handle:
            policy = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Terminal command allowlist could not be loaded from {_POLICY_PATH}") from exc

    if not isinstance(policy, dict) or not policy.get("allowed"):
        raise RuntimeError(f"Terminal command allowlist at {_POLICY_PATH} is empty or malformed")
    return policy


def _base_command(tokens: list[str]) -> str:
    """The binary being invoked, with any directory prefix stripped.

    ``/usr/bin/git`` and ``git`` are the same command for allowlist purposes;
    comparing the raw token would let a full path bypass the lookup.
    """
    return Path(tokens[0]).name


def timeout_for(command: str) -> int:
    """Execution ceiling in seconds for ``command``.

    Builds, installs and test runs legitimately take minutes; everything else
    is held to the short default so a hung command cannot hold the session.
    """
    policy = _load_policy()
    default = int(policy.get("default_timeout_seconds", 30))
    try:
        tokens = shlex.split(command)
    except ValueError:
        return default
    if not tokens:
        return default
    if _base_command(tokens) in set(policy.get("long_running", [])):
        return int(policy.get("long_timeout_seconds", 300))
    return default


def _check_constraints(policy: dict[str, Any], base: str, tokens: list[str]) -> CommandRejection | None:
    constraints = (policy.get("constraints") or {}).get(base)
    if not constraints:
        return None

    args = tokens[1:]

    allowed_subcommands = constraints.get("allowed_subcommands")
    if allowed_subcommands:
        subcommand = next((a for a in args if not a.startswith("-")), None)
        if subcommand is None:
            return CommandRejection(f"{base} requires a subcommand", base)
        if subcommand not in set(allowed_subcommands):
            return CommandRejection(f"{base} subcommand {subcommand!r} is not approved", base)

    denied = set(constraints.get("denied_flags") or [])
    for arg in args:
        if arg in denied:
            return CommandRejection(f"{base} flag {arg!r} is not permitted", base)
    return None


def validate_command(command: str) -> CommandRejection | None:
    """Return a rejection when ``command`` may not run, or ``None`` to allow.

    Empty commands are allowed through: the terminal tool uses them to poll
    for further output from a still-running process, and they reach no shell.
    """
    if not command or not command.strip():
        return None

    if any(token in command for token in _METACHARACTERS):
        return CommandRejection("command contains shell metacharacters", None)

    if _SUBSTITUTION_PATTERN.search(command):
        return CommandRejection("command contains variable or command substitution", None)

    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        # Unbalanced quoting — the shell would parse this differently than we
        # just did, so the allowlist decision would not be trustworthy.
        return CommandRejection(f"command could not be parsed ({exc})", None)

    if not tokens:
        return CommandRejection("command is empty after parsing", None)

    base = _base_command(tokens)

    if base in _DYNAMIC_EXEC:
        return CommandRejection(f"{base!r} executes arbitrary code and is not permitted", base)

    policy = _load_policy()
    if base not in set(policy["allowed"]):
        return CommandRejection(f"{base!r} is not an approved command", base)

    return _check_constraints(policy, base, tokens)


def enforce(command: str) -> CommandRejection | None:
    """Validate ``command`` and record the decision.

    Both outcomes are logged: approvals give the audit trail its denominator,
    without which a spike in rejections cannot be distinguished from a spike
    in traffic.
    """
    rejection = validate_command(command)
    if rejection is None:
        logger.info(
            "terminal allowlist: approved",
            extra={"command": command, "timeout_seconds": timeout_for(command)},
        )
        return None

    logger.warning(
        "terminal allowlist: rejected",
        extra={
            "command": command,
            "base_command": rejection.base_command,
            "rejection_reason": rejection.reason,
        },
    )
    return rejection

# Reviewed: allowlist policy is loaded once per process and fails closed.
