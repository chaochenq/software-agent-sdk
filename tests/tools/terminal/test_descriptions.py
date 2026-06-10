"""SDK-7: terminal tool descriptions proactively warn against literal args.

Motivation: trajectory analysis of a Nemotron 550B SWE-Bench run showed three
"expensive" instances each hit the SDK-5 literal-arg guard 31–61 times. The
in-band hint stopped each failure but never changed the model's prior, so it
kept making the same mistake. SDK-7 puts the warning in the *system-visible*
tool description so the model sees it on every turn, before it ever emits a
malformed call. These tests pin the warning to both shell variants so it
cannot silently drift away.
"""

from openhands.tools.terminal.descriptions import (
    UNIX_TOOL_DESCRIPTION,
    WINDOWS_TOOL_DESCRIPTION,
)


def test_unix_description_warns_about_literal_arguments() -> None:
    desc = UNIX_TOOL_DESCRIPTION
    # The warning must appear and be marked visually distinct so it stands out
    # to attention-weighted summarisers.
    assert "Command Argument Format" in desc
    assert "shell command" in desc
    assert "Python" in desc and "JSON" in desc
    # Concrete recovery paths must both be mentioned.
    assert "file_editor" in desc
    assert "heredoc" in desc or "<<'EOF'" in desc
    # At least one rejection example so the model can pattern-match.
    assert "[{" in desc or '["' in desc


def test_windows_description_warns_about_literal_arguments() -> None:
    desc = WINDOWS_TOOL_DESCRIPTION
    assert "Command Argument Format" in desc
    assert "file_editor" in desc
    assert "Python" in desc and "JSON" in desc


def test_warning_appears_before_command_execution_section() -> None:
    """Models are weight-sensitive to *where* guidance appears. The warning
    must be the very first section so it's read before the model starts
    composing a command."""
    for desc in (UNIX_TOOL_DESCRIPTION, WINDOWS_TOOL_DESCRIPTION):
        warn_idx = desc.find("Command Argument Format")
        exec_idx = desc.find("### Command Execution")
        assert warn_idx != -1 and exec_idx != -1
        assert warn_idx < exec_idx, (
            "Literal-arg warning must appear above the Command Execution section"
        )


def test_descriptions_remain_compact() -> None:
    """Tool descriptions are paid for on every turn on non-caching providers.
    The added warning should be on the order of ~20 lines, not pages."""
    for desc in (UNIX_TOOL_DESCRIPTION, WINDOWS_TOOL_DESCRIPTION):
        # Budget: < 3 KB each. Today both are well under 2 KB.
        assert len(desc) < 3000, (
            f"Tool description grew past the 3 KB budget ({len(desc)} bytes)"
        )
