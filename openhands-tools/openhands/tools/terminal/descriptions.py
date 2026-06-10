"""User-facing terminal tool descriptions by shell family."""

# Shared paragraph that proactively warns the model against the
# Python-/JSON-literal-as-command pattern. This block addresses a real-world
# failure mode observed in long-context, non-cached models (e.g. Nemotron 550B)
# where the agent would otherwise burn dozens of turns per task repeating the
# same malformed call shape. Keep this short — every byte is paid for on every
# turn on non-caching providers.
_LITERAL_ARG_GUIDANCE = "\n".join(
    [
        "### ⚠️ Command Argument Format (read this first)",
        "* The `command` argument must be a **shell command**, not a Python or",
        "  JSON literal. Do NOT put a `dict`, `list`, multi-line code block,",
        "  or other structured data directly into `command`.",
        "* If you need to execute code that has structured data or multiple",
        "  lines, use ONE of:",
        "  1. **Write a script first**, then run it: use `file_editor` with",
        '     `command="create"` to write `/tmp/run.py`, then call this tool',
        '     with `command="python /tmp/run.py"`.',
        "  2. **Inline heredoc**, e.g.:",
        "         python - <<'EOF'",
        "         DATABASES = {'default': {...}}",
        "         # your code",
        "         EOF",
        "* Examples that will be REJECTED:",
        "  - `command=\"[{'default': {...}}, ['apps'], ['code...']]\"`",
        '  - `command=\'["some", "list"]\'`',
        '  - `command=\'{"key": "value"}\'`',
    ]
)


UNIX_TOOL_DESCRIPTION = "\n".join(
    [
        "Execute a shell command in the terminal within a persistent shell session.",
        "",
        "",
        _LITERAL_ARG_GUIDANCE,
        "",
        "### Command Execution",
        "* One command at a time: You can only execute one shell command at a time.",
        "  If you need to run multiple commands sequentially, use `&&` or `;`.",
        "* Persistent session: Environment variables, virtual environments, and",
        "  working directory changes persist across commands.",
        "* Soft timeout: Commands pause for confirmation after 10 seconds without",
        "  new output unless you provide a longer `timeout`.",
        "* Shell options: Do NOT use `set -e`, `set -eu`, or `set -euo pipefail`.",
        "  The runtime may not support them reliably.",
        "",
        "### Long-running Commands",
        "* For commands that may run indefinitely, run them in the background and",
        "  redirect output to a file, e.g. `python3 app.py > server.log 2>&1 &`.",
        "* For long-running commands, set the `timeout` parameter accordingly.",
        "* If a command returns exit code `-1`, it hit the soft timeout and is",
        "  still running. With `is_input=true`, you can:",
        "  - Send empty `command` to retrieve additional logs",
        "  - Send text to STDIN of the running process",
        "  - Send control commands like `C-c`, `C-d`, or `C-z`",
        "  - Send navigation keys like `UP`, `DOWN`, `LEFT`, `RIGHT`, `TAB`,",
        "    `ESC`, `BS`, `HOME`, `END`, `PGUP`, and `PGDN`",
        "  - Send any `C-<letter>` Ctrl sequence such as `C-a`, `C-e`, or `C-l`",
        "",
        "### Best Practices",
        "* Verify a parent directory exists before creating files or directories.",
        "* Prefer absolute paths and avoid excessive use of `cd`.",
        "",
        "### Output Handling",
        "* Large output may be truncated before being returned.",
        "",
        "### Terminal Reset",
        "* Set `reset=true` to create a fresh terminal session if the current one",
        "  becomes unresponsive.",
        "* Resetting the terminal clears environment variables, working directory",
        "  changes, and running processes.",
    ]
)

WINDOWS_TOOL_DESCRIPTION = "\n".join(
    [
        (
            "Execute a shell command in the terminal within a persistent "
            "PowerShell session."
        ),
        "",
        "",
        _LITERAL_ARG_GUIDANCE,
        "",
        "### Command Execution",
        "* One command at a time: You can only execute one PowerShell command at a",
        "  time. If you need multiple commands, prefer `;` to chain them.",
        "* Persistent session: Environment variables, modules, and working",
        "  directory changes persist across commands.",
        "* Soft timeout: Commands pause for confirmation after 10 seconds without",
        "  new output unless you provide a longer `timeout`.",
        "* PowerShell syntax: Prefer native cmdlets such as `Get-ChildItem` or",
        "  `Set-Location`, or common aliases like `ls`, `cd`, and `pwd`.",
        "",
        "### Long-running Commands",
        "* For commands that may run indefinitely, prefer background jobs such as",
        "  `Start-Job -ScriptBlock { python app.py } | Receive-Job -Wait`.",
        "* For long-running commands, set the `timeout` parameter accordingly.",
        "* If a command returns exit code `-1`, it hit the soft timeout and is",
        "  still running. With `is_input=true`, you can:",
        "  - Send empty `command` to retrieve additional logs",
        "  - Send text to STDIN of the running process",
        "  - Send control commands like `C-c`",
        "  - Send navigation keys like `UP`, `DOWN`, `LEFT`, `RIGHT`, `TAB`,",
        "    `ESC`, `BS`, `HOME`, `END`, `PGUP`, and `PGDN`",
        "  - Send any `C-<letter>` Ctrl sequence such as `C-a`, `C-e`, or `C-l`",
        "",
        "### Best Practices",
        "* Verify a parent directory exists before creating files or directories.",
        "* Prefer absolute paths and avoid excessive use of `cd` or `Set-Location`.",
        "* Use PowerShell environment variable syntax like `$env:NAME = 'value'`",
        "  and `$env:NAME` when manipulating environment variables directly.",
        "",
        "### Output Handling",
        "* Large output may be truncated before being returned.",
        "",
        "### Terminal Reset",
        "* Set `reset=true` to create a fresh PowerShell session if the current",
        "  one becomes unresponsive.",
        "* Resetting the terminal clears loaded modules, environment variables,",
        "  working directory changes, and running processes.",
    ]
)
