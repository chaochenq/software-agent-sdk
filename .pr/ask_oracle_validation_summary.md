# ask_oracle validation evidence

This directory is temporary PR evidence for the `ask_oracle` tool implementation.

## What changed

- Added `ask_oracle`, a read-only tool in **`openhands-tools`** (`openhands/tools/ask_oracle/`). It is a one-shot, tool-less sub-agent: it consults a stronger LLM for a stateless second opinion.
- The Oracle model is a saved LLM profile resolved **by convention** under the name `oracle` (`ORACLE_PROFILE_NAME`). There is no agent setting and no wiring in `OpenHandsAgentSettings`/`model.py`: users add `Tool(name="ask_oracle")` and the tool resolves the `oracle` profile from the conversation's profile store at run time. This mirrors how `TaskToolSet` lives in `openhands-tools` and resolves sub-agents by registered name.
- Removed the previous `OpenHandsAgentSettings.oracle_llm_profile` field and the `AskOracleTool` SDK built-in (and its `BUILT_IN_TOOL_CLASSES` entry). The SDK no longer references the tool at all, so there is no `openhands-sdk` → `openhands-tools` dependency.
- The active conversation LLM is not switched. The Oracle call sends only the Oracle system prompt plus the agent's question and optional context, without forwarding conversation history or tools.

## Live validation

Evidence file: `.pr/ask_oracle_live_validation.json`

Command run:

```bash
OPENHANDS_SUPPRESS_BANNER=1 \
OPENAI_API_KEY="$OPENAI_API_KEY" \
LITELLM_API_KEY="$LITELLM_API_KEY" \
uv run python .pr/ask_oracle_live_validation.py
```

Validated profiles:

- Regular profile: `openai/gpt-5-nano` with OpenAI direct API key.
- Oracle profile: `litellm_proxy/openai/gpt-5-mini` with the eval LiteLLM key.
- Eval proxy base URL: `https://llm-proxy.eval.all-hands.dev`, found in `openhands-sdk/openhands/sdk/agent/base.py`.

Result summary:

- Primary direct OpenAI profile returned: `primary profile live check ok`.
- `ask_oracle` loaded the saved `oracle` profile from an isolated temporary profile store.
- Tool observation was successful (`observation_is_error: false`).
- Oracle response identified itself as the Oracle profile and explained why an agent should ask for a second opinion when stuck.
- The temporary profile store was removed in a `finally` block after the run.

## Validation commands

Pre-commit command run on changed files: passed.

### Targeted tests

Command run:

```bash
uv run pytest \
  tests/tools/ask_oracle/test_ask_oracle.py \
  tests/sdk/tool/test_builtins.py \
  tests/sdk/test_settings.py::test_llm_agent_settings_export_schema_groups_sections \
  tests/sdk/test_settings.py::test_export_agent_settings_schema_emits_variant_tagged_sections \
  tests/examples/test_examples.py::test_directory_example_is_discovered
```

Result: `10 passed`.

### Example execution

Command run:

```bash
uv run pytest tests/examples/test_examples.py --run-examples -k 55_ask_oracle_tool
```

Result: `1 passed, 66 deselected`.

