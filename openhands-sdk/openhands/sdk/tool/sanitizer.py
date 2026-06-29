"""Adversarial-content sanitization for tool arguments (MT-PA-001).

LLM-generated tool calls may carry prompt-injection payloads in their string
arguments — for example content scraped from a web page or read out of a file
that tries to override the agent's instructions ("ignore previous
instructions", fake ``system:`` role prefixes, ``<|im_start|>`` tags, etc.).
Because the agent forwards these arguments to tool executors (and tool results
are later fed back into the model context), an unsanitized payload can steer the
agent regardless of the model's own intent.

This module strips/neutralizes the most common injection patterns from tool
arguments *before dispatch*. It is intentionally conservative: it neutralizes
only the matched injection phrase (replacing it with ``_NEUTRALIZED_MARKER``)
rather than discarding the whole argument, to minimize collateral damage to
legitimate content.

Sanitization can be toggled globally or per-tool via environment variables (see
``is_sanitization_enabled``) so it can be disabled for testing or for tools
where it interferes with legitimate use.

Scope / limitations (defense-in-depth):
    This control filters injection patterns out of *tool-call arguments*. It is
    one layer, not a complete mitigation. It does NOT clean attacker content
    before it enters the model context (that is MT-PA-009, browser/file output
    verification), validate model outputs (MT-PA-014), or authorize tool calls
    independently of the LLM (MT-PA-003). A model whose reasoning is already
    poisoned can still emit a benign-looking-but-malicious tool call; the
    per-tool argument validators (e.g. path-traversal and control-character
    checks) are the last line of defense for that case.
"""

from __future__ import annotations

import os
import re
import unicodedata

from openhands.sdk.logger import get_logger
from openhands.sdk.tool.schema import Action


logger = get_logger(__name__)

# Replacement token inserted in place of a detected injection pattern. Kept
# human-readable so it is obvious in logs and tool output that content was
# filtered.
_NEUTRALIZED_MARKER = "[filtered:prompt-injection]"

# Regex-based filters for common prompt-injection patterns. Each entry is a
# (rule_name, compiled_pattern) tuple. Patterns are deliberately bounded (no
# unbounded ``.*``) to avoid catastrophic backtracking on large arguments and
# to avoid over-matching legitimate prose.
INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # "ignore/disregard/forget [all] [previous|prior|above] instructions/rules"
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\b[^.\n]{0,40}?"
            r"\b(?:previous|prior|above|earlier|all|any)\b[^.\n]{0,40}?"
            r"\b(?:instructions?|prompts?|rules?|directives?|context)\b",
            re.IGNORECASE,
        ),
    ),
    # Chat-template / role markers: <|im_start|>, <|system|>, <system>, etc.
    (
        "role_template_tag",
        re.compile(
            r"<\s*\|?\s*(?:im_start|im_end|system|assistant|user)\s*\|?\s*>",
            re.IGNORECASE,
        ),
    ),
    # Bracketed role/instruction tags: [SYSTEM], [/INST], [ASSISTANT]
    (
        "bracket_role_tag",
        re.compile(
            r"\[\s*/?\s*(?:system|inst|assistant|user)\s*\]",
            re.IGNORECASE,
        ),
    ),
    # Inline role prefixes at the start of a line: "system:", "assistant:"
    (
        "inline_role_prefix",
        re.compile(r"(?im)^[ \t]*(?:system|assistant)[ \t]*:[ \t]*"),
    ),
    # "new/updated/revised instructions:" or "new system prompt:"
    (
        "new_instructions",
        re.compile(
            r"\b(?:new|updated|revised|additional)\s+"
            r"(?:instructions?|system\s+prompt)\b\s*:?",
            re.IGNORECASE,
        ),
    ),
    # Persona/jailbreak switches: "you are now", "developer mode", "DAN mode"
    (
        "persona_override",
        re.compile(
            r"\byou\s+are\s+now\b|\b(?:developer|admin|root|god)\s+mode\b",
            re.IGNORECASE,
        ),
    ),
    # Delimiter injection: closing chat-template tags (</system>) and JSON role
    # objects that try to forge a new turn ({"role": "system", ...}).
    (
        "delimiter_injection",
        re.compile(
            r"</\s*(?:system|assistant|user|instruction)\s*>"
            r'|\{\s*"role"\s*:\s*"(?:system|assistant)"',
            re.IGNORECASE,
        ),
    ),
    # Encoded-payload smuggling: calls to decoders commonly used to hide
    # instructions (base64_decode/atob/rot13/hex_decode/url_decode).
    (
        "encoded_payload",
        re.compile(
            r"\b(?:base64_decode|atob|rot13|hex_decode|url_decode)\s*\(",
            re.IGNORECASE,
        ),
    ),
]


# Common Cyrillic/Greek characters that are visual look-alikes ("confusables")
# of ASCII letters, mapped to their ASCII equivalent. NFKC does NOT collapse
# these cross-script homoglyphs, so we fold them explicitly before matching to
# defeat payloads like "іgnore all previous іnstructions" (Cyrillic і).
_HOMOGLYPH_TRANSLATION = str.maketrans(
    {
        # Cyrillic lowercase
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
        "к": "k", "м": "m", "т": "t", "н": "h", "в": "b", "і": "i", "ј": "j",
        "ѕ": "s", "ԛ": "q", "ѡ": "w", "ɡ": "g",
        # Cyrillic uppercase
        "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "Х": "X", "У": "Y",
        "К": "K", "М": "M", "Т": "T", "Н": "H", "В": "B", "І": "I", "Ј": "J",
        "Ѕ": "S",
        # Greek
        "α": "a", "ο": "o", "ν": "v", "ρ": "p", "ι": "i",
        "Α": "A", "Β": "B", "Ε": "E", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M",
        "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X", "Ζ": "Z",
    }
)


def _normalize_for_detection(text: str) -> str:
    """Canonicalize text so obfuscated injection payloads still match.

    Three steps:
    1. Remove zero-width / invisible formatting characters (Unicode category
       ``Cf``) and C0/C1 control characters (except tab/newline/CR) used to
       split injection keywords (e.g. ``ig\\u200bnore``).
    2. Apply NFKC normalization so compatibility variants (e.g. full-width
       ``ｉｇｎｏｒｅ``) collapse to their ASCII equivalents.
    3. Fold common cross-script homoglyphs (Cyrillic/Greek look-alikes) to
       ASCII, which NFKC does not handle.
    """
    stripped = "".join(
        c
        for c in text
        if c in ("\t", "\n", "\r") or unicodedata.category(c) not in ("Cf", "Cc")
    )
    normalized = unicodedata.normalize("NFKC", stripped)
    return normalized.translate(_HOMOGLYPH_TRANSLATION)


def sanitize_text(text: str) -> tuple[str, list[str]]:
    """Neutralize known prompt-injection patterns in a single string.

    Detection runs against a Unicode-normalized copy (see
    ``_normalize_for_detection``) to defeat homoglyph/zero-width obfuscation;
    when any pattern fires, the normalized+neutralized text is returned so the
    obfuscation is also stripped. Returns the (possibly modified) text and the
    list of rule names that fired.
    """
    fired: list[str] = []
    cleaned = _normalize_for_detection(text)
    for name, pattern in INJECTION_PATTERNS:
        cleaned, n = pattern.subn(_NEUTRALIZED_MARKER, cleaned)
        if n:
            fired.append(name)
    return cleaned, fired


def sanitize_tool_call(action: Action) -> tuple[Action, dict[str, list[str]]]:
    """Sanitize the string arguments of a tool action before dispatch.

    Iterates over the action's string-valued fields, neutralizes injection
    patterns in each, and returns a copy of the action with the cleaned values
    plus a report mapping each affected field to the rules that fired. If
    nothing matched, the original action is returned unchanged.
    """
    updates: dict[str, str] = {}
    report: dict[str, list[str]] = {}
    for field_name in type(action).model_fields:
        value = getattr(action, field_name, None)
        if isinstance(value, str) and value:
            cleaned, fired = sanitize_text(value)
            if fired:
                updates[field_name] = cleaned
                report[field_name] = fired
    if not updates:
        return action, report
    # Action models are frozen; produce a sanitized deep copy so nested
    # objects are not shared with / mutated on the original action.
    return action.model_copy(update=updates, deep=True), report


def is_sanitization_enabled(tool_name: str | None = None) -> bool:
    """Whether tool-input sanitization is active for the given tool.

    Controlled by environment variables (useful for testing):

    * ``OH_TOOL_INPUT_SANITIZATION`` — set to ``false``/``0``/``no`` to disable
      sanitization globally. Enabled by default.
    * ``OH_TOOL_SANITIZATION_DISABLED_TOOLS`` — comma-separated list of tool
      names for which sanitization is skipped.
    """
    if os.getenv("OH_TOOL_INPUT_SANITIZATION", "true").strip().lower() in (
        "false",
        "0",
        "no",
    ):
        return False
    if tool_name:
        disabled = {
            t.strip()
            for t in os.getenv("OH_TOOL_SANITIZATION_DISABLED_TOOLS", "").split(",")
            if t.strip()
        }
        if tool_name in disabled:
            return False
    return True
