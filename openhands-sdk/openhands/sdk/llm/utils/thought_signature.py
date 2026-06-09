"""Helpers for handling Vertex Gemini thought signatures.

When Vertex AI Gemini is used with ``reasoning_effort`` enabled, the provider
returns a ``thoughtSignature`` field on each function-calling turn. LiteLLM
encodes that signature into the OpenAI-shaped ``tool_call.id`` by appending
``__thought__<base64-blob>`` to the canonical call id::

    call_f0be918123f4462bb482dd9df123__thought__AY89a18oWjPi7IVOiw5FIMB22r9...

The signature is required on the *immediately following* tool-result turn so
the model can resume from its previous reasoning state. It is **not** consumed
on any later turn, but the SDK currently re-ships every signature in every
subsequent prompt because they live on the event log. On long agent runs this
can be the dominant cost driver: a single 278 KB signature replayed across 30
turns equals millions of prompt tokens.

The utilities in this module identify and strip the ``__thought__`` suffix so
the SDK can keep signatures on the most recent turn(s) and drop them from
archival history without changing the canonical call id.
"""

from __future__ import annotations


THOUGHT_SIGNATURE_MARKER = "__thought__"


def has_thought_signature(tool_call_id: str | None) -> bool:
    """Return True if ``tool_call_id`` carries a Vertex thought signature."""
    return bool(tool_call_id) and THOUGHT_SIGNATURE_MARKER in tool_call_id


def strip_thought_signature(tool_call_id: str) -> str:
    """Return the canonical call id with any thought-signature suffix removed.

    Non-Gemini ids (Anthropic ``toolu_*``, OpenAI ``call_*`` without a
    signature, ACP ids, etc.) are returned unchanged.
    """
    if not tool_call_id:
        return tool_call_id
    marker_index = tool_call_id.find(THOUGHT_SIGNATURE_MARKER)
    if marker_index == -1:
        return tool_call_id
    return tool_call_id[:marker_index]
