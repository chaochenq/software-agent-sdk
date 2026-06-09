"""Tests for the thought_signature utility module."""

from openhands.sdk.llm.utils.thought_signature import (
    THOUGHT_SIGNATURE_MARKER,
    has_thought_signature,
    strip_thought_signature,
)


def test_marker_constant_value():
    """The marker is the literal substring LiteLLM emits for Vertex Gemini."""
    assert THOUGHT_SIGNATURE_MARKER == "__thought__"


class TestHasThoughtSignature:
    def test_returns_true_for_gemini_id(self):
        gemini_id = "call_f0be918123f4462bb482dd9df123__thought__AY89a18oWjPi"
        assert has_thought_signature(gemini_id) is True

    def test_returns_false_for_openai_id(self):
        assert has_thought_signature("call_abc123def456") is False

    def test_returns_false_for_anthropic_id(self):
        assert has_thought_signature("toolu_01ABCdef") is False

    def test_returns_false_for_empty(self):
        assert has_thought_signature("") is False

    def test_returns_false_for_none(self):
        assert has_thought_signature(None) is False


class TestStripThoughtSignature:
    def test_strips_gemini_signature(self):
        gemini_id = "call_f0be918123f4462bb482dd9df123__thought__AY89a18oWjPi"
        assert strip_thought_signature(gemini_id) == "call_f0be918123f4462bb482dd9df123"

    def test_returns_openai_id_unchanged(self):
        assert strip_thought_signature("call_abc123def456") == "call_abc123def456"

    def test_returns_anthropic_id_unchanged(self):
        assert strip_thought_signature("toolu_01ABCdef") == "toolu_01ABCdef"

    def test_empty_string_returns_empty(self):
        assert strip_thought_signature("") == ""

    def test_strips_huge_signature(self):
        # The pathological case observed in the wild: a 278 KB signature
        # blob appended to a 32-char id.
        big_blob = "A" * 278_000
        gemini_id = f"call_f0be918123f4462bb482dd9df123__thought__{big_blob}"
        result = strip_thought_signature(gemini_id)
        assert result == "call_f0be918123f4462bb482dd9df123"
        # The stripped id no longer carries a signature.
        assert has_thought_signature(result) is False

    def test_stripping_is_idempotent(self):
        gemini_id = "call_f0be918123f4462bb482dd9df123__thought__AY89a18oWjPi"
        stripped_once = strip_thought_signature(gemini_id)
        assert strip_thought_signature(stripped_once) == stripped_once

    def test_strips_only_first_marker_occurrence(self):
        # If a signature blob ever happens to contain the marker again, we
        # still want everything from the first occurrence onward removed.
        weird = "call_x__thought__blob__thought__more"
        assert strip_thought_signature(weird) == "call_x"
