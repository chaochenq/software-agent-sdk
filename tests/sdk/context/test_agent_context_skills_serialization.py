"""Tests for the auto-loaded-skills serialization optimisation.

When ``load_user_skills`` / ``load_public_skills`` are enabled,
``AgentContext._load_auto_skills`` resolves every matching skill and
appends it to ``self.skills`` at model-validation time. Persisting
those skills on the wire is pure duplication — the validator runs
again on every deserialization and rebuilds the same set deterministically.

The serializer drops auto-loaded skills from ``model_dump`` output but
keeps the in-memory ``self.skills`` intact so runtime prompt rendering
behaves exactly as today. These tests pin both halves of that contract.
"""

from __future__ import annotations

from unittest.mock import patch

from openhands.sdk.context import AgentContext
from openhands.sdk.skills import Skill


def _make_skill(name: str, content: str = "auto skill body") -> Skill:
    return Skill(
        name=name,
        content=content,
        source=f"/fake/{name}.md",
    )


class TestAutoLoadedSkillsSerialization:
    def test_auto_loaded_skills_drop_from_serialized_output(self):
        """``model_dump`` omits auto-loaded skills (the headline payload win)."""
        auto = {
            "auto-1": _make_skill("auto-1"),
            "auto-2": _make_skill("auto-2"),
            "auto-3": _make_skill("auto-3"),
        }
        with patch(
            "openhands.sdk.context.agent_context.load_available_skills",
            return_value=auto,
        ):
            ctx = AgentContext(load_public_skills=True)

        # In-memory: the validator populated all three.
        assert {s.name for s in ctx.skills} == {"auto-1", "auto-2", "auto-3"}
        # On the wire: dropped entirely (no caller passed explicit skills).
        dumped = ctx.model_dump()
        assert dumped["skills"] == []

    def test_explicit_skills_survive_serialization(self):
        """Caller-supplied skills are NOT dropped — only the auto-loaded ones."""
        explicit = _make_skill("user-explicit", "this one is mine")
        auto = {"auto-only": _make_skill("auto-only")}
        with patch(
            "openhands.sdk.context.agent_context.load_available_skills",
            return_value=auto,
        ):
            ctx = AgentContext(load_public_skills=True, skills=[explicit])

        # In-memory: both present.
        assert {s.name for s in ctx.skills} == {"user-explicit", "auto-only"}
        # On the wire: only the explicit one.
        dumped = ctx.model_dump()
        assert [s["name"] for s in dumped["skills"]] == ["user-explicit"]

    def test_explicit_skill_shadows_an_auto_one(self):
        """When the explicit list collides with an auto-loaded name, the
        explicit one wins in-memory AND on the wire (it was never marked
        as auto-loaded — the auto-load step skipped it).
        """
        explicit = _make_skill("shared-name", "user version")
        auto = {"shared-name": _make_skill("shared-name", "auto version")}
        with patch(
            "openhands.sdk.context.agent_context.load_available_skills",
            return_value=auto,
        ):
            ctx = AgentContext(load_public_skills=True, skills=[explicit])

        assert len(ctx.skills) == 1
        assert ctx.skills[0].content == "user version"
        dumped = ctx.model_dump()
        assert len(dumped["skills"]) == 1
        assert dumped["skills"][0]["name"] == "shared-name"
        assert dumped["skills"][0]["content"] == "user version"

    def test_round_trip_via_serialized_output_re_resolves_auto_skills(self):
        """Deserializing the trimmed payload + re-validating repopulates the
        auto-loaded skills via the same auto-load path. This is what makes
        dropping them on the wire safe: the receiver sees the same
        in-memory shape as the sender.
        """
        auto = {
            "auto-a": _make_skill("auto-a"),
            "auto-b": _make_skill("auto-b"),
        }
        with patch(
            "openhands.sdk.context.agent_context.load_available_skills",
            return_value=auto,
        ):
            ctx = AgentContext(load_public_skills=True)
            dumped = ctx.model_dump()
            # Re-validating the trimmed payload should re-fire the auto-load
            # validator and rebuild the in-memory list.
            roundtripped = AgentContext.model_validate(dumped)

        assert {s.name for s in roundtripped.skills} == {"auto-a", "auto-b"}

    def test_disabled_auto_load_leaves_explicit_skills_on_the_wire(self):
        """With both flags off, ``_load_auto_skills`` is a no-op — every
        skill is treated as explicit and serialized normally.
        """
        explicit = [_make_skill("foo"), _make_skill("bar")]
        ctx = AgentContext(skills=explicit)
        assert {s.name for s in ctx.skills} == {"foo", "bar"}
        dumped = ctx.model_dump()
        assert {s["name"] for s in dumped["skills"]} == {"foo", "bar"}

    def test_migration_stored_skills_matching_loader_are_marked_auto_loaded(self):
        """Migration path: a conversation stored before this PR carries the
        resolved auto-loaded skills inlined on ``skills``. On reload, the
        validator must recognise those as auto-loaded (the persisted
        skill equals what the loader produces) and drop them from the
        next serialization — otherwise the bloat persists until the
        conversation is recreated.
        """
        stored = [_make_skill("auto-x"), _make_skill("auto-y")]
        # Loader returns the same skill objects (matches what was
        # persisted under the old SDK).
        loader_output = {
            "auto-x": _make_skill("auto-x"),
            "auto-y": _make_skill("auto-y"),
        }
        with patch(
            "openhands.sdk.context.agent_context.load_available_skills",
            return_value=loader_output,
        ):
            ctx = AgentContext(load_public_skills=True, skills=stored)

        # No duplicates: skills stayed at 2 (migration recognised them).
        assert {s.name for s in ctx.skills} == {"auto-x", "auto-y"}
        # And the serialized payload drops them.
        dumped = ctx.model_dump()
        assert dumped["skills"] == []

    def test_migration_stored_skills_diverged_from_loader_stay_explicit(self):
        """If the persisted skill content no longer matches what the
        loader would produce (user edited the file, marketplace updated),
        the on-disk version wins — treat it as explicit and keep it on
        the wire. This is the safe default: we don't drop content
        someone may have intentionally customised.
        """
        stored = [_make_skill("auto-x", "old persisted content")]
        loader_output = {"auto-x": _make_skill("auto-x", "new loader content")}
        with patch(
            "openhands.sdk.context.agent_context.load_available_skills",
            return_value=loader_output,
        ):
            ctx = AgentContext(load_public_skills=True, skills=stored)

        # The stored ("old") version wins in-memory.
        assert len(ctx.skills) == 1
        assert ctx.skills[0].content == "old persisted content"
        # And it stays on the wire — not treated as auto-loaded.
        dumped = ctx.model_dump()
        assert len(dumped["skills"]) == 1
        assert dumped["skills"][0]["content"] == "old persisted content"

    def test_payload_shrinks_to_explicit_only(self):
        """Concrete byte-count assertion: a 40-skill auto-load with a single
        explicit skill should serialize approximately the size of the
        single explicit skill, not 40+1.
        """
        import json

        auto = {f"auto-{i}": _make_skill(f"auto-{i}", "x" * 1000) for i in range(40)}
        explicit = _make_skill("user", "y" * 100)
        with patch(
            "openhands.sdk.context.agent_context.load_available_skills",
            return_value=auto,
        ):
            ctx = AgentContext(load_public_skills=True, skills=[explicit])

        dumped = ctx.model_dump()
        skills_bytes = len(json.dumps(dumped["skills"]))
        # The 40 auto skills total ~40 KB; the single explicit one is
        # ~150 B. The serialized ``skills`` list must clearly be the
        # explicit-only size, not the full set.
        assert skills_bytes < 1000, (
            f"serialized skills should be ~1 explicit skill (~150 B), "
            f"got {skills_bytes} B — auto skills leaked into output"
        )
