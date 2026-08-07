"""Per-conversation resource budgets for the agent loop (MT-002).

An agent run is driven by whatever the LLM emits, and the LLM's output is shaped
by conversation content that is frequently untrusted — a repository file, a
fetched page, a tool observation. Three of those knobs are unbounded by default
and each turns into an availability problem, or a cost one, under a prompt that
is trying:

* **Iterations.** Nothing stops the loop from stepping forever. A prompt that
  induces a "retry, that failed" cycle burns tokens indefinitely, on the
  operator's account.
* **Tool calls per response.** A single response may carry an arbitrary number
  of tool calls, all dispatched. That is a fan-out amplifier: one LLM turn into
  hundreds of subprocesses.
* **Observation size.** A tool that reads a large file puts every byte into the
  next request. Beyond the direct cost this is the cheapest way to blow the
  context window and evict the system prompt.

The budgets live here rather than on the run loop because they are properties of
a *conversation*, and the same `Agent` instance may serve several. Keying by
conversation id keeps one conversation's spend from exhausting another's.
"""

from __future__ import annotations

import threading
from collections import OrderedDict


# Chosen to sit well above any legitimate task while still bounding a runaway
# loop to a recoverable cost. Operators who need more can raise the field; the
# point is that the default is finite.
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_MAX_TOOL_CALLS_PER_STEP = 10

# The counter map is itself a resource. A long-lived process serving many
# conversations would grow it without bound — an unbounded budget of exactly the
# kind this module exists to prevent. Callers release a finished conversation,
# but relying on every finish path remembering to do so is how leaks happen, so
# the map also evicts its oldest entry past this many tracked conversations.
MAX_TRACKED_CONVERSATIONS = 10_000

# Bytes, applied to a single observation's rendered text.
DEFAULT_MAX_OBSERVATION_BYTES = 10 * 1024

_TRUNCATION_NOTICE = (
    "\n\n[Observation truncated: exceeded the {limit}-byte per-observation "
    "budget. Narrow the command or read the file in ranges.]"
)


class ResourceExhaustionError(RuntimeError):
    """A conversation exceeded one of its declared resource budgets."""


class IterationBudget:
    """Counts agent steps per conversation and refuses to exceed the budget.

    Thread-safe: the async and sync step paths can both charge concurrently, and
    a lost increment here is a lost enforcement.
    """

    def __init__(
        self,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_tracked_conversations: int = MAX_TRACKED_CONVERSATIONS,
    ) -> None:
        self._max = max_iterations
        self._max_tracked = max_tracked_conversations
        # Insertion-ordered, so evicting the oldest entry is a `popitem` on the
        # front rather than a scan.
        self._counts: OrderedDict[str, int] = OrderedDict()
        self._lock = threading.Lock()

    def charge(self, conversation_id: str) -> int:
        """Record one step. Raises once the budget is spent.

        Returns the new count so callers can log progress toward the limit.
        """
        with self._lock:
            count = self._counts.get(conversation_id, 0) + 1
            if count > self._max:
                raise ResourceExhaustionError(
                    f"Conversation exceeded its iteration budget of {self._max} "
                    f"steps. Increase `max_iterations` if the task legitimately "
                    f"needs more."
                )
            self._counts[conversation_id] = count
            self._counts.move_to_end(conversation_id)
            while len(self._counts) > self._max_tracked:
                # Evicting an active conversation resets its budget rather than
                # denying it, which is the safer direction to fail: the limit is
                # a runaway guard, not a quota to be enforced at the cost of
                # killing legitimate work.
                self._counts.popitem(last=False)
            return count

    def spent(self, conversation_id: str) -> int:
        with self._lock:
            return self._counts.get(conversation_id, 0)

    def release(self, conversation_id: str) -> None:
        """Drop a finished conversation's counter.

        Without this the map grows for the lifetime of a long-lived process —
        an unbounded budget of a different kind.
        """
        with self._lock:
            self._counts.pop(conversation_id, None)


def enforce_tool_call_limit(
    tool_call_count: int,
    limit: int = DEFAULT_MAX_TOOL_CALLS_PER_STEP,
) -> None:
    """Reject a response that fans out beyond ``limit`` tool calls.

    Rejecting rather than truncating is deliberate: silently dropping calls
    leaves the model believing work happened that did not, and it will act on
    that belief. A refusal it can see is recoverable.
    """
    if tool_call_count > limit:
        raise ResourceExhaustionError(
            f"LLM response requested {tool_call_count} tool calls, above the "
            f"per-step limit of {limit}. Refusing to dispatch."
        )


def bound_observation_text(
    text: str,
    limit: int = DEFAULT_MAX_OBSERVATION_BYTES,
) -> tuple[str, bool]:
    """Clamp one observation to ``limit`` bytes.

    Returns ``(text, was_truncated)``. Measured in bytes rather than characters
    because the cost being bounded is the serialised payload, and a multi-byte
    character costs more than one. Truncation is on a character boundary, so the
    result stays valid text rather than a split code point.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text, False

    notice = _TRUNCATION_NOTICE.format(limit=limit)
    budget = max(limit - len(notice.encode("utf-8")), 0)
    clipped = encoded[:budget].decode("utf-8", errors="ignore")
    return clipped + notice, True


def budget_snapshot(budget: IterationBudget, conversation_id: str) -> dict[str, int]:
    """How much of a conversation's iteration budget is left.

    Exposed so an operator can see a run approaching its ceiling before it is
    refused. A budget that only announces itself by raising gives no warning.
    """
    spent = budget.spent(conversation_id)
    return {"spent": spent, "limit": budget._max, "remaining": max(budget._max - spent, 0)}
