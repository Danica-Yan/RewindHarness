"""Run state shared by the kernel (loop.py) and the durability layer.

`LoopState` and `RunAbort` used to live in loop.py, which forced
checkpoint.py -- a persistence module -- to import the agent runner. They
are plain data + an exception with no dependency on the runner, so they
live here, below both: loop.py, checkpoint.py, fork.py and worker.py all
import them from this module, and nothing in the durability layer depends
on the kernel any more.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .events import Event

ARG_ERRORS = ("TOOL_ARG_INVALID", "HALLUCINATED_TOOL")


class RunAbort(Exception):
    """Ends a run with a terminal status. Raised by the loop (budgets, loop
    kill, invalid args) and by a checkpointer whose write was fenced out."""

    def __init__(self, status: str, reason: str | None = None) -> None:
        super().__init__(f"{status}: {reason}")
        self.status = status
        self.reason = reason


@dataclass
class StepStart:
    """Where a step began: what a checkpoint taken *during* the step (intent,
    tool) must restore so that resuming re-enters the step exactly once --
    the trace position before its first event, the conversation before the
    model's reply, the counters before its tokens and tool calls were
    counted, and the env as the step found it (the step's `step_start`
    hash, which forks start from)."""
    seq: int
    messages: list[dict]
    state: "LoopState"
    env_hash: str


@dataclass
class LoopState:
    """Loop counters that must survive a fork/resume; rebuildable from events."""
    tokens: int = 0
    call_counts: dict[str, int] = field(default_factory=dict)
    consecutive_invalid: int = 0

    def bump_call(self, name: str, args_hash: str) -> int:
        key = f"{name}|{args_hash}"
        self.call_counts[key] = self.call_counts.get(key, 0) + 1
        return self.call_counts[key]

    def note_result(self, error_class: str | None) -> None:
        self.consecutive_invalid = self.consecutive_invalid + 1 if error_class in ARG_ERRORS else 0

    @classmethod
    def from_events(cls, events: list[Event]) -> LoopState:
        st = cls()
        for e in events:
            if e.type == "llm_call" and "response" in e.data:
                u = e.data["response"].get("usage") or {}
                st.tokens += u.get("input_tokens", 0) + u.get("output_tokens", 0)
            elif e.type == "tool_call":
                st.bump_call(e.data["name"], e.data["args_hash"])
                st.note_result(e.data["error_class"])
        return st
