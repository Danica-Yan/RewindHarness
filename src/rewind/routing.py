"""Per-turn tool routing (context pruning) for the agent loop.

A router looks at each new user turn (plus the agent's message just before it) and decides which tools,
and which version of the system prompt, the agent needs from now on. The loop then sends only those
tool schemas and that prompt to the LLM. Routing is cumulative within a run: a later turn can add
tools, never take them away. A run falls back to the full toolset when the router isn't confident,
when the agent calls the escalation tool, or when the agent calls a registered tool that isn't in its
current route.

rewind itself knows nothing about intents or tau-bench: a concrete router (e.g.
experiments/intent_funnel) implements `ToolRouter.update`.

Determinism: every routing decision is recorded as a `route` trace event. The routed system prompt
goes into the SnapshotStore, and the event keeps a reference to it. Replay serves those recorded
decisions back through `RecordedRouter`, so a replay needs neither the router's model nor its data.
`LoopState` carries the current route, so checkpoints, resume and fork restore it with no special case.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from .events import Event
from .tools import Tool

ESCALATE_TOOL = "request_full_toolset"


@dataclass
class RouteUpdate:
    """The route after a user turn. tools=None means the full toolset (fallback); prompt=None means the
    unmodified system prompt."""
    units: list[str]                 # router-specific route units (e.g. intents), cumulative
    full: bool
    tools: list[str] | None
    prompt: str | None
    detail: dict[str, Any] = field(default_factory=dict)  # router diagnostics for the trace


class ToolRouter(Protocol):
    def update(self, units: list[str], full: bool, user_text: str, agent_text: str,
               full_prompt: str) -> RouteUpdate:
        """Route after one new user turn, given the current cumulative units and fallback flag.
        Must be deterministic in its inputs."""
        ...


class RecordedRouter:
    """Serves a recorded run's routing decisions in order (for replay). Raises if the replay asks for
    more decisions than were recorded, the same contract as ReplayProvider."""

    def __init__(self, events: list[Event], snapshots: Any) -> None:
        self.decisions = [e for e in events if e.type == "route" and e.data.get("reason") == "user_turn"]
        self.snapshots = snapshots
        self.i = 0

    def update(self, units, full, user_text, agent_text, full_prompt) -> RouteUpdate:
        if self.i >= len(self.decisions):
            raise RuntimeError("replay requested more routing decisions than were recorded")
        d = self.decisions[self.i].data
        self.i += 1
        prompt = self.snapshots.get(d["prompt_ref"])["prompt"] if d.get("prompt_ref") else None
        return RouteUpdate(units=list(d["units"]), full=d["full"], tools=d["tools"], prompt=prompt,
                           detail=d.get("detail", {}))


class _NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def escalate_tool() -> Tool:
    """Safety valve offered in every routed prompt. Calling it switches the run to the full toolset."""
    return Tool(
        name=ESCALATE_TOOL,
        description=("Call this if helping the user needs a tool that is not in your current tool list "
                     "(for example a different kind of order change). All tools become available."),
        args_model=_NoArgs, fn=lambda env, args: "All tools are now available.", mutating=False)
