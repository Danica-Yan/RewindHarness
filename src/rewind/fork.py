"""Counterfactual fork: restore env + history at step k, optionally replace the
model's action at step k, then continue live.

The prefix (steps < k) is copied into the child trace flagged `inherited`; a
production version would share it by reference instead of copying."""
from __future__ import annotations

import copy
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from .events import Event
from .loop import AgentRunner, LoopState, RunResult
from .providers import LLMResponse, ToolCall
from .snapshot import SnapshotStore
from .trace import TraceWriter
from dataclasses import asdict


class ForkPatch(BaseModel):
    kind: Literal["replace_action"] = "replace_action"
    text: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)  # {"name":..., "args": {...}}

    def to_response(self) -> LLMResponse:
        return LLMResponse(
            text=self.text,
            tool_calls=[ToolCall(id="", name=c["name"], args=c.get("args", {}))
                        for c in self.tool_calls])


async def fork_run(base_events: list[Event], snapshots: SnapshotStore, *, at_step: int,
                   runner: AgentRunner, env_factory: Callable[[dict], Any],
                   trace: TraceWriter, patch: ForkPatch | None = None,
                   system_prompt: str | None = None) -> RunResult:
    """`system_prompt`, if given, replaces the system message of the rebuilt
    history -- the counterfactual "what if this context had been in effect
    from the start" used to screen context edits (tool-description edits
    take effect through `runner.registry`, which is read on every call)."""
    start = base_events[0]
    if start.type != "run_start":
        raise ValueError("trace must begin with run_start")
    step_start = next((e for e in base_events if e.type == "step_start" and e.step == at_step), None)
    if step_start is None:
        raise ValueError(f"no step_start for step {at_step} in base trace")

    env = env_factory(snapshots.get(step_start.data["env_hash"]))
    prefix = [e for e in base_events
              if e.type not in ("run_start", "run_end") and e.step < at_step]
    state = LoopState.from_events(prefix)
    messages = [copy.deepcopy(e.data["message"]) for e in prefix if e.type == "message"]
    if system_prompt is not None:
        if not messages or messages[0].get("role") != "system":
            raise ValueError("cannot override the system prompt: history has no system message")
        messages[0] = {**messages[0], "content": system_prompt}

    child_start = {**start.data, "cfg": asdict(runner.cfg),
                   "parent_run_id": start.run_id, "fork_step": at_step,
                   "patch": patch.model_dump() if patch else None,
                   "system_prompt_override": system_prompt is not None}
    trace.emit("run_start", 0, child_start)
    for e in prefix:
        trace.emit(e.type, e.step, e.data, inherited=True)

    forced = patch.to_response() if patch else None
    return await runner.resume(env, messages, state, at_step, trace, forced)
