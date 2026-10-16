"""Trace event schema.

Every event carries (run_id, seq); seq is strictly increasing within a run.
Large payloads are inlined in `data` in this skeleton; swap them for
content-addressed blob refs once traces get big (see plan section 4.1).
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EventType = Literal[
    "run_start",   # task, cfg, initial_env_hash (+ fork lineage)
    "message",     # every message appended to the conversation
    "step_start",  # env_hash before the step's LLM call (fork points)
    "llm_call",    # one provider *attempt*: response or error
    "tool_call",   # one tool execution (or guard-blocked call)
    "env_snapshot",
    "guard",       # loop warning/kill, repair prompt, fallback switch
    "error",       # transient tool errors etc.
    "run_end",
]


class Event(BaseModel):
    run_id: str
    seq: int
    step: int
    type: EventType
    ts: str
    inherited: bool = False  # True for prefix events copied into a fork
    data: dict[str, Any] = Field(default_factory=dict)
