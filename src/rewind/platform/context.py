"""Who is making the tool call that is executing right now.

A tool function only receives (env, args). Executors that must know more --
the sandbox (which tenant's workspace and container pool), the MCP gateway
(an idempotency key for the server) -- read it from here. The agent loop
sets it around every tool execution; `asyncio.to_thread` copies the context
into the worker thread, and concurrently running read-only calls each run in
their own task, so they never see each other's value.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

from .identity import Principal


@dataclass(frozen=True)
class CallContext:
    principal: Principal | None
    run_id: str
    action_id: str

    @property
    def idempotency_key(self) -> str:
        """Stable across crash/resume: a resumed run replays the same action id."""
        return f"{self.run_id}:{self.action_id}"


CALL_CONTEXT: ContextVar[CallContext | None] = ContextVar("rewind_call_context", default=None)


def current_call() -> CallContext | None:
    return CALL_CONTEXT.get()
