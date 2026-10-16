"""Deterministic replay.

Re-runs the harness against a fresh env restored from the initial snapshot, with
a provider that returns the *recorded* LLM attempts (responses and failures) and
asserts each request hash matches. Tools are re-executed, so a passing replay
proves the harness + tools are deterministic given the recorded model outputs.
Then the two traces are diffed on stable fields; the first mismatch is reported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .events import Event
from .loop import AgentConfig, AgentRunner
from .providers import LLMResponse, MalformedOutput, ProviderError
from .snapshot import SnapshotStore
from .tools import ToolRegistry
from .trace import TraceWriter
from .util import canonical_json, request_hash

VOLATILE_KEYS = {"latency_ms"}


class ReplayMismatch(Exception):
    pass


class ReplayProvider:
    def __init__(self, events: list[Event]) -> None:
        self.attempts = [e for e in events if e.type == "llm_call"]
        self.i = 0

    async def complete(self, messages, tools, params) -> LLMResponse:
        if self.i >= len(self.attempts):
            raise ReplayMismatch("replay requested more LLM calls than were recorded")
        rec = self.attempts[self.i]
        self.i += 1
        if request_hash(messages, tools, params) != rec.data["request_hash"]:
            raise ReplayMismatch(
                f"request hash mismatch at step {rec.step} (recorded LLM attempt #{self.i}): "
                "the harness now builds a different prompt than when the run was recorded")
        err = rec.data.get("error")
        if err:
            if err["kind"] == "malformed":
                raise MalformedOutput(err["message"])
            raise ProviderError(err["message"], err["retryable"], err.get("status"))
        return LLMResponse.model_validate(rec.data["response"])


async def _no_sleep(_: float) -> None:
    return None


def stable(e: Event) -> tuple:
    data = {k: v for k, v in e.data.items() if k not in VOLATILE_KEYS}
    if e.type == "llm_call":
        # A fork-patched response is served by the replay provider as "primary";
        # provenance of the response is not harness behaviour.
        data.pop("patched", None)
        if data.get("provider") == "patched":
            data["provider"] = "primary"
    return (e.step, e.type, canonical_json(data))


def first_divergence(a: list[Event], b: list[Event]) -> dict[str, Any] | None:
    xa = [e for e in a if e.type != "run_start"]
    xb = [e for e in b if e.type != "run_start"]
    for i in range(max(len(xa), len(xb))):
        ea = stable(xa[i]) if i < len(xa) else None
        eb = stable(xb[i]) if i < len(xb) else None
        if ea != eb:
            return {"index": i, "expected": ea, "actual": eb}
    return None


@dataclass
class ReplayReport:
    ok: bool
    reason: str | None
    divergence: dict[str, Any] | None
    n_events: int


async def replay_run(events: list[Event], snapshots: SnapshotStore, registry: ToolRegistry,
                     env_factory: Callable[[dict], Any],
                     cfg_override: dict[str, Any] | None = None) -> ReplayReport:
    start = events[0]
    if start.type != "run_start":
        raise ValueError("trace must begin with run_start")
    d = start.data
    cfg = AgentConfig(**{**d["cfg"], **(cfg_override or {})})
    rp = ReplayProvider(events)
    runner = AgentRunner(rp, registry, cfg, snapshots,
                         fallback=rp if d.get("has_fallback") else None, sleep=_no_sleep)
    env = env_factory(snapshots.get(d["initial_env_hash"]))
    trace = TraceWriter(run_id=start.run_id + "~replay")
    try:
        await runner.run(env, d["task"], trace)
    except ReplayMismatch as e:
        return ReplayReport(False, str(e), None, len(events))
    div = first_divergence(events, trace.events)
    return ReplayReport(div is None, None if div is None else "trace diverged", div, len(events))
