"""Run, fork and replay one benchmark episode -- for any `Benchmark`.

This is the orchestration layer between the runtime (loop / fork / replay
/ trace) and everything that evaluates or improves an agent (scripts,
`evolve/`). It used to live inside the tau-bench plugin, which made the
plugin depend on fork/replay/loop; now the plugin only says how to build,
restore and grade its env (`Benchmark`), and the episode mechanics here are
written once for every benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .benchmark import Benchmark, Grade
from .environment import Environment
from .events import Event
from .fork import fork_run
from .loop import AgentConfig, AgentRunner, RunResult
from .replay import ReplayMismatch, ReplayProvider, replay_run
from .snapshot import SnapshotStore
from .tools import ToolRegistry
from .trace import TraceWriter


@dataclass
class Episode:
    result: RunResult
    env: Environment
    events: list[Event]
    reward: Grade


async def run_episode(bench: Benchmark, split: str, index: int, *, provider: Any, user_llm: Any,
                      store: SnapshotStore, cfg: AgentConfig | None = None,
                      registry: ToolRegistry | None = None, run_id: str | None = None,
                      trace_path: str | None = None) -> Episode:
    cfg = cfg or bench.agent_config()
    registry = registry or bench.build_registry()
    env, first = await bench.start_episode(split, index, store=store, user=user_llm)
    runner = AgentRunner(provider, registry, cfg, store)
    with TraceWriter(run_id or f"{bench.name}-{split}-{index}", path=trace_path) as trace:
        rr = await runner.run(env, first, trace)
        events = list(trace.events)
    return Episode(rr, env, events, bench.grade(env))


async def fork_episode(bench: Benchmark, base_events: list[Event], at_step: int, *, provider: Any,
                       user_llm: Any, store: SnapshotStore, cfg: AgentConfig | None = None,
                       registry: ToolRegistry | None = None, patch: Any = None,
                       run_id: str | None = None, trace_path: str | None = None,
                       system_prompt: str | None = None) -> Episode:
    """Restore env + counterparty at `at_step` of a recorded episode and
    continue live. With `patch=None` this is the *null fork* used to
    calibrate how much a continuation varies on its own; with a `ForkPatch`
    it is a counterfactual. `cfg` defaults to the parent's recorded config,
    so a context edit is expressed by passing a different cfg (system
    prompt)."""
    if cfg is None:
        cfg = AgentConfig(**base_events[0].data["cfg"])
    registry = registry or bench.build_registry()
    holder: dict[str, Environment] = {}

    def env_factory(state: dict) -> Environment:
        holder["env"] = bench.restore_env(state, store=store, user=user_llm)
        return holder["env"]

    runner = AgentRunner(provider, registry, cfg, store)
    rid = run_id or f"{base_events[0].run_id}~fork{at_step}"
    with TraceWriter(rid, path=trace_path) as trace:
        rr = await fork_run(base_events, store, at_step=at_step, runner=runner,
                            env_factory=env_factory, trace=trace, patch=patch,
                            system_prompt=system_prompt)
        events = list(trace.events)
    env = holder["env"]
    return Episode(rr, env, events, bench.grade(env))


async def replay_episode(bench: Benchmark, events: list[Event], store: SnapshotStore,
                         registry: ToolRegistry | None = None):
    """Deterministic replay of a recorded episode: recorded model responses
    AND recorded counterparty replies, no API calls."""
    registry = registry or bench.build_registry()
    user = bench.recorded_user(events)
    return await replay_run(events, store, registry,
                            lambda s: bench.restore_env(s, store=store, user=user))


class FidelityResult(BaseModel):
    step: int
    ok: bool
    error: str | None = None


async def fork_fidelity(bench: Benchmark, events: list[Event], store: SnapshotStore,
                        steps: list[int] | None = None,
                        registry: ToolRegistry | None = None) -> list[FidelityResult]:
    """API-free check that forking a recorded episode at step k restores
    EXACTLY what the agent saw: fork at k, then serve the parent's own
    recorded model responses and counterparty replies from k on, with every
    request hash-checked against the recording (ReplayProvider). Any
    difference in restored DB content or order, dialogue state or message
    history shows up as a request-hash mismatch; the final env must also
    hash the same."""
    registry = registry or bench.build_registry()
    all_steps = sorted({e.step for e in events if e.type == "step_start"})
    parent_end = next(e for e in reversed(events) if e.type == "run_end").data["env_hash"]
    out: list[FidelityResult] = []
    for k in (steps or all_steps):
        rest = [e for e in events if e.step >= k]
        try:
            ep = await fork_episode(bench, events, k, provider=ReplayProvider(rest),
                                    user_llm=bench.recorded_user(rest), store=store,
                                    registry=registry, run_id=f"{events[0].run_id}~fid{k}")
        except (ReplayMismatch, RuntimeError) as e:
            out.append(FidelityResult(step=k, ok=False, error=str(e)))
            continue
        end = next(e for e in reversed(ep.events) if e.type == "run_end").data["env_hash"]
        out.append(FidelityResult(step=k, ok=end == parent_end,
                                  error=None if end == parent_end else "final env differs"))
    return out
