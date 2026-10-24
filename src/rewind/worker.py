"""Single job-attempt execution: claim -> cold/warm start -> run -> complete.

Call `run_worker_step` once per available worker slot to drain the queue. A
worker that crashes mid-run simply never returns from this call (see
`SimulatedCrash` below, used only in tests to model that) — its job stays
LEASED until the lease expires, at which point another call to
`run_worker_step` claims it (bumping the fencing token) and resumes from the
latest checkpoint.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .checkpoint import StepCheckpointer
from .coordination import CoordinatorStore, JobSpec
from .loop import AgentConfig, AgentRunner, RunResult
from .providers import LLMResponse
from .snapshot import SnapshotStore
from .state import LoopState
from .tools import ToolRegistry
from .trace import TraceWriter


class SimulatedCrash(Exception):
    """Raised only by test/demo debug hooks to model a process dying at an
    exact point. Deliberately not a RunAbort subclass so the loop's own
    except-RunAbort handling does not swallow it: it must propagate all the
    way out of run_worker_step uncaught, exactly like a real process death."""


async def submit_job(coordinator: CoordinatorStore, snapshots: SnapshotStore, run_id: str, task: str,
                     cfg: AgentConfig, initial_env: dict[str, Any], priority: int = 0) -> str:
    ref = snapshots.put(initial_env)
    spec = JobSpec(run_id=run_id, task=task, cfg=asdict(cfg), initial_env_ref=ref)
    return await coordinator.submit(spec, priority=priority)


# Historical name from when the sqlite backend was synchronous and needed a
# separate sync wrapper; every backend is async now.
submit_job_async = submit_job


async def run_worker_step(coordinator: CoordinatorStore, snapshots: SnapshotStore, blobs: SnapshotStore,
                          registry: ToolRegistry, provider, env_factory: Callable[[dict], Any],
                          trace_dir: str | Path, owner: str | None = None,
                          lease_seconds: float = 30, fallback=None, sleep=None, clock=None,
                          debug_hook: Callable[[str, int], None] | None = None
                          ) -> tuple[RunResult, Any] | None:
    """Claim and fully execute (or resume) one job. Returns (result, env) or
    None if no job is available to claim right now."""
    owner = owner or ("worker-" + uuid.uuid4().hex[:8])
    job = await coordinator.claim(owner, lease_seconds=lease_seconds)
    if job is None:
        return None

    cfg = AgentConfig(**job.spec.cfg)
    trace_path = Path(trace_dir) / f"{job.run_id}.jsonl"
    checkpointer = StepCheckpointer(coordinator, job.job_id, job.run_id, job.fencing_token,
                                    snapshots, blobs)
    runner = AgentRunner(provider, registry, cfg, snapshots, clock=clock, fallback=fallback,
                         sleep=sleep, checkpointer=checkpointer, debug_hook=debug_hook)

    ckpt = await coordinator.latest_checkpoint(job.run_id)
    if ckpt is None:
        env = env_factory(snapshots.get(job.spec.initial_env_ref))
        trace = TraceWriter(job.run_id, trace_path, clock=clock)
        result = await runner.run(env, job.spec.task, trace)
    else:
        env = env_factory(snapshots.get(ckpt.env_hash))
        bundle = blobs.get(ckpt.bundle_ref)
        messages = bundle["messages"]
        state = LoopState(**bundle["loop_state"])
        trace = TraceWriter.resume_from(trace_path, job.run_id, ckpt.seq, clock=clock)
        if ckpt.kind in ("intent", "tool"):
            # Step not yet fully confirmed: replay its recorded action from idx 0.
            # For "tool", env already reflects any calls that durably completed,
            # so their action_id is in env.applied and they won't re-run for real
            # (see loop.py's comment above the "tool" checkpoint call).
            start_step = ckpt.step
            forced = LLMResponse.model_validate(bundle["action"]) if bundle["action"] else None
        else:
            start_step = ckpt.step + 1                     # step confirmed: start the next one fresh
            forced = None
        result = await runner.resume(env, messages, state, start_step, trace, forced)
    trace.close()

    if result.status != "fenced_out":
        await coordinator.complete(job.job_id, job.fencing_token, result.status,
                                   {"steps": result.steps, "env_hash": result.env_hash})
    return result, env
