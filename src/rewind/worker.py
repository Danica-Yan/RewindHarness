"""Single job-attempt execution: claim -> cold/warm start -> run -> complete.

Call `run_worker_step` once per available worker slot to drain the queue. A
worker that crashes mid-run simply never returns from this call (see
`SimulatedCrash` below, used only in tests to model that) — its job stays
LEASED until the lease expires, at which point another call to
`run_worker_step` claims it (bumping the fencing token) and resumes from the
latest checkpoint.

While the job runs, a background task renews the lease every
`lease_seconds / 3` (`heartbeat_every`), so a worker that is alive but slow
(a real LLM call easily outlasts a 30 s lease) keeps its job; only a worker
that stopped running -- crashed, killed, or partitioned away long enough --
loses it. A renewal the coordinator rejects means another worker already
holds a newer fencing token; renewal then stops, and this worker's next
checkpoint write aborts the run (`fenced_out`).
"""
from __future__ import annotations

import asyncio
import contextlib
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


async def _keep_lease(coordinator: CoordinatorStore, job_id: str, fencing_token: int,
                      lease_seconds: float, every: float) -> None:
    """Renew the lease until cancelled or until the coordinator says we no
    longer own the job. Uses real time on purpose (not the runner's injected
    `sleep`, which tests replace with a no-op)."""
    while True:
        await asyncio.sleep(every)
        try:
            ok = await coordinator.heartbeat(job_id, fencing_token, lease_seconds)
        except Exception:  # transient coordinator error: the lease may still be valid, retry next tick
            continue
        if not ok:
            return


async def run_worker_step(coordinator: CoordinatorStore, snapshots: SnapshotStore, blobs: SnapshotStore,
                          registry: ToolRegistry, provider, env_factory: Callable[[dict], Any],
                          trace_dir: str | Path, owner: str | None = None,
                          lease_seconds: float = 30, fallback=None, sleep=None, clock=None,
                          debug_hook: Callable[[str, int], None] | None = None,
                          heartbeat_every: float | None = None
                          ) -> tuple[RunResult, Any] | None:
    """Claim and fully execute (or resume) one job. Returns (result, env) or
    None if no job is available to claim right now. `heartbeat_every`:
    seconds between lease renewals (default `lease_seconds / 3`; 0 = never)."""
    owner = owner or ("worker-" + uuid.uuid4().hex[:8])
    job = await coordinator.claim(owner, lease_seconds=lease_seconds)
    if job is None:
        return None
    every = lease_seconds / 3 if heartbeat_every is None else heartbeat_every
    keeper = (asyncio.create_task(_keep_lease(coordinator, job.job_id, job.fencing_token, lease_seconds, every))
              if every > 0 else None)
    try:
        return await _execute(coordinator, job, snapshots, blobs, registry, provider, env_factory,
                              trace_dir, fallback, sleep, clock, debug_hook, keeper)
    finally:
        await _stop(keeper)   # also on a (simulated) crash: a dead worker must not keep renewing


async def _stop(task: asyncio.Task | None) -> None:
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def _execute(coordinator, job, snapshots, blobs, registry, provider, env_factory, trace_dir,
                   fallback, sleep, clock, debug_hook, keeper) -> tuple[RunResult, Any]:

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
        step_env_hash = None
        if ckpt.kind in ("intent", "tool"):
            # Step not yet fully confirmed: re-enter it once with its recorded
            # action. The checkpoint holds the step's start (trace position,
            # messages, counters, starting env hash), so nothing is duplicated.
            # For "tool", env already reflects any calls that durably completed,
            # so their action_id is in env.applied and they won't re-run for real
            # (see loop.py's comment above the "intent" checkpoint call).
            start_step = ckpt.step
            forced = LLMResponse.model_validate(bundle["action"]) if bundle["action"] else None
            step_env_hash = bundle.get("step_env_hash")
        else:
            start_step = ckpt.step + 1                     # step confirmed: start the next one fresh
            forced = None
        result = await runner.resume(env, messages, state, start_step, trace, forced,
                                     step_env_hash=step_env_hash)
    trace.close()
    await _stop(keeper)

    if result.status != "fenced_out":
        await coordinator.complete(job.job_id, job.fencing_token, result.status,
                                   {"steps": result.steps, "env_hash": result.env_hash})
    return result, env
