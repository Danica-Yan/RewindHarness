#!/usr/bin/env python3
"""W6 scalability experiment: fixed job count, worker concurrency 1/2/4/8,
measure throughput. Runs against a REAL Postgres (`PgCoordinator`), so every
claim/checkpoint/complete is a genuine network round trip — concurrency
matters here because the bottleneck is I/O wait on those round trips, not
CPU, which is exactly why this is a meaningful curve even on a 2-vCPU
machine (see the honesty note below).

Uses an in-process mock LLM (zero real latency, zero cost) so the curve
isolates the coordinator/checkpoint machinery's own scalability rather than
an external API's rate limits — that's a separate, already-covered concern
(see dist/gateway.py's rate limiter and circuit breaker).

Usage:
  python scripts/chaos_scale_experiment.py --jobs 40 --workers 1,2,4,8

Honesty note: the plan asks for 200 runs; the dev machine has 2 vCPUs, so 200
runs at high concurrency would mostly measure asyncio scheduling overhead
rather than anything about the protocol, and would take a while. Default
here is a smaller N (override with --jobs) — the shape of the curve
(does throughput improve with concurrency, and where does it flatten out)
is the thing worth reporting, not a specific absolute number that only
means something on real multi-core hardware.
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rewind.dist.pg_coordinator import PgCoordinator  # noqa: E402
from rewind.env import ToyRetailEnv, build_registry  # noqa: E402
from rewind.loop import AgentConfig  # noqa: E402
from rewind.providers import LLMResponse, ToolCall  # noqa: E402
from rewind.snapshot import SnapshotStore  # noqa: E402
from rewind.worker import run_worker_step, submit_job_async  # noqa: E402

DSN = "postgresql://rewind:rewind@127.0.0.1:5432/rewind_test"
SYS = "You are a retail support agent."
TASK = "I'm u1, cancel my headphones order."


def _call(name, **args):
    return LLMResponse(tool_calls=[ToolCall(name=name, args=args)],
                       usage={"input_tokens": 10, "output_tokens": 5})


def _text(t):
    return LLMResponse(text=t, usage={"input_tokens": 10, "output_tokens": 5})


SCRIPT = [_call("list_orders", user_id="u1"),
         _call("cancel_order", order_id="O1", reason="ordered_by_mistake"),
         _text("done")]


class MockProvider:
    """Zero-latency scripted provider — isolates coordinator/checkpoint
    overhead from anything provider-side (see dist/gateway.py for that)."""

    async def complete(self, messages, tools, params):
        step = sum(1 for m in messages if m.get("role") == "assistant") + 1
        return SCRIPT[min(step - 1, len(SCRIPT) - 1)]


async def run_trial(n_jobs: int, n_workers: int, dsn: str) -> float:
    coord = await PgCoordinator.connect(dsn)
    async with coord.pool.acquire() as conn:
        await conn.execute("TRUNCATE jobs, checkpoints")
    tmp = tempfile.mkdtemp()
    snaps = SnapshotStore(root=str(Path(tmp) / "snap"))
    for i in range(n_jobs):
        await submit_job_async(coord, snaps, f"scale-{n_workers}w-{i}", TASK,
                               AgentConfig(system_prompt=SYS), ToyRetailEnv().state,)

    async def worker_loop(worker_id: int) -> int:
        registry = build_registry()
        n_done = 0
        while True:
            result = await run_worker_step(coord, snaps, snaps, registry, MockProvider(),
                                           ToyRetailEnv.restore, Path(tmp) / "trace",
                                           owner=f"scale-w{worker_id}", lease_seconds=10)
            if result is None:
                return n_done
            n_done += 1

    t0 = time.monotonic()
    completed_counts = await asyncio.gather(*[worker_loop(i) for i in range(n_workers)])
    elapsed = time.monotonic() - t0
    assert sum(completed_counts) == n_jobs, (completed_counts, n_jobs)
    await coord.close()
    return elapsed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=40)
    p.add_argument("--workers", default="1,2,4,8")
    p.add_argument("--dsn", default=DSN)
    p.add_argument("--repeats", type=int, default=3, help="repeats per worker count, reports median")
    args = p.parse_args()

    print(f"{'workers':>8} {'jobs':>6} {'median_s':>10} {'jobs/s':>10}")
    for w in [int(x) for x in args.workers.split(",")]:
        elapsed_samples = [asyncio.run(run_trial(args.jobs, w, args.dsn)) for _ in range(args.repeats)]
        med = statistics.median(elapsed_samples)
        print(f"{w:>8} {args.jobs:>6} {med:>10.3f} {args.jobs / med:>10.2f}")


if __name__ == "__main__":
    main()
