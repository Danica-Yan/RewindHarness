#!/usr/bin/env python3
"""W6 recovery-time experiment: repeatedly kill a REAL worker OS process
with a REAL SIGKILL at a point of genuine durable progress, start a second
real process, and measure how long recovery actually takes — reported as
p50/p99 over N trials. This is the statistical companion to
`tests/test_chaos.py::test_real_sigkill_mid_run_and_a_second_real_process_completes_it`,
which proves the mechanism is correct once; this script runs it repeatedly
to characterize its *latency*.

"Recovery time" here is measured from the moment worker A's lease
*actually* expires (not from the kill itself, since the lease grace period
is a deliberate, configured part of the design — killing faster than the
lease doesn't buy real recovery time, it just wastes the remaining lease)
to the moment worker B's claim of the now-reclaimable job lands in Postgres.

Usage:
  python scripts/chaos_recovery_experiment.py --trials 20 --lease-seconds 2
"""
from __future__ import annotations

import argparse
import asyncio
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rewind.dist.pg_coordinator import PgCoordinator  # noqa: E402
from rewind.env import ToyRetailEnv  # noqa: E402
from rewind.loop import AgentConfig  # noqa: E402
from rewind.snapshot import SnapshotStore  # noqa: E402
from rewind.worker import submit_job_async  # noqa: E402

DSN = "postgresql://rewind:rewind@127.0.0.1:5432/rewind_test"
SYS = "You are a retail support agent."
TASK = "I'm u1, cancel my headphones order."


async def one_trial(trial_idx: int, lease_seconds: float, dsn: str, src_dir: str) -> float:
    coord = await PgCoordinator.connect(dsn)
    async with coord.pool.acquire() as conn:
        await conn.execute("TRUNCATE jobs, checkpoints")
    tmp = Path(tempfile.mkdtemp())
    snap_dir, trace_dir = tmp / "snap", tmp / "trace"
    run_id = f"recovery-{trial_idx}"
    job_id = await submit_job_async(coord, SnapshotStore(root=str(snap_dir)), run_id, TASK,
                                    AgentConfig(system_prompt=SYS), ToyRetailEnv().state)

    pid_file = tmp / "a.pid"
    env = dict(os.environ, PYTHONPATH=src_dir)
    proc_a = subprocess.Popen(
        [sys.executable, "-m", "rewind.dist.chaos_worker",
         "--dsn", dsn, "--snapshot-dir", str(snap_dir), "--trace-dir", str(trace_dir),
         "--owner", "recA", "--lease-seconds", str(lease_seconds), "--step-delay", "0.5",
         "--pid-file", str(pid_file)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for _ in range(100):
        if pid_file.exists():
            break
        await asyncio.sleep(0.02)

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        row = await coord.pool.fetchrow(
            "SELECT 1 FROM checkpoints WHERE run_id=$1 AND step=1", run_id)
        if row is not None:
            break
        await asyncio.sleep(0.02)

    pid = int(pid_file.read_text())
    os.kill(pid, signal.SIGKILL)
    proc_a.wait(timeout=5)

    lease_row = await coord.pool.fetchrow("SELECT lease_expires_at FROM jobs WHERE job_id=$1", job_id)
    lease_expires_at = lease_row["lease_expires_at"].timestamp()
    now = time.time()
    if lease_expires_at > now:
        await asyncio.sleep(lease_expires_at - now)

    t_lease_expired = time.monotonic()
    proc_b = subprocess.Popen(
        [sys.executable, "-m", "rewind.dist.chaos_worker",
         "--dsn", dsn, "--snapshot-dir", str(snap_dir), "--trace-dir", str(trace_dir),
         "--owner", "recB", "--lease-seconds", "30", "--step-delay", "0"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    proc_b.wait(timeout=30)
    t_recovered = time.monotonic()
    await coord.close()
    return t_recovered - t_lease_expired


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--lease-seconds", type=float, default=2.0)
    p.add_argument("--dsn", default=DSN)
    args = p.parse_args()
    src_dir = str(Path(__file__).resolve().parents[1] / "src")

    latencies = []
    for i in range(args.trials):
        lat = asyncio.run(one_trial(i, args.lease_seconds, args.dsn, src_dir))
        latencies.append(lat)
        print(f"trial {i:>3}: recovery latency = {lat:.3f}s")

    latencies.sort()
    p50 = statistics.median(latencies)
    p99_idx = min(len(latencies) - 1, int(0.99 * len(latencies)))
    p99 = latencies[p99_idx]
    print(f"\nN={len(latencies)}  p50={p50:.3f}s  p99={p99:.3f}s  "
         f"min={latencies[0]:.3f}s  max={latencies[-1]:.3f}s")


if __name__ == "__main__":
    main()
