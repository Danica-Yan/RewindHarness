"""PgCoordinator tests against a real Postgres.

These are the "real infra" tier: unlike tests/test_checkpoint.py (which
exhaustively crashes at every single checkpoint hook against sqlite, cheaply,
in-process), these exercise the SAME worker.py/StepCheckpointer code path
against actual network round-trips and real row-level locking, to prove the
sqlite-vs-Postgres duck-typing (`_await.maybe_await`) actually works and that
`FOR UPDATE SKIP LOCKED` behaves as claimed. A couple of representative crash
points, not all seven — that exhaustiveness is what the sqlite tier is for.

Needs a real Postgres reachable at REWIND_PG_DSN (default:
postgresql://rewind:rewind@127.0.0.1:5432/rewind_test). Skips with a clear
reason, not a failure, if it can't connect.
"""
from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

asyncpg = pytest.importorskip("asyncpg", reason="needs the [dist] extra")
from conftest import call, text

from rewind import (AgentConfig, LoopState, RunAbort, ScriptedProvider, SimulatedCrash,
                    SnapshotStore, TraceWriter, run_worker_step)
from rewind.checkpoint import StepCheckpointer
from rewind.dist.pg_coordinator import PgCoordinator
from rewind.env import ToyRetailEnv, build_registry, grade
from rewind.worker import submit_job_async

DSN = os.environ.get("REWIND_PG_DSN", "postgresql://rewind:rewind@127.0.0.1:5432/rewind_test")
SYS = "You are a retail support agent."
TASK = "I'm u1, cancel my headphones order."
CANCEL_O1 = dict(order_id="O1", reason="ordered_by_mistake")
SCRIPT = [call("list_orders", user_id="u1"), call("cancel_order", **CANCEL_O1), text("done")]


async def _connect() -> PgCoordinator:
    try:
        coord = await PgCoordinator.connect(DSN)
    except (OSError, asyncpg.PostgresError, ConnectionError) as e:
        pytest.skip(f"Postgres not reachable at {DSN}: {e}")
    # `rewind_test` is scratch space for this suite only. claim() picks "the
    # oldest available job" with no way to target a specific one, so a row
    # left over from an earlier test run (this database persists across
    # separate `pytest` invocations, unlike the in-memory sqlite Coordinator)
    # could get claimed instead of the job this test just submitted, whose
    # initial_env_ref only exists in *this* test's fresh in-memory
    # SnapshotStore. Start every test from a clean slate.
    async with coord.pool.acquire() as conn:
        await conn.execute("TRUNCATE jobs, checkpoints")
    return coord


def counting_registry():
    reg = build_registry()
    counter = {"n": 0}
    orig = reg._tools["cancel_order"].fn

    def wrapped(env, a):
        counter["n"] += 1
        return orig(env, a)

    reg._tools["cancel_order"].fn = wrapped
    return reg, counter


def test_claim_lease_and_fencing_over_real_network():
    async def go():
        coord = await _connect()
        try:
            snaps = SnapshotStore()
            run_id = "pg-lease-" + os.urandom(4).hex()
            await submit_job_async(coord, snaps, run_id, TASK, AgentConfig(system_prompt=SYS),
                                   ToyRetailEnv().state)
            job_a = await coord.claim("A", lease_seconds=1)
            assert job_a is not None
            assert await coord.claim("B", lease_seconds=1) is None    # A's lease still valid

            await asyncio.sleep(1.2)                                  # real wall-clock expiry
            job_b = await coord.claim("B", lease_seconds=30)
            assert job_b is not None and job_b.fencing_token == job_a.fencing_token + 1
        finally:
            await coord.close()
    asyncio.run(go())


def test_checkpoint_write_is_token_gated_over_real_network():
    async def go():
        coord = await _connect()
        try:
            snaps, blobs = SnapshotStore(), SnapshotStore()
            run_id = "pg-fence-" + os.urandom(4).hex()
            await submit_job_async(coord, snaps, run_id, TASK, AgentConfig(system_prompt=SYS),
                                   ToyRetailEnv().state)
            job = await coord.claim("A", lease_seconds=1)
            await asyncio.sleep(1.2)
            newer = await coord.claim("B", lease_seconds=30)
            assert newer.fencing_token == job.fencing_token + 1

            zombie = StepCheckpointer(coord, job.job_id, run_id, job.fencing_token, snaps, blobs)
            with pytest.raises(RunAbort) as ei:
                await zombie.save("step", 1, TraceWriter(run_id), ToyRetailEnv(), [], LoopState())
            assert ei.value.status == "fenced_out"

            current = StepCheckpointer(coord, job.job_id, run_id, newer.fencing_token, snaps, blobs)
            await current.save("step", 1, TraceWriter(run_id), ToyRetailEnv(), [], LoopState())

            ckpt = await coord.latest_checkpoint(run_id)
            assert ckpt.fencing_token == newer.fencing_token
        finally:
            await coord.close()
    asyncio.run(go())


def test_worker_crash_recovery_against_real_postgres():
    """The same claim: a crash mid-tool-call doesn't double-apply a mutating
    tool, now proven against real Postgres instead of sqlite."""
    async def go():
        coord = await _connect()
        try:
            registry, counter = counting_registry()
            snaps, blobs = SnapshotStore(), SnapshotStore()
            run_id = "pg-crash-" + os.urandom(4).hex()
            tmp = tempfile.mkdtemp()
            await submit_job_async(coord, snaps, run_id, TASK, AgentConfig(system_prompt=SYS),
                                   ToyRetailEnv().state)

            def crash_hook(tag, step):
                if (tag, step) == ("after_tool_call", 2):
                    raise SimulatedCrash()

            with pytest.raises(SimulatedCrash):
                await run_worker_step(coord, snaps, blobs, registry, ScriptedProvider(list(SCRIPT)),
                                      ToyRetailEnv.restore, tmp, owner="A", lease_seconds=1,
                                      debug_hook=crash_hook)

            await asyncio.sleep(1.2)                                  # real wall-clock lease expiry
            res, env = await run_worker_step(coord, snaps, blobs, registry, ScriptedProvider(SCRIPT[2:]),
                                             ToyRetailEnv.restore, tmp, owner="B", lease_seconds=30)
            assert res.status == "completed"
            assert grade(env)
            assert counter["n"] == 1, "cancel_order must run exactly once despite the crash"
        finally:
            await coord.close()
    asyncio.run(go())


def test_concurrent_claims_from_real_connections_never_double_deliver():
    """FOR UPDATE SKIP LOCKED under real concurrent connections: N jobs, N
    concurrent claimers, every job claimed by exactly one — the property
    sqlite's single-connection Coordinator could never actually exercise."""
    async def go():
        coord = await _connect()
        try:
            snaps = SnapshotStore()
            n = 8
            prefix = "pg-concurrent-" + os.urandom(4).hex()
            for i in range(n):
                await submit_job_async(coord, snaps, f"{prefix}-{i}", TASK,
                                       AgentConfig(system_prompt=SYS), ToyRetailEnv().state)

            results = await asyncio.gather(*[coord.claim(f"worker-{i}", lease_seconds=30)
                                             for i in range(n)])
            claimed_run_ids = [j.run_id for j in results if j is not None]
            assert len(claimed_run_ids) == n
            assert len(set(claimed_run_ids)) == n, "every job must be claimed by exactly one worker"
        finally:
            await coord.close()
    asyncio.run(go())


def test_claim_query_uses_the_partial_index_not_a_seq_scan():
    """Regression test for a real gap: `jobs` originally had no index at all
    supporting claim()'s WHERE clause, so every claim was an O(n) sequential
    scan once completed/failed history piled up (invisible at test scale,
    real at production scale). `jobs_claimable_idx` (a partial index on
    PENDING/LEASED rows only -- `now()` can't appear in an index predicate,
    so the lease-expiry half of the WHERE clause still runs as a filter,
    just over a tiny pre-pruned set instead of the whole table) should make
    the planner pick an index scan even once there's a large amount of
    COMPLETED/FAILED history sharing the table. This seeds 5,000 historical
    rows (COMPLETED/FAILED) plus a handful of real active ones and reads
    back Postgres's own EXPLAIN output -- the only way to actually prove a
    query plan, as opposed to asserting on wall-clock time, which is noisy
    and would make this test flaky."""
    async def go():
        coord = await _connect()
        try:
            async with coord.pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO jobs (job_id, run_id, spec, status, priority, created_at)
                       SELECT 'hist_'||g, 'r'||g, '{}', (ARRAY['COMPLETED','FAILED'])[1 + (g % 2)],
                              0, now() - (g || ' seconds')::interval
                       FROM generate_series(1, 5000) g""")
                await conn.execute(
                    """INSERT INTO jobs (job_id, run_id, spec, status, priority, created_at)
                       VALUES ('active_1', 'r-active', '{}', 'PENDING', 0, now())""")
                rows = await conn.fetch(
                    """EXPLAIN (FORMAT TEXT)
                       SELECT job_id FROM jobs
                       WHERE status='PENDING' OR (status='LEASED' AND lease_expires_at < now())
                       ORDER BY priority DESC, created_at
                       FOR UPDATE SKIP LOCKED LIMIT 1""")
            plan_text = "\n".join(r["QUERY PLAN"] for r in rows)
            assert "jobs_claimable_idx" in plan_text, (
                f"claim()'s query should use the partial index, not a seq scan:\n{plan_text}")
            assert "Seq Scan on jobs" not in plan_text, (
                f"claim()'s query fell back to a full table scan:\n{plan_text}")
        finally:
            await coord.close()
    asyncio.run(go())
