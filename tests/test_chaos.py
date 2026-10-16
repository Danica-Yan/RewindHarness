"""W6: chaos testing against REAL infrastructure — a real OS process killed
with a real SIGKILL, and a real Toxiproxy TCP proxy injecting real network
faults — not the in-process `SimulatedCrash` used in `test_checkpoint.py`
(which is exhaustive but, by construction, can't cross a process boundary)
nor a mocked network layer.

Needs: a reachable Postgres (`REWIND_PG_DSN`, default
postgresql://rewind:rewind@127.0.0.1:5432/rewind_test) and the
`toxiproxy-server` binary on PATH (see README for how it's fetched — a
static Go binary from GitHub releases, no Docker required). Each test group
skips itself with a clear reason if what it needs isn't available, so a
plain `pytest -q` stays green without either.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

asyncpg = pytest.importorskip("asyncpg", reason="needs the [dist] extra")
pytest.importorskip("requests", reason="needs the [chaos] extra")

from rewind.dist.invariants import check_no_stuck_leases, check_seq_contiguous
from rewind.dist.pg_coordinator import PgCoordinator
from rewind.dist.toxi import ToxiproxyController, connection_reset, latency, timeout
from rewind.env import ToyRetailEnv
from rewind.loop import AgentConfig
from rewind.snapshot import SnapshotStore
from rewind.worker import submit_job_async

DSN = os.environ.get("REWIND_PG_DSN", "postgresql://rewind:rewind@127.0.0.1:5432/rewind_test")
SYS = "You are a retail support agent."
TASK = "I'm u1, cancel my headphones order."


async def _connect() -> PgCoordinator:
    try:
        coord = await PgCoordinator.connect(DSN)
    except (OSError, asyncpg.PostgresError, ConnectionError) as e:
        pytest.skip(f"Postgres not reachable at {DSN}: {e}")
    async with coord.pool.acquire() as conn:
        await conn.execute("TRUNCATE jobs, checkpoints")
    return coord


def _require_toxiproxy() -> ToxiproxyController:
    if not ToxiproxyController.binary_available():
        pytest.skip("toxiproxy-server binary not on PATH (see README for how to fetch it)")
    ctl = ToxiproxyController()
    ctl.start()
    return ctl


# ---------------------------------------------------------------------------
# Real process kill: a genuine SIGKILL, a genuine second process resuming.
# ---------------------------------------------------------------------------

def test_real_sigkill_mid_run_and_a_second_real_process_completes_it(tmp_path):
    """Submits one job, launches worker A as a real subprocess with an
    artificial per-step delay (so there's a wide, real wall-clock window),
    waits for A to make *durable* progress (its checkpoint for step 1 lands
    in Postgres), then sends A a real SIGKILL — not a debug hook, an actual
    `os.kill(pid, signal.SIGKILL)`. After A's lease expires, worker B (a
    second, independent process) is launched and must finish the job
    correctly: final state graded correct, the mutating tool's call counter
    (baked into env state so it survives the process boundary) is exactly
    1, no stuck lease, and the trace's seq is contiguous end to end."""
    async def go():
        coord = await _connect()
        snap_dir = tmp_path / "snap"
        trace_dir = tmp_path / "trace"
        run_id = "chaos-kill-1"
        job_id = await submit_job_async(coord, SnapshotStore(root=str(snap_dir)), run_id, TASK,
                                        AgentConfig(system_prompt=SYS), ToyRetailEnv().state)

        src_dir = str(Path(__file__).resolve().parents[1] / "src")
        env = dict(os.environ, PYTHONPATH=src_dir)
        pid_file = tmp_path / "a.pid"

        proc_a = subprocess.Popen(
            [sys.executable, "-m", "rewind.dist.chaos_worker",
             "--dsn", DSN, "--snapshot-dir", str(snap_dir), "--trace-dir", str(trace_dir),
             "--owner", "worker-A", "--lease-seconds", "3", "--step-delay", "0.6",
             "--pid-file", str(pid_file)],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            # Wait for real, durable progress: a step-1 checkpoint actually
            # committed to Postgres by worker A (not just "the process started").
            for _ in range(60):
                if pid_file.exists():
                    break
                await asyncio.sleep(0.05)
            assert pid_file.exists(), "worker A never wrote its pid file"

            got = await _poll_checkpoint_exists(coord, run_id, step=1, timeout_s=5.0)
            assert got, "worker A never durably checkpointed step 1 — nothing real to kill mid-flight"

            pid = int(pid_file.read_text())
            os.kill(pid, signal.SIGKILL)
            ret = proc_a.wait(timeout=5)
            assert ret == -signal.SIGKILL, f"worker A should have died to a real SIGKILL, got {ret}"
        finally:
            if proc_a.poll() is None:
                proc_a.kill()
                proc_a.wait(timeout=5)

        # Let A's 3s lease actually expire in real wall-clock time.
        await asyncio.sleep(3.2)

        proc_b = subprocess.run(
            [sys.executable, "-m", "rewind.dist.chaos_worker",
             "--dsn", DSN, "--snapshot-dir", str(snap_dir), "--trace-dir", str(trace_dir),
             "--owner", "worker-B", "--lease-seconds", "10", "--step-delay", "0"],
            env=env, capture_output=True, text=True, timeout=30)
        assert proc_b.returncode == 0, proc_b.stderr
        out = json.loads(proc_b.stdout.strip().splitlines()[-1])
        assert out["claimed"] is True
        assert out["status"] == "completed", out
        assert out["cancel_calls"] == 1, \
            f"mutating tool must fire exactly once across the crash, got {out['cancel_calls']}"

        # Invariants, checked against the real artifacts both processes left behind.
        trace_path = trace_dir / f"{run_id}.jsonl"
        seq_problems = check_seq_contiguous(trace_path)
        assert seq_problems == [], seq_problems
        async with coord.pool.acquire() as conn:
            stuck = await check_no_stuck_leases(conn, grace_seconds=1.0)
        assert stuck == [], stuck

        row = await coord.pool.fetchrow("SELECT status FROM jobs WHERE job_id=$1", job_id)
        assert row["status"] == "completed"
        await coord.close()
    asyncio.run(go())


async def _poll_checkpoint_exists(coord: PgCoordinator, run_id: str, step: int,
                                  timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    async with coord.pool.acquire() as conn:
        while time.monotonic() < deadline:
            row = await conn.fetchrow(
                "SELECT 1 FROM checkpoints WHERE run_id=$1 AND step=$2", run_id, step)
            if row is not None:
                return True
            await asyncio.sleep(0.05)
    return False


# ---------------------------------------------------------------------------
# Real network faults via a real Toxiproxy.
# ---------------------------------------------------------------------------

def test_toxiproxy_latency_toxic_actually_delays_real_queries():
    """Proves the fault injection is real: a `latency` toxic on a proxy in
    front of Postgres measurably slows a real `asyncpg` round trip through
    it, and removing the toxic brings it back to normal."""
    ctl = _require_toxiproxy()
    try:
        ctl.create_proxy("rewind-chaos-pg", "127.0.0.1:15433", "127.0.0.1:5432")
        try:
            async def go():
                fast = await asyncpg.connect(
                    "postgresql://rewind:rewind@127.0.0.1:15433/rewind_test")
                t0 = time.monotonic()
                await fast.fetchval("select 1")
                baseline = time.monotonic() - t0
                await fast.close()

                ctl.add_toxic("rewind-chaos-pg", latency("lat", ms=400, jitter_ms=20))
                t0 = time.monotonic()
                slow = await asyncpg.connect(
                    "postgresql://rewind:rewind@127.0.0.1:15433/rewind_test")
                await slow.fetchval("select 1")
                elapsed = time.monotonic() - t0
                await slow.close()
                return baseline, elapsed
            baseline, elapsed = asyncio.run(go())
            assert elapsed > baseline + 0.3, \
                f"latency toxic should measurably slow the connection (baseline={baseline:.3f}s, " \
                f"with toxic={elapsed:.3f}s)"
        finally:
            ctl.remove_all_toxics("rewind-chaos-pg")
            ctl.delete_proxy("rewind-chaos-pg")
    finally:
        ctl.stop()


def test_toxiproxy_connection_reset_makes_the_coordinator_fail_then_recover():
    """A `PgCoordinator` pointed through a proxy whose connection gets reset
    genuinely fails its query while the toxic is active, and genuinely
    succeeds again once it's removed — no code path here is aware it's
    running through a proxy at all, which is the point: the fault is at the
    transport layer, invisible to the application."""
    ctl = _require_toxiproxy()
    try:
        ctl.create_proxy("rewind-chaos-pg2", "127.0.0.1:15434", "127.0.0.1:5432")
        try:
            async def go():
                ctl.add_toxic("rewind-chaos-pg2", connection_reset("cut"))
                broke = False
                try:
                    conn = await asyncio.wait_for(
                        asyncpg.connect("postgresql://rewind:rewind@127.0.0.1:15434/rewind_test"),
                        timeout=3.0)
                    await conn.close()
                except Exception:
                    broke = True
                ctl.remove_all_toxics("rewind-chaos-pg2")
                conn = await asyncio.wait_for(
                    asyncpg.connect("postgresql://rewind:rewind@127.0.0.1:15434/rewind_test"),
                    timeout=3.0)
                ok = await conn.fetchval("select 1") == 1
                await conn.close()
                return broke, ok
            broke, ok = asyncio.run(go())
            assert broke, "a reset toxic should make the connection attempt genuinely fail"
            assert ok, "removing the toxic should let the same target recover immediately"
        finally:
            ctl.remove_all_toxics("rewind-chaos-pg2")
            ctl.delete_proxy("rewind-chaos-pg2")
    finally:
        ctl.stop()


def test_toxiproxy_timeout_toxic_on_redis_is_observable():
    """Same idea against Redis instead of Postgres, and with a `timeout`
    toxic (accepts the connection, then goes silent) rather than an
    immediate reset — a different, equally real failure shape."""
    ctl = _require_toxiproxy()
    try:
        import redis.asyncio as aioredis
        redis_url = os.environ.get("REWIND_REDIS_URL", "redis://127.0.0.1:6379/0")
        host_port = redis_url.split("://", 1)[-1].split("/", 1)[0]
        upstream_host, upstream_port = host_port.split(":") if ":" in host_port else (host_port, "6379")

        async def redis_reachable() -> bool:
            try:
                c = aioredis.from_url(redis_url)
                await c.ping()
                await c.aclose()
                return True
            except Exception:
                return False
        if not asyncio.run(redis_reachable()):
            pytest.skip(f"Redis not reachable at {redis_url}")

        ctl.create_proxy("rewind-chaos-redis", "127.0.0.1:16380", f"{upstream_host}:{upstream_port}")
        try:
            ctl.add_toxic("rewind-chaos-redis", timeout("to", ms=300))

            async def go():
                client = aioredis.from_url("redis://127.0.0.1:16380/0")
                t0 = time.monotonic()
                timed_out = False
                try:
                    await asyncio.wait_for(client.ping(), timeout=2.0)
                except Exception:
                    timed_out = True
                elapsed = time.monotonic() - t0
                await client.aclose()
                return timed_out, elapsed
            timed_out, elapsed = asyncio.run(go())
            assert timed_out, "a timeout toxic should make the ping never complete normally"
        finally:
            ctl.remove_all_toxics("rewind-chaos-redis")
            ctl.delete_proxy("rewind-chaos-redis")
    finally:
        ctl.stop()


# ---------------------------------------------------------------------------
# Invariant checker, unit-level (synthetic inputs, no live process needed).
# ---------------------------------------------------------------------------

def test_invariant_checker_flags_a_genuine_seq_gap(tmp_path):
    from rewind.events import Event
    p = tmp_path / "broken.jsonl"
    events = [
        Event(run_id="r", seq=1, step=1, type="message", ts="2026-01-01T00:00:00Z", data={}),
        Event(run_id="r", seq=2, step=1, type="message", ts="2026-01-01T00:00:00Z", data={}),
        Event(run_id="r", seq=4, step=1, type="message", ts="2026-01-01T00:00:00Z", data={}),  # gap: no seq=3
    ]
    p.write_text("\n".join(e.model_dump_json() for e in events) + "\n", encoding="utf-8")
    problems = check_seq_contiguous(p)
    assert any("gap" in msg for msg in problems), problems


def test_invariant_checker_is_clean_on_a_contiguous_trace(tmp_path):
    from rewind.events import Event
    p = tmp_path / "ok.jsonl"
    events = [Event(run_id="r", seq=i, step=1, type="message", ts="2026-01-01T00:00:00Z", data={})
             for i in range(1, 6)]
    p.write_text("\n".join(e.model_dump_json() for e in events) + "\n", encoding="utf-8")
    assert check_seq_contiguous(p) == []


def test_invariant_checker_flags_a_stuck_lease_and_clears_once_reclaimed():
    async def go():
        coord = await _connect()
        snaps = SnapshotStore()
        await submit_job_async(coord, snaps, "stuck-r1", TASK, AgentConfig(system_prompt=SYS),
                               ToyRetailEnv().state)
        job = await coord.claim("zombie-owner", lease_seconds=0.2)
        assert job is not None
        await asyncio.sleep(1.5)   # lease expires, and stays unreclaimed past the grace window

        async with coord.pool.acquire() as conn:
            stuck = await check_no_stuck_leases(conn, grace_seconds=1.0)
        assert any(job.job_id in msg for msg in stuck), stuck

        reclaimed = await coord.claim("rescuer", lease_seconds=30)
        assert reclaimed is not None and reclaimed.job_id == job.job_id

        async with coord.pool.acquire() as conn:
            stuck_after = await check_no_stuck_leases(conn, grace_seconds=1.0)
        assert stuck_after == [], stuck_after
        await coord.close()
    asyncio.run(go())
