"""Platform P1: quotas (usage entitlement).

Offline tier: rule/bucket semantics, the in-memory store (all-or-nothing,
exactly-once, thread races), enforcement in the loop, replay, crash at every
checkpoint. Postgres tier: the same races against a real database; skips
itself when Postgres is not reachable (REWIND_PG_DSN).
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import threading

import pytest
from conftest import call, run, text
from test_platform import INJECTED_RUN, _no_sleep, ev, go, platform_registry, principal, tool_msgs

from rewind import AgentConfig, Coordinator, ScriptedProvider, SimulatedCrash, SnapshotStore, replay_run
from rewind import run_worker_step, submit_job
from rewind.env import ToyRetailEnv, grade
from rewind.platform import (CapabilityDescriptor, Charge, InMemoryApprovalStore, InMemoryQuotaStore,
                             QuotaMeter, QuotaRule)
from rewind.util import FakeClock

DSN = os.environ.get("REWIND_PG_DSN", "postgresql://rewind:rewind@127.0.0.1:5432/rewind_test")
LIST = CapabilityDescriptor(id="local.list_orders", name="list_orders")
EMAIL = CapabilityDescriptor(id="local.send_email", name="send_email", risk="external")


def rule(**kw) -> QuotaRule:
    base = {"id": "q", "capabilities": ["local.list_orders"], "per": "tenant", "limit": 2}
    return QuotaRule.model_validate({**base, **kw})


def with_quota(rules, store=None, now=lambda: 1000.0, approvals=None):
    reg = platform_registry(approvals)
    store = store if store is not None else InMemoryQuotaStore()
    reg.quota = QuotaMeter(rules, store, now=now)
    return reg, store


def lists(*users):
    return [call("list_orders", user_id=u) for u in users]


# ---------------------------------------------------------------------------
# rules and buckets
# ---------------------------------------------------------------------------
def test_subjects_are_tenant_qualified_and_windows_are_fixed():
    t = {"now": 3599.0}
    m = QuotaMeter([rule(per="user", window_seconds=3600)], InMemoryQuotaStore(), now=lambda: t["now"])
    a = m.charges(principal(tenant="t1"), LIST, "r1")[0]
    b = m.charges(principal(tenant="t2"), LIST, "r1")[0]
    assert a.key != b.key and "t1/u1" in a.key                 # same user id, different tenants
    t["now"] = 3600.0
    c = m.charges(principal(tenant="t1"), LIST, "r1")[0]
    assert c.key != a.key and (a.reset_at, c.reset_at) == (3600.0, 7200.0)
    assert m.charges(principal(), EMAIL, "r1") == []           # no rule covers it: free


def test_example_rules_file_loads():
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "data/access_policies/example_quotas.json"
    rules = [QuotaRule.model_validate(r) for r in json.loads(path.read_text())]
    assert {r.per for r in rules} == {"tenant", "run", "user"}


def test_rule_validation():
    with pytest.raises(ValueError):
        rule(limit=0)
    with pytest.raises(ValueError):
        rule(capabilities=[])
    with pytest.raises(ValueError, match="duplicate"):
        QuotaMeter([rule(), rule()], InMemoryQuotaStore())


# ---------------------------------------------------------------------------
# in-memory store
# ---------------------------------------------------------------------------
def C(key, limit, cost=1, reset=None):
    return Charge(key, key, limit, cost, reset)


def test_charge_is_all_or_nothing():
    s = InMemoryQuotaStore()
    s.used["b"] = 1
    r = s.charge("x1", [C("a", 5), C("b", 1)], now=10.0)
    assert (r.ok, r.exceeded) == (False, "b") and s.used.get("a", 0) == 0
    assert s.charge("x2", [C("a", 5)], now=10.0).ok and s.used["a"] == 1


def test_charge_is_exactly_once_per_charge_id():
    s = InMemoryQuotaStore()
    first = s.charge("run:s1:c0", [C("a", 5)], now=0.0)
    again = s.charge("run:s1:c0", [C("a", 5)], now=0.0)
    assert first.ok and again.ok and again.deduplicated and s.used["a"] == 1
    s.used["b"] = 1
    no = s.charge("run:s2:c0", [C("b", 1, reset=60.0)], now=15.0)
    assert (no.ok, no.retry_after) == (False, 45.0)
    s.used["b"] = 0                                            # room again ...
    assert not s.charge("run:s2:c0", [C("b", 1)], now=15.0).ok  # ... but the recorded answer stands


def test_threads_racing_for_the_last_units_never_overspend():
    s = InMemoryQuotaStore()
    results, barrier = [], threading.Barrier(20)

    def worker(i):
        barrier.wait()
        results.append(s.charge(f"c{i}", [C("hot", 5)], now=0.0).ok)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sum(results) == 5 and s.used["hot"] == 5


# ---------------------------------------------------------------------------
# in the loop
# ---------------------------------------------------------------------------
def test_call_over_the_limit_is_refused_and_does_not_run():
    reg, store = with_quota([rule(window_seconds=3600)])
    res, env, trace, *_ = go(lists("u1", "u2", "u3") + [text("done")], principal(), registry=reg)
    q = [a["quota"] for a in ev(trace, "authz")]
    assert [x["ok"] for x in q] == [True, True, False]
    assert q[2]["exceeded"] == "q" and q[2]["retry_after"] == 2600.0
    third = tool_msgs(trace)[2]["content"]
    assert third.startswith("ERROR[QUOTA_EXCEEDED]") and "resets in 2600s" in third
    assert [t["ok"] for t in ev(trace, "tool_call")] == [True, True, False]
    assert sum(store.used.values()) == 2


def test_tenant_bucket_is_shared_across_agents_but_not_tenants():
    store = InMemoryQuotaStore()
    reg, _ = with_quota([rule()], store)
    go(lists("u1", "u2") + [text("a")], principal(agent="agent-a"), registry=reg, run_id="ra")
    _, _, t_b, *_ = go(lists("u1") + [text("b")], principal(agent="agent-b"), registry=reg, run_id="rb")
    _, _, t_c, *_ = go(lists("u1") + [text("c")], principal(tenant="t2"), registry=reg, run_id="rc")
    assert ev(t_b, "authz")[0]["quota"]["ok"] is False
    assert ev(t_c, "authz")[0]["quota"]["ok"] is True


def test_denied_calls_are_not_charged():
    reg, store = with_quota([rule(capabilities=["local.*"])])
    go([call("write_file", path="/workspaces/t1/a", content="x"), text("x")],
       principal(roles=("support",)), registry=reg)            # hidden: not an editor
    assert store.used == {}


def test_a_step_paused_for_approval_charges_nothing():
    approvals = InMemoryApprovalStore()
    reg, store = with_quota([rule(capabilities=["local.list_orders", "local.send_email"], limit=10)],
                            approvals=approvals)
    turn2 = INJECTED_RUN[1].model_copy(deep=True)
    turn2.tool_calls.insert(0, call("list_orders", user_id="u1").tool_calls[0])
    res, *_ = go([INJECTED_RUN[0], turn2], principal(), registry=reg)
    assert res.status == "awaiting_approval" and store.used == {} and len(approvals.pending()) == 1


def test_replay_serves_recorded_quota_and_never_touches_the_store():
    reg, store = with_quota([rule()])
    res, env, trace, _p, snaps, _ = go(lists("u1", "u2", "u3") + [text("done")], principal(), registry=reg)
    before = dict(store.used)
    rep = run(replay_run(trace.events, snaps, reg, ToyRetailEnv.restore))
    assert rep.ok, rep
    assert store.used == before
    store.used.clear()                                         # even with all the room back,
    assert run(replay_run(trace.events, snaps, reg, ToyRetailEnv.restore)).ok   # the refusal replays


# ---------------------------------------------------------------------------
# crash at every checkpoint: each action charged exactly once
# ---------------------------------------------------------------------------
SCRIPT = [call("list_orders", user_id="u1"), call("cancel_order", order_id="O1", reason="ordered_by_mistake"),
          text("done")]
PER_RUN = [rule(id="per-run", capabilities=["local.*"], per="run", limit=10)]


async def _hook_points():
    reg, _ = with_quota(PER_RUN)
    coord, snaps = Coordinator(clock=FakeClock()), SnapshotStore()
    cfg = AgentConfig(system_prompt="s", principal=principal().model_dump(mode="json"))
    await submit_job(coord, snaps, "dry", "t", cfg, ToyRetailEnv().state)
    pts: list[tuple[str, int]] = []
    await run_worker_step(coord, snaps, snaps, reg, ScriptedProvider(list(SCRIPT)), ToyRetailEnv.restore,
                          tempfile.mkdtemp(), sleep=_no_sleep, debug_hook=lambda t, s: pts.append((t, s)))
    return pts


CRASH_POINTS = asyncio.run(_hook_points())
# An empty list would make the parametrized test below skip silently.
assert len(CRASH_POINTS) >= 5, CRASH_POINTS


@pytest.mark.parametrize("tag,step", CRASH_POINTS, ids=[f"{t}@{s}" for t, s in CRASH_POINTS])
def test_crash_anywhere_charges_each_action_once(tag, step):
    async def go_():
        reg, store = with_quota(PER_RUN)
        clock = FakeClock()
        coord, snaps = Coordinator(clock=clock), SnapshotStore()
        cfg = AgentConfig(system_prompt="s", principal=principal().model_dump(mode="json"))
        await submit_job(coord, snaps, "r1", "t", cfg, ToyRetailEnv().state)
        tmp = tempfile.mkdtemp()

        def crash(t, s):
            if (t, s) == (tag, step):
                raise SimulatedCrash(t)

        with pytest.raises(SimulatedCrash):
            await run_worker_step(coord, snaps, snaps, reg, ScriptedProvider(list(SCRIPT)),
                                  ToyRetailEnv.restore, tmp, owner="A", clock=clock, sleep=_no_sleep,
                                  debug_hook=crash)
        clock.advance(31)
        res, env = await run_worker_step(coord, snaps, snaps, reg, ScriptedProvider(SCRIPT[step:]),
                                         ToyRetailEnv.restore, tmp, owner="B", clock=clock, sleep=_no_sleep)
        assert res.status == "completed" and grade(env)
        assert store.used == {"per-run|r1|-": 2}, (tag, step, store.used)   # two tool calls, two units
    asyncio.run(go_())


# ---------------------------------------------------------------------------
# Postgres tier
# ---------------------------------------------------------------------------
async def _pg():
    asyncpg = pytest.importorskip("asyncpg", reason="needs the [dist] extra")
    from rewind.platform import PgQuotaStore
    try:
        store = await PgQuotaStore.connect(DSN)
    except (OSError, asyncpg.PostgresError, ConnectionError) as e:
        pytest.skip(f"Postgres not reachable at {DSN}: {e}")
    async with store.pool.acquire() as conn:
        await conn.execute("TRUNCATE quota_buckets, quota_charges")
    return store


async def _used(store, key):
    async with store.pool.acquire() as conn:
        return await conn.fetchval("SELECT used FROM quota_buckets WHERE key=$1", key)


def test_pg_concurrent_charges_never_overspend():
    async def go_():
        store = await _pg()
        try:
            rs = await asyncio.gather(*(store.charge(f"c{i}", [C("hot", 5)], 0.0) for i in range(20)))
            assert sum(r.ok for r in rs) == 5 and await _used(store, "hot") == 5
            assert {r.exceeded for r in rs if not r.ok} == {"hot"}
        finally:
            await store.close()
    asyncio.run(go_())


def test_pg_same_charge_id_is_charged_once_under_concurrency():
    async def go_():
        store = await _pg()
        try:
            rs = await asyncio.gather(*(store.charge("run:s1:c0", [C("a", 100)], 0.0) for _ in range(10)))
            assert all(r.ok for r in rs) and sum(not r.deduplicated for r in rs) == 1
            assert await _used(store, "a") == 1
        finally:
            await store.close()
    asyncio.run(go_())


def test_pg_all_or_nothing_and_recorded_refusal():
    async def go_():
        store = await _pg()
        try:
            assert (await store.charge("x0", [C("b", 1)], 0.0)).ok
            r = await store.charge("x1", [C("a", 5), C("b", 1, reset=60.0)], 15.0)
            assert (r.ok, r.exceeded, r.retry_after) == (False, "b", 45.0)
            assert not await _used(store, "a")                 # rolled back to the savepoint (row too)
            again = await store.charge("x1", [C("a", 5)], 15.0)
            assert (again.ok, again.deduplicated) == (False, True)
        finally:
            await store.close()
    asyncio.run(go_())


def test_pg_store_drives_the_loop():
    async def go_():
        store = await _pg()
        try:
            reg, _ = with_quota([rule()], store)
            from rewind import AgentRunner, TraceWriter
            cfg = AgentConfig(system_prompt="s", principal=principal().model_dump(mode="json"))
            runner = AgentRunner(ScriptedProvider(lists("u1", "u2", "u3") + [text("d")]), reg, cfg,
                                 SnapshotStore(), sleep=_no_sleep)
            trace = TraceWriter("pg-1")
            res = await runner.run(ToyRetailEnv(), "t", trace)
            assert res.status == "completed"
            assert [a["quota"]["ok"] for a in ev(trace, "authz")] == [True, True, False]
        finally:
            await store.close()
    asyncio.run(go_())
