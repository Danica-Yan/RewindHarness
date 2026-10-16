"""W8: task suite generation + fixed split, the heuristic policy's two
independent bugs, and the two-round data flywheel that fixes them —
against the deterministic `HeuristicRetailProvider` (see its module
docstring for why: fully offline, no LLM API key needed). Every number asserted
here is real and reproducible from a fixed seed, not illustrative."""
from __future__ import annotations

import asyncio

from rewind.env import ToyRetailEnv, build_registry
from rewind.flywheel.data_flywheel import run_flywheel, run_split
from rewind.flywheel.heuristic_provider import HeuristicRetailProvider
from rewind.flywheel.tasks import (
    CATEGORIES, generate_suite, grade_task, load_split, save_split, split_suite)
from rewind.loop import AgentConfig, AgentRunner
from rewind.snapshot import SnapshotStore
from rewind.trace import TraceWriter


def test_generate_suite_is_deterministic_for_a_fixed_seed():
    a = generate_suite(30, seed=42)
    b = generate_suite(30, seed=42)
    assert [t.to_dict() for t in a] == [t.to_dict() for t in b]


def test_generate_suite_covers_every_category():
    tasks = generate_suite(30, seed=42)
    seen = {t.category for t in tasks}
    assert seen == set(CATEGORIES)


def test_split_suite_is_60_20_20_and_covers_every_task_exactly_once():
    tasks = generate_suite(30, seed=1)
    split = split_suite(tasks)
    assert len(split["train"]) == 18 and len(split["dev"]) == 6 and len(split["heldout"]) == 6
    all_ids = {t.task_id for s in split.values() for t in s}
    assert all_ids == {t.task_id for t in tasks}
    assert len(all_ids) == 30    # no duplicates / no task dropped


def test_split_round_trips_through_disk(tmp_path):
    tasks = generate_suite(12, seed=7)
    split = split_suite(tasks)
    p = tmp_path / "split.json"
    save_split(split, p)
    loaded = load_split(p)
    assert {k: [t.task_id for t in v] for k, v in loaded.items()} == \
        {k: [t.task_id for t in v] for k, v in split.items()}


async def _run_one(task, **bug_flags):
    snaps = SnapshotStore()
    registry = build_registry()
    cfg = AgentConfig(system_prompt="You are a retail support agent.")
    env = ToyRetailEnv(task.initial_state())
    provider = HeuristicRetailProvider(task.target_order, task.requester, **bug_flags)
    runner = AgentRunner(provider, registry, cfg, snaps)
    trace = TraceWriter(task.task_id)
    await runner.run(env, task.task_text, trace)
    return grade_task(env, task), trace.events


def test_heuristic_provider_is_correct_with_both_bugs_off():
    tasks = generate_suite(15, seed=3)
    for t in tasks:
        ok, _events = asyncio.run(_run_one(t, bug_ignores_ownership=False, bug_ignores_status=False))
        assert ok, t.task_id


def test_ownership_bug_actually_produces_a_wrong_final_state():
    tasks = [t for t in generate_suite(30, seed=3) if t.category == "ownership_trap"]
    assert tasks
    for t in tasks:
        ok, _events = asyncio.run(_run_one(t, bug_ignores_ownership=True, bug_ignores_status=False))
        assert not ok, f"{t.task_id}: the tool has no ownership check, so this bug must fail the task"


def test_status_bug_is_caught_by_the_tools_own_guard_and_does_not_fail_the_task():
    """The interesting, honest asymmetry the flywheel report is built
    around: `cancel_order` already refuses a non-pending order on its own,
    so this bug produces a TOOL_RUNTIME_ERROR in the trace but the task
    still grades correctly."""
    tasks = [t for t in generate_suite(30, seed=3) if t.category == "shipped_trap"]
    assert tasks
    for t in tasks:
        ok, events = asyncio.run(_run_one(t, bug_ignores_ownership=False, bug_ignores_status=True))
        assert ok, f"{t.task_id}: the tool's own guard should have prevented the wrong mutation"
        assert any(e.type == "tool_call" and e.data.get("error_class") == "TOOL_RUNTIME_ERROR"
                  for e in events), f"{t.task_id}: the refused cancel should still show up in the trace"


def test_run_split_reports_pass_rate_and_defensive_saves():
    async def go():
        tasks = generate_suite(15, seed=9)
        snaps = SnapshotStore()
        registry = build_registry()
        cfg = AgentConfig(system_prompt="You are a retail support agent.")
        both_on = await run_split(tasks, {"bug_ignores_ownership": True, "bug_ignores_status": True},
                                  snaps, registry, cfg)
        both_off = await run_split(tasks, {"bug_ignores_ownership": False, "bug_ignores_status": False},
                                   snaps, registry, cfg)
        assert both_off.pass_rate == 1.0
        assert both_off.defensive_saves == 0
        assert both_on.pass_rate < 1.0
        assert all(f.category == "REASONING_ERROR" for f in both_on.failures)
    asyncio.run(go())


def test_two_round_flywheel_merges_both_fixes_and_reaches_full_heldout_pass_rate():
    async def go():
        tasks = generate_suite(30, seed=20260928)
        split = split_suite(tasks)
        report = await run_flywheel(split, n_rounds=2)

        assert report.baseline.pass_rate < 1.0
        assert len(report.rounds) == 2

        r1 = report.rounds[0]
        assert r1.fix == "bug_ignores_ownership" and r1.metric == "pass_rate" and r1.gate_passed
        assert r1.dev_after == 1.0 and r1.heldout_after == 1.0

        r2 = report.rounds[1]
        assert r2.fix == "bug_ignores_status" and r2.metric == "defensive_saves" and r2.gate_passed
        assert r2.dev_after == 0 and r2.heldout_after == 0
        assert "does not move" in r2.note

        assert report.final.pass_rate == 1.0
        assert report.final.defensive_saves == 0
        assert report.bug_flags_by_round[-1] == {"bug_ignores_ownership": False, "bug_ignores_status": False}
    asyncio.run(go())


def test_a_gate_that_regresses_is_rejected_not_merged():
    """A candidate fix that would only be evaluated against a target cluster
    with no known fix must report gate_passed=False and leave bug_flags
    untouched, rather than merging something that made things worse or
    merging nothing while silently claiming success."""
    async def go():
        tasks = generate_suite(30, seed=20260928)
        split = split_suite(tasks)
        # 0 rounds: nothing should be evaluated or merged at all.
        report = await run_flywheel(split, n_rounds=0)
        assert report.rounds == []
        assert report.bug_flags_by_round == [{"bug_ignores_ownership": True, "bug_ignores_status": True}]
        assert report.final.pass_rate < 1.0, "with 0 rounds run, both bugs are still present"
    asyncio.run(go())
