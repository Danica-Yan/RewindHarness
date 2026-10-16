"""W1 followup: the real tau-bench retail-domain adapter.

Needs the optional `tau_bench` dependency (not on PyPI -- see
`env/taubench_adapter.py`'s module docstring for the install command, or set
REWIND_TAUBENCH_SRC to a clone). Skips with a clear reason, not a failure,
if it isn't importable -- same pattern as the Postgres/Redis/chaos tiers.

These tests do NOT need an LLM: they exercise the golden-trajectory sanity
check (replay the task's own recorded ground truth, grade with tau-bench's
real reward convention), which is the only thing honestly runnable without
an API key. See that module's docstring for why this is not a baseline.
"""
from __future__ import annotations

import asyncio

import pytest

from rewind.env.taubench_adapter import (
    TauBenchRetailEnv, _ensure_importable, build_registry, golden_provider, load_tasks,
    replay_grade, unsupported_reason,
)

try:
    _ensure_importable()  # actually probes for tau_bench, unlike the plain import above
    _IMPORT_ERROR = None
except ImportError as e:
    _IMPORT_ERROR = e

from rewind.loop import AgentConfig, AgentRunner
from rewind.providers import LLMResponse, ScriptedProvider
from rewind.snapshot import SnapshotStore
from rewind.trace import TraceWriter

pytestmark = pytest.mark.skipif(
    _IMPORT_ERROR is not None,
    reason=f"tau_bench not importable: {_IMPORT_ERROR}")


def _run(coro):
    return asyncio.run(coro)


async def _run_golden(task, registry):
    env = TauBenchRetailEnv()
    runner = AgentRunner(golden_provider(task), registry, AgentConfig(system_prompt="x", max_steps=30),
                         SnapshotStore())
    result = await runner.run(env, task.instruction, TraceWriter("t"))
    return result, env


def test_build_registry_has_all_16_real_retail_tools():
    reg = build_registry()
    names = {s["name"] for s in reg.schemas()}
    assert len(names) == 16
    assert "cancel_pending_order" in names and "get_order_details" in names


def test_snapshot_restore_round_trips_real_tau_bench_data():
    env = TauBenchRetailEnv()
    before = env.db_hash()
    snap = env.snapshot()
    env.data["orders"]["#W0000000"] = {"made up": True}  # mutate the live copy
    assert env.db_hash() != before
    restored = TauBenchRetailEnv.restore(snap)
    assert restored.db_hash() == before


def test_golden_trajectory_passes_replay_grade_on_real_train_tasks():
    """Sample across the real train split (not all 500 -- each task deep-
    copies a ~1550-record DB per step, so the full split is a ~9 minute
    script (scripts/taubench_baseline.py), not a unit test). 500/500 passed
    when actually run in full; this is the fast regression-sized sample."""
    reg = build_registry()
    tasks = load_tasks("train")[:8]
    for i, task in enumerate(tasks):
        result, env = _run(_run_golden(task, reg))
        assert result.status == "completed", f"task[{i}]: {result.status}/{result.reason}"
        assert replay_grade(env, task), f"task[{i}] failed golden-trajectory replay"


def test_dropping_a_mutating_action_fails_grading():
    """Negative control: proves replay_grade isn't trivially true. Uses a
    real train task with >=2 non-dialogue actions and drops the last one."""
    reg = build_registry()
    task = next(t for t in load_tasks("train")
               if sum(1 for a in t.actions if a.name not in ("respond", "transfer_to_human_agents")) >= 2)
    kept = [a for a in task.actions if a.name not in ("respond", "transfer_to_human_agents")][:-1]
    script = [LLMResponse(tool_calls=[{"id": f"x{i}", "name": a.name, "args": a.kwargs}])
             for i, a in enumerate(kept)]
    script.append(LLMResponse(text="done"))

    async def go():
        env = TauBenchRetailEnv()
        runner = AgentRunner(ScriptedProvider(script), reg, AgentConfig(system_prompt="x", max_steps=30),
                             SnapshotStore())
        await runner.run(env, task.instruction, TraceWriter("neg"))
        return env
    env = _run(go())
    assert not replay_grade(env, task)


def test_unsupported_reason_flags_outputs_tasks_not_train_tasks():
    train_task = load_tasks("train")[0]
    assert unsupported_reason(train_task) is None  # train tasks never carry `outputs`
    test_tasks_with_outputs = [t for t in load_tasks("test") if t.outputs]
    assert test_tasks_with_outputs
    assert unsupported_reason(test_tasks_with_outputs[0]) is not None
