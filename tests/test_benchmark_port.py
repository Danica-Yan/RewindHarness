"""The Benchmark port's result-row contract and gold calls.

`check_row` is what stands between a benchmark's results and evolve, so it
must accept every shape a real benchmark produces (tested against
tau-bench's own rows below) and reject each way a row can break the
contract (tested one defect at a time)."""
from __future__ import annotations

import asyncio
import copy

import pytest

from rewind.benchmark import (
    RowContractError, check_row, row_ctx_version, row_trace, set_ctx_version, validate_row,
)

GOOD = {
    "task_id": 3, "trial": 0, "reward": 1.0,
    "traj": [{"role": "system", "content": "policy"},
             {"role": "assistant", "content": "", "tool_calls": [
                 {"id": "c1", "type": "function",
                  "function": {"name": "get_order_details", "arguments": "{\"order_id\": \"#W1\"}"}}]},
             {"role": "tool", "tool_call_id": "c1", "content": "{}"}],
    "info": {"rewind": {"trace": "runs/x.jsonl", "ctx_version": 2}, "anything_else": 1},
    "benchmark_specific_field": "is allowed",
}


def test_a_well_formed_row_passes():
    assert check_row(GOOD) == []
    assert validate_row(GOOD) is GOOD


def test_rows_from_benchmark_run_py_without_rewind_info_pass():
    row = copy.deepcopy(GOOD)
    row["info"] = {"reward_info": {}}
    assert check_row(row) == []
    assert row_trace(row) is None and row_ctx_version(row) == 0


@pytest.mark.parametrize("mutate,expect", [
    (lambda r: r.pop("task_id"), "missing 'task_id'"),
    (lambda r: r.update(trial="0"), "'trial' is str"),
    (lambda r: r.update(reward=True), "'reward' is bool"),
    (lambda r: r.pop("traj"), "missing 'traj'"),
    (lambda r: r["traj"].append("not a message"), "is not a chat message"),
    (lambda r: r["traj"][1]["tool_calls"][0].update(function={"arguments": "{}"}), "without function.name"),
    (lambda r: r.pop("info"), "missing 'info'"),
    (lambda r: r["info"].update(rewind="runs/x.jsonl"), "info.rewind is not a dict"),
    (lambda r: r["info"]["rewind"].update(trace=7), "info.rewind.trace is not a path string"),
    (lambda r: r["info"]["rewind"].update(ctx_version="2"), "ctx_version is not an int"),
], ids=["no-task_id", "trial-str", "reward-bool", "no-traj", "bad-message", "bad-tool-call",
        "no-info", "rewind-not-dict", "trace-not-str", "ctx-version-str"])
def test_each_contract_break_is_reported(mutate, expect):
    row = copy.deepcopy(GOOD)
    mutate(row)
    problems = check_row(row)
    assert any(expect in p for p in problems), problems
    with pytest.raises(RowContractError):
        validate_row(row, "test")


def test_accessors_read_and_write_the_nested_fields():
    row = copy.deepcopy(GOOD)
    assert row_trace(row) == "runs/x.jsonl" and row_ctx_version(row) == 2
    bare = {"task_id": 1, "trial": 0, "reward": 0.0, "traj": []}
    set_ctx_version(bare, 5)
    assert row_ctx_version(bare) == 5 and check_row(bare) == []


# ---------------------------------------------------------------------------
# tau-bench as a Benchmark
# ---------------------------------------------------------------------------
try:
    from rewind.env.taubench_adapter import _ensure_importable
    _ensure_importable()
    _TAU_ERROR = None
except Exception as e:  # pragma: no cover
    _TAU_ERROR = e
needs_tau = pytest.mark.skipif(_TAU_ERROR is not None, reason=f"tau_bench not importable: {_TAU_ERROR}")


@needs_tau
def test_tau_gold_calls_are_the_tool_calls_of_the_gold_solution():
    from rewind.env.taubench_benchmark import TauBenchRetail
    from rewind.env.taubench_dialogue import build_dialogue_registry, load_task, n_tasks
    tau, tools = TauBenchRetail(), set(build_dialogue_registry()._tools)  # noqa: SLF001
    for split in ("train", "test"):
        for i in range(n_tasks(split)):
            gold = tau.gold_calls(split, i)
            assert all(a.name in tools for a in gold), (split, i)
            assert len(gold) == sum(a.name != "respond" for a in load_task(split, i).actions)


def test_tau_gold_calls_drop_the_respond_pseudo_action(monkeypatch):
    """No retail task currently has a gold `respond` (it is tau-bench's
    "say this to the user" action), so check the filter on a synthetic task."""
    from types import SimpleNamespace

    import rewind.env.taubench_benchmark as tb
    A = SimpleNamespace
    task = A(instruction="x", actions=[A(name="respond", kwargs={"content": "hi"}),
                                       A(name="cancel_pending_order", kwargs={"order_id": "#W1"})])
    monkeypatch.setattr(tb, "load_task", lambda split, i: task)
    assert [a.name for a in tb.TauBenchRetail().gold_calls("train", 0)] == ["cancel_pending_order"]


@needs_tau
def test_tau_result_rows_satisfy_the_contract():
    from tests.test_taubench_dialogue import _first_task, golden_dialogue, scripted_user

    from rewind.env.taubench_benchmark import TauBenchRetail
    from rewind.experiment import run_episode
    from rewind.providers import ScriptedProvider
    from rewind.snapshot import SnapshotStore
    tau = TauBenchRetail()
    idx = _first_task("train", with_outputs=False)
    script, replies = golden_dialogue("train", idx)
    ep = asyncio.run(run_episode(tau, "train", idx, provider=ScriptedProvider(script),
                                 user_llm=scripted_user(["hi", *replies]), store=SnapshotStore()))
    row = tau.result_record(ep.env, ep.result, 0, "runs/t.jsonl", ep.events)
    assert check_row(row) == []
    assert row_trace(row) == "runs/t.jsonl" and row["reward"] == 1.0
