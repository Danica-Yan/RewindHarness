"""Unit tests for the failure taxonomy classifier, clustering, and the
auto-paired-patch extraction — W7's "classify the full failure batch"
and "auto-pair `patch_for_step`" items, using `conftest.call/text` scenarios
that construct each category deliberately (see `scripts/failure_batch.py`
for the fuller batch version of the same idea)."""
from __future__ import annotations

import asyncio

import pytest
from conftest import call, text

from rewind import AgentConfig, AgentRunner, ScriptedProvider, SnapshotStore, TraceWriter
from rewind.env import ToyRetailEnv, build_registry, grade
from rewind.flywheel.failures import (
    auto_patch_for_step, category_distribution, classify, classify_worker_crash, cluster,
    top_clusters)
from rewind.providers import MalformedOutput, ProviderError

SYS = "You are a retail support agent."
TASK = "I'm u1, cancel my headphones order."


async def _no_sleep(_d: float) -> None:
    pass


async def _run(script, cfg=None) -> tuple[list, object, bool]:
    snaps = SnapshotStore()
    registry = build_registry()
    env = ToyRetailEnv()
    trace = TraceWriter("t")
    runner = AgentRunner(ScriptedProvider(list(script)), registry, cfg or AgentConfig(system_prompt=SYS),
                         snaps, sleep=_no_sleep)
    result = await runner.run(env, TASK, trace)
    return trace.events, result, grade(env)


@pytest.mark.parametrize("script,expected_category", [
    ([ProviderError("boom", retryable=True, status=500)] * 5, "LLM_API_ERROR"),
    ([MalformedOutput(), MalformedOutput()], "OUTPUT_MALFORMED"),
])
def test_classify_provider_level_failures(script, expected_category):
    events, result, ok = asyncio.run(_run(script))
    rec = classify(events, result, ok)
    assert rec is not None and rec.category == expected_category


def test_classify_tool_arg_invalid():
    calls = [call("cancel_order", order_id=oid, reason="not_a_real_reason") for oid in ["O1", "O2", "O3"]]
    events, result, ok = asyncio.run(_run(calls))
    rec = classify(events, result, ok)
    assert rec.category == "TOOL_ARG_INVALID"
    assert rec.root_cause_tool == "cancel_order"


def test_classify_hallucinated_tool():
    calls = [call("delete_order", order_id=oid) for oid in ["O1", "O2", "O3"]]
    events, result, ok = asyncio.run(_run(calls))
    rec = classify(events, result, ok)
    assert rec.category == "HALLUCINATED_TOOL"


def test_classify_tool_runtime_error_does_not_abort_but_is_classified():
    calls = [call("cancel_order", order_id="O2", reason="ordered_by_mistake"), text("Sorry.")]
    events, result, ok = asyncio.run(_run(calls))
    assert result.status == "completed" and not ok      # a *recoverable* error, run finishes normally
    rec = classify(events, result, ok)
    assert rec.category == "TOOL_RUNTIME_ERROR"
    assert rec.root_cause_tool == "cancel_order"


def test_classify_loop():
    same = call("get_order", order_id="O1")
    events, result, ok = asyncio.run(_run([same] * 6))
    rec = classify(events, result, ok)
    assert rec.category == "LOOP"


def test_classify_budget_exceeded():
    script = [call("get_order", order_id=x) for x in ["O1", "O2", "O3"]] + [text("done")]
    events, result, ok = asyncio.run(_run(script, cfg=AgentConfig(system_prompt=SYS, max_steps=2)))
    rec = classify(events, result, ok)
    assert rec.category == "BUDGET_EXCEEDED"
    assert rec.error_message == "max_steps"


def test_classify_reasoning_error_when_every_call_succeeds_but_outcome_is_wrong():
    script = [call("list_orders", user_id="u1"),
             call("cancel_order", order_id="O3", reason="ordered_by_mistake"), text("Done.")]
    events, result, ok = asyncio.run(_run(script))
    assert result.status == "completed" and not ok
    rec = classify(events, result, ok)
    assert rec.category == "REASONING_ERROR"
    assert rec.root_cause_tool == "cancel_order"


def test_classify_returns_none_for_an_actually_successful_run():
    script = [call("list_orders", user_id="u1"),
             call("cancel_order", order_id="O1", reason="ordered_by_mistake"), text("Done.")]
    events, result, ok = asyncio.run(_run(script))
    assert ok is True
    assert classify(events, result, ok) is None


def test_classify_worker_crash_is_a_constructed_record_not_derived_from_trace():
    rec = classify_worker_crash("run-x")
    assert rec.category == "WORKER_CRASH" and rec.run_id == "run-x"


def test_cluster_groups_by_category_and_tool():
    records = [classify_worker_crash("a"), classify_worker_crash("b")]
    script = [call("cancel_order", order_id="O2", reason="ordered_by_mistake"), text("Sorry.")]
    events, result, ok = asyncio.run(_run(script))
    records.append(classify(events, result, ok))

    buckets = cluster(records)
    assert buckets[("WORKER_CRASH", None)] == records[:2]
    assert len(buckets[("TOOL_RUNTIME_ERROR", "cancel_order")]) == 1

    top = top_clusters(records, n=5)
    assert top[0] == (("WORKER_CRASH", None), 2)

    dist = category_distribution(records)
    assert dist["WORKER_CRASH"] == 2 and dist["TOOL_RUNTIME_ERROR"] == 1


def test_auto_patch_for_step_pulls_the_action_from_a_reference_trace():
    good_script = [call("list_orders", user_id="u1"),
                   call("cancel_order", order_id="O1", reason="ordered_by_mistake"), text("Done.")]
    events, _result, ok = asyncio.run(_run(good_script))
    assert ok is True

    patch1 = auto_patch_for_step(events, 1)
    assert patch1 is not None and patch1.tool_calls == [{"name": "list_orders", "args": {"user_id": "u1"}}]

    patch2 = auto_patch_for_step(events, 2)
    assert patch2 is not None
    assert patch2.tool_calls == [{"name": "cancel_order",
                                  "args": {"order_id": "O1", "reason": "ordered_by_mistake"}}]


def test_auto_patch_for_step_returns_none_past_the_end_of_the_reference_trace():
    good_script = [call("cancel_order", order_id="O1", reason="ordered_by_mistake"), text("Done.")]
    events, _result, ok = asyncio.run(_run(good_script))
    assert ok is True
    assert auto_patch_for_step(events, 99) is None
