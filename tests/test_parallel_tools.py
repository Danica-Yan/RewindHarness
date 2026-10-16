"""Parallel tool-call execution within a single agent step (W1-followup-era
addition after W1-W8 -- see the "next steps" ablation list in
the dev log, which flagged parallel tool calls as not yet done).

The real motivation: a tool-calling LLM can put several tool calls in ONE
turn (it generated all of them before seeing any of their results, so none
can depend on another's outcome). `loop.py`'s original `_loop` executed
them one at a time regardless -- correct, but it pays every call's latency
serially even when several are independent reads. `AgentRunner._run_tool_batch`
now runs 2+ NON-MUTATING calls from the same turn concurrently (their actual
work is dispatched to a thread pool via `asyncio.to_thread` inside
`_tool_step`), while MUTATING calls always stay sequential in their original
relative order -- a later mutation can depend on an earlier one's effect on
`env` (see `scripts/taubench_round1_wiki.patch` for a real tau-bench example
of exactly this: modifying items before address permanently locks the order).

These tests are entirely offline (`ToyRetailEnv` + `ScriptedProvider`, no
real LLM/API key needed) -- same spirit as the rest of this repo's harness
tests. The latency assertions use REAL `time.sleep` inside a tool function
(dispatched to a worker thread), because a fake/injected clock can't prove
two things genuinely overlapped in wall-clock time -- it can only prove the
harness *thinks* time passed, which is exactly what a real-concurrency claim
must not rely on. The scale note: `max_concurrency` is bounded by Python's
default thread pool (and a step's own tool-call count) rather than any limit
coded here -- a step with a very large fan-out of reads would need an
explicit semaphore to cap worker-thread usage, which this repo does not need
yet since no real task has shown steps with more than a handful of tool
calls.
"""
from __future__ import annotations

import asyncio
import time

import pytest
from conftest import Sleeps, call, run, text

from rewind import AgentConfig, AgentRunner, LLMResponse, SnapshotStore, ToolCall, TraceWriter
from rewind.env import ToyRetailEnv, build_registry
from rewind.tools import Tool, ToolRegistry
from pydantic import BaseModel


def multi_call(*pairs):
    """pairs: (name, args_dict) tuples -- several tool calls in ONE turn."""
    return LLMResponse(tool_calls=[ToolCall(name=n, args=a) for n, a in pairs],
                       usage={"input_tokens": 10, "output_tokens": 5})


def slow_registry(delay: float):
    """build_registry()'s tools, but get_order/list_orders each really sleep
    `delay` seconds (via time.sleep, so it genuinely blocks only the worker
    thread it runs on, not the event loop) -- enough to make a concurrency
    win measurable in wall-clock time, not just in event counts."""
    reg = build_registry()
    for name in ("get_order", "list_orders"):
        orig = reg._tools[name].fn

        def wrapped(env, a, _orig=orig):
            time.sleep(delay)
            return _orig(env, a)

        reg._tools[name].fn = wrapped
    return reg


def make_runner(provider, registry, cfg=None):
    snaps = SnapshotStore()
    runner = AgentRunner(provider, registry, cfg or AgentConfig(system_prompt="sys"),
                         snaps, sleep=Sleeps())
    return runner, ToyRetailEnv(), TraceWriter("t-1"), snaps


class _FakeScripted:
    """Like ScriptedProvider but takes a pre-built list of LLMResponse
    (ScriptedProvider itself is fine for this -- just spelling it out here
    for readability of what each step returns)."""
    def __init__(self, script):
        self.script = list(script)

    async def complete(self, messages, tools, params):
        return self.script.pop(0)


DELAY = 0.15


def test_three_readonly_calls_in_one_turn_run_concurrently_not_serially():
    registry = slow_registry(DELAY)
    resp = multi_call(("get_order", {"order_id": "O1"}), ("get_order", {"order_id": "O2"}),
                      ("get_order", {"order_id": "O3"}))
    runner, env, trace, _snaps = make_runner(_FakeScripted([resp, text("done")]), registry)

    t0 = time.perf_counter()
    res = run(runner.run(env, "look up my orders", trace))
    elapsed = time.perf_counter() - t0

    assert res.status == "completed"
    # Sequential would cost >= 3*DELAY (~0.45s); concurrent costs ~1*DELAY. A
    # generous cutoff of 2*DELAY keeps this robust on a slow/loaded CI box
    # while still failing hard if the batch silently went back to serial.
    assert elapsed < 2 * DELAY, f"expected the 3 reads to overlap, took {elapsed:.3f}s"

    tool_events = [e for e in trace.events if e.type == "tool_call"]
    assert len(tool_events) == 3
    assert all(e.data["ok"] for e in tool_events)
    # One checkpoint-equivalent hook for the whole concurrent batch, not one
    # per call -- see debug_hook assertion in the hook-sequence test below for
    # the explicit version of this; here just confirm all 3 actually ran.
    ids = {e.data["call_id"] for e in tool_events}
    assert len(ids) == 3


def test_readonly_batch_outcomes_and_messages_preserve_original_call_order():
    registry = slow_registry(0.01)
    resp = multi_call(("get_order", {"order_id": "O2"}), ("get_order", {"order_id": "O1"}),
                      ("get_order", {"order_id": "O3"}))
    runner, env, trace, _snaps = make_runner(_FakeScripted([resp, text("done")]), registry)
    run(runner.run(env, "t", trace))

    tool_msgs = [m for e in trace.events if e.type == "message"
                for m in [e.data["message"]] if m["role"] == "tool"]
    # Despite running concurrently (and possibly finishing in any real-time
    # order), the three tool-result messages must land in the SAME order the
    # model asked for them in -- each one's own content has to match ITS call
    # (O2's order, then O1's, then O3's), not just "three results, order TBD".
    assert "O2" in tool_msgs[0]["content"]
    assert "O1" in tool_msgs[1]["content"]
    assert "O3" in tool_msgs[2]["content"]


def test_single_tool_call_step_keeps_original_hook_sequence():
    """0 or 1 non-mutating calls in a step -- including every existing
    demo/test scenario -- must take the exact old path: "after_tool_call"
    per call, never the new "after_readonly_batch" hook. This is what makes
    this feature safe to add without touching test_checkpoint.py's exhaustive
    crash-point coverage (verified separately by that suite still passing)."""
    registry = build_registry()
    script = [call("list_orders", user_id="u1"), call("cancel_order", order_id="O1",
                                                       reason="ordered_by_mistake"), text("done")]
    points = []
    snaps = SnapshotStore()
    runner = AgentRunner(_FakeScripted(script), registry, AgentConfig(system_prompt="sys"),
                         snaps, sleep=Sleeps(), debug_hook=lambda tag, step: points.append(tag))
    res = run(runner.run(ToyRetailEnv(), "t", TraceWriter("t-1")))
    assert res.status == "completed"
    assert points == ["after_intent_checkpoint", "after_tool_call", "after_step_checkpoint",
                      "after_intent_checkpoint", "after_tool_call", "after_step_checkpoint",
                      "after_intent_checkpoint"]


class _OrderedArgs(BaseModel):
    pass


def two_mutating_tool_registry():
    """Two toy mutating tools, B's correctness depends on A having already
    run (B reads a flag A sets) -- a minimal stand-in for the real
    address-before-items ordering dependency `taubench_round1_wiki.patch`
    fixes in tau-bench. Proves the mutating batch really does stay
    sequential in original order, not just "happens to work"."""
    reg = ToolRegistry()

    @reg.tool("step_a", "set the flag", _OrderedArgs, mutating=True)
    def step_a(env, _a):
        env.state["flag_set"] = True
        return {"ok": True}

    @reg.tool("step_b", "requires the flag already set", _OrderedArgs, mutating=True)
    def step_b(env, _a):
        from rewind.tools import ToolError
        if not env.state.get("flag_set"):
            raise ToolError("flag not set yet -- step_a must run first")
        env.state["b_ran_after_a"] = True
        return {"ok": True}

    return reg


def test_mutating_calls_in_one_turn_stay_sequential_in_original_order():
    registry = two_mutating_tool_registry()
    resp = multi_call(("step_a", {}), ("step_b", {}))
    points = []
    snaps = SnapshotStore()
    runner = AgentRunner(_FakeScripted([resp, text("done")]), registry,
                         AgentConfig(system_prompt="sys"), snaps, sleep=Sleeps(),
                         debug_hook=lambda tag, step: points.append(tag))
    env = ToyRetailEnv()
    res = run(runner.run(env, "t", TraceWriter("t-1")))

    assert res.status == "completed"
    assert env.state.get("b_ran_after_a") is True
    # two mutating calls -> two individual "after_tool_call" hooks, never the
    # concurrent-batch hook (that one is reserved for non-mutating calls).
    assert points.count("after_tool_call") == 2
    assert "after_readonly_batch" not in points


def test_mixed_batch_runs_readonly_concurrently_and_mutating_sequentially():
    registry = slow_registry(DELAY)
    resp = multi_call(("get_order", {"order_id": "O1"}), ("get_order", {"order_id": "O2"}),
                      ("cancel_order", {"order_id": "O1", "reason": "ordered_by_mistake"}))
    points = []
    snaps = SnapshotStore()
    runner = AgentRunner(_FakeScripted([resp, text("done")]), registry,
                         AgentConfig(system_prompt="sys"), snaps, sleep=Sleeps(),
                         debug_hook=lambda tag, step: points.append(tag))
    env = ToyRetailEnv()

    t0 = time.perf_counter()
    res = run(runner.run(env, "t", TraceWriter("t-1")))
    elapsed = time.perf_counter() - t0

    assert res.status == "completed"
    assert env.state["orders"]["O1"]["status"] == "cancelled"
    assert elapsed < 2 * DELAY, f"the 2 reads should have overlapped, took {elapsed:.3f}s"
    # one batch hook for the 2 concurrent reads, one per-call hook for the
    # single mutating call -- never 3 separate "after_tool_call" hooks.
    assert points.count("after_readonly_batch") == 1
    assert points.count("after_tool_call") == 1
