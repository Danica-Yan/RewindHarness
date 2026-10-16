"""Tests for the counterfactual root-cause attribution algorithm.

Split to match the module: `bisect_attribute` is tested against a fake,
fully-controlled oracle (no agents involved, so we can construct exactly
the monotonic / non-monotonic / all-fail cases and count oracle calls);
`flip_rate_via_fork` is tested end-to-end against the real toy env, reusing
the same bad trace fork.py's tests already use.
"""
from __future__ import annotations

import asyncio

import pytest
from conftest import call, run, text

from rewind import (AgentConfig, AttributionResult, ForkPatch, ScriptedProvider, SnapshotStore,
                    TraceWriter, bisect_attribute, flip_rate_via_fork)
from rewind.env import ToyRetailEnv, build_registry, grade
from rewind.loop import AgentRunner

CANCEL_O1 = dict(order_id="O1", reason="ordered_by_mistake")


# ---------------------------------------------------------------------------
# pure bisection algorithm: fake oracle, no agents
# ---------------------------------------------------------------------------

def counting_oracle(rates: dict[int, float]):
    calls = {"n": 0, "asked": []}

    async def flip_rate(step: int) -> float:
        calls["n"] += 1
        calls["asked"].append(step)
        return rates[step]

    return flip_rate, calls


def test_finds_earliest_flipping_step_monotonic():
    async def go():
        candidates = [1, 2, 3, 4, 5]
        rates = {1: 0.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 1.0}
        oracle, calls = counting_oracle(rates)
        res = await bisect_attribute(candidates, oracle, threshold=0.66)
        assert res.root_step == 3
        assert res.monotonic is True
        # O(log N) oracle calls, not O(N): 5 candidates should need far fewer than 5 calls
        assert calls["n"] < len(candidates)
        assert calls["n"] == res.oracle_calls
    asyncio.run(go())


def test_no_candidate_flips_returns_none():
    async def go():
        candidates = [1, 2, 3]
        rates = {1: 0.0, 2: 0.0, 3: 0.33}
        oracle, calls = counting_oracle(rates)
        res = await bisect_attribute(candidates, oracle, threshold=0.66)
        assert res.root_step is None
        assert calls["n"] == 1          # only ever needs to check the last candidate
    asyncio.run(go())


def test_all_candidates_flip_picks_first():
    async def go():
        candidates = [1, 2, 3, 4]
        rates = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}
        oracle, _ = counting_oracle(rates)
        res = await bisect_attribute(candidates, oracle, threshold=0.66)
        assert res.root_step == 1
    asyncio.run(go())


def test_empty_candidates_returns_none():
    async def go():
        async def oracle(step):
            raise AssertionError("should never be called")
        res = await bisect_attribute([], oracle)
        assert res.root_step is None and res.oracle_calls == 0
    asyncio.run(go())


def test_nonmonotonic_trace_flags_and_corrects():
    """Two independent mistakes: naive bisection over [1..5] would land on
    whichever one it happens to probe first; the neighborhood re-check must
    catch that an earlier candidate *also* flips and report the earlier one,
    with monotonic=False so callers know not to trust "earliest" blindly."""
    async def go():
        candidates = [1, 2, 3, 4, 5]
        # both step 2 and step 4 independently flip the outcome; nothing
        # about 1..2 vs 3..5 is a clean threshold, so bisection's initial
        # probe can land anywhere >= 2, and the neighborhood check must
        # still recover the true earliest flip point (2) when it does.
        rates = {1: 0.0, 2: 1.0, 3: 0.0, 4: 1.0, 5: 1.0}
        oracle, _ = counting_oracle(rates)
        res = await bisect_attribute(candidates, oracle, threshold=0.66, verify_neighborhood=3)
        assert res.root_step == 2
        assert res.monotonic is False
    asyncio.run(go())


def test_neighborhood_window_bounds_the_correction():
    """If the earlier flip is outside the verify window, bisection reports
    its (wrong, non-earliest) answer honestly rather than silently missing
    it — but still flags nothing false, since within its window it found
    no contradiction. This documents the real limitation named in the plan:
    verify_neighborhood trades completeness for O(log N) + O(window) cost."""
    async def go():
        candidates = [1, 2, 3, 4, 5, 6, 7, 8]
        rates = {i: 0.0 for i in candidates}
        rates[1] = 1.0   # true earliest flip, far outside a narrow window
        rates[6] = 1.0
        rates[7] = rates[8] = 1.0
        oracle, _ = counting_oracle(rates)
        res = await bisect_attribute(candidates, oracle, threshold=0.66, verify_neighborhood=1)
        assert res.root_step == 6          # bisection's own probe, window too narrow to reach step 1
        assert res.monotonic is True        # no contradiction found *within* the window it checked
    asyncio.run(go())


# ---------------------------------------------------------------------------
# integration: real fork_run against the toy env
# ---------------------------------------------------------------------------

BAD = [call("list_orders", user_id="u1"),
      call("cancel_order", order_id="O2", reason="ordered_by_mistake"),   # O2 is shipped -> fails
      text("Sorry, I could not cancel your order.")]


async def _bad_trace():
    snaps = SnapshotStore()
    registry = build_registry()
    cfg = AgentConfig(system_prompt="sys")
    trace = TraceWriter("bad-1")
    env = ToyRetailEnv()
    await AgentRunner(ScriptedProvider(list(BAD)), registry, cfg, snaps).run(env, "cancel my order", trace)
    assert not grade(env)
    return trace.events, snaps, registry, cfg


def test_flip_rate_via_fork_distinguishes_real_root_cause_from_noop():
    async def go():
        events, snaps, registry, cfg = await _bad_trace()

        def patch_for(step: int) -> ForkPatch | None:
            if step == 1:
                return ForkPatch(tool_calls=[{"name": "list_orders", "args": {"user_id": "u1"}}])
            if step == 2:
                return ForkPatch(tool_calls=[{"name": "cancel_order", "args": CANCEL_O1}])
            return None

        def provider_for(step: int, sample_idx: int):
            # "Fixing" step 1 changes nothing real, so a faithful re-ask of the
            # model would repeat the same mistake at step 2; "fixing" step 2
            # is the actual correction, so the model just needs to wrap up.
            if step == 1:
                return ScriptedProvider([call("cancel_order", order_id="O2",
                                              reason="ordered_by_mistake"), text("Sorry.")])
            return ScriptedProvider([text("Done.")])

        async def flip_rate(step: int) -> float:
            return await flip_rate_via_fork(events, snaps, registry, cfg, ToyRetailEnv.restore,
                                            grade, step, patch_for(step), provider_for, n_samples=2)

        r1 = await flip_rate(1)
        r2 = await flip_rate(2)
        assert r1 == 0.0
        assert r2 == 1.0

        res = await bisect_attribute([1, 2], flip_rate, threshold=0.66)
        assert res.root_step == 2
        assert res.flip_rates == {1: 0.0, 2: 1.0}
    asyncio.run(go())


def test_flip_rate_via_fork_samples_are_independent_runs():
    """n_samples must not share state across forks (each is its own trace/env)."""
    async def go():
        events, snaps, registry, cfg = await _bad_trace()
        patch = ForkPatch(tool_calls=[{"name": "cancel_order", "args": CANCEL_O1}])

        def provider_for(step, i):
            return ScriptedProvider([text(f"done #{i}")])

        rate = await flip_rate_via_fork(events, snaps, registry, cfg, ToyRetailEnv.restore, grade,
                                        2, patch, provider_for, n_samples=4)
        assert rate == 1.0
    asyncio.run(go())
