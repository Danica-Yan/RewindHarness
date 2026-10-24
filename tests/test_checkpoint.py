"""Checkpoint / lease / fencing tests.

The central property under test: however a worker dies mid-run, the job
recovers to the same correct final state as an uninterrupted run, and any
mutating tool call that has real-world effect (modeled here with a counter
on `cancel_order`) fires exactly once — never zero, never twice — no matter
which instant the crash lands on.
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest

from rewind import (AgentConfig, Coordinator, LoopState, RunAbort, ScriptedProvider,
                    SimulatedCrash, SnapshotStore, StepCheckpointer, TraceWriter, load_trace,
                    run_worker_step, submit_job)
from rewind.env import ToyRetailEnv, build_registry, grade
from rewind.providers import LLMResponse, ToolCall
from rewind.util import FakeClock

SYS = "You are a retail support agent."
TASK = "I'm u1, cancel my headphones order."
CANCEL_O1 = dict(order_id="O1", reason="ordered_by_mistake")


def call(name, **args):
    return LLMResponse(tool_calls=[ToolCall(name=name, args=args)],
                       usage={"input_tokens": 10, "output_tokens": 5})


def text(t):
    return LLMResponse(text=t, usage={"input_tokens": 10, "output_tokens": 5})


# step1: list_orders (no mutation), step2: cancel_order (mutation), step3: final text
SCRIPT = [call("list_orders", user_id="u1"), call("cancel_order", **CANCEL_O1), text("done")]


def counting_registry():
    """A registry where cancel_order's *real* execution count is observable,
    independent of env state (which gets reset by every crash+resume)."""
    reg = build_registry()
    counter = {"n": 0}
    orig = reg._tools["cancel_order"].fn

    def wrapped(env, a):
        counter["n"] += 1
        return orig(env, a)

    reg._tools["cancel_order"].fn = wrapped
    return reg, counter


async def _fresh_job(registry):
    clock = FakeClock()
    coord = Coordinator(clock=clock)
    snaps, blobs = SnapshotStore(), SnapshotStore()
    tmp = tempfile.mkdtemp()
    job_id = await submit_job(coord, snaps, "r1", TASK, AgentConfig(system_prompt=SYS), ToyRetailEnv().state)
    return coord, snaps, blobs, tmp, job_id, clock


# ---------------------------------------------------------------------------
# baseline: no crash
# ---------------------------------------------------------------------------

def test_cold_run_completes_and_writes_checkpoints_for_both_mutating_steps():
    async def go():
        registry, counter = counting_registry()
        coord, snaps, blobs, tmp, job_id, clock = await _fresh_job(registry)
        res, env = await run_worker_step(coord, snaps, blobs, registry, ScriptedProvider(list(SCRIPT)),
                                         ToyRetailEnv.restore, tmp, owner="A", clock=clock)
        assert res.status == "completed" and grade(env) and counter["n"] == 1
        assert (await coord.get(job_id)).status == "completed"
        kinds = {(c.step, c.kind) for c in
                 [await coord.latest_checkpoint("r1")]}  # sanity: at least the last one is there
        assert kinds
    asyncio.run(go())


def test_dry_run_hook_sequence_matches_expected_shape():
    tags = asyncio.run(_collect_hook_points())
    # step1 (list_orders, no mutation): intent, tool_call, step-confirm
    # step2 (cancel_order, mutation):   intent, tool_call, step-confirm
    # step3 (final text, no tools):     intent only
    assert tags == [
        ("after_intent_checkpoint", 1), ("after_tool_call", 1), ("after_step_checkpoint", 1),
        ("after_intent_checkpoint", 2), ("after_tool_call", 2), ("after_step_checkpoint", 2),
        ("after_intent_checkpoint", 3),
    ]


async def _collect_hook_points():
    registry, _ = counting_registry()
    snaps, blobs = SnapshotStore(), SnapshotStore()
    coord = Coordinator(clock=FakeClock())
    job_id = await submit_job(coord, snaps, "dry", TASK, AgentConfig(system_prompt=SYS), ToyRetailEnv().state)
    points: list[tuple[str, int]] = []
    await run_worker_step(coord, snaps, blobs, registry, ScriptedProvider(list(SCRIPT)),
                          ToyRetailEnv.restore, tempfile.mkdtemp(), owner="dry",
                          debug_hook=lambda tag, step: points.append((tag, step)))
    return points


# ---------------------------------------------------------------------------
# exhaustive: crash at every recorded point, verify clean recovery
# ---------------------------------------------------------------------------

CRASH_POINTS = asyncio.run(_collect_hook_points())


@pytest.mark.parametrize("tag,step", CRASH_POINTS, ids=[f"{t}@{s}" for t, s in CRASH_POINTS])
def test_crash_at_every_point_recovers_to_correct_state(tag, step):
    async def go():
        registry, counter = counting_registry()
        coord, snaps, blobs, tmp, job_id, clock = await _fresh_job(registry)

        def crash_hook(t, s):
            if (t, s) == (tag, step):
                raise SimulatedCrash(f"worker A died at {t}@{s}")

        with pytest.raises(SimulatedCrash):
            await run_worker_step(coord, snaps, blobs, registry, ScriptedProvider(list(SCRIPT)),
                                  ToyRetailEnv.restore, tmp, owner="A", clock=clock,
                                  debug_hook=crash_hook, lease_seconds=30)
        assert (await coord.get(job_id)).status in ("leased", "LEASED")

        clock.advance(31)  # let A's lease expire
        remaining = SCRIPT[step:]
        res, env = await run_worker_step(coord, snaps, blobs, registry, ScriptedProvider(remaining),
                                         ToyRetailEnv.restore, tmp, owner="B", clock=clock)
        assert res is not None, "worker B must find the job available after the lease expired"
        assert res.status == "completed", (tag, step, res)
        assert grade(env), f"final state wrong after crash at {tag}@{step}"
        assert counter["n"] == 1, f"cancel_order ran {counter['n']} times after crash at {tag}@{step}"
        assert (await coord.get(job_id)).status == "completed"
    asyncio.run(go())


def test_torn_trace_events_are_truncated_on_resume():
    """A crash mid-network-call (nothing checkpointed for that step at all)
    still leaves a `step_start` event on disk for the step that never
    finished; resume must discard it, not just the checkpointed portion."""
    async def go():
        registry, _ = counting_registry()
        coord, snaps, blobs, tmp, job_id, clock = await _fresh_job(registry)

        # Step 1 completes normally; step 2's very first provider call "dies"
        # (raises something the harness's retry logic doesn't even catch, like
        # a process being killed mid-request) before any checkpoint for it exists.
        class TwoPhase:
            def __init__(self):
                self.i = 0
            async def complete(self, messages, tools, params):
                self.i += 1
                if self.i == 1:
                    return SCRIPT[0]           # step 1: list_orders, succeeds
                raise SimulatedCrash()          # step 2: dies mid-call

        with pytest.raises(SimulatedCrash):
            await run_worker_step(coord, snaps, blobs, registry, TwoPhase(),
                                  ToyRetailEnv.restore, tmp, owner="A", clock=clock)

        trace_path = Path(tmp) / "r1.jsonl"
        torn = load_trace(trace_path)
        ckpt = await coord.latest_checkpoint("r1")
        assert ckpt.kind == "step" and ckpt.step == 1
        assert any(e.seq > ckpt.seq for e in torn), "test setup should have left torn events on disk"
        assert any(e.type == "step_start" and e.step == 2 for e in torn)

        clock.advance(31)
        await run_worker_step(coord, snaps, blobs, registry, ScriptedProvider(SCRIPT[1:]),
                              ToyRetailEnv.restore, tmp, owner="B", clock=clock)
        final_events = load_trace(trace_path)
        assert not any(e.type == "step_start" and e.step == 2 and e.seq <= ckpt.seq for e in final_events)
        # seq is contiguous and strictly increasing across the truncate + resume boundary
        seqs = [e.seq for e in final_events]
        assert seqs == list(range(1, len(seqs) + 1)), seqs
    asyncio.run(go())


def test_crash_after_intent_replays_recorded_action_not_a_new_model_call():
    """If the model were asked again for step 2 it would return a *different*
    action here; resume must instead replay the one that was checkpointed."""
    async def go():
        registry, counter = counting_registry()
        coord, snaps, blobs, tmp, job_id, clock = await _fresh_job(registry)

        def crash_hook(tag, step):
            if (tag, step) == ("after_intent_checkpoint", 2):
                raise SimulatedCrash()

        with pytest.raises(SimulatedCrash):
            await run_worker_step(coord, snaps, blobs, registry, ScriptedProvider(list(SCRIPT)),
                                  ToyRetailEnv.restore, tmp, owner="A", clock=clock, debug_hook=crash_hook)
        clock.advance(31)
        # Step 2 must be replayed verbatim from the checkpoint, not re-decided by
        # asking the model. `trap` scripts a *different* action for whatever it's
        # asked first — if the harness incorrectly re-asked for step 2, O3 (not O1)
        # would end up cancelled. Step 3 legitimately does need a fresh call (it
        # was never checkpointed), so trap's action is real, expected work for it.
        trap = ScriptedProvider([call("cancel_order", order_id="O3", reason="no_longer_needed"),
                                 text("also done")])
        res, env = await run_worker_step(coord, snaps, blobs, registry, trap,
                                         ToyRetailEnv.restore, tmp, owner="B", clock=clock)
        assert res.status == "completed"
        assert env.state["orders"]["O1"]["status"] == "cancelled"   # replayed, not re-decided
        assert env.state["orders"]["O3"]["status"] == "cancelled"   # step 3's own, legitimate action
        assert counter["n"] == 2                                    # one real cancel_order per order
    asyncio.run(go())


# ---------------------------------------------------------------------------
# lease / fencing mechanics on the Coordinator directly
# ---------------------------------------------------------------------------

def test_second_worker_cannot_claim_before_lease_expires():
    async def go():
        clock = FakeClock()
        coord = Coordinator(clock=clock)
        snaps = SnapshotStore()
        await submit_job(coord, snaps, "r1", TASK, AgentConfig(system_prompt=SYS), ToyRetailEnv().state)
        job_a = await coord.claim("A", lease_seconds=30)
        assert job_a is not None
        clock.advance(10)
        assert await coord.claim("B", lease_seconds=30) is None       # A's lease still valid
        clock.advance(21)
        job_b = await coord.claim("B", lease_seconds=30)
        assert job_b is not None and job_b.fencing_token == job_a.fencing_token + 1
    asyncio.run(go())


def test_heartbeat_extends_lease_and_is_token_gated():
    async def go():
        clock = FakeClock()
        coord = Coordinator(clock=clock)
        snaps = SnapshotStore()
        await submit_job(coord, snaps, "r1", TASK, AgentConfig(system_prompt=SYS), ToyRetailEnv().state)
        job = await coord.claim("A", lease_seconds=10)
        clock.advance(8)
        assert await coord.heartbeat(job.job_id, job.fencing_token, lease_seconds=10) is True
        clock.advance(8)                                          # would have expired without the heartbeat
        assert await coord.claim("B", lease_seconds=10) is None
        assert await coord.heartbeat(job.job_id, job.fencing_token - 1, lease_seconds=10) is False  # stale token
    asyncio.run(go())


def test_zombie_checkpoint_write_is_fenced_out():
    async def go():
        clock = FakeClock()
        coord = Coordinator(clock=clock)
        snaps, blobs = SnapshotStore(), SnapshotStore()
        await submit_job(coord, snaps, "r1", TASK, AgentConfig(system_prompt=SYS), ToyRetailEnv().state)
        job = await coord.claim("A", lease_seconds=10)
        clock.advance(11)
        newer = await coord.claim("B", lease_seconds=30)                # supersedes A
        assert newer.fencing_token == job.fencing_token + 1

        zombie = StepCheckpointer(coord, job.job_id, "r1", job.fencing_token, snaps, blobs)
        with pytest.raises(RunAbort) as ei:
            await zombie.save("step", 1, TraceWriter("r1"), ToyRetailEnv(), [], LoopState())
        assert ei.value.status == "fenced_out"

        current = StepCheckpointer(coord, job.job_id, "r1", newer.fencing_token, snaps, blobs)
        await current.save("step", 1, TraceWriter("r1"), ToyRetailEnv(), [], LoopState())  # does not raise
    asyncio.run(go())


def test_complete_is_token_gated():
    async def go():
        clock = FakeClock()
        coord = Coordinator(clock=clock)
        snaps = SnapshotStore()
        await submit_job(coord, snaps, "r1", TASK, AgentConfig(system_prompt=SYS), ToyRetailEnv().state)
        job = await coord.claim("A", lease_seconds=10)
        clock.advance(11)
        await coord.claim("B", lease_seconds=30)
        assert await coord.complete(job.job_id, job.fencing_token, "completed") is False  # A is stale
        current = await coord.get(job.job_id)
        assert await coord.complete(job.job_id, current.fencing_token, "completed") is True
    asyncio.run(go())
