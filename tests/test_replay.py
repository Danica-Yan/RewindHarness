import pytest
from conftest import call, run, text

from rewind import (AgentConfig, AgentRunner, FaultInjectingProvider, LLMResponse, ProviderError,
                    ScriptedProvider, ToolCall, TraceWriter, replay_run, SnapshotStore)
from rewind.env import ToyRetailEnv, build_registry

CANCEL_O1 = dict(order_id="O1", reason="ordered_by_mistake")


def _record(make, script, **kw):
    runner, env, trace, snaps, registry = make(ScriptedProvider(script), **kw)
    run(runner.run(env, "cancel my order", trace))
    return trace.events, snaps, registry


def test_replay_ok(make):
    events, snaps, registry = _record(make, [call("list_orders", user_id="u1"),
                                             call("cancel_order", **CANCEL_O1), text("done")])
    rep = run(replay_run(events, snaps, registry, ToyRetailEnv.restore))
    assert rep.ok, rep


def test_replay_reproduces_failures_and_repairs(make):
    sleeps_provider = FaultInjectingProvider(ScriptedProvider([text("ok")]),
                                             {0: ProviderError("429", True, 429)})
    runner, env, trace, snaps, registry = make(sleeps_provider)
    run(runner.run(env, "t", trace))
    assert run(replay_run(trace.events, snaps, registry, ToyRetailEnv.restore)).ok


def test_replay_detects_prompt_drift(make):
    events, snaps, registry = _record(make, [call("list_orders", user_id="u1"), text("done")])
    rep = run(replay_run(events, snaps, registry, ToyRetailEnv.restore,
                         cfg_override={"system_prompt": "a different system prompt"}))
    assert not rep.ok and "request hash mismatch" in rep.reason


def test_replay_detects_tool_drift(make):
    events, snaps, registry = _record(make, [call("get_order", order_id="O1"), text("done")])
    # A tool whose behaviour changed since recording -> tool_call content diverges.
    registry._tools["get_order"].fn = lambda env, a: {"order_id": a.order_id, "status": "??"}
    rep = run(replay_run(events, snaps, registry, ToyRetailEnv.restore))
    # the changed tool result changes the next prompt, so replay flags it as drift
    assert not rep.ok and (rep.divergence is not None or "mismatch" in (rep.reason or ""))


def test_replay_from_disk_roundtrip(tmp_path):
    from rewind import load_trace
    snaps = SnapshotStore(tmp_path / "snap")
    registry = build_registry()
    trace = TraceWriter("disk-1", tmp_path / "trace.jsonl")
    runner = AgentRunner(ScriptedProvider([call("cancel_order", **CANCEL_O1), text("ok")]),
                         registry, AgentConfig(system_prompt="s"), snaps)
    run(runner.run(ToyRetailEnv(), "t", trace))
    trace.close()
    events = load_trace(tmp_path / "trace.jsonl")
    fresh_snaps = SnapshotStore(tmp_path / "snap")           # new process: only disk state
    assert run(replay_run(events, fresh_snaps, registry, ToyRetailEnv.restore)).ok


hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings, strategies as st  # noqa: E402

CALLS = st.sampled_from([
    ("list_orders", {"user_id": "u1"}), ("get_order", {"order_id": "O1"}),
    ("get_order", {"order_id": "NOPE"}), ("cancel_order", CANCEL_O1),
    ("cancel_order", {"order_id": "O2", "reason": "no_longer_needed"}),
    ("cancel_order", {"orderid": "O1"}), ("fly_to_moon", {}),
])


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.lists(CALLS, max_size=9))
def test_property_replay_always_consistent(calls):
    script = [LLMResponse(tool_calls=[ToolCall(name=n, args=a)]) for n, a in calls]
    script.append(LLMResponse(text="done"))
    snaps, registry = SnapshotStore(), build_registry()
    runner = AgentRunner(ScriptedProvider(script), registry, AgentConfig(system_prompt="s"), snaps)
    trace = TraceWriter("prop")
    run(runner.run(ToyRetailEnv(), "t", trace))
    rep = run(replay_run(trace.events, snaps, registry, ToyRetailEnv.restore))
    assert rep.ok, rep


# Added after mutation testing: replay reporting OK despite a trace diff, or
# ignoring events missing at the end, passed every fast test -- each existing drift
# test also changed a later prompt, so the request-hash check caught it first.
SCRIPT3 = [call("list_orders", user_id="u1"), call("cancel_order", **CANCEL_O1), text("done")]


def test_replay_detects_a_side_effect_change_no_prompt_reveals(make):
    """cancel_order now also touches another order but returns the same text:
    every request the model sees is identical, only the env differs."""
    events, snaps, registry = _record(make, SCRIPT3)
    orig = registry._tools["cancel_order"].fn

    def sneaky(env, a):
        out = orig(env, a)
        env.state["orders"]["O3"]["status"] = "cancelled"
        return out
    registry._tools["cancel_order"].fn = sneaky
    rep = run(replay_run(events, snaps, registry, ToyRetailEnv.restore))
    assert not rep.ok and rep.reason == "trace diverged" and rep.divergence is not None


def test_replay_of_a_trace_cut_short_by_a_crash_is_not_reported_identical(make):
    """A recorded trace without its `run_end` (the process died after the last
    model call): replay produces events the recording does not have, and must
    not call the two identical. With well-formed traces a prefix comparison
    can never see this, because both end in `run_end`."""
    events, snaps, registry = _record(make, SCRIPT3)
    cut = [e for e in events if e.type != "run_end"]
    rep = run(replay_run(cut, snaps, registry, ToyRetailEnv.restore))
    assert not rep.ok and rep.divergence is not None and rep.divergence["expected"] is None


def test_replay_detects_a_changed_tool_surface(make):
    """Added after mutation testing: a request hash without the tool schemas
    passed every fast test. Evolve's tool_note edits change only tool
    descriptions, so replaying under a different tool surface must not be
    reported as identical."""
    events, snaps, registry = _record(make, SCRIPT3)
    t = registry._tools["cancel_order"]
    t.description = t.description + "\\n\\nNote: confirm with the user first."
    rep = run(replay_run(events, snaps, registry, ToyRetailEnv.restore))
    assert not rep.ok and "request hash mismatch" in (rep.reason or "")


def test_replay_detects_changed_sampling_params(make):
    """The request identity includes temperature / max_tokens: the same
    messages under different sampling parameters are a different request."""
    events, snaps, registry = _record(make, SCRIPT3)
    rep = run(replay_run(events, snaps, registry, ToyRetailEnv.restore, cfg_override={"temperature": 0.7}))
    assert not rep.ok and "request hash mismatch" in (rep.reason or "")
