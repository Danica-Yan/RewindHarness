from conftest import call, run, text

from rewind import ForkPatch, ScriptedProvider, TraceWriter, fork_run, replay_run, AgentRunner
from rewind.env import ToyRetailEnv, grade
from rewind.replay import stable

CANCEL_O1 = dict(order_id="O1", reason="ordered_by_mistake")
BAD = [call("list_orders", user_id="u1"),
       call("cancel_order", order_id="O2", reason="ordered_by_mistake"),   # shipped -> error
       text("Sorry, I could not cancel it.")]


def _fork(make, at_step, patch, cont):
    runner, env, trace, snaps, registry = make(ScriptedProvider(BAD))
    run(runner.run(env, "cancel my order", trace))
    base = trace.events
    child_runner = AgentRunner(ScriptedProvider(cont), registry, runner.cfg, snaps, sleep=runner.sleep)
    child = TraceWriter("child-1")
    holder = {}

    def factory(state):
        holder["env"] = ToyRetailEnv.restore(state)
        return holder["env"]

    res = run(fork_run(base, snaps, at_step=at_step, runner=child_runner, env_factory=factory,
                       trace=child, patch=patch))
    return base, child, res, holder["env"], snaps, registry, env


def test_fork_flips_outcome_and_shares_prefix(make):
    patch = ForkPatch(tool_calls=[{"name": "cancel_order", "args": CANCEL_O1}])
    base, child, res, fenv, *_ , base_env = _fork(make, 2, patch, [text("Done.")])
    assert not grade(base_env) and grade(fenv) and res.status == "completed"

    start = child.events[0]
    assert start.type == "run_start"
    assert start.data["parent_run_id"] == "t-1" and start.data["fork_step"] == 2

    prefix_base = [stable(e) for e in base if e.type not in ("run_start", "run_end") and e.step < 2]
    prefix_child = [stable(e) for e in child.events if e.inherited]
    assert prefix_base == prefix_child and len(prefix_base) > 0
    patched = [e for e in child.events if e.type == "llm_call" and e.data.get("patched")]
    assert len(patched) == 1 and patched[0].step == 2


def test_fork_without_patch_resamples_live(make):
    # no patch: step 2 is re-sampled from the (new) provider instead of the recorded action
    _, child, res, fenv, *_ = _fork(make, 2, None, [call("cancel_order", **CANCEL_O1), text("ok")])
    assert res.status == "completed" and grade(fenv)


def test_forked_trace_is_itself_replayable(make):
    patch = ForkPatch(tool_calls=[{"name": "cancel_order", "args": CANCEL_O1}])
    _, child, _, _, snaps, registry, _ = _fork(make, 2, patch, [text("Done.")])
    assert run(replay_run(child.events, snaps, registry, ToyRetailEnv.restore)).ok


def test_fork_restores_loop_state(make):
    # two identical calls happen before the fork point; the third identical call must be
    # blocked because loop counters were rebuilt from the prefix.
    script = [call("get_order", order_id="O1")] * 2 + [text("x")]
    runner, env, trace, snaps, registry = make(ScriptedProvider(script))
    run(runner.run(env, "t", trace))
    child_runner = AgentRunner(ScriptedProvider([call("get_order", order_id="O1"), text("y")]),
                               registry, runner.cfg, snaps, sleep=runner.sleep)
    child = TraceWriter("c")
    run(fork_run(trace.events, snaps, at_step=3, runner=child_runner,
                 env_factory=ToyRetailEnv.restore, trace=child))
    third = [e for e in child.events if e.type == "tool_call" and not e.inherited]
    assert third[0].data["error_class"] == "LOOP_DETECTED"


def test_fork_bad_step_raises(make):
    runner, env, trace, snaps, registry = make(ScriptedProvider(BAD))
    run(runner.run(env, "t", trace))
    import pytest
    with pytest.raises(ValueError):
        run(fork_run(trace.events, snaps, at_step=99, runner=runner,
                     env_factory=ToyRetailEnv.restore, trace=TraceWriter("x")))
