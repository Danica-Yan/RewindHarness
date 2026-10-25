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


def test_fork_counts_every_tokens_spent_including_the_forced_action(make):
    """Added after mutation testing: dropping the forced (counterfactual)
    action's tokens, or not rebuilding the prefix's tokens, left every fast test
    green. A fork must account for the parent's tokens before the fork point,
    the forced action, and everything after -- otherwise its token budget is
    wrong and a later resume of it sees a different count than the live run."""
    from rewind import LLMResponse, LoopState, ToolCall
    runner, env, trace, snaps, registry = make(ScriptedProvider(BAD))
    run(runner.run(env, "cancel my order", trace))

    prefix_events = [e for e in trace.events if e.step < 2]
    state = LoopState.from_events(prefix_events)
    assert state.tokens == 15                      # rebuilt from step 1's model call
    forced = LLMResponse(tool_calls=[ToolCall(name="cancel_order", args=CANCEL_O1)],
                         usage={"input_tokens": 7, "output_tokens": 3})
    start = next(e for e in trace.events if e.type == "step_start" and e.step == 2)
    child = TraceWriter("child-tokens")
    child_runner = AgentRunner(ScriptedProvider([text("done")]), registry, runner.cfg, snaps,
                               sleep=runner.sleep)
    res = run(child_runner.resume(ToyRetailEnv.restore(snaps.get(start.data["env_hash"])),
                                  [e.data["message"] for e in prefix_events if e.type == "message"],
                                  state, 2, child, forced))
    assert res.status == "completed"
    assert res.tokens == 15 + 10 + 15              # prefix + forced action + step 3
    # the child's own events (steps >= 2) rebuild to exactly what it spent
    assert LoopState.from_events(child.events).tokens == 10 + 15


def test_fork_under_a_new_system_prompt_sends_the_new_prompt(make):
    """Added after mutation testing: a fork that silently kept the old system
    prompt passed every fast test. Evolve screens every context edit this way
    ("what if this prompt had been in effect"), so a no-op here would make
    every screened edit look exactly as good as no edit."""
    runner, env, trace, snaps, registry = make(ScriptedProvider(BAD))
    run(runner.run(env, "cancel my order", trace))
    sent = []

    class Spy(ScriptedProvider):
        async def complete(self, messages, tools, params):
            sent.append(messages)
            return await super().complete(messages, tools, params)
    child_runner = AgentRunner(Spy([text("ok")]), registry, runner.cfg, snaps, sleep=runner.sleep)
    run(fork_run(trace.events, snaps, at_step=2, runner=child_runner, env_factory=ToyRetailEnv.restore,
                 trace=TraceWriter("c"), system_prompt="NEW POLICY"))
    assert sent and sent[0][0] == {"role": "system", "content": "NEW POLICY"}
