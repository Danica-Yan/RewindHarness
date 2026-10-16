from conftest import Sleeps, call, run, text

from rewind import (AgentConfig, FaultInjectingProvider, MalformedOutput, ProviderError,
                    ScriptedProvider)
from rewind.env import grade

CANCEL_O1 = dict(order_id="O1", reason="ordered_by_mistake")


def test_happy_path(make):
    p = ScriptedProvider([call("list_orders", user_id="u1"), call("cancel_order", **CANCEL_O1),
                          text("done")])
    runner, env, trace, *_ = make(p)
    res = run(runner.run(env, "cancel my order", trace))
    assert res.status == "completed" and res.steps == 3 and grade(env)
    seqs = [e.seq for e in trace.events]
    assert seqs == list(range(1, len(seqs) + 1))            # strictly increasing, no gaps
    assert trace.events[-1].type == "run_end"


def test_invalid_args_self_correct(make):
    p = ScriptedProvider([call("cancel_order", orderid="O1"), call("cancel_order", **CANCEL_O1),
                          text("done")])
    runner, env, trace, *_ = make(p)
    res = run(runner.run(env, "t", trace))
    assert res.status == "completed" and grade(env)
    errs = [e for e in trace.events if e.type == "tool_call" and not e.data["ok"]]
    assert [e.data["error_class"] for e in errs] == ["TOOL_ARG_INVALID"]


def test_consecutive_invalid_aborts(make):
    p = ScriptedProvider([call("cancel_order", orderid="O1"), call("cancel_order", orderid="O2"),
                          call("cancel_order", orderid="O3")])
    runner, env, trace, *_ = make(p)
    res = run(runner.run(env, "t", trace))
    assert res.status == "invalid_args" and res.steps == 3


def test_loop_warn_then_kill(make):
    p = ScriptedProvider([call("get_order", order_id="O1")] * 6)
    runner, env, trace, *_ = make(p)
    res = run(runner.run(env, "t", trace))
    assert res.status == "loop" and res.steps == 5
    classes = [e.data["error_class"] for e in trace.events if e.type == "tool_call"]
    assert classes[2:4] == ["LOOP_DETECTED", "LOOP_DETECTED"] and classes[4] == "LOOP_KILLED"


def test_max_steps_budget(make):
    p = ScriptedProvider([call("list_orders", user_id="u1"), call("list_orders", user_id="u2"),
                          call("list_orders", user_id="u3")])
    runner, env, trace, *_ = make(p, AgentConfig(system_prompt="sys", max_steps=2))
    res = run(runner.run(env, "t", trace))
    assert res.status == "budget_exceeded" and res.reason == "max_steps" and res.steps == 2


def test_backoff_retries_then_succeeds(make):
    sleeps = Sleeps()
    inner = ScriptedProvider([text("ok")])
    p = FaultInjectingProvider(inner, {0: ProviderError("429", True, 429),
                                       1: ProviderError("503", True, 503)})
    runner, env, trace, *_ = make(p, sleep=sleeps)
    res = run(runner.run(env, "t", trace))
    assert res.status == "completed" and len(sleeps.calls) == 2
    assert sleeps.calls[1] > sleeps.calls[0] * 0.9          # exponential growth (with jitter)
    assert sum(1 for e in trace.events if e.type == "llm_call" and "error" in e.data) == 2


def test_non_retryable_error_aborts(make):
    p = FaultInjectingProvider(ScriptedProvider([text("ok")]), fail_always=ProviderError("401", False, 401))
    runner, env, trace, *_ = make(p)
    assert run(runner.run(env, "t", trace)).status == "provider_error"


def test_fallback_provider(make):
    primary = FaultInjectingProvider(ScriptedProvider([]), fail_always=ProviderError("503", True, 503))
    fallback = ScriptedProvider([text("from fallback")])
    runner, env, trace, *_ = make(primary, AgentConfig(system_prompt="sys", max_llm_retries=1),
                                  fallback=fallback)
    res = run(runner.run(env, "t", trace))
    assert res.status == "completed" and res.final_text == "from fallback"
    assert any(e.data.get("provider") == "fallback" for e in trace.events if e.type == "llm_call")


def test_malformed_output_repaired_once(make):
    p = ScriptedProvider([MalformedOutput("bad json"), text("ok")])
    runner, env, trace, *_ = make(p)
    res = run(runner.run(env, "t", trace))
    assert res.status == "completed"
    assert any(e.type == "guard" and e.data["guard"] == "repair_prompt" for e in trace.events)


def test_malformed_twice_aborts(make):
    p = ScriptedProvider([MalformedOutput("x"), MalformedOutput("y")])
    runner, env, trace, *_ = make(p)
    assert run(runner.run(env, "t", trace)).status == "malformed"
