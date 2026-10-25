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


# ---------------------------------------------------------------------------
# Budgets and retry counts. Added after mutation testing: removing the
# wall-clock or token budget, or retrying one time too many, left every
# fast test green.
# ---------------------------------------------------------------------------
from pydantic import BaseModel  # noqa: E402

from rewind import AgentRunner, SnapshotStore, Tool, TraceWriter, TransientToolError  # noqa: E402
from rewind.env import ToyRetailEnv, build_registry  # noqa: E402
from rewind.util import FakeClock  # noqa: E402


class Ticking:
    """Each model call takes `dt` seconds of (fake) time."""
    def __init__(self, clock, dt, script):
        self.clock, self.dt, self.inner = clock, dt, ScriptedProvider(script)

    async def complete(self, messages, tools, params):
        self.clock.advance(self.dt)
        return await self.inner.complete(messages, tools, params)


LOOKUPS = [call("list_orders", user_id=f"u{i}") for i in range(1, 9)]


def test_wall_clock_budget_stops_the_run():
    clock = FakeClock()
    runner = AgentRunner(Ticking(clock, 10, LOOKUPS), build_registry(),
                         AgentConfig(system_prompt="sys", wall_seconds=25), SnapshotStore(), clock=clock)
    res = run(runner.run(ToyRetailEnv(), "t", TraceWriter("t-1")))
    # checks happen before each step: 0, 10, 20 are within 25 s; 30 is not
    assert res.status == "budget_exceeded" and res.reason == "wall_time" and res.steps == 3


def test_token_budget_stops_the_run(make):
    runner, env, trace, *_ = make(ScriptedProvider(LOOKUPS), AgentConfig(system_prompt="sys", max_tokens=40))
    res = run(runner.run(env, "t", trace))
    # 15 tokens per call: 0, 15, 30 are within 40; 45 is not
    assert res.status == "budget_exceeded" and res.reason == "max_tokens"
    assert res.steps == 3 and res.tokens == 45


def test_llm_retries_exactly_the_configured_number_of_times(make):
    sleeps = Sleeps()
    p = FaultInjectingProvider(ScriptedProvider([text("ok")]), fail_always=ProviderError("503", True, 503))
    runner, env, trace, *_ = make(p, AgentConfig(system_prompt="sys", max_llm_retries=2), sleep=sleeps)
    res = run(runner.run(env, "t", trace))
    assert res.status == "provider_error"
    assert p.attempts == 3 and len(sleeps.calls) == 2          # 1 attempt + 2 retries


class _NoArgs(BaseModel):
    pass


def _flaky_registry(fail_times):
    reg = build_registry()
    n = {"calls": 0}

    def fn(env, args):
        n["calls"] += 1
        if n["calls"] <= fail_times:
            raise TransientToolError("upstream timeout")
        return {"ok": True}
    reg.register(Tool("ping", "flaky read-only tool", _NoArgs, fn))
    return reg, n


def _run_ping(reg, retries):
    runner = AgentRunner(ScriptedProvider([call("ping"), text("done")]), reg,
                         AgentConfig(system_prompt="sys", max_transient_tool_retries=retries),
                         SnapshotStore(), sleep=Sleeps())
    trace = TraceWriter("t-1")
    res = run(runner.run(ToyRetailEnv(), "t", trace))
    return res, [e for e in trace.events if e.type == "tool_call"]


def test_transient_tool_error_is_retried_until_it_succeeds():
    reg, n = _flaky_registry(fail_times=2)
    res, calls = _run_ping(reg, retries=2)
    assert res.status == "completed" and n["calls"] == 3
    assert [c.data["ok"] for c in calls] == [True]             # one call as the model sees it


def test_transient_tool_error_retries_are_bounded():
    reg, n = _flaky_registry(fail_times=99)
    res, calls = _run_ping(reg, retries=2)
    assert n["calls"] == 3                                      # 1 attempt + 2 retries, then give up
    assert calls[0].data["error_class"] == "TOOL_TRANSIENT_ERROR" and res.status == "completed"


def test_invalid_args_limit_counts_consecutive_errors_only(make):
    """Added after mutation testing: a counter that never reset on a valid call
    passed every fast test, because none made more than max_invalid_args (2)
    mistakes in total. Three mistakes, each corrected by a valid call, must
    not abort the run."""
    bad = lambda o: call("get_order", orderid=o)  # noqa: E731  (wrong arg name)
    good = lambda o: call("get_order", order_id=o)  # noqa: E731
    p = ScriptedProvider([bad("O1"), good("O1"), bad("O2"), good("O2"), bad("O3"), good("O3"), text("done")])
    runner, env, trace, *_ = make(p)
    res = run(runner.run(env, "t", trace))
    assert res.status == "completed", res
    errs = [e for e in trace.events if e.type == "tool_call" and not e.data["ok"]]
    assert len(errs) == 3
