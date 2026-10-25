"""The Evolve gate's decision rules, end to end, on a benchmark whose
outcomes are fully controlled.

Added after mutation testing. With the existing suite -- including the
8-minute tau-bench evolve tests, whose fake agents never produce these
situations -- each of these gate bugs passed every test: never rejecting a
*significantly harmful* edit, never stopping early, counting noise as
confirmed regressions, allowing one confirmed regression too many, and
not counting a remembered fix the incumbent solves 2 of 3 times as broken.
(Ignoring the regression memory altogether was caught by
test_evolve_meta.py; it is covered here too, in milliseconds.) Here every
task's outcome under every context is chosen by the test:

* a task is solved iff the agent calls `solve`;
* the agent skips `solve` for tasks the candidate context "breaks" (the
  candidate is recognised by a marker in the system prompt), and for tasks
  that are just failing right now regardless of context (`flaky`);
* `fail_first[(context, task)] = n`: the first n episodes fail, the rest pass.
"""
from __future__ import annotations

import asyncio
import copy
from collections import defaultdict
from dataclasses import dataclass

from pydantic import BaseModel

from rewind import AgentConfig, LLMResponse, SnapshotStore, Tool, ToolCall
from rewind.evolve.gate import Efficacy, GatePolicy, gate
from rewind.evolve.runner import run_batch
from rewind.evolve.surface import ContextVersion
from rewind.providers import to_openai_messages
from rewind.tools import ToolRegistry

MARK = "MARKER-CANDIDATE"
CANDIDATE = ContextVersion(version=1, parent=0, rules={"r": f"When the user asks for help: {MARK}."})
INCUMBENT = ContextVersion()


class TaskEnv:
    def __init__(self, state):
        self.state = copy.deepcopy(state)

    @property
    def applied(self):
        return self.state["_applied"]

    def snapshot(self):
        return copy.deepcopy(self.state)


class _NoArgs(BaseModel):
    pass


def _solve(env, args):
    env.state["solved"] = True
    return "solved"


@dataclass
class Graded:
    reward: float


class GateBench:
    name = "gate-toy"
    write_tools = frozenset({"solve"})

    def task(self, split, i):
        return type("T", (), {"instruction": f"task {i}"})()

    def gold_calls(self, split, i):
        return [type("A", (), {"name": "solve", "kwargs": {}})()]

    def n_tasks(self, split):
        return 40

    def policy_text(self):
        return "You solve tasks."

    def agent_config(self, **kw):
        return AgentConfig(**{"system_prompt": self.policy_text(), **kw})

    def build_registry(self):
        reg = ToolRegistry()
        reg.register(Tool("solve", "solve the task", _NoArgs, _solve, mutating=True))
        return reg

    async def start_episode(self, split, i, *, store, user):
        return TaskEnv({"task": i, "solved": False, "_applied": {}}), f"task {i}"

    def restore_env(self, state, *, store, user):
        return TaskEnv(state)

    def recorded_user(self, events):
        return None

    def grade(self, env):
        return Graded(1.0 if env.state["solved"] else 0.0)

    def result_record(self, env, result, trial, trace_path, events):
        return {"task_id": env.state["task"], "trial": trial, "reward": self.grade(env).reward,
                "traj": to_openai_messages([e.data["message"] for e in events if e.type == "message"]),
                "info": {"rewind": {"trace": trace_path}}}

    def failure_signature(self, row, split):
        return None


class Agent:
    """Decides per episode from the system prompt (which context) and the
    task message (which task). Stateless across calls of one episode except
    for `fail_first`, which counts episodes per (context, task)."""
    def __init__(self, breaks=frozenset(), flaky=frozenset(), fail_first=None):
        self.breaks, self.flaky = set(breaks), set(flaky)
        self.fail_first = dict(fail_first or {})
        self.episodes = defaultdict(int)

    async def complete(self, messages, tools, params):
        if any(m["role"] == "assistant" for m in messages):
            return LLMResponse(text="done")
        ctx = "candidate" if MARK in messages[0]["content"] else "incumbent"
        task = int(next(m["content"] for m in messages if m["role"] == "user").split()[-1])
        n = self.episodes[(ctx, task)]
        self.episodes[(ctx, task)] += 1
        fail = (task in self.flaky or (ctx == "candidate" and task in self.breaks)
                or n < self.fail_first.get((ctx, task), 0))
        return LLMResponse(text="I give up.") if fail else LLMResponse(tool_calls=[ToolCall(name="solve", args={})])


def passing_rows(tasks):
    """What the incumbent recorded: every task passed (one trial)."""
    return [{"task_id": t, "trial": 0, "reward": 1.0, "traj": [], "info": {}} for t in tasks]


EFFECT = Efficacy(tasks=[100], cand_passes=5, cand_n=5, null_passes=0, null_n=5, p_value=0.004,
                  lift=1.0, passed=True)


def run_gate(tmp_path, agent, pool, store=None, **kw):
    policy = kw.pop("policy", GatePolicy(regression_k=0))
    return asyncio.run(gate(CANDIDATE, kw.pop("incumbent_rows", passing_rows(pool)), pool, split="train",
                            out=tmp_path / "gate.json", bench=GateBench(), provider=agent, user_llm=None,
                            store=store or SnapshotStore(), policy=policy, incumbent=INCUMBENT,
                            prior_efficacy=EFFECT, log=lambda m: None, **kw))


def test_a_harmless_candidate_is_accepted(tmp_path):
    v = run_gate(tmp_path, Agent(), list(range(20)))
    assert v.accepted, v.reason
    assert v.regressions == [] and v.delta == 0.0


def test_a_significantly_harmful_candidate_is_rejected_and_stopped_early(tmp_path):
    v = run_gate(tmp_path, Agent(breaks=set(range(20))), list(range(20)))
    assert not v.accepted and "significantly worse" in v.reason, v.reason
    assert v.stopped_early and v.n_tasks == 10          # the first half was enough


def test_breakage_that_reproduces_is_rejected(tmp_path):
    v = run_gate(tmp_path, Agent(breaks={3, 4}), list(range(20)))
    assert v.regressions == [3, 4] and v.ci[1] >= 0     # not "significantly worse" on the pool
    assert v.confirmed_regressions == [3, 4]
    assert not v.accepted and "2 confirmed regressions" in v.reason, v.reason


def test_one_reproducible_breakage_is_within_budget(tmp_path):
    v = run_gate(tmp_path, Agent(breaks={3}), list(range(20)))
    assert v.confirmed_regressions == [3] and v.accepted, v.reason


def test_failures_that_also_hit_the_incumbent_are_noise_not_regressions(tmp_path):
    """Tasks 8 and 9 passed when the incumbent was recorded but fail now
    under ANY context: the candidate did not break them."""
    v = run_gate(tmp_path, Agent(flaky={8, 9}), list(range(20)))
    assert v.regressions == [8, 9] and v.confirmed_regressions == []
    assert v.accepted, v.reason


def _remembered(tmp_path, tasks, store):
    """Regression memory entries: (task, trace of an incumbent run that solved it)."""
    rows = asyncio.run(run_batch("train", tasks, 1, tmp_path / "fixed.json", ctx=INCUMBENT, bench=GateBench(),
                                 provider=Agent(), user_llm=None, store=store, log=lambda m: None))
    return [(r["task_id"], r["info"]["rewind"]["trace"]) for r in rows]


def test_breaking_a_remembered_fix_is_rejected(tmp_path):
    pool = list(range(10))
    store = SnapshotStore()
    suite = _remembered(tmp_path, [30], store)
    v = run_gate(tmp_path, Agent(breaks={30}), pool, store=store, regression_suite=suite,
                 policy=GatePolicy(regression_k=3))
    assert v.regression_check["broken"] == [30]
    assert not v.accepted and "remembered fixes" in v.reason, v.reason


def test_a_remembered_fix_the_incumbent_solves_2_of_3_times_still_counts(tmp_path):
    """The documented threshold: the incumbent still solves it (>= 2 of 3)
    and the candidate never does."""
    pool = list(range(10))
    store = SnapshotStore()
    suite = _remembered(tmp_path, [30], store)
    agent = Agent(breaks={30}, fail_first={("incumbent", 30): 1})   # incumbent: fail, pass, pass
    v = run_gate(tmp_path, agent, pool, store=store, regression_suite=suite, policy=GatePolicy(regression_k=3))
    t = v.regression_check["tasks"][0]
    assert (t["candidate"], t["incumbent"]) == (0, 2)
    assert v.regression_check["broken"] == [30] and not v.accepted


def test_reused_efficacy_that_did_not_pass_rejects_without_running_anything(tmp_path):
    eff = EFFECT.model_copy(update={"passed": False})
    agent = Agent()
    v = asyncio.run(gate(CANDIDATE, passing_rows(range(10)), list(range(10)), split="train",
                         out=tmp_path / "g.json", bench=GateBench(), provider=agent, user_llm=None,
                         store=SnapshotStore(), policy=GatePolicy(), incumbent=INCUMBENT,
                         prior_efficacy=eff, log=lambda m: None))
    assert not v.accepted and "no proven effect" in v.reason and not agent.episodes


def test_a_significantly_harmful_candidate_is_rejected_without_early_stopping(tmp_path):
    """The pool's CI must reject on its own, not only through the early stop."""
    v = run_gate(tmp_path, Agent(breaks=set(range(20))), list(range(20)),
                 policy=GatePolicy(regression_k=0, futility_stop=False))
    assert not v.stopped_early and v.n_tasks == 20 and v.ci[1] < 0
    assert not v.accepted and "significantly worse" in v.reason, v.reason
    assert v.confirm_episodes == 0                  # no confirmation spent on a clearly bad edit
