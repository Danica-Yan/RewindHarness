"""A second Benchmark, built on the offline toy retail env.

Two jobs:

* evidence that the Benchmark port is not tau-bench-shaped: experiment.py
  and evolve's batch runner drive this benchmark unchanged;
* the row-contract check at the boundary: added after mutation testing,
  when removing `validate_row` from `run_batch` left every fast test green. A
  benchmark that writes a malformed row must be stopped when it writes it.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from rewind import AgentConfig, LLMResponse, ScriptedProvider, SnapshotStore, ToolCall
from rewind.benchmark import Benchmark, FailureSignature, RowContractError, check_row
from rewind.env import ToyRetailEnv, build_registry, grade
from rewind.evolve.runner import run_batch
from rewind.evolve.surface import ContextVersion
from rewind.experiment import fork_episode, replay_episode, run_episode
from rewind.providers import to_openai_messages

TASKS = [SimpleNamespace(instruction="I'm u1, cancel my headphones order (O1).",
                         actions=[SimpleNamespace(name="cancel_order",
                                                  kwargs={"order_id": "O1", "reason": "ordered_by_mistake"})])]


@dataclass
class Graded:
    reward: float


class ToyBench:
    name = "toy-retail"
    write_tools = frozenset({"cancel_order"})

    def __init__(self, broken_rows: bool = False) -> None:
        self.broken_rows = broken_rows

    def task(self, split, index):
        return TASKS[index]

    def gold_calls(self, split, index):
        return list(TASKS[index].actions)

    def n_tasks(self, split):
        return len(TASKS)

    def policy_text(self):
        return "You are a retail support agent."

    def agent_config(self, **overrides):
        return AgentConfig(**{"system_prompt": self.policy_text(), **overrides})

    def build_registry(self):
        return build_registry()

    async def start_episode(self, split, index, *, store, user):
        return ToyRetailEnv(), TASKS[index].instruction

    def restore_env(self, state, *, store, user):
        return ToyRetailEnv.restore(state)

    def recorded_user(self, events):
        return None                                   # no counterparty: text ends the run

    def grade(self, env):
        return Graded(1.0 if grade(env) else 0.0)

    def result_record(self, env, result, trial, trace_path, events):
        row = {"task_id": 0, "trial": trial, "reward": self.grade(env).reward,
               "traj": to_openai_messages([e.data["message"] for e in events if e.type == "message"]),
               "info": {"rewind": {"trace": trace_path}}}
        if self.broken_rows:
            row["reward"] = str(row["reward"])        # e.g. a benchmark that serialises scores as text
        return row

    def failure_signature(self, row, split):
        return None if row["reward"] >= 1.0 else FailureSignature(category="WRONG_ACTION")


def agent():
    return ScriptedProvider([LLMResponse(tool_calls=[ToolCall(
        name="cancel_order", args={"order_id": "O1", "reason": "ordered_by_mistake"})]),
        LLMResponse(text="Done.")])


def test_toy_benchmark_implements_the_port():
    assert isinstance(ToyBench(), Benchmark)


def test_experiment_runs_forks_and_replays_a_non_tau_benchmark():
    bench, store = ToyBench(), SnapshotStore()
    ep = asyncio.run(run_episode(bench, "train", 0, provider=agent(), user_llm=None, store=store))
    assert ep.reward.reward == 1.0
    assert asyncio.run(replay_episode(bench, ep.events, store)).ok
    nofix = ScriptedProvider([LLMResponse(text="I can't help with that.")])
    fk = asyncio.run(fork_episode(bench, ep.events, 1, provider=nofix, user_llm=None, store=store))
    assert fk.reward.reward == 0.0                   # counterfactual: no cancel, task fails


def test_run_batch_writes_contract_rows_for_a_non_tau_benchmark(tmp_path):
    rows = asyncio.run(run_batch("train", [0], 2, tmp_path / "rows.json", ctx=ContextVersion(),
                                 bench=ToyBench(), provider=_FreshAgent(), user_llm=None,
                                 store=SnapshotStore(), log=lambda m: None))
    assert len(rows) == 2 and all(check_row(r) == [] for r in rows)
    assert {r["info"]["rewind"]["ctx_version"] for r in rows} == {0}


def test_run_batch_rejects_a_row_that_breaks_the_contract(tmp_path):
    with pytest.raises(RowContractError, match="'reward' is str"):
        asyncio.run(run_batch("train", [0], 1, tmp_path / "rows.json", ctx=ContextVersion(),
                              bench=ToyBench(broken_rows=True), provider=_FreshAgent(), user_llm=None,
                              store=SnapshotStore(), log=lambda m: None))
    assert not (tmp_path / "rows.json").exists()     # nothing malformed reached the results file


class _FreshAgent:
    """Stateless, so one instance can serve every episode of a batch
    (run_batch shares a single provider): cancel first, then say done."""
    async def complete(self, messages, tools, params):
        if not any(m["role"] == "assistant" for m in messages):
            return LLMResponse(tool_calls=[ToolCall(
                name="cancel_order", args={"order_id": "O1", "reason": "ordered_by_mistake"})])
        return LLMResponse(text="Done.")
