"""tau-bench retail as a `Benchmark` (see rewind/benchmark.py).

Everything evolve and experiment.py need from tau-bench goes through one
`TauBenchRetail` instance. The class only delegates to the adapter modules
next to it (taubench_adapter, taubench_dialogue, taubench_failures); it
does not run episodes itself, so this plugin depends on the tool and env
layers but never on loop/fork/replay.

`failure_signature` lives here rather than in evolve because it reads
tau-bench's own result format (`traj` in OpenAI chat format, gold actions
of a tau-bench task): it is how *this* benchmark explains a failure.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..benchmark import FailureSignature
from ..events import Event
from ..snapshot import SnapshotStore
from .taubench_adapter import MUTATING_TOOLS
from .taubench_dialogue import (
    Reward, TauBenchDialogueEnv, TauUserSim, build_dialogue_registry, faithful_config, grade,
    load_task, n_tasks, recorded_user_llm, result_record, wiki,
)
from .taubench_failures import action_shape, classify_taubench

_NUM = re.compile(r"#?W?\d+")


def _norm_error(msg: str) -> str:
    return _NUM.sub("<n>", msg.replace("Error:", "").strip())[:80]


def _agent_writes(traj: list[dict]) -> list[tuple[str, dict]]:
    """Mutating calls the agent made, in order (first call of each turn, as
    tau-bench's agent executes only that one)."""
    out = []
    for m in traj:
        if m.get("role") != "assistant":
            continue
        for c in (m.get("tool_calls") or [])[:1]:
            fn = c.get("function") or {}
            if fn.get("name") in MUTATING_TOOLS:
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                out.append((fn["name"], args))
    return out


class TauBenchRetail:
    """tau-bench retail domain. `user` (the counterparty handle) is the user
    simulator's LLM: a `UserLLM` such as `litellm_user_llm()`."""

    name = "tau-retail"

    # -- tasks -------------------------------------------------------------------
    @property
    def write_tools(self) -> frozenset[str]:
        return frozenset(MUTATING_TOOLS)

    def task(self, split: str, index: int) -> Any:
        return load_task(split, index)

    def n_tasks(self, split: str) -> int:
        return n_tasks(split)

    # -- agent -------------------------------------------------------------------
    def policy_text(self) -> str:
        return wiki()

    def agent_config(self, **overrides: Any):
        return faithful_config(**overrides)

    def build_registry(self):
        return build_dialogue_registry()

    # -- episodes ----------------------------------------------------------------
    async def start_episode(self, split: str, index: int, *, store: SnapshotStore,
                            user: Any) -> tuple[TauBenchDialogueEnv, str]:
        env = TauBenchDialogueEnv(store, TauUserSim(user), split, index)
        first = await env.start()
        return env, first

    def restore_env(self, state: dict[str, Any], *, store: SnapshotStore,
                    user: Any) -> TauBenchDialogueEnv:
        return TauBenchDialogueEnv.restore(state, store=store, user_llm=user)

    def recorded_user(self, events: list[Event]) -> Any:
        return recorded_user_llm(events)

    # -- results -----------------------------------------------------------------
    def grade(self, env: TauBenchDialogueEnv) -> Reward:
        return grade(env)

    def result_record(self, env: TauBenchDialogueEnv, result: Any, trial: int,
                      trace_path: str | None, events: list[Event]) -> dict:
        return result_record(env, result, trial, trace_path, events)

    def failure_signature(self, row: dict, split: str) -> FailureSignature | None:
        rec = classify_taubench(row)
        if rec is None:
            return None
        if rec.category == "TOOL_RUNTIME_ERROR":  # only unrecovered errors land here
            return FailureSignature(category=rec.category, tool=rec.last_tool_error_tool,
                                    detail=_norm_error(rec.last_tool_error or ""))
        if rec.category == "WRONG_OUTPUT":
            return FailureSignature(category=rec.category, detail="missing required output")
        # action failures: find the first mutating tool where agent and truth differ
        task = load_task(split, row["task_id"])
        gt = [(a.name, a.kwargs) for a in task.actions if a.name in MUTATING_TOOLS]
        agent = _agent_writes(row.get("traj", []))
        gt_names, ag_names = [n for n, _ in gt], [n for n, _ in agent]
        if gt_names == ag_names:
            for (gn, ga), (_, aa) in zip(gt, agent):
                diff = sorted(k for k in set(ga) | set(aa) if ga.get(k) != aa.get(k))
                if diff:
                    return FailureSignature(category=rec.category, tool=gn,
                                            detail="wrong args: " + ",".join(diff))
            return FailureSignature(category=rec.category, detail="same calls, state still differs")
        missing = [n for n in gt_names if n not in ag_names]
        tool = missing[0] if missing else next((n for n in ag_names if n not in gt_names), None)
        return FailureSignature(category=rec.category, tool=tool, detail=action_shape(rec))
