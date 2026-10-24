"""Fork screening: a cheap filter before the expensive gate.

For a candidate context, fork each held-out ("B") stable failure from step 1
-- same first user line, the candidate's system prompt / tool notes in effect
from the start -- `n` times and compare its pass rate with THAT task's own
null-fork rate from the stability filter (P0: baselines differ a lot by
task, so a global threshold would be wrong).

Forking from step 1 rather than from a mid-episode step on purpose: a
context edit changes behaviour from the first turn, and splicing a new
system prompt into a history produced under the old one would measure "rule
suddenly appears at step k", a biased counterfactual (design doc, difficulty 2).

Budget: forks are sequential per task and stop as soon as the task can no
longer count as improved; tasks stop as soon as the candidate can no longer
reach `min_improved`.
"""
from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel

from ..benchmark import Benchmark
from ..experiment import fork_episode
from ..snapshot import SnapshotStore
from ..trace import load_trace


class ScreenPolicy(BaseModel):
    n: int = 3                     # forks per task
    tau: float = 0.5               # task improved if pass_rate >= null_rate + tau
    min_improved_frac: float = 0.5  # share of B tasks that must improve


class TaskScreen(BaseModel):
    task_id: int
    null_rate: float
    passes: int
    forks: int
    improved: bool


class ScreenResult(BaseModel):
    edit_ids: list[str]
    passed: bool
    tasks: list[TaskScreen]
    forks: int
    reason: str


async def screen(edit_ids: list[str], b_tasks: list[tuple[int, str, float]], *,
                 system_prompt: str, registry: Any, store: SnapshotStore, bench: Benchmark, provider: Any,
                 user_llm: Any, policy: ScreenPolicy = ScreenPolicy(),
                 log: Callable[[str], None] = print) -> ScreenResult:
    """`b_tasks`: (task_id, parent trace path, null-fork pass rate)."""
    need = max(1, round(policy.min_improved_frac * len(b_tasks) + 1e-9))
    results: list[TaskScreen] = []
    total_forks = 0
    for j, (task_id, trace, null_rate) in enumerate(b_tasks):
        improved_so_far = sum(t.improved for t in results)
        if improved_so_far + (len(b_tasks) - j) < need:
            break  # cannot reach the bar any more
        events = load_trace(trace)
        target = null_rate + policy.tau  # pass rate this task must reach
        passes = forks = 0
        for i in range(policy.n):
            ep = await fork_episode(bench, events, 1, provider=provider, user_llm=user_llm, store=store,
                                    registry=registry, system_prompt=system_prompt,
                                    run_id=f"{events[0].run_id}~screen-{i}")
            forks += 1
            passes += ep.reward.reward >= 1.0
            if (passes + (policy.n - forks)) / policy.n < target:
                break  # even all remaining passing would not reach the target
            if passes / policy.n >= target:
                break  # already reached
        total_forks += forks
        rate = passes / policy.n
        results.append(TaskScreen(task_id=task_id, null_rate=null_rate, passes=passes, forks=forks,
                                  improved=rate >= target))
        log(f"    screen task {task_id:3d}: {passes} pass in {forks} forks "
            f"(null {null_rate:.2f}) -> {'improved' if results[-1].improved else 'no'}")
    n_imp = sum(t.improved for t in results)
    ok = n_imp >= need
    return ScreenResult(edit_ids=edit_ids, passed=ok, tasks=results, forks=total_forks,
                        reason=f"{n_imp}/{len(b_tasks)} held-out stable failures improved (need {need})")
