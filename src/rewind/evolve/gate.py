"""The acceptance gate: the only place an edit can be committed.

Candidate and incumbent are compared on the gate pool (train tasks the
proposer has never been shown), paired by task, with a bootstrap CI over
tasks. Accept only if ALL of:

  * CI lower bound of the mean per-task change > min_delta
  * regressions (incumbent passed, candidate failed) <= regression_budget
  * the candidate's added context fits max_ctx_tokens

API budget:
  * the incumbent's results are reused (base run for v0; the accepted
    candidate's own gate run afterwards), never re-run
  * one trial per task by default -- the bootstrap unit is the task, so more
    tasks beat more trials per task at the same cost
  * futility stop: the pool is run in two halves; after the first, a
    candidate whose CI upper bound is already below 0 is rejected
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel

from .runner import run_batch
from .stats import paired_bootstrap_ci, per_task_rewards, pooled_pass1, task_means
from .surface import ContextVersion


class GatePolicy(BaseModel):
    trials: int = 1
    min_delta: float = 0.0
    regression_budget: int = 2
    max_ctx_tokens: int = 1500
    futility_stop: bool = True
    bootstrap: int = 5000
    seed: int = 0


class GateVerdict(BaseModel):
    accepted: bool
    reason: str
    n_tasks: int
    delta: float
    ci: tuple[float, float]
    incumbent_pass1: float
    candidate_pass1: float
    regressions: list[int]
    improvements: list[int]
    episodes: int
    stopped_early: bool = False
    rows_path: str


def _compare(inc: dict, cand: dict, tasks: list[int], policy: GatePolicy):
    common = [t for t in tasks if t in inc and t in cand]
    mi, mc = task_means(inc), task_means(cand)
    delta = sum(mc[t] - mi[t] for t in common) / len(common) if common else 0.0
    ci = paired_bootstrap_ci(common, inc, cand, policy.bootstrap, policy.seed)
    regs = sorted(t for t in common if mi[t] >= 1.0 and mc[t] < 1.0)
    imps = sorted(t for t in common if mi[t] < 1.0 and mc[t] >= 1.0)
    return common, delta, ci, regs, imps


async def gate(candidate: ContextVersion, incumbent_rows: list[dict], pool: list[int], *,
               split: str, out: str | Path, provider: Any, user_llm: Any, store: Any,
               policy: GatePolicy = GatePolicy(), concurrency: int = 4,
               log: Callable[[str], None] = print) -> GateVerdict:
    if candidate.token_size() > policy.max_ctx_tokens:
        return GateVerdict(accepted=False, reason=f"context too large ({candidate.token_size()} tokens)",
                           n_tasks=0, delta=0.0, ci=(0.0, 0.0), incumbent_pass1=0.0,
                           candidate_pass1=0.0, regressions=[], improvements=[], episodes=0,
                           rows_path=str(out))
    inc = per_task_rewards([r for r in incumbent_rows if r["task_id"] in set(pool)])
    pool = [t for t in pool if t in inc]
    order = list(pool)
    random.Random(policy.seed).shuffle(order)
    halves = [order[: len(order) // 2], order] if policy.futility_stop else [order]
    rows: list[dict] = []
    stopped = False
    for h, part in enumerate(halves):
        rows = await run_batch(split, part, policy.trials, out, ctx=candidate, provider=provider,
                               user_llm=user_llm, store=store, concurrency=concurrency, log=log)
        cand = per_task_rewards(rows)
        common, delta, ci, regs, imps = _compare(inc, cand, part, policy)
        log(f"  gate {'half' if h == 0 and len(halves) == 2 else 'full'}: {len(common)} tasks, "
            f"delta {delta:+.3f}, CI [{ci[0]:+.3f}, {ci[1]:+.3f}], "
            f"{len(imps)} fixed, {len(regs)} broken")
        if h == 0 and len(halves) == 2 and ci[1] < 0:
            stopped = True
            break
    cand = per_task_rewards(rows)
    common, delta, ci, regs, imps = _compare(inc, cand, pool, policy)
    if stopped:
        ok, why = False, f"futility stop after half the pool: CI upper bound {ci[1]:+.3f} < 0"
    elif ci[0] <= policy.min_delta:
        ok, why = False, f"CI lower bound {ci[0]:+.3f} <= {policy.min_delta:+.3f}"
    elif len(regs) > policy.regression_budget:
        ok, why = False, f"{len(regs)} regressions > budget {policy.regression_budget}"
    else:
        ok, why = True, f"delta {delta:+.3f}, CI [{ci[0]:+.3f}, {ci[1]:+.3f}], {len(regs)} regressions"
    sub_inc = {t: inc[t] for t in common}
    sub_cand = {t: cand[t] for t in common}
    return GateVerdict(accepted=ok, reason=why, n_tasks=len(common), delta=delta, ci=ci,
                       incumbent_pass1=pooled_pass1(sub_inc)[0], candidate_pass1=pooled_pass1(sub_cand)[0],
                       regressions=regs, improvements=imps,
                       episodes=sum(len(v) for v in sub_cand.values()), stopped_early=stopped,
                       rows_path=str(out))
