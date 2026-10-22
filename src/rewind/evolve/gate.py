"""The acceptance gate: the only place an edit can be committed.

Two modes.

**targeted** (default) -- the standard for a fix aimed at one failure
pattern, and the only one with usable power when the pattern is a handful of
tasks in a pool whose baseline is ~90%:

  1. efficacy: on the held-out stable failures of the cluster (the screening
     "B" tasks -- never shown to the proposer), fork from the start
     `efficacy_n` FRESH times under the candidate (screening forks are not
     reused: they selected the candidate, so they are biased). Pooled passes
     must beat the same tasks' null-fork passes by a one-sided Fisher exact
     test (p < alpha) and by at least `min_lift` in rate.
  2. safety: on the gate pool (train tasks never shown to the proposer or to
     screening), paired with the incumbent: NOT significantly worse (CI upper
     bound >= 0) and the point change >= -max_drop.

  Efficacy runs first (10-15 episodes); a candidate that fails it never
  spends the pool run.

**superiority** -- the whole pool must improve: CI lower bound > min_delta.
Only meaningful when the targeted pattern is a sizeable share of the pool.

Raw regression counts are reported but are not a veto by default: at a ~90%
baseline, a no-op re-run alone "breaks" several tasks (P0: passing parents
re-pass only 86-96% of the time). The CI accounts for that noise; a count
does not.

API budget: incumbent results are reused, one trial per pool task, the pool
runs in two halves and stops after the first if harm is already significant.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel

from ..env.taubench_dialogue import fork_episode
from ..trace import load_trace
from .runner import run_batch, version_setup
from .stats import paired_bootstrap_ci, per_task_rewards, pooled_pass1, task_means
from .surface import ContextVersion


class GatePolicy(BaseModel):
    mode: Literal["targeted", "superiority"] = "targeted"
    trials: int = 1
    # targeted: efficacy
    efficacy_n: int = 5
    alpha: float = 0.05
    min_lift: float = 0.3
    # targeted: safety
    max_drop: float = 0.02
    # superiority
    min_delta: float = 0.0
    # both
    regression_budget: int | None = None
    max_ctx_tokens: int = 1500
    futility_stop: bool = True
    bootstrap: int = 5000
    seed: int = 0


class Efficacy(BaseModel):
    tasks: list[int]
    cand_passes: int
    cand_n: int
    null_passes: int
    null_n: int
    p_value: float
    lift: float
    passed: bool


class GateVerdict(BaseModel):
    accepted: bool
    reason: str
    mode: str = "targeted"
    efficacy: Efficacy | None = None
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


def fisher_greater(k1: int, n1: int, k0: int, n0: int) -> float:
    """One-sided Fisher exact p-value for 'rate 1 > rate 0'."""
    k, n = k1 + k0, n1 + n0
    denom = math.comb(n, k)
    return sum(math.comb(n1, x) * math.comb(n0, k - x)
               for x in range(k1, min(n1, k) + 1) if 0 <= k - x <= n0) / denom


async def efficacy(candidate: ContextVersion, targets: list[tuple[int, str, int, int]], *,
                   policy: GatePolicy, provider: Any, user_llm: Any, store: Any,
                   log: Callable[[str], None] = print) -> tuple[Efficacy, int]:
    """`targets`: (task_id, parent trace, null passes, null forks)."""
    cfg, reg = version_setup(candidate)
    k1 = n1 = 0
    for task_id, trace, _, _ in targets:
        events = load_trace(trace)
        passes = 0
        for i in range(policy.efficacy_n):
            ep = await fork_episode(events, 1, provider=provider, user_llm=user_llm, store=store,
                                    registry=reg, system_prompt=cfg.system_prompt,
                                    run_id=f"{events[0].run_id}~eff{candidate.version}-{i}")
            passes += ep.reward.reward >= 1.0
        k1 += passes
        n1 += policy.efficacy_n
        log(f"    efficacy task {task_id:3d}: {passes}/{policy.efficacy_n} pass")
    k0 = sum(t[2] for t in targets)
    n0 = sum(t[3] for t in targets)
    p = fisher_greater(k1, n1, k0, n0)
    lift = (k1 / n1 if n1 else 0.0) - (k0 / n0 if n0 else 0.0)
    eff = Efficacy(tasks=[t[0] for t in targets], cand_passes=k1, cand_n=n1, null_passes=k0, null_n=n0,
                   p_value=p, lift=lift, passed=p < policy.alpha and lift >= policy.min_lift)
    return eff, n1


def _compare(inc: dict, cand: dict, tasks: list[int], policy: GatePolicy):
    common = [t for t in tasks if t in inc and t in cand]
    mi, mc = task_means(inc), task_means(cand)
    delta = sum(mc[t] - mi[t] for t in common) / len(common) if common else 0.0
    ci = paired_bootstrap_ci(common, inc, cand, policy.bootstrap, policy.seed)
    regs = sorted(t for t in common if mi[t] >= 1.0 and mc[t] < 1.0)
    imps = sorted(t for t in common if mi[t] < 1.0 and mc[t] >= 1.0)
    return common, delta, ci, regs, imps


def _verdict(**kw) -> GateVerdict:
    base = dict(n_tasks=0, delta=0.0, ci=(0.0, 0.0), incumbent_pass1=0.0, candidate_pass1=0.0,
                regressions=[], improvements=[], episodes=0)
    base.update(kw)
    return GateVerdict(**base)


async def gate(candidate: ContextVersion, incumbent_rows: list[dict], pool: list[int], *,
               split: str, out: str | Path, provider: Any, user_llm: Any, store: Any,
               policy: GatePolicy = GatePolicy(), targets: list[tuple[int, str, int, int]] = (),
               concurrency: int = 4, log: Callable[[str], None] = print) -> GateVerdict:
    if candidate.token_size() > policy.max_ctx_tokens:
        return _verdict(accepted=False, mode=policy.mode, rows_path=str(out),
                        reason=f"context too large ({candidate.token_size()} tokens)")
    episodes = 0
    eff = None
    if policy.mode == "targeted":
        if not targets:
            return _verdict(accepted=False, mode=policy.mode, rows_path=str(out),
                            reason="targeted gate needs held-out target tasks")
        eff, used = await efficacy(candidate, list(targets), policy=policy, provider=provider,
                                   user_llm=user_llm, store=store, log=log)
        episodes += used
        log(f"  efficacy: {eff.cand_passes}/{eff.cand_n} vs null {eff.null_passes}/{eff.null_n}, "
            f"lift {eff.lift:+.2f}, one-sided Fisher p = {eff.p_value:.4f}")
        if not eff.passed:
            return _verdict(accepted=False, mode=policy.mode, efficacy=eff, episodes=episodes,
                            rows_path=str(out),
                            reason=f"no proven effect on held-out targets "
                                   f"(p = {eff.p_value:.3f}, lift {eff.lift:+.2f})")

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
        log(f"  pool {'half' if h == 0 and len(halves) == 2 else 'full'}: {len(common)} tasks, "
            f"delta {delta:+.3f}, CI [{ci[0]:+.3f}, {ci[1]:+.3f}], {len(imps)} fixed, {len(regs)} broken")
        if h == 0 and len(halves) == 2 and ci[1] < 0:
            stopped = True
            break
    cand = per_task_rewards(rows)
    common, delta, ci, regs, imps = _compare(inc, cand, [t for t in pool if t in cand], policy)
    episodes += sum(len(cand[t]) for t in common)

    if stopped or ci[1] < 0:
        ok, why = False, f"significantly worse on the pool: CI upper bound {ci[1]:+.3f} < 0"
    elif policy.regression_budget is not None and len(regs) > policy.regression_budget:
        ok, why = False, f"{len(regs)} regressions > budget {policy.regression_budget}"
    elif policy.mode == "targeted" and delta < -policy.max_drop:
        ok, why = False, f"pool drop {delta:+.3f} exceeds {policy.max_drop:.2f}"
    elif policy.mode == "superiority" and ci[0] <= policy.min_delta:
        ok, why = False, f"CI lower bound {ci[0]:+.3f} <= {policy.min_delta:+.3f}"
    else:
        ok = True
        why = (f"targets {eff.cand_passes}/{eff.cand_n} vs null {eff.null_passes}/{eff.null_n} "
               f"(p = {eff.p_value:.3f}); pool delta {delta:+.3f}, CI [{ci[0]:+.3f}, {ci[1]:+.3f}]"
               if eff else f"pool delta {delta:+.3f}, CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    sub_inc = {t: inc[t] for t in common}
    sub_cand = {t: cand[t] for t in common}
    return _verdict(accepted=ok, reason=why, mode=policy.mode, efficacy=eff, n_tasks=len(common),
                    delta=delta, ci=ci, incumbent_pass1=pooled_pass1(sub_inc)[0],
                    candidate_pass1=pooled_pass1(sub_cand)[0], regressions=regs, improvements=imps,
                    episodes=episodes, stopped_early=stopped, rows_path=str(out))
