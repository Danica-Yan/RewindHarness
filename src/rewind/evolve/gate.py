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
     screening), paired with the incumbent, one trial each: NOT significantly
     worse (CI upper bound >= 0).
  3. confirmation: every task that flipped (broke or got fixed) is re-run
     `confirm_k` more times under BOTH versions. A regression is confirmed
     only if the candidate fails EVERY re-run AND the incumbent passes EVERY
     re-run -- i.e. the task is reliable without the edit and reliably broken
     with it. More than `max_confirmed_regressions` confirmed -> reject.

     Calibration (simulation in the dev log, per-task flakiness fitted to the
     P0 null-fork data, k = 2): a harmless edit is rejected ~3% of the time,
     5 truly broken tasks are caught ~99% of the time. The weaker variant
     "incumbent passes at least one re-run" rejected ~13% of harmless edits,
     because coin-flip tasks often fail twice in a row.

  Why confirmation instead of a drop margin: the original "drop <= 2 pt"
  rule rejected a HARMLESS edit 18-24% of the time at this pool size and
  baseline (simulation in the dev log) -- it measured re-run noise, not
  harm. Confirming the handful of flipped tasks costs ~4 x flips episodes
  and separates reproducible breakage from noise, the same way the
  stability filter separates stable failures from bad luck.

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

from ..benchmark import Benchmark, ResultRow, set_ctx_version, validate_row
from ..experiment import fork_episode
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
    max_drop: float | None = None      # optional point-drop cap (miscalibrated at ~90% baselines)
    confirm_k: int = 2                 # re-runs per flipped task, per version
    max_confirmed_regressions: int = 1
    regression_k: int = 3              # forks per regression-memory task, per version (0 = off)
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
    per_task: dict[int, int] = {}      # task -> candidate passes (of efficacy_n)


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
    confirmed_regressions: list[int] = []
    confirmed_fixes: list[int] = []
    confirm_episodes: int = 0
    regression_check: dict | None = None
    rows_path: str


def fisher_greater(k1: int, n1: int, k0: int, n0: int) -> float:
    """One-sided Fisher exact p-value for 'rate 1 > rate 0'."""
    k, n = k1 + k0, n1 + n0
    denom = math.comb(n, k)
    return sum(math.comb(n1, x) * math.comb(n0, k - x)
               for x in range(k1, min(n1, k) + 1) if 0 <= k - x <= n0) / denom


async def efficacy(candidate: ContextVersion, targets: list[tuple[int, str, int, int]], *,
                   policy: GatePolicy, bench: Benchmark, provider: Any, user_llm: Any, store: Any,
                   log: Callable[[str], None] = print,
                   rows_out: str | Path | None = None) -> tuple[Efficacy, int]:
    """`targets`: (task_id, parent trace, null passes, null forks).
    `rows_out`: also write every fork as a results row (with its
    conversation), e.g. to see which skills the agent loaded."""
    from .runner import write_rows
    cfg, reg = version_setup(candidate, bench=bench)
    k1 = n1 = 0
    rows: list[ResultRow] = []
    per_task: dict[int, int] = {}
    for task_id, trace, _, _ in targets:
        events = load_trace(trace)
        passes = 0
        for i in range(policy.efficacy_n):
            ep = await fork_episode(bench, events, 1, provider=provider, user_llm=user_llm, store=store,
                                    registry=reg, system_prompt=cfg.system_prompt,
                                    run_id=f"{events[0].run_id}~eff{candidate.version}-{i}")
            passes += ep.reward.reward >= 1.0
            if rows_out is not None:
                row = validate_row(bench.result_record(ep.env, ep.result, i, None, ep.events),
                                   f"{bench.name} efficacy fork of task {task_id}")
                set_ctx_version(row, candidate.version)
                rows.append(row)
        if rows_out is not None:
            write_rows(rows_out, rows)
        k1 += passes
        n1 += policy.efficacy_n
        per_task[task_id] = passes
        log(f"    efficacy task {task_id:3d}: {passes}/{policy.efficacy_n} pass")
    k0 = sum(t[2] for t in targets)
    n0 = sum(t[3] for t in targets)
    p = fisher_greater(k1, n1, k0, n0)
    lift = (k1 / n1 if n1 else 0.0) - (k0 / n0 if n0 else 0.0)
    eff = Efficacy(tasks=[t[0] for t in targets], cand_passes=k1, cand_n=n1, null_passes=k0, null_n=n0,
                   p_value=p, lift=lift, passed=p < policy.alpha and lift >= policy.min_lift,
                   per_task=per_task)
    return eff, n1


async def regression_check(candidate: ContextVersion, incumbent: ContextVersion,
                           suite: list[tuple[int, str]], *, k: int, bench: Benchmark, provider: Any, user_llm: Any,
                           store: Any, log: Callable[[str], None] = print) -> dict:
    """Regression memory: every task an accepted version fixed is re-checked
    by every later gate. Each suite task is forked from step 1, `k` times
    under the candidate and `k` under the incumbent. A task counts as broken
    when the incumbent still solves it (>= 2 of 3, scaled to k) and the
    candidate never does; the suite as a whole fails on any broken task or
    on a one-sided Fisher test (incumbent > candidate, p < 0.05)."""
    from ..evolve.runner import version_setup as _vs
    out = {"tasks": [], "broken": [], "cand_passes": 0, "inc_passes": 0, "n": 0, "episodes": 0}
    need = max(1, round(2 * k / 3))
    for task_id, trace in suite:
        events = load_trace(trace)
        res = {}
        for name, ctx in (("candidate", candidate), ("incumbent", incumbent)):
            cfg, reg = _vs(ctx, bench=bench)
            passes = 0
            for i in range(k):
                ep = await fork_episode(bench, events, 1, provider=provider, user_llm=user_llm, store=store,
                                        registry=reg, system_prompt=cfg.system_prompt,
                                        run_id=f"{events[0].run_id}~reg{ctx.version}-{i}")
                passes += ep.reward.reward >= 1.0
            res[name] = passes
        out["tasks"].append({"task_id": task_id, **res})
        out["cand_passes"] += res["candidate"]
        out["inc_passes"] += res["incumbent"]
        out["n"] += k
        out["episodes"] += 2 * k
        if res["candidate"] == 0 and res["incumbent"] >= need:
            out["broken"].append(task_id)
        log(f"    regression task {task_id:3d}: candidate {res['candidate']}/{k}, incumbent {res['incumbent']}/{k}")
    out["p_value"] = fisher_greater(out["inc_passes"], out["n"], out["cand_passes"], out["n"]) if out["n"] else 1.0
    out["passed"] = not out["broken"] and out["p_value"] >= 0.05
    return out


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


async def gate(candidate: ContextVersion, incumbent_rows: list[ResultRow], pool: list[int], *,
               split: str, out: str | Path, bench: Benchmark, provider: Any, user_llm: Any, store: Any,
               policy: GatePolicy = GatePolicy(), targets: list[tuple[int, str, int, int]] = (),
               concurrency: int = 4, log: Callable[[str], None] = print,
               incumbent: ContextVersion | None = None,
               prior_efficacy: Efficacy | None = None,
               regression_suite: list[tuple[int, str]] = ()) -> GateVerdict:
    """`incumbent` enables flip confirmation (re-runs under the incumbent);
    `prior_efficacy` reuses an efficacy result instead of re-running it
    (re-gating a candidate under a revised policy)."""
    if candidate.token_size() > policy.max_ctx_tokens:
        return _verdict(accepted=False, mode=policy.mode, rows_path=str(out),
                        reason=f"context too large ({candidate.token_size()} tokens)")
    episodes = 0
    eff = None
    if policy.mode == "targeted" and prior_efficacy is not None:
        eff = prior_efficacy
        log(f"  efficacy (reused): {eff.cand_passes}/{eff.cand_n} vs null {eff.null_passes}/{eff.null_n}, "
            f"p = {eff.p_value:.4f}")
        if not eff.passed:
            return _verdict(accepted=False, mode=policy.mode, efficacy=eff, rows_path=str(out),
                            reason="no proven effect on held-out targets (reused result)")
    elif policy.mode == "targeted":
        if not targets:
            return _verdict(accepted=False, mode=policy.mode, rows_path=str(out),
                            reason="targeted gate needs held-out target tasks")
        eff, used = await efficacy(candidate, list(targets), policy=policy, bench=bench, provider=provider,
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
    rows: list[ResultRow] = []
    stopped = False
    for h, part in enumerate(halves):
        rows = await run_batch(split, part, policy.trials, out, ctx=candidate, bench=bench, provider=provider,
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

    conf_regs: list[int] = []
    conf_fixes: list[int] = []
    conf_eps = 0
    harmful = stopped or ci[1] < 0
    if not harmful and incumbent is not None and policy.confirm_k > 0 and (regs or imps):
        conf_regs, conf_fixes, conf_eps = await confirm_flips(
            candidate, incumbent, regs, imps, split=split, out=Path(out), k=policy.confirm_k,
            bench=bench, provider=provider, user_llm=user_llm, store=store, concurrency=concurrency, log=log)
        episodes += conf_eps
        log(f"  confirmation: {len(conf_regs)}/{len(regs)} regressions and "
            f"{len(conf_fixes)}/{len(imps)} fixes reproduce")

    reg_chk = None
    early_reject = harmful or (len(conf_regs) > policy.max_confirmed_regressions) or (
        policy.regression_budget is not None and len(regs) > policy.regression_budget) or (
        policy.max_drop is not None and policy.mode == "targeted" and delta < -policy.max_drop)
    suite = [s for s in regression_suite if not eff or s[0] not in eff.tasks]
    if not early_reject and incumbent is not None and policy.regression_k > 0 and suite:
        reg_chk = await regression_check(candidate, incumbent, suite, k=policy.regression_k,
                                         bench=bench, provider=provider, user_llm=user_llm, store=store, log=log)
        episodes += reg_chk["episodes"]
        log(f"  regression memory: {len(suite)} tasks, candidate {reg_chk['cand_passes']}/{reg_chk['n']} vs "
            f"incumbent {reg_chk['inc_passes']}/{reg_chk['n']}, broken {reg_chk['broken']}")

    if harmful:
        ok, why = False, f"significantly worse on the pool: CI upper bound {ci[1]:+.3f} < 0"
    elif policy.regression_budget is not None and len(regs) > policy.regression_budget:
        ok, why = False, f"{len(regs)} regressions > budget {policy.regression_budget}"
    elif policy.max_drop is not None and policy.mode == "targeted" and delta < -policy.max_drop:
        ok, why = False, f"pool drop {delta:+.3f} exceeds {policy.max_drop:.2f}"
    elif len(conf_regs) > policy.max_confirmed_regressions:
        ok, why = False, (f"{len(conf_regs)} confirmed regressions (tasks {conf_regs}) > "
                          f"{policy.max_confirmed_regressions}: the breakage reproduces")
    elif reg_chk is not None and not reg_chk["passed"]:
        ok, why = False, (f"breaks remembered fixes: tasks {reg_chk['broken']}" if reg_chk["broken"] else
                          f"worse on remembered fixes: {reg_chk['cand_passes']}/{reg_chk['n']} vs "
                          f"{reg_chk['inc_passes']}/{reg_chk['n']} (p = {reg_chk['p_value']:.3f})")
    elif policy.mode == "superiority" and ci[0] <= policy.min_delta:
        ok, why = False, f"CI lower bound {ci[0]:+.3f} <= {policy.min_delta:+.3f}"
    else:
        ok = True
        conf = (f"; {len(conf_regs)} of {len(regs)} regressions reproduce"
                if incumbent is not None and policy.confirm_k > 0 and regs else "")
        why = (f"targets {eff.cand_passes}/{eff.cand_n} vs null {eff.null_passes}/{eff.null_n} "
               f"(p = {eff.p_value:.3f}); pool delta {delta:+.3f}, CI [{ci[0]:+.3f}, {ci[1]:+.3f}]{conf}"
               if eff else f"pool delta {delta:+.3f}, CI [{ci[0]:+.3f}, {ci[1]:+.3f}]{conf}")
    sub_inc = {t: inc[t] for t in common}
    sub_cand = {t: cand[t] for t in common}
    return _verdict(accepted=ok, reason=why, mode=policy.mode, efficacy=eff, n_tasks=len(common),
                    delta=delta, ci=ci, incumbent_pass1=pooled_pass1(sub_inc)[0],
                    candidate_pass1=pooled_pass1(sub_cand)[0], regressions=regs, improvements=imps,
                    episodes=episodes, stopped_early=stopped, confirmed_regressions=conf_regs,
                    confirmed_fixes=conf_fixes, confirm_episodes=conf_eps, regression_check=reg_chk,
                    rows_path=str(out))


def confirm_paths(out: Path) -> tuple[Path, Path]:
    """Separate files, so the gate's own rows stay one trial per task (they
    become the next version's incumbent rows)."""
    return (out.with_name(out.stem + "_confirm_candidate.json"),
            out.with_name(out.stem + "_confirm_incumbent.json"))


async def confirm_flips(candidate: ContextVersion, incumbent: ContextVersion, regs: list[int],
                        imps: list[int], *, split: str, out: Path, k: int, bench: Benchmark, provider: Any,
                        user_llm: Any, store: Any, concurrency: int = 4,
                        log: Callable[[str], None] = print) -> tuple[list[int], list[int], int]:
    flips = sorted(set(regs) | set(imps))
    c_out, i_out = confirm_paths(out)
    c_rows = await run_batch(split, flips, k, c_out, ctx=candidate, bench=bench, provider=provider, user_llm=user_llm,
                             store=store, concurrency=concurrency, log=log, trial_offset=1)
    i_rows = await run_batch(split, flips, k, i_out, ctx=incumbent, bench=bench, provider=provider, user_llm=user_llm,
                             store=store, concurrency=concurrency, log=log, trial_offset=1)
    c, i = per_task_rewards(c_rows), per_task_rewards(i_rows)
    passed = lambda v: [r >= 1.0 for r in v]  # noqa: E731
    conf_regs = [t for t in regs if c.get(t) and i.get(t)
                 and not any(passed(c[t])) and all(passed(i[t]))]
    conf_fixes = [t for t in imps if c.get(t) and i.get(t)
                  and all(passed(c[t])) and not any(passed(i[t]))]
    return conf_regs, conf_fixes, sum(len(v) for v in c.values()) + sum(len(v) for v in i.values())
