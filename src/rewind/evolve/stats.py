"""Statistics shared by evaluation scripts and the evolution loop.

Moved out of scripts/taubench_multitrial_compare.py so the gate can call the
same code the round-1 analysis used (same numbers, same seeds), plus:

* `pass_hat_k` -- tau-bench's own pass^k estimator (unbiased, from c
  successes in n trials per task), not just "all n trials passed".
* `wilson_interval` -- the lower bound the prevalence rule uses, so a
  cluster of 3 failures out of 34 does not look like a 9% problem.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable

Rewards = dict[int, list[float]]  # task_id -> reward per trial


def per_task_rewards(results: Iterable[dict]) -> Rewards:
    by_task: dict[int, list[float]] = defaultdict(list)
    for r in results:
        by_task[r["task_id"]].append(r["reward"])
    return dict(by_task)


def _ok(r: float) -> bool:
    return r >= 1.0


def pooled_pass1(by_task: Rewards) -> tuple[float, int]:
    """Share of all episodes (task x trial) that passed."""
    rs = [r for v in by_task.values() for r in v]
    return (sum(map(_ok, rs)) / len(rs) if rs else 0.0), len(rs)


def pass_all(by_task: Rewards) -> tuple[float, int]:
    """Share of tasks whose every trial passed (= pass^n for n trials)."""
    n = len(by_task)
    return (sum(1 for v in by_task.values() if all(map(_ok, v))) / n if n else 0.0), n


def pass_hat_k(by_task: Rewards, k: int) -> float:
    """tau-bench's pass^k: mean over tasks of C(c, k) / C(n, k), where c of
    the task's n trials passed. Tasks with fewer than k trials are skipped."""
    vals = []
    for v in by_task.values():
        n, c = len(v), sum(map(_ok, v))
        if n >= k:
            vals.append(math.comb(c, k) / math.comb(n, k))
    return sum(vals) / len(vals) if vals else 0.0


def task_means(by_task: Rewards) -> dict[int, float]:
    return {t: sum(v) / len(v) for t, v in by_task.items() if v}


def mean_delta(common: list[int], a: Rewards, b: Rewards) -> float:
    ma, mb = task_means(a), task_means(b)
    d = [mb[t] - ma[t] for t in common]
    return sum(d) / len(d) if d else 0.0


def paired_bootstrap_ci(common: list[int], a: Rewards, b: Rewards, n_resamples: int = 5000,
                        seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """CI of the mean per-task change b - a, resampling tasks (the unit that
    is actually independent; trials of one task are not)."""
    rng = random.Random(seed)
    ma, mb = task_means(a), task_means(b)
    deltas = [mb[t] - ma[t] for t in common]
    n = len(deltas)
    if n == 0:
        return (0.0, 0.0)
    boots = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(n_resamples))
    lo = boots[int(alpha / 2 * n_resamples)]
    hi = boots[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]
    return (lo, hi)


def regressions(common: list[int], a: Rewards, b: Rewards, *, was_at_least: float = 1.0,
                now_at_most: float = 1 / 3) -> list[int]:
    """Tasks that were reliable under `a` (mean >= was_at_least) and are now
    mostly failing under `b` (mean <= now_at_most) -- the gate's regression
    budget counts these."""
    ma, mb = task_means(a), task_means(b)
    return sorted(t for t in common if ma[t] >= was_at_least and mb[t] <= now_at_most + 1e-9)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - m), min(1.0, c + m))
