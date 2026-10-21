"""rewind.evolve.stats: the gate's statistics."""
from __future__ import annotations

import math

from rewind.evolve.stats import (
    paired_bootstrap_ci, pass_all, pass_hat_k, per_task_rewards, pooled_pass1, regressions,
    wilson_interval,
)


def _rows(d):
    return [{"task_id": t, "reward": r, "trial": i} for t, v in d.items() for i, r in enumerate(v)]


def test_pass_hat_k_matches_tau_bench_estimator():
    by = per_task_rewards(_rows({0: [1, 1, 0], 1: [1, 1, 1], 2: [0, 0, 0]}))
    assert pass_hat_k(by, 1) == (2 / 3 + 1 + 0) / 3
    assert math.isclose(pass_hat_k(by, 2), (1 / 3 + 1 + 0) / 3)
    assert pass_hat_k(by, 3) == pass_all(by)[0] == 1 / 3
    assert pooled_pass1(by) == (5 / 9, 9)


def test_bootstrap_ci_separates_real_gap_from_noise():
    a = {t: [0.0, 0.0, 1.0] for t in range(60)}
    b = {t: [1.0, 1.0, 1.0] for t in range(60)}
    lo, hi = paired_bootstrap_ci(list(range(60)), a, b, n_resamples=2000)
    assert lo > 0.5
    same = paired_bootstrap_ci(list(range(60)), a, a, n_resamples=2000)
    assert same == (0.0, 0.0)


def test_regressions_counts_reliable_tasks_that_broke():
    a = {0: [1, 1, 1], 1: [1, 1, 1], 2: [0, 1, 0]}
    b = {0: [0, 0, 1], 1: [1, 1, 0], 2: [0, 0, 0]}
    assert regressions([0, 1, 2], a, b) == [0]


def test_wilson_lower_bound_used_by_prevalence_rule():
    lo, _ = wilson_interval(3, 34)
    assert 0.029 < lo < 0.031
    lo12, _ = wilson_interval(12, 34)
    assert 0.21 < lo12 < 0.22
    assert wilson_interval(0, 0) == (0.0, 1.0)
