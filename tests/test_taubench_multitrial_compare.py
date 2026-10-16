"""Unit tests for scripts/taubench_multitrial_compare.py's statistics
helpers -- constructed multi-trial fixtures (not real data; the real
multi-trial comparison needs a real local rerun with --num-trials, see
docs/DEVLOG.zh-CN.md, multi-trial section), checking pooled pass^1, pass^k, and that the
paired bootstrap CI behaves sanely: wide/including 0 for a small sample
with no real difference, narrow/excluding 0 for a larger sample with an
actual gap between the two conditions.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from taubench_multitrial_compare import (  # noqa: E402
    mean_delta, paired_bootstrap_ci, pass_k, per_task_rewards, pooled_pass1,
)


def _records(task_rewards: dict[int, list[float]]) -> list[dict]:
    return [{"task_id": t, "reward": r, "trial": i}
            for t, rewards in task_rewards.items() for i, r in enumerate(rewards)]


def test_per_task_rewards_groups_by_task_id():
    recs = _records({0: [1.0, 0.0, 1.0], 1: [0.0, 0.0, 0.0]})
    by_task = per_task_rewards(recs)
    assert by_task == {0: [1.0, 0.0, 1.0], 1: [0.0, 0.0, 0.0]}


def test_pooled_pass1_is_fraction_of_all_task_trial_samples():
    by_task = {0: [1.0, 0.0, 1.0], 1: [0.0, 0.0, 1.0]}
    rate, n = pooled_pass1(by_task)
    assert n == 6
    assert abs(rate - 3 / 6) < 1e-9


def test_pass_k_requires_every_trial_to_succeed():
    by_task = {0: [1.0, 1.0, 1.0], 1: [1.0, 0.0, 1.0], 2: [0.0, 0.0, 0.0]}
    rate, n_tasks = pass_k(by_task)
    assert n_tasks == 3
    assert abs(rate - 1 / 3) < 1e-9  # only task 0 passed every trial


def test_mean_delta_is_average_of_per_task_differences():
    a = {0: [1.0, 1.0], 1: [0.0, 0.0]}  # per-task means: 1.0, 0.0
    b = {0: [0.0, 0.0], 1: [1.0, 1.0]}  # per-task means: 0.0, 1.0
    delta = mean_delta([0, 1], a, b)
    # task 0: 0.0 - 1.0 = -1.0; task 1: 1.0 - 0.0 = +1.0; mean = 0.0
    assert abs(delta - 0.0) < 1e-9


def test_bootstrap_ci_includes_zero_when_no_real_difference():
    # identical per-task distributions -> every per-task delta is exactly 0
    a = {t: [1.0, 0.0, 1.0] for t in range(10)}
    b = {t: [1.0, 0.0, 1.0] for t in range(10)}
    lo, hi = paired_bootstrap_ci(list(range(10)), a, b, n_resamples=500, seed=0)
    assert lo <= 0.0 <= hi


def test_bootstrap_ci_excludes_zero_for_a_large_consistent_gap():
    # every task's reward is uniformly 1.0 higher under b than under a
    a = {t: [0.0, 0.0, 0.0] for t in range(30)}
    b = {t: [1.0, 1.0, 1.0] for t in range(30)}
    lo, hi = paired_bootstrap_ci(list(range(30)), a, b, n_resamples=500, seed=0)
    assert lo > 0.0 and hi > 0.0
