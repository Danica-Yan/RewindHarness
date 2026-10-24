#!/usr/bin/env python3
"""Compare two MULTI-trial tau-bench results jsons (each produced by
`run.py --num-trials N`, which writes N trials per task into ONE file --
see scripts/run_taubench_multitrial.sh) with an actual statistical method,
instead of the single-point pass^1 comparison that scripts/taubench_compare.py
does (which is fine for exploring WHICH tasks flipped, but -- as the real
noise-floor result in this repo's README showed -- not fine for deciding
whether a difference is real).

What it computes, per run:
- pooled pass^1: mean reward over every (task, trial) sample -- N = n_tasks *
  n_trials observations, so a more stable point estimate than a single
  trial's pass^1.
- pass^k: tau-bench's own definition -- fraction of tasks where ALL trials
  of that task succeeded (reward >= 1.0 every time).

What it computes for the A vs B comparison:
- per-task mean reward under A and under B (averaged over that task's
  trials), and the per-task delta (B - A).
- mean delta across tasks (this is the net pass^1 difference, same
  quantity scripts/taubench_compare.py reports from a single trial each,
  but now averaged over multiple trials per task instead of one).
- a paired bootstrap 95% confidence interval for that mean delta: resample
  task ids with replacement (same resample used for both A and B, since
  they're paired on task_id), recompute the mean delta, repeat 5000 times,
  report the 2.5th/97.5th percentile. If the interval does NOT contain 0,
  the difference is unlikely to be pure noise at the 95% level, given the
  trials actually run; if it DOES contain 0, you cannot distinguish the
  observed delta from noise with this many trials, and the honest
  conclusion is "not enough evidence either way", not "no effect".

Usage:
    python scripts/taubench_multitrial_compare.py run_a.json run_b.json \\
        [--label-a unpatched] [--label-b patched] [--bootstrap 5000] [--seed 0]
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")
from rewind.env.taubench_failures import load_results  # noqa: E402


# The statistics now live in rewind.evolve.stats so the evolution gate uses
# exactly this code; re-exported here under the original names.
from rewind.evolve.stats import (  # noqa: E402,F401
    mean_delta, paired_bootstrap_ci, pass_all as pass_k, per_task_rewards, pooled_pass1,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_a_json")
    ap.add_argument("run_b_json")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a = per_task_rewards(load_results(args.run_a_json))
    b = per_task_rewards(load_results(args.run_b_json))

    a_trials = {len(v) for v in a.values()}
    b_trials = {len(v) for v in b.values()}
    print(f"{args.label_a}: {len(a)} tasks, trial counts per task: {sorted(a_trials)}")
    print(f"{args.label_b}: {len(b)} tasks, trial counts per task: {sorted(b_trials)}")
    if len(a_trials) > 1 or len(b_trials) > 1:
        print("WARNING: not every task has the same number of trials in one of these files "
              "(a crashed/incomplete run?) -- per-task means below still work, but pooled "
              "pass^1 weights tasks unevenly.\n")
    else:
        print()

    a_pass1, a_n = pooled_pass1(a)
    b_pass1, b_n = pooled_pass1(b)
    a_passk, a_ntasks = pass_k(a)
    b_passk, b_ntasks = pass_k(b)
    print(f"{args.label_a:12s} pooled pass^1 = {a_pass1:.4f}  (n={a_n} task*trial samples)   "
          f"pass^k = {a_passk:.4f}  ({a_ntasks} tasks, all trials must pass)")
    print(f"{args.label_b:12s} pooled pass^1 = {b_pass1:.4f}  (n={b_n} task*trial samples)   "
          f"pass^k = {b_passk:.4f}  ({b_ntasks} tasks, all trials must pass)")

    common = sorted(set(a) & set(b))
    missing = sorted(set(a) ^ set(b))
    if missing:
        print(f"\nWARNING: {len(missing)} task ids only in one file -- comparing the "
              f"{len(common)} shared ones.")

    delta = mean_delta(common, a, b)
    lo, hi = paired_bootstrap_ci(common, a, b, args.bootstrap, args.seed)
    print(f"\nMean per-task delta ({args.label_b} - {args.label_a}): {delta:+.4f}")
    print(f"95% paired bootstrap CI ({args.bootstrap} resamples): [{lo:+.4f}, {hi:+.4f}]")
    if lo <= 0.0 <= hi:
        print("CI includes 0 -- cannot distinguish this delta from noise with the trials run. "
              "This is a real 'not enough evidence' result, not evidence of 'no effect' -- more "
              "trials would narrow the interval.")
    else:
        direction = "an improvement" if delta > 0 else "a regression"
        print(f"CI excludes 0 -- {args.label_b} looks like {direction} relative to {args.label_a} "
              "that is unlikely to be pure run-to-run noise, given the trials actually run.")


if __name__ == "__main__":
    main()
