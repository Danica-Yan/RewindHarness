#!/usr/bin/env python3
"""Real failure-mode report over a tau-bench results json (the file
tau-bench's own `run.py` writes to `--log-dir`, e.g.
`results/tool-calling-<model>-<temp>_range_<start>--<end>_user-<user_model>-<strategy>_<timestamp>.json`).

Usage:
    python scripts/taubench_failure_report.py results/your-results-file.json

Prints the category distribution over every reward<1.0 task, then for the
two action-shaped categories, a breakdown of *why* (never attempted the
mutating action / missing one of several / extra one / same tools different
args) from `flywheel/taubench_failures.action_shape`.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter

sys.path.insert(0, "src")
from rewind.env.taubench_failures import action_shape, load_results, summarize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_json")
    args = ap.parse_args()

    results = load_results(args.results_json)
    n_total = len(results)
    records, dist = summarize(results)
    n_fail = len(records)
    n_ok = n_total - n_fail

    print(f"{n_total} tasks total -- {n_ok} passed, {n_fail} failed (pass^1 = {n_ok / n_total:.4f})\n")
    print("Failure category distribution:")
    for cat, count in dist.most_common():
        print(f"  {cat:40s} {count:3d}  ({count / n_fail:.1%} of failures)")

    action_cats = {"WRONG_ACTION", "ACTION_MISMATCH_DESPITE_RIGHT_OUTPUT"}
    shaped = [r for r in records if r.category in action_cats]
    if shaped:
        print(f"\nOf the {len(shaped)} action-mismatch failures, by shape:")
        shapes = Counter(action_shape(r) for r in shaped)
        for shape, count in shapes.most_common():
            print(f"  {count:3d}  {shape}")

    tool_err = [r for r in records if r.category == "TOOL_RUNTIME_ERROR"]
    if tool_err:
        print(f"\nOf the {len(tool_err)} TOOL_RUNTIME_ERROR failures, distinct error messages:")
        msgs = Counter(r.last_tool_error for r in tool_err)
        for msg, count in msgs.most_common():
            print(f"  {count:3d}  {msg}")

    print("\nPer-task detail (task_id, category, reward):")
    for r in sorted(records, key=lambda r: r.task_id):
        print(f"  task {r.task_id:>4}  {r.category:40s} reward={r.reward}")


if __name__ == "__main__":
    main()
