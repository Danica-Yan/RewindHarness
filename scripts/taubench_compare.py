#!/usr/bin/env python3
"""Generic two-run diff for tau-bench results jsons -- same task_id space,
same split, same model, reports pass^1 for each, which tasks flipped in
either direction, and (optionally) how a named subset of "target" task ids
did. Used for two different real comparisons in this project, which is why
it doesn't hard-code "patch" language:

- round 1 vs baseline (did the wiki.md patch actually help?) --
  `taubench_round1_compare.py` is a thin wrapper around this with
  --targets 72,98,105 (the tasks the patch specifically targeted).
- baseline vs a second, unpatched baseline rerun (the "noise floor" --
  how much do two *identical*-condition runs differ just from run-to-run
  variance?) -- see scripts/run_taubench_noise_floor.sh. No target subset
  makes sense here since nothing was deliberately targeted; the whole
  point is seeing how much flips with nothing changed.

Usage:
    python scripts/taubench_compare.py run_a.json run_b.json \\
        [--label-a baseline] [--label-b "round 1"] [--targets 72,98,105]
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")
from rewind.flywheel.taubench_failures import load_results  # noqa: E402


def by_task(results: list[dict]) -> dict[int, float]:
    return {r["task_id"]: r["reward"] for r in results}


def compare(a_json: str, b_json: str, label_a: str = "run A", label_b: str = "run B",
           targets: list[int] | None = None) -> None:
    a = by_task(load_results(a_json))
    b = by_task(load_results(b_json))

    common = sorted(set(a) & set(b))
    missing_in_b = sorted(set(a) - set(b))
    extra_in_b = sorted(set(b) - set(a))
    if missing_in_b or extra_in_b:
        print(f"WARNING: task_id sets differ -- missing in {label_b}: {missing_in_b}, "
              f"extra in {label_b}: {extra_in_b}. Comparing only the {len(common)} shared task ids.\n")

    a_pass = sum(1 for t in common if a[t] >= 1.0)
    b_pass = sum(1 for t in common if b[t] >= 1.0)
    n = len(common)

    print(f"{label_a:20s} pass^1 = {a_pass}/{n} = {a_pass / n:.4f}")
    print(f"{label_b:20s} pass^1 = {b_pass}/{n} = {b_pass / n:.4f}")
    print(f"delta: {b_pass - a_pass:+d} tasks\n")

    flipped_to_pass = [t for t in common if a[t] < 1.0 and b[t] >= 1.0]
    flipped_to_fail = [t for t in common if a[t] >= 1.0 and b[t] < 1.0]
    still_fail = [t for t in common if a[t] < 1.0 and b[t] < 1.0]
    total_flips = len(flipped_to_pass) + len(flipped_to_fail)

    print(f"Flipped to PASS in {label_b} ({len(flipped_to_pass)}): {flipped_to_pass}")
    print(f"Flipped to FAIL in {label_b} ({len(flipped_to_fail)}): {flipped_to_fail}")
    print(f"Still failing in both ({len(still_fail)}): {still_fail}")
    print(f"\nTotal churn (either direction): {total_flips} tasks "
          f"({total_flips / n:.1%} of the {n} shared tasks) for a net delta of {b_pass - a_pass:+d}.")
    if total_flips and abs(b_pass - a_pass) < total_flips:
        print("Net delta is smaller than total churn -- some or all of it could be run-to-run "
              "noise rather than a real effect of whatever differs between the two runs. See "
              "docs/DEVLOG.zh-CN.md (round 1 / noise-floor sections) for why this matters before crediting "
              "any intervention with the net number alone.")

    if targets:
        hit = set(targets) & set(flipped_to_pass)
        print(f"\nOf the {len(targets)} specifically targeted tasks ({sorted(targets)}): "
              f"{len(hit)}/{len(targets)} flipped to PASS ({sorted(hit)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_a_json")
    ap.add_argument("run_b_json")
    ap.add_argument("--label-a", default="run A")
    ap.add_argument("--label-b", default="run B")
    ap.add_argument("--targets", default=None, help="comma-separated task ids, e.g. 72,98,105")
    args = ap.parse_args()
    targets = [int(x) for x in args.targets.split(",")] if args.targets else None
    compare(args.run_a_json, args.run_b_json, args.label_a, args.label_b, targets)


if __name__ == "__main__":
    main()
