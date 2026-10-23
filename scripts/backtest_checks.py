#!/usr/bin/env python3
"""Backtest policy checks over recorded rewind episodes -- NO API calls.

For every tool call in every recorded episode, evaluates each check against
the DB state and conversation at that moment and reports where it would have
fired: on passing episodes (should be ~0), on gold actions inside failures
(would have stopped a right action), and on off-gold calls inside failures
(catches). Needs the snapshot store the episodes were recorded with.

Usage:
  export REWIND_TAUBENCH_SRC=../tau-bench
  python scripts/backtest_checks.py --results runs/p1/train_base.json --store-dir runs/p1/snapshots
  # several result files (e.g. train + test runs), one check only, examples to a file:
  python scripts/backtest_checks.py --results a.json --results b.json --split train \\
      --only confirm_before_write --out runs/p1/backtest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rewind.evolve.backtest import taubench_backtest
from rewind.policy import Check, PolicySet
from rewind.snapshot import SnapshotStore

DEFAULT_CHECKS = Path(__file__).resolve().parents[1] / "data/policy_checks/taubench_retail.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", action="append", required=True,
                    help="rewind results json (run.py format, with info.rewind.trace); repeatable")
    ap.add_argument("--store-dir", default="runs/p1/snapshots")
    ap.add_argument("--split", default="train", help="task split the results come from (for gold actions)")
    ap.add_argument("--checks", default=str(DEFAULT_CHECKS), help="JSON list of checks")
    ap.add_argument("--only", action="append", help="check id(s) to run (default: all)")
    ap.add_argument("--out", help="write summary + example firings as JSON here")
    ap.add_argument("--examples", type=int, default=3, help="example firings printed per bucket")
    a = ap.parse_args()

    checks = [Check.model_validate(c) for c in json.loads(Path(a.checks).read_text(encoding="utf-8"))]
    if a.only:
        checks = [c for c in checks if c.id in a.only]
    rows = [r for f in a.results for r in json.loads(Path(f).read_text(encoding="utf-8"))]
    rep = taubench_backtest(PolicySet(checks), rows, store=SnapshotStore(a.store_dir), split=a.split)

    print(f"{len(rows)} episodes ({rep.n_pass} passed, {rep.n_fail} failed), "
          f"{len(rep.skipped)} skipped (no trace / snapshot)\n")
    hdr = f"{'check':28s} {'passing eps hit':>16s} {'fires/calls on passing':>23s} " \
          f"{'gold calls hit (fail)':>22s} {'catches':>8s} {'failed eps caught':>18s}"
    print(hdr)
    print("-" * len(hdr))
    for s in rep.summaries():
        print(f"{s['check']:28s} {s['passing_episodes_hit']:>7d} / {s['passing_episodes']:<6d} "
              f"{s['fires_on_passing_calls']:>15s} ({s['false_fire_rate']:5.1%}) "
              f"{s['fires_on_gold_calls_in_failures']:>22d} {s['catches']:>8d} "
              f"{s['failing_episodes_caught']:>9d} / {s['failing_episodes']:<6d}")
    js = rep.to_json(max_examples=20)
    if a.examples:
        for key, ex in js["examples"].items():
            print(f"\n{key}:")
            for e in ex[: a.examples]:
                print(f"  task {e['task_id']} t{e['trial']} step {e['step']}: {e['tool']}"
                      f"({json.dumps(e['args'], ensure_ascii=False)[:100]}) -- {e['reason']}")
    for s in rep.skipped[:5]:
        print(f"skipped: {s}")
    if a.out:
        Path(a.out).write_text(json.dumps(js, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nwrote {a.out}")
    print("\nReading it: 'fires on passing' are calls in episodes that ended well -- a check that fires "
          "there would have interrupted them (live, the agent can usually recover by asking or looking "
          "up, but each block costs a turn). 'catches' are calls in failed episodes that the gold "
          "solution never makes. Whether blocking turns a failure into a pass needs real episodes.")


if __name__ == "__main__":
    main()
