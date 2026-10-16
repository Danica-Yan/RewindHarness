#!/usr/bin/env python3
"""W8 items 2-3: run the two-round data flywheel against the persisted
task split and print a report — baseline, each round's gate decision, and
heldout after each merge. See `flywheel/data_flywheel.py`'s module
docstring for what each round actually fixes and why round 2's metric is
deliberately not pass rate.

Usage:
  python scripts/make_task_split.py        # once, if data/task_split.json doesn't exist yet
  python scripts/data_flywheel_report.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rewind.flywheel.data_flywheel import run_flywheel  # noqa: E402
from rewind.flywheel.failures import top_clusters  # noqa: E402
from rewind.flywheel.tasks import load_split  # noqa: E402

SPLIT_PATH = Path(__file__).resolve().parents[1] / "data" / "task_split.json"


async def main_async() -> None:
    if not SPLIT_PATH.exists():
        print(f"{SPLIT_PATH} not found — run scripts/make_task_split.py first.")
        return
    split = load_split(SPLIT_PATH)
    report = await run_flywheel(split, n_rounds=2)

    print(f"baseline (both bugs present): train pass_rate={report.baseline.pass_rate:.1%} "
         f"({report.baseline.passed}/{report.baseline.total})")
    print(f"  train failure clusters: {top_clusters(report.baseline.failures)}")

    for r in report.rounds:
        status = "MERGED" if r.gate_passed else "REJECTED"
        print(f"\nround {r.round_no} [{status}]  target_cluster={r.target_cluster}  fix={r.fix}")
        if r.metric == "pass_rate":
            print(f"  dev pass_rate: {r.dev_before:.1%} -> {r.dev_after:.1%}" if r.dev_after is not None
                 else f"  dev pass_rate: {r.dev_before:.1%} (candidate not evaluated)")
        else:
            print(f"  dev defensive_saves: {r.dev_before:.0f} -> {r.dev_after:.0f}" if r.dev_after is not None
                 else f"  dev defensive_saves: {r.dev_before:.0f} (candidate not evaluated)")
        if r.heldout_after is not None:
            unit = "pass_rate" if r.metric == "pass_rate" else "defensive_saves"
            val = f"{r.heldout_after:.1%}" if r.metric == "pass_rate" else f"{r.heldout_after:.0f}"
            print(f"  heldout {unit} after merge: {val}")
        if r.note:
            print(f"  note: {r.note}")

    print(f"\nfinal heldout: pass_rate={report.final.pass_rate:.1%} "
         f"({report.final.passed}/{report.final.total})  defensive_saves={report.final.defensive_saves}")
    print(f"bug flags after all rounds: {report.bug_flags_by_round[-1]}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
