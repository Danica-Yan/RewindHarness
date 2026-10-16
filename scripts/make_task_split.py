#!/usr/bin/env python3
"""W8 item 1: generate the task suite and a fixed 60/20/20 split, persisted
to disk (`data/task_split.json`). Deterministic (seeded) — rerunning this
regenerates byte-for-byte the same split, but the point of persisting it is
that downstream flywheel rounds should read the *file*, not regenerate on
every run, so the split can't silently drift between rounds or machines.

Usage:
  python scripts/make_task_split.py [--n 30] [--out data/task_split.json]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rewind.flywheel.tasks import generate_suite, save_split, split_suite  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=20260928)
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "task_split.json"))
    args = p.parse_args()

    tasks = generate_suite(args.n, seed=args.seed)
    split = split_suite(tasks)
    save_split(split, args.out)
    for name, ts in split.items():
        by_cat = {}
        for t in ts:
            by_cat[t.category] = by_cat.get(t.category, 0) + 1
        print(f"{name:>8}: {len(ts):>3} tasks  {by_cat}")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
