#!/usr/bin/env python3
"""Free (no API calls) check that forks restore exactly what the agent saw.

For each recorded episode, fork at every step (or a sample of steps) and
replay the parent's own recorded model responses and user replies; every
request to the model must hash-match the recording and the final env must
match. Run this BEFORE spending money on taubench_fork_calibration.py: if
fidelity is not 100%, calibration numbers measure a bug, not noise.

Usage:
  python scripts/taubench_fork_fidelity.py runs/p0/test_rewind.json \
      --store-dir runs/p0/snapshots --max-episodes 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rewind.env.taubench_dialogue import build_dialogue_registry, fork_fidelity  # noqa: E402
from rewind.snapshot import SnapshotStore  # noqa: E402
from rewind.trace import load_trace  # noqa: E402


async def main_async(args: argparse.Namespace) -> int:
    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = [r for r in rows if (r["info"].get("rewind") or {}).get("trace")]
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.max_episodes]
    store = SnapshotStore(args.store_dir)
    registry = build_dialogue_registry()
    n_ok = n = 0
    for r in rows:
        events = load_trace(r["info"]["rewind"]["trace"])
        res = await fork_fidelity(events, store, registry=registry)
        bad = [x for x in res if not x.ok]
        n += len(res)
        n_ok += len(res) - len(bad)
        tag = "ok " if not bad else "BAD"
        print(f"{tag} task {r['task_id']:3d} trial {r['trial']}: {len(res) - len(bad)}/{len(res)} steps"
              + (f"  first bad step {bad[0].step}: {bad[0].error[:120]}" if bad else ""), flush=True)
    print(f"\nfork fidelity: {n_ok}/{n} fork points exact")
    return 0 if n_ok == n else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results")
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--max-episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    sys.exit(asyncio.run(main_async(ap.parse_args())))


if __name__ == "__main__":
    main()
