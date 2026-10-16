#!/usr/bin/env python3
"""W1 (real tau-bench integration): golden-trajectory sanity check for the
retail-domain adapter (`env/taubench_adapter.py`).

This is NOT a model baseline. It replays each task's own recorded
ground-truth tool calls through the real adapter and real tau-bench reward
convention, so a correct adapter should score ~100% by construction -- the
point is proving the real-tools/real-data/real-reward wiring is correct
*before* ever spending an API call on it. A real baseline (an actual model
attempting these tasks against a real or human user simulator) needs an LLM
API key (see scripts/run_taubench_deepseek.sh for that run); see the module docstring in
`env/taubench_adapter.py` and the README/project-plan "next steps" for what
that next step looks like.

Scope: train + dev splits only. Those grade purely on the data-mutation
hash; the 37/115 test-split tasks with non-empty `outputs` need the
dialogue layer this adapter does not simulate, and are reported separately
(see `unsupported_reason`).

Usage:
  pip install "tau_bench @ git+https://github.com/sierra-research/tau-bench.git"
  python scripts/taubench_baseline.py
  python scripts/taubench_baseline.py --split test --limit 115
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rewind.env.taubench_adapter import (  # noqa: E402
    TauBenchRetailEnv, build_registry, golden_provider, load_tasks, replay_grade,
    unsupported_reason,
)
from rewind.loop import AgentConfig, AgentRunner  # noqa: E402
from rewind.snapshot import SnapshotStore  # noqa: E402
from rewind.trace import TraceWriter  # noqa: E402


async def run_one(task, registry) -> tuple[str, str | None, bool | None]:
    """Returns (run_status, skip_reason, passed). `passed` is None when
    skip_reason is set (task needs the dialogue layer, not attempted)."""
    reason = unsupported_reason(task)
    if reason is not None:
        return "skipped", reason, None
    env = TauBenchRetailEnv()
    provider = golden_provider(task)
    cfg = AgentConfig(system_prompt="You are a retail customer service agent.", max_steps=30)
    runner = AgentRunner(provider, registry, cfg, SnapshotStore())
    result = await runner.run(env, task.instruction, TraceWriter("taubench"))
    if result.status != "completed":
        return "run_error", f"harness status={result.status} reason={result.reason}", False
    return "completed", None, replay_grade(env, task)


async def run_split(split: str, limit: int | None, registry) -> None:
    tasks = load_tasks(split)
    if limit is not None:
        tasks = tasks[:limit]
    n_pass = n_fail = n_skip = 0
    failures: list[tuple[int, str]] = []
    t0 = time.time()
    for i, task in enumerate(tasks):
        status, reason, passed = await run_one(task, registry)
        if status == "skipped":
            n_skip += 1
        elif passed:
            n_pass += 1
        else:
            n_fail += 1
            failures.append((i, reason or "hash mismatch against ground truth"))
    graded = n_pass + n_fail
    elapsed = time.time() - t0
    print(f"\n[{split}] {len(tasks)} tasks, {elapsed:.1f}s "
          f"({n_skip} skipped: needs dialogue layer; see unsupported_reason)")
    if graded:
        print(f"  golden-trajectory replay: {n_pass}/{graded} = {n_pass/graded:.1%}")
    for i, reason in failures[:10]:
        print(f"  FAIL task[{i}]: {reason}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "dev", "test", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None, help="cap tasks per split")
    args = ap.parse_args()

    registry = build_registry()
    splits = ["train", "dev", "test"] if args.split == "all" else [args.split]
    print("Golden-trajectory sanity check against the REAL tau-bench retail domain.")
    print("This is NOT a model baseline -- see this script's module docstring.")
    for split in splits:
        asyncio.run(run_split(split, args.limit, registry))


if __name__ == "__main__":
    main()
