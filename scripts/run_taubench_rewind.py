#!/usr/bin/env python3
"""Run real tau-bench retail episodes INSIDE rewind's loop (P0).

Same model setup as scripts/run_taubench_deepseek.sh (deepseek-chat agent at
temperature 0, deepseek-chat user simulator, tool-calling behaviour), but the
episodes run through rewind's AgentRunner, so every episode leaves:

  * a JSONL trace          (--trace-dir/<split>-<task>-t<trial>.jsonl)
  * content-addressed env snapshots incl. the user simulator's state
                           (--store-dir, shared by all episodes; the 1.3 MB DB
                            is stored once per distinct content)
  * a results row in tau-bench's own run.py format (--out)

so the existing analysis scripts work unchanged, and any episode can later be
replayed, forked or attributed.

The results file is rewritten after every episode, and (task, trial) pairs
already in it are skipped -- an interrupted run resumes where it stopped.

Usage:
  export DEEPSEEK_API_KEY=sk-...
  export REWIND_TAUBENCH_SRC=../tau-bench      # or pip install the benchmark
  python scripts/run_taubench_rewind.py --split test --num-trials 1 \
      --out runs/p0/test_baseline.json

  # P0 parity check against the run.py baseline (70.4% pass^1 on test):
  python scripts/taubench_multitrial_compare.py <run.py results> runs/p0/test_baseline.json
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rewind.env.taubench_dialogue import litellm_user_llm, n_tasks  # noqa: E402
from rewind.evolve.runner import run_batch  # noqa: E402
from rewind.evolve.surface import ContextVersion  # noqa: E402
from rewind.evolve.stats import pass_hat_k, per_task_rewards, pooled_pass1  # noqa: E402
from rewind.providers import DeepSeekProvider  # noqa: E402
from rewind.snapshot import SnapshotStore  # noqa: E402
from rewind.env.taubench_benchmark import TauBenchRetail  # noqa: E402

TAU = TauBenchRetail()


def parse_ids(spec: str | None, total: int) -> list[int]:
    if not spec:
        return list(range(total))
    ids: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            ids.extend(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))
    return [i for i in ids if 0 <= i < total]


async def main_async(args: argparse.Namespace) -> None:
    out = Path(args.out)
    store = SnapshotStore(args.store_dir or out.parent / "snapshots")
    ids = parse_ids(args.task_ids, n_tasks(args.split))
    rows = await run_batch(args.split, ids, args.num_trials, out, ctx=ContextVersion(),
                           provider=DeepSeekProvider(model=args.model),
                           user_llm=litellm_user_llm(model=args.user_model, provider=args.user_provider),
                           store=store, trace_dir=args.trace_dir, concurrency=args.concurrency,
                           log=lambda m: print(m, flush=True), bench=TAU)
    by_task = per_task_rewards(rows)
    p1, n = pooled_pass1(by_task)
    k_max = min(len(v) for v in by_task.values()) if by_task else 0
    print(f"\npooled pass^1 = {p1:.4f} over {n} episodes, {len(by_task)} tasks")
    for k in range(1, k_max + 1):
        print(f"pass^{k} = {pass_hat_k(by_task, k):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--task-ids", help="e.g. 0-49 or 3,7,12 (default: all)")
    ap.add_argument("--num-trials", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", required=True, help="results json (tau-bench run.py format)")
    ap.add_argument("--trace-dir")
    ap.add_argument("--store-dir")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--user-model", default="deepseek-chat")
    ap.add_argument("--user-provider", default="deepseek")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
