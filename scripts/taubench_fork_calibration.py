#!/usr/bin/env python3
"""P0 gate: how much does a forked continuation vary on its own?

For recorded episodes from run_taubench_rewind.py, fork each one with NO
patch at several points (a fraction of the way through the episode) and
continue live N times. Two things come out of it:

1. Fork at the start (fraction 0 -> step 1) is a rerun that shares only the
   user's opening line. Its pass rate per task should match the task's
   pass rate over ordinary reruns -- if it doesn't, fork restore is broken
   (wrong env / simulator state), not merely noisy.
2. Agreement with the parent's outcome as the fork point moves later. This
   is the ceiling on what fork screening can tell you: if null forks from
   the root-cause step already disagree with the parent half the time, a
   patched fork's flip rate must be read against that noise floor.

Run scripts/taubench_fork_fidelity.py first (free): if forks do not restore
exactly what the agent saw, these numbers measure that bug, not noise.

Usage:
  export DEEPSEEK_API_KEY=sk-...
  python scripts/taubench_fork_calibration.py runs/p0/test_baseline.json \
      --store-dir runs/p0/snapshots --n 5 --max-episodes 20 \
      --out runs/p0/fork_calibration.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rewind.env.taubench_dialogue import litellm_user_llm  # noqa: E402
from rewind.experiment import fork_episode  # noqa: E402
from rewind.env.taubench_benchmark import TauBenchRetail  # noqa: E402
TAU = TauBenchRetail()  # noqa: E402
from rewind.providers import DeepSeekProvider  # noqa: E402
from rewind.snapshot import SnapshotStore  # noqa: E402
from rewind.trace import load_trace  # noqa: E402


def fork_step(events: list, frac: float) -> int:
    steps = sorted({e.step for e in events if e.type == "step_start"})
    return steps[min(len(steps) - 1, int(frac * len(steps)))]


def pick_episodes(rows: list[dict], k: int, seed: int) -> list[dict]:
    """Half failures, half successes (as available), trial 0 only."""
    rows = [r for r in rows if r["trial"] == 0 and (r["info"].get("rewind") or {}).get("trace")]
    rng = random.Random(seed)
    fail = [r for r in rows if r["reward"] < 1.0]
    ok = [r for r in rows if r["reward"] >= 1.0]
    rng.shuffle(fail)
    rng.shuffle(ok)
    n_fail = min(len(fail), max(k // 2, k - len(ok)))  # top up from either side
    n_ok = min(len(ok), k - n_fail)
    return fail[:n_fail] + ok[:n_ok]


async def main_async(args: argparse.Namespace) -> None:
    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    store = SnapshotStore(args.store_dir)
    provider = DeepSeekProvider(model=args.model)
    user_llm = litellm_user_llm(model=args.user_model, provider=args.user_provider)
    fracs = [float(x) for x in args.fractions.split(",")]
    episodes = pick_episodes(rows, args.max_episodes, args.seed)
    sem = asyncio.Semaphore(args.concurrency)
    out_rows: list[dict] = []

    async def one(row: dict, frac: float, i: int) -> None:
        events = load_trace(row["info"]["rewind"]["trace"])
        k = fork_step(events, frac)
        async with sem:
            try:
                ep = await fork_episode(TAU, events, k, provider=provider, user_llm=user_llm,
                                        store=store, run_id=f"{events[0].run_id}~null{k}-{i}")
            except Exception as e:
                print(f"[error] task {row['task_id']} frac {frac} #{i}: {e}", flush=True)
                return
        out_rows.append({"task_id": row["task_id"], "parent_reward": row["reward"], "frac": frac,
                         "fork_step": k, "sample": i, "reward": ep.reward.reward,
                         "status": ep.result.status, "reason": ep.result.reason,
                         "steps": ep.result.steps, "r_actions": ep.reward.r_actions})
        print(f"task {row['task_id']:3d} frac {frac:.2f} step {k:2d} #{i}: "
              f"parent={row['reward']:.0f} fork={ep.reward.reward:.0f}", flush=True)

    await asyncio.gather(*(one(r, f, i) for r in episodes for f in fracs for i in range(args.n)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out_rows, indent=1), encoding="utf-8")

    # agreement with the parent outcome, by fork position
    agree: dict[float, list[int]] = defaultdict(list)
    for r in out_rows:
        agree[r["frac"]].append(int((r["reward"] >= 1.0) == (r["parent_reward"] >= 1.0)))
    print("\nfork position -> agreement with parent outcome")
    for f in sorted(agree):
        v = agree[f]
        print(f"  {f:.2f}: {sum(v) / len(v):.3f}  (n={len(v)})")

    # check 1: fork-at-start pass rate vs. ordinary reruns of the same tasks
    if args.reruns:
        rr = json.loads(Path(args.reruns).read_text(encoding="utf-8"))
        rerun_rate: dict[int, list[float]] = defaultdict(list)
        for r in rr:
            rerun_rate[r["task_id"]].append(float(r["reward"] >= 1.0))
        start = [r for r in out_rows if r["frac"] == min(fracs)]
        tasks = sorted({r["task_id"] for r in start} & set(rerun_rate))
        if tasks:
            f_rate = sum(r["reward"] >= 1.0 for r in start if r["task_id"] in tasks) / \
                sum(1 for r in start if r["task_id"] in tasks)
            r_rate = sum(sum(rerun_rate[t]) / len(rerun_rate[t]) for t in tasks) / len(tasks)
            print(f"\nfork-at-start pass rate {f_rate:.3f} vs rerun pass rate {r_rate:.3f} "
                  f"on {len(tasks)} shared tasks")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="results json from run_taubench_rewind.py")
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--reruns", help="multi-trial results json of the same tasks (for check 1)")
    ap.add_argument("--fractions", default="0,0.25,0.5,0.75")
    ap.add_argument("--n", type=int, default=5, help="null forks per (episode, position)")
    ap.add_argument("--max-episodes", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--user-model", default="deepseek-chat")
    ap.add_argument("--user-provider", default="deepseek")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
