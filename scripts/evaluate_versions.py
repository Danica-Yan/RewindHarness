#!/usr/bin/env python3
"""Does what the loop learned on TRAIN carry over to TEST? Compare ledger
versions on a split the loop never saw.

  plan     cost of the comparison -- NO API calls
  run      run each version on every task of the split (resumable; results
           of a version you already have can be reused with --reuse)
  compare  pass^1 per version, paired bootstrap CI of the change (tasks are
           the resampling unit), and the change by task type -- NO API calls

The loop never reads test results: they are written under
<run-dir>/eval/, which no Evolve command looks at.

Usage:
  export DEEPSEEK_API_KEY=sk-... REWIND_TAUBENCH_SRC=../tau-bench
  python scripts/evaluate_versions.py plan    --versions 0 3
  python scripts/evaluate_versions.py run     --versions 0 3 --reuse 0=runs/p0/test_rewind_3trials.json
  python scripts/evaluate_versions.py compare --versions 0 3 --reuse 0=runs/p0/test_rewind_3trials.json
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rewind.env.taubench_dialogue import n_tasks  # noqa: E402
from rewind.evolve.ledger import EditLedger  # noqa: E402
from rewind.evolve.runner import load_rows  # noqa: E402
from rewind.evolve.skills import task_writes  # noqa: E402
from rewind.evolve.stats import (  # noqa: E402
    mean_delta, paired_bootstrap_ci, per_task_rewards, pooled_pass1, task_means,
)


def out_path(run_dir: str, split: str, v: int) -> Path:
    return Path(run_dir) / "eval" / f"{split}_v{v}.json"


def rows_for(a: argparse.Namespace, v: int) -> list[dict]:
    if v in a.reuse_map:
        return load_rows(a.reuse_map[v])
    return load_rows(out_path(a.run_dir, a.split, v))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "run", "compare"])
    ap.add_argument("--versions", type=int, nargs="+", required=True)
    ap.add_argument("--run-dir", default="runs/p1")
    ap.add_argument("--store-dir", default="runs/p1/snapshots")
    ap.add_argument("--split", default="test")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--reuse", action="append", default=[], metavar="V=PATH",
                    help="existing results for version V on this split (e.g. a baseline run)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    a.reuse_map = {int(k): v for k, v in (x.split("=", 1) for x in a.reuse)}
    for v, path in a.reuse_map.items():
        if not Path(path).exists():
            sys.exit(f"--reuse {v}={path}: no such file (find it with: find runs -name '*.json' | grep -i test)")
        if not load_rows(path):
            sys.exit(f"--reuse {v}={path}: no result rows in this file")
    if a.split == "train":
        sys.exit("evaluate on a split the loop did not learn from (e.g. test)")
    led = EditLedger(Path(a.run_dir) / "ledger.sqlite")
    tasks = list(range(n_tasks(a.split)))

    if a.cmd in ("plan", "run"):
        todo = {}
        for v in a.versions:
            led.version(v)  # raises if unknown
            if v in a.reuse_map:
                print(f"  v{v}: reusing {a.reuse_map[v]} ({len(load_rows(a.reuse_map[v]))} rows)")
                continue
            have = {(r["task_id"], r["trial"]) for r in load_rows(out_path(a.run_dir, a.split, v))}
            todo[v] = sum((t, k) not in have for t in tasks for k in range(a.trials))
            print(f"  v{v}: {todo[v]} episodes to run ({len(have)} already done)")
        print(f"total: {sum(todo.values())} episodes on {len(tasks)} {a.split} tasks x {a.trials} trial(s)")
        if a.cmd == "plan" or not todo or not (a.yes or input("run? [y/N] ").strip().lower() == "y"):
            return
        from rewind.env.taubench_dialogue import litellm_user_llm
        from rewind.evolve.runner import run_batch
        from rewind.providers import DeepSeekProvider
        from rewind.snapshot import SnapshotStore
        for v in todo:
            asyncio.run(run_batch(a.split, tasks, a.trials, out_path(a.run_dir, a.split, v), ctx=led.version(v),
                                  provider=DeepSeekProvider(model=a.model),
                                  user_llm=litellm_user_llm(model=a.model, provider="deepseek"),
                                  store=SnapshotStore(a.store_dir), concurrency=a.concurrency,
                                  log=lambda m: print(m, flush=True)))
        return

    # compare
    by_v = {v: per_task_rewards(rows_for(a, v)) for v in a.versions}
    for v, rw in by_v.items():
        p, n = pooled_pass1(rw)
        print(f"v{v}: pass^1 {p:.1%} over {len(rw)} tasks ({n} episodes)")
    base = a.versions[0]
    for v in a.versions[1:]:
        common = sorted(set(by_v[base]) & set(by_v[v]))
        d = mean_delta(common, by_v[base], by_v[v])
        lo, hi = paired_bootstrap_ci(common, by_v[base], by_v[v])
        verdict = ("better" if lo > 0 else "worse" if hi < 0 else "no detectable difference")
        print(f"\nv{v} vs v{base} on {len(common)} common tasks: {d:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] "
              f"-> {verdict}")
        ma, mb = task_means(by_v[base]), task_means(by_v[v])
        groups = defaultdict(list)
        for t in common:
            for w in sorted(task_writes(a.split, t)) or ["(no write)"]:
                groups[w].append(mb[t] - ma[t])
        print("  by task type (mean change, tasks):")
        for w, ds in sorted(groups.items()):
            print(f"    {w:34s} {sum(ds) / len(ds):+.3f}  ({len(ds)})")
    print("\nA learned rule targets a narrow situation, so the overall change is expected to be small; "
          "'no detectable difference' with a CI near zero is evidence of no harm, not of no effect.")


if __name__ == "__main__":
    main()
