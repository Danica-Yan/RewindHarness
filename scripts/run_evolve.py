#!/usr/bin/env python3
"""Self-improvement loop (P1) on tau-bench retail train tasks.

  plan      worst-case API use of the next round -- NO API calls
  round     run one round (asks for confirmation unless --yes)
  status    versions, decisions, shown tasks -- NO API calls
  rollback  make an earlier version current -- NO API calls

The ledger, stability cache, gate results, round reports and SFT data all
live under --run-dir. First use registers the baseline results file as v0.

Usage:
  export DEEPSEEK_API_KEY=sk-... REWIND_TAUBENCH_SRC=../tau-bench
  python scripts/run_evolve.py plan  --base runs/p1/train_base.json --store-dir runs/p1/snapshots
  python scripts/run_evolve.py round --base runs/p1/train_base.json --store-dir runs/p1/snapshots \
      --budget 250
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rewind.env.taubench_dialogue import build_dialogue_registry, litellm_user_llm  # noqa: E402
from rewind.evolve.gate import GatePolicy  # noqa: E402
from rewind.evolve.ledger import EditLedger  # noqa: E402
from rewind.evolve.loop import EvolveConfig, plan, run_round  # noqa: E402
from rewind.evolve.prevalence import PrevalencePolicy  # noqa: E402
from rewind.evolve.proposer import LLMProposer, deepseek_json_llm  # noqa: E402
from rewind.evolve.screen import ScreenPolicy  # noqa: E402
from rewind.providers import DeepSeekProvider  # noqa: E402
from rewind.snapshot import SnapshotStore  # noqa: E402


def make_cfg(a: argparse.Namespace) -> EvolveConfig:
    return EvolveConfig(
        split=a.split, run_dir=a.run_dir,
        prevalence=PrevalencePolicy(min_distinct_tasks=a.min_tasks, min_share_lb=a.min_share_lb),
        stability_n=a.stability_n, screen=ScreenPolicy(n=a.screen_n, tau=a.tau),
        gate=GatePolicy(mode=a.gate_mode, trials=a.gate_trials, efficacy_n=a.efficacy_n,
                        max_drop=a.max_drop, regression_budget=a.regression_budget),
        n_candidates=a.candidates, max_clusters_per_round=a.max_clusters,
        max_members_per_cluster=a.max_members, budget_episodes=a.budget, concurrency=a.concurrency)


def open_ledger(a: argparse.Namespace) -> EditLedger:
    led = EditLedger(Path(a.run_dir) / "ledger.sqlite")
    if a.base:
        led.init_base(a.base)
    try:
        led.version(0)
    except KeyError:
        sys.exit(f"no baseline registered in {a.run_dir}: pass --base <train results json> once")
    return led


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "round", "status", "rollback"])
    ap.add_argument("--base", help="baseline results json (registered as v0 on first use)")
    ap.add_argument("--run-dir", default="runs/p1")
    ap.add_argument("--store-dir", default="runs/p1/snapshots")
    ap.add_argument("--split", default="train")
    ap.add_argument("--budget", type=int, default=250, help="hard cap on episodes this round")
    ap.add_argument("--yes", action="store_true", help="do not ask before spending API calls")
    ap.add_argument("--to", type=int, help="rollback target version")
    ap.add_argument("--min-tasks", type=int, default=4)
    ap.add_argument("--min-share-lb", type=float, default=0.05)
    ap.add_argument("--stability-n", type=int, default=5)
    ap.add_argument("--screen-n", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--gate-trials", type=int, default=1)
    ap.add_argument("--gate-mode", choices=["targeted", "superiority"], default="targeted")
    ap.add_argument("--efficacy-n", type=int, default=5)
    ap.add_argument("--max-drop", type=float, default=0.02)
    ap.add_argument("--regression-budget", type=int, default=None,
                    help="optional hard cap on raw regressions (noisy at high baselines)")
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--max-clusters", type=int, default=2)
    ap.add_argument("--max-members", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--model", default="deepseek-chat")
    a = ap.parse_args()

    led = open_ledger(a)
    cfg = make_cfg(a)
    if a.cmd == "status":
        cur = led.current()
        print(f"current version: v{cur.version}")
        for v, edits in led.lineage(cur.version):
            for e in edits:
                print(f"  v{v}: {e['kind']} {e['target']}: {e.get('text')}")
        print(f"tasks shown to proposer/screening: {sorted(led.shown())}")
        for d in led.decisions():
            print(f"  round {d['round']} v{d['version_from']} {d['stage']:9s} "
                  f"{'ACCEPT' if d['accepted'] else 'reject':6s} {d['edit_id'] or '':16s} {d['reason']}")
        return
    if a.cmd == "rollback":
        print(f"current is now v{led.rollback(a.to).version}")
        return

    print(plan(cfg, led))
    if a.cmd == "plan":
        return
    if not a.yes and input("\nrun this round? [y/N] ").strip().lower() != "y":
        return
    registry = build_dialogue_registry()
    proposer = LLMProposer(deepseek_json_llm(model=a.model), sorted(registry._tools))  # noqa: SLF001
    rep = asyncio.run(run_round(cfg, led, provider=DeepSeekProvider(model=a.model),
                                user_llm=litellm_user_llm(model=a.model, provider="deepseek"),
                                store=SnapshotStore(a.store_dir), proposer=proposer,
                                log=lambda m: print(m, flush=True)))
    print(json.dumps({"round": rep.round, "from": rep.version_from, "to": rep.version_to,
                      "episodes_used": rep.episodes_used, "proposer_calls": rep.proposer_calls},
                     indent=1))


if __name__ == "__main__":
    main()
