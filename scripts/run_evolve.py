#!/usr/bin/env python3
"""Self-improvement loop (P1) on tau-bench retail train tasks.

  plan      worst-case API use of the next round -- NO API calls
  round     run one round (asks for confirmation unless --yes)
  status    versions, decisions, shown tasks -- NO API calls
  rollback  make an earlier version current -- NO API calls
  gate-pending  gate edits that passed screening but were not gated (budget)
  retry     another proposal for a cluster whose A/B split exists, with the
            rejection feedback shown to the proposer (no stability re-run)
  regate    re-decide an earlier gate rejection under the current policy
            (reuses efficacy + pool episodes; only flip confirmation is new)

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
from rewind.evolve.loop import (  # noqa: E402
    EvolveConfig, _gate_reserve, backfill_feedback, gate_pending, pending_candidates, plan,
    regate, regate_candidates, retry_candidates, retry_cluster, run_round,
)
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
                        max_drop=a.max_drop, confirm_k=a.confirm_k,
                        max_confirmed_regressions=a.max_confirmed_regressions,
                        regression_budget=a.regression_budget),
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
    ap.add_argument("cmd", choices=["plan", "round", "status", "rollback", "gate-pending", "retry",
                                    "regate"])
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
    ap.add_argument("--max-drop", type=float, default=None,
                    help="optional point-drop cap; off by default (miscalibrated, see dev log)")
    ap.add_argument("--confirm-k", type=int, default=2, help="re-runs per flipped task, per version")
    ap.add_argument("--max-confirmed-regressions", type=int, default=1)
    ap.add_argument("--cluster", help="retry: which cluster (default: the most recent one)")
    ap.add_argument("--max-attempts", type=int, default=3, help="retry: proposer batches per cluster")
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

    def clients():
        return (DeepSeekProvider(model=a.model), litellm_user_llm(model=a.model, provider="deepseek"),
                SnapshotStore(a.store_dir))

    def ask(prompt: str) -> bool:
        return a.yes or input(prompt).strip().lower() == "y"

    if a.cmd == "regate":
        n = backfill_feedback(led, cfg)
        cands = regate_candidates(led)
        if not cands:
            print("nothing to re-gate" + (f" (attached feedback to {n} old rejection(s))" if n else ""))
            return
        for c in cands:
            v = c["verdict"]
            flips = len(v["regressions"]) + len(v["improvements"])
            print(f"decision {c['decision_id']}: {c['edit'].kind} {c['edit'].target}\n  was: {v['reason']}\n"
                  f"  efficacy and {v['n_tasks']} pool episodes are reused; new cost: confirm "
                  f"{flips} flipped tasks x {cfg.gate.confirm_k} re-runs x 2 versions = "
                  f"{flips * cfg.gate.confirm_k * 2} episodes"
                  + (" (+ the unfinished half of the pool)" if v.get("stopped_early") else ""))
        if not ask("\nre-gate now? [y/N] "):
            return
        prov, user, store = clients()
        res = asyncio.run(regate(cfg, led, provider=prov, user_llm=user, store=store,
                                 log=lambda m: print(m, flush=True)))
        print(json.dumps([{"target": r["edit"]["target"], "was": r["was"], "accepted": r["verdict"]["accepted"],
                           "reason": r["verdict"]["reason"]} for r in res], indent=1, ensure_ascii=False))
        return

    if a.cmd == "retry":
        backfill_feedback(led, cfg)
        cands = [c for c in retry_candidates(led, cfg) if a.cluster in (None, c["cluster"])]
        if not cands:
            print("no cluster to retry at this version")
            return
        c = cands[-1] if a.cluster is None else cands[0]
        rows = json.loads(Path(led.rows_path(led.current().version)).read_text())
        pool = sorted({r["task_id"] for r in rows} - led.shown())
        worst = cfg.n_candidates * len(c["B"]) * cfg.screen.n + \
            _gate_reserve(cfg, pool, len(c["B"]))
        print(f"retry {c['cluster']} (attempt {c['attempts'] + 1} of max {a.max_attempts})\n"
              f"  A={[r.task_id for r in c['A']]} B={[r.task_id for r in c['B']]}\n"
              f"  proposer: 1 call, sees this cluster's rejections with their feedback\n"
              f"  worst case: <= {cfg.n_candidates * len(c['B']) * cfg.screen.n} screening forks + "
              f"<= {_gate_reserve(cfg, pool, len(c['B']))} gate episodes = {worst}; budget {cfg.budget_episodes}")
        for r in led.rejected_for_prompt(c["cluster"])[:3]:
            if r.get("feedback"):
                print("\n  feedback the proposer will see:\n    " + r["feedback"].replace("\n", "\n    "))
                break
        if not ask("\nretry now? [y/N] "):
            return
        prov, user, store = clients()
        registry = build_dialogue_registry()
        proposer = LLMProposer(deepseek_json_llm(model=a.model), sorted(registry._tools))  # noqa: SLF001
        rep = asyncio.run(retry_cluster(cfg, led, provider=prov, user_llm=user, store=store,
                                        proposer=proposer, cluster_key=c["cluster"],
                                        max_attempts=a.max_attempts, log=lambda m: print(m, flush=True)))
        print(json.dumps({"round": rep.round, "from": rep.version_from, "to": rep.version_to,
                          "episodes_used": rep.episodes_used}, indent=1))
        return

    if a.cmd == "gate-pending":
        pend = pending_candidates(led, cfg)
        if not pend:
            print("no edit is waiting for the gate")
            return
        for p in pend:
            e = p["edit"]
            pool = len({r["task_id"] for r in json.loads(Path(led.rows_path(led.current().version))
                                                         .read_text())} - led.shown())
            print(f"pending: {e.kind} {e.target}: {e.text!r}\n  cluster {p['cluster']}, "
                  f"targets {[t[0] for t in p['targets']]} (null passes "
                  f"{[f'{t[2]}/{t[3]}' for t in p['targets']]})\n  cost <= "
                  f"{len(p['targets']) * cfg.gate.efficacy_n} efficacy forks, then <= "
                  f"{pool * cfg.gate.trials} pool episodes only if efficacy passes")
        if not a.yes and input("\ngate now? [y/N] ").strip().lower() != "y":
            return
        res = asyncio.run(gate_pending(cfg, led, provider=DeepSeekProvider(model=a.model),
                                       user_llm=litellm_user_llm(model=a.model, provider="deepseek"),
                                       store=SnapshotStore(a.store_dir),
                                       log=lambda m: print(m, flush=True)))
        print(json.dumps([{"target": r["edit"]["target"], "accepted": r["verdict"]["accepted"],
                           "reason": r["verdict"]["reason"], "episodes": r["verdict"]["episodes"]}
                          for r in res], indent=1, ensure_ascii=False))
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
