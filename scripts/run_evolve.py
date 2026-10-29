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
  consolidate  for each learned item, test the version without it on the
            fixes it is remembered for; remove it if that costs nothing
  reflect   one LLM call: rewrite the proposer's lessons from the ledger
            history -> <run-dir>/meta/proposed_lessons.json (not adopted yet)
  meta-eval old vs proposed lessons on recorded clusters (validation +
            screening); adopts the proposed lessons only if they win
  report    funnel, episodes per stage and per accepted edit, and what each
            filter saved -- NO API calls (--out file.md to save it)

Regression memory: every task an accepted version fixed is re-checked by
every later gate; `status` lists them (versions from before it existed are
backfilled automatically, no API).

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
    EvolveConfig, _gate_reserve, backfill_feedback, backfill_memory, gate_pending, pending_candidates,
    plan, regate, regate_candidates, retry_candidates, retry_cluster, run_round,
)
from rewind.evolve import consolidate as consol  # noqa: E402
from rewind.evolve import integrity, meta  # noqa: E402
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
                        max_confirmed_regressions=a.max_confirmed_regressions, regression_k=a.regression_k,
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
                                    "regate", "consolidate", "reflect", "meta-eval", "report"])
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
    ap.add_argument("--regression-k", type=int, default=3, help="forks per remembered fix, per version (0 = off)")
    ap.add_argument("--item", help="consolidate: only this rule/note/check/skill")
    ap.add_argument("--accept-judge-change", metavar="REASON",
                    help="the gate/screening code or settings changed on purpose: record why and continue")
    ap.add_argument("--meta-clusters", type=int, default=2, help="meta-eval: clusters to compare on")
    ap.add_argument("--out", help="report: also write the markdown here")
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
    if (n := backfill_memory(led, cfg)):
        print(f"regression memory: backfilled {n} fixed task(s) from earlier versions")
    if a.cmd == "report":
        from rewind.evolve.report import report
        md = report(led, cfg)
        print(md)
        if a.out:
            Path(a.out).write_text(md, encoding="utf-8")
        return
    if a.cmd not in ("status", "plan", "rollback", "reflect"):
        try:
            integrity.verify(led, cfg, a.accept_judge_change)
        except integrity.JudgeChanged as e:
            sys.exit(str(e))
    if a.cmd == "status":
        cur = led.current()
        print(f"current version: v{cur.version}")
        jh = integrity.history(led)
        print(f"judge: {jh[-1]['id'] if jh else 'not recorded yet'} ({len(jh)} recorded change(s))")
        print(f"regression memory: {[t for t, _ in led.regression_suite()] or 'empty'}")
        print("proposer lessons: " + ("none" if not meta.lessons(led) else ""))
        for x in meta.lessons(led):
            print(f"  - {x}")
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

    if a.cmd == "consolidate":
        items = [i for i in consol.plan(led, cfg.gate) if a.item in (None, i["target"])]
        if not items:
            print("nothing learned yet")
            return
        for i in items:
            print(f"  {i['kind']} {i['target']}: remembered fixes {i['evidence'] or 'none (skipped)'}; "
                  f"<= {i['cost']} episodes")
        if not ask("\nablate now? [y/N] "):
            return
        prov, user, store = clients()
        res = asyncio.run(consol.ablate(led, cfg.gate, provider=prov, user_llm=user, store=store, only=a.item,
                                        log=lambda m: print(m, flush=True)))
        print(json.dumps(res, indent=1, ensure_ascii=False))
        return

    if a.cmd == "reflect":
        exp = meta.experience(led)
        print(f"{len(exp)} proposals in the ledger; current lessons: {meta.lessons(led) or 'none'}")
        if not exp:
            return
        if not ask("one LLM call to propose new lessons? [y/N] "):
            return
        new, why, note, raw = asyncio.run(meta.reflect(led, deepseek_json_llm(model=a.model, temperature=0.3)))
        out = Path(a.run_dir) / "meta" / "proposed_lessons.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        (out.parent / "last_reflection_raw.txt").write_text(raw, encoding="utf-8")  # never lose a call
        if note:
            print(note)
        if not new:
            print(f"nothing adoptable; raw output kept in {out.parent / 'last_reflection_raw.txt'}")
            return
        out.write_text(json.dumps({"lessons": new, "rationale": why}, ensure_ascii=False, indent=1))
        print("proposed lessons:\n" + "\n".join(f"  - {x}" for x in new) + f"\n\nrationale: {why}\n\n"
              f"wrote {out}; `meta-eval` decides whether they are adopted")
        return

    if a.cmd == "meta-eval":
        src = Path(a.run_dir) / "meta" / "proposed_lessons.json"
        if not src.exists():
            sys.exit("no proposed lessons: run `reflect` first")
        new, note = meta.validate_lessons(json.loads(src.read_text())["lessons"])
        if not new:
            sys.exit(f"proposed lessons invalid: {note}")
        cl = meta.meta_clusters(led, cfg)[: a.meta_clusters]
        if not cl:
            print("no cluster with a recorded A/B split to compare on")
            return
        print(f"compare old vs proposed lessons on {[c['cluster'] for c in cl]}: "
              f"{2 * len(cl)} proposer calls, <= {meta.meta_cost(cfg, cl)} screening forks")
        if not ask("run? [y/N] "):
            return
        prov, user, store = clients()
        registry = build_dialogue_registry()
        res = asyncio.run(meta.meta_eval(cfg, led, new, cl, llm=deepseek_json_llm(model=a.model),
                                         tool_names=sorted(registry._tools), provider=prov,  # noqa: SLF001
                                         user_llm=user, store=store, log=lambda m: print(m, flush=True)))
        print(json.dumps(res, indent=1, ensure_ascii=False))
        return

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
        proposer = LLMProposer(deepseek_json_llm(model=a.model), sorted(registry._tools),  # noqa: SLF001
                               lessons=meta.lessons(led))
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
    proposer = LLMProposer(deepseek_json_llm(model=a.model), sorted(registry._tools),  # noqa: SLF001
                               lessons=meta.lessons(led))
    rep = asyncio.run(run_round(cfg, led, provider=DeepSeekProvider(model=a.model),
                                user_llm=litellm_user_llm(model=a.model, provider="deepseek"),
                                store=SnapshotStore(a.store_dir), proposer=proposer,
                                log=lambda m: print(m, flush=True)))
    print(json.dumps({"round": rep.round, "from": rep.version_from, "to": rep.version_to,
                      "episodes_used": rep.episodes_used, "proposer_calls": rep.proposer_calls},
                     indent=1))


if __name__ == "__main__":
    main()
