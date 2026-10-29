#!/usr/bin/env python3
"""Skill library: induce procedures from successful episodes, gate them, read usage.

  plan    groups of passing episodes per write tool, the examples each skill
          would be induced from, which failures can be gate targets (stable,
          never shown), and the API cost of induce + gate -- NO API
  induce  one LLM call per group -> <run-dir>/skills/candidates.json (validated)
  gate    gate all valid candidates as ONE new version (asks first)
  usage   which skills episodes loaded, and pass rates loaded vs not -- NO API

Induction reads PASSING train episodes (and with --contrast also failures of
"A"-half tasks); the gate's targets are STABLE failures the inducer never saw,
so evidence and evaluation never overlap.

Usage:
  export DEEPSEEK_API_KEY=sk-... REWIND_TAUBENCH_SRC=../tau-bench
  python scripts/induce_skills.py plan   --base runs/p1/train_base.json
  python scripts/induce_skills.py induce --base runs/p1/train_base.json
  python scripts/induce_skills.py gate   --base runs/p1/train_base.json --store-dir runs/p1/snapshots
  python scripts/induce_skills.py usage  --results runs/p1/gate/v2_skills_eff_candidate.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rewind.evolve.ledger import EditLedger  # noqa: E402
from rewind.evolve.loop import (  # noqa: E402
    EvolveConfig, _pool, gate_skill_library, skill_gate_cost, skill_target_plan,
)
from rewind.evolve.runner import load_rows, version_setup  # noqa: E402
from rewind.evolve.skills import (  # noqa: E402
    covers_from_ledger, induce, induce_messages, skill_groups, skill_usage,
)
from rewind.evolve.surface import Edit, validate_edit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "induce", "gate", "usage"])
    ap.add_argument("--base", help="baseline results json (passing episodes = evidence, failing = targets)")
    ap.add_argument("--run-dir", default="runs/p1")
    ap.add_argument("--store-dir", default="runs/p1/snapshots")
    ap.add_argument("--split", default="train")
    ap.add_argument("--min-passing", type=int, default=5, help="min distinct passing tasks per group")
    ap.add_argument("--examples", type=int, default=4, help="episodes shown per induction call")
    ap.add_argument("--max-targets", type=int, default=8)
    ap.add_argument("--contrast", action="store_true",
                    help="show failed episodes (A-half tasks) next to the successes when inducing")
    ap.add_argument("--efficacy-n", type=int, default=5)
    ap.add_argument("--results", action="append", help="usage: results json(s)")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--accept-judge-change", metavar="REASON",
                    help="gate code/settings changed on purpose: record why and continue")
    a = ap.parse_args()

    run = Path(a.run_dir)
    led = EditLedger(run / "ledger.sqlite")
    cfg = EvolveConfig(split=a.split, run_dir=a.run_dir, concurrency=a.concurrency)
    cfg.gate.efficacy_n = a.efficacy_n
    cand_path = run / "skills" / "candidates.json"

    if a.cmd == "usage":
        rows = [r for f in a.results or [] for r in json.loads(Path(f).read_text(encoding="utf-8"))]
        covers = covers_from_ledger(led)
        print(f"{len(rows)} episodes; skills in the current version: {sorted(led.current().skills) or 'none'}\n")
        print(f"{'skill':32s} {'for tool':32s} {'relevant':>8s} {'loaded':>7s} {'pass|loaded':>12s} "
              f"{'pass|not':>9s} {'elsewhere':>9s}")
        fmt = lambda x: "-" if x is None else f"{x:.0%}"  # noqa: E731
        for u in skill_usage(rows, a.split, covers):
            print(f"{u.skill:32s} {u.covers_tool or '?':32s} {u.relevant:>8d} {u.loaded:>7d} "
                  f"{fmt(u.pass_loaded):>12s} {fmt(u.pass_not_loaded):>9s} {u.loaded_elsewhere:>9d}")
        print("\nObservational: the agent decides when to load, so loaded-vs-not is a diagnostic, not proof.")
        return

    if not a.base:
        sys.exit("--base <baseline results json> is required")
    base_rows = load_rows(a.base)
    inc = led.current()
    _, reg = version_setup(inc)
    tools = sorted(reg._tools)  # noqa: SLF001
    covered = {e["source_cluster"].split(":", 1)[1] for _v, es in led.lineage(inc.version) for e in es
               if e.get("kind") == "skill" and str(e.get("source_cluster", "")).startswith("skill:")}
    groups = skill_groups(base_rows, a.split, min_passing=a.min_passing, k=a.examples, skip_tools=covered,
                          contrast=a.contrast)

    def target_summary(tp: dict, n_targets: int, pool: list[int]) -> str:
        return (f"  targets: {len(tp['failures'])} unseen failures of these types, "
                f"{len(tp['stable'])} known stable, {len(tp['missing'])} not yet assessed "
                f"(<= {tp['assess_cost']} null forks to assess)\n"
                f"  gate:    <= {tp['assess_cost'] + skill_gate_cost(cfg, pool, n_targets)} episodes in total "
                f"(up to {n_targets} targets x {a.efficacy_n} forks x 2 versions, {len(pool)} pool tasks, "
                f"flip confirmation)")

    if a.cmd == "plan":
        print(f"current version v{inc.version}; {len(base_rows)} baseline episodes "
              f"({sum(r['reward'] >= 1 for r in base_rows)} passed)\n")
        for g in groups:
            ex = [base_rows[i]["task_id"] for i in g.examples]
            fx = [base_rows[i]["task_id"] for i in g.failing_examples]
            print(f"  {g.tool:32s} {len(g.task_ids):3d} passing tasks; examples {ex}"
                  + (f"; failures shown {fx}" if a.contrast else ""))
        prompt_chars = sum(len(m["content"]) for g in groups
                           for m in induce_messages(g, base_rows, a.split, inc, tools))
        evidence = {base_rows[i]["task_id"] for g in groups for i in g.examples + g.failing_examples}
        tp = skill_target_plan(cfg, base_rows, {g.tool for g in groups}, evidence)
        n_t = min(a.max_targets, len(tp["stable"]) + len(tp["missing"]))
        pool = _pool(load_rows(led.rows_path(inc.version)), led.shown() | evidence
                     | {r["task_id"] for r in tp["failures"]})
        print(f"\ninduce: {len(groups)} LLM calls (~{prompt_chars // 4} prompt tokens in total), no episodes")
        print(target_summary(tp, n_t, pool))
        return

    if a.cmd == "induce":
        if not groups:
            print("no group with enough passing episodes")
            return
        print(f"inducing {len(groups)} skills ({len(groups)} LLM calls)")
        if not (a.yes or input("go? [y/N] ").strip().lower() == "y"):
            return
        from rewind.evolve.proposer import deepseek_json_llm
        res = asyncio.run(induce(groups, base_rows, a.split, inc, deepseek_json_llm(model=a.model, temperature=0.3),
                                 tools))
        out = []
        for g, e, raw in res:
            why = "unparseable output" if e is None else validate_edit(
                e, inc, known_tools=set(tools), max_tokens=cfg.gate.max_ctx_tokens)
            out.append({"tool": g.tool, "valid": why is None, "why": why,
                        "edit": e.model_dump() if e else None, "raw": raw})
            print(f"\n[{'ok' if why is None else 'INVALID: ' + why}] {g.tool}")
            if e:
                body = json.loads(e.text)
                print(f"  {e.target}: {body.get('description')}")
                for i, s in enumerate(body.get("steps") or [], 1):
                    print(f"    {i}. {s}")
        cand_path.parent.mkdir(parents=True, exist_ok=True)
        if cand_path.exists():  # keep the previous induction for comparison
            cand_path.replace(cand_path.with_name("candidates.prev.json"))
        cand_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nwrote {cand_path}; review it, edit or drop skills by hand if you like, then run `gate`")
        return

    # gate
    from rewind.evolve import integrity
    try:
        integrity.verify(led, cfg, a.accept_judge_change)
    except integrity.JudgeChanged as e:
        sys.exit(str(e))
    if not cand_path.exists():
        sys.exit(f"no candidates at {cand_path}: run `induce` first")
    edits = [Edit.model_validate(c["edit"]) for c in json.loads(cand_path.read_text(encoding="utf-8"))
             if c.get("valid") and c.get("edit")]
    tools_cov = {e.source_cluster.split(":", 1)[1] for e in edits}
    evidence = {t for e in edits for t in e.evidence_tasks}
    tp = skill_target_plan(cfg, base_rows, tools_cov, evidence)
    n_t = min(a.max_targets, len(tp["stable"]) + len(tp["missing"]))
    pool = _pool(load_rows(led.rows_path(inc.version)), led.shown() | evidence
                 | {r["task_id"] for r in tp["failures"]})
    print(f"gate {len(edits)} skills as one version on top of v{inc.version}: {[e.target for e in edits]}\n"
          f"  known stable targets: {[r['task_id'] for r in tp['stable']]}\n" + target_summary(tp, n_t, pool))
    if not (a.yes or input("gate now? [y/N] ").strip().lower() == "y"):
        return
    from rewind.env.taubench_dialogue import litellm_user_llm
    from rewind.providers import DeepSeekProvider
    from rewind.snapshot import SnapshotStore
    res = asyncio.run(gate_skill_library(cfg, led, edits, base_rows, max_targets=a.max_targets,
                                         provider=DeepSeekProvider(model=a.model),
                                         user_llm=litellm_user_llm(model=a.model, provider="deepseek"),
                                         store=SnapshotStore(a.store_dir), log=lambda m: print(m, flush=True)))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
