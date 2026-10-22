#!/usr/bin/env python3
"""Explain a gate decision from the ledger (free: no API calls).

For a gated edit, print the rule, the efficacy test on held-out targets, the
pool comparison, and WHERE the regressions and improvements landed: for each
mutating tool, how concentrated the broken tasks are on tasks that use it
compared with the pool, with a one-sided binomial tail -- "if breakage were
random over the pool, how likely is at least this many broken tasks using
this tool?". Concentration far above the pool rate is evidence of a real side
effect rather than re-run noise.

Usage:
  python scripts/evolve_gate_report.py                      # latest gate decision
  python scripts/evolve_gate_report.py --edit-id e_ab12...  # a specific edit
  python scripts/evolve_gate_report.py --group modify_pending_order_items,exchange_delivered_order_items
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rewind.env.taubench_adapter import MUTATING_TOOLS  # noqa: E402
from rewind.env.taubench_dialogue import load_task  # noqa: E402


def tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def tools_of(split: str, t: int) -> set[str]:
    return {a.name for a in load_task(split, t).actions if a.name in MUTATING_TOOLS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="runs/p1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--edit-id")
    ap.add_argument("--group", help="comma-separated tools to test as one group")
    a = ap.parse_args()

    db = sqlite3.connect(str(Path(a.run_dir) / "ledger.sqlite"))
    q = "SELECT edit_json, data_json, accepted, reason, cluster FROM decisions WHERE stage='gate'"
    row = (db.execute(q + " AND edit_id=? ORDER BY id DESC LIMIT 1", (a.edit_id,)).fetchone()
           if a.edit_id else db.execute(q + " ORDER BY id DESC LIMIT 1").fetchone())
    if row is None:
        sys.exit("no gate decision found")
    edit, v, accepted, reason, cluster = json.loads(row[0]), json.loads(row[1]), row[2], row[3], row[4]

    print(f"rule [{edit['kind']} {edit['target']}] from cluster {cluster}:\n  {edit['text']}\n")
    print(f"verdict: {'ACCEPT' if accepted else 'REJECT'} -- {reason}")
    eff = v.get("efficacy")
    if eff:
        print(f"efficacy on held-out targets {eff['tasks']}: {eff['cand_passes']}/{eff['cand_n']} "
              f"vs null {eff['null_passes']}/{eff['null_n']}, lift {eff['lift']:+.2f}, "
              f"one-sided Fisher p = {eff['p_value']:.4f}")
    print(f"pool: {v['n_tasks']} tasks, pass^1 {v['incumbent_pass1']:.3f} -> {v['candidate_pass1']:.3f}, "
          f"delta {v['delta']:+.3f}, 95% CI [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}], "
          f"{len(v['improvements'])} fixed, {len(v['regressions'])} broken, {v['episodes']} episodes\n")

    pool_rows = json.loads(Path(v["rows_path"]).read_text(encoding="utf-8"))
    pool = sorted({r["task_id"] for r in pool_rows})
    tools = {t: tools_of(a.split, t) for t in pool}
    broken, fixed = v["regressions"], v["improvements"]

    groups = [(name, {name}) for name in sorted(MUTATING_TOOLS)]
    if a.group:
        g = set(a.group.split(","))
        groups.insert(0, (" + ".join(sorted(g)), g))
    print(f"{'tool (group)':58s} {'pool':>6s} {'broken':>8s} {'P(>=k)':>8s} {'fixed':>7s}")
    for name, g in groups:
        p = sum(1 for t in pool if tools[t] & g) / len(pool)
        kb = sum(1 for t in broken if tools.get(t, set()) & g)
        kf = sum(1 for t in fixed if tools.get(t, set()) & g)
        if kb == 0 and kf == 0 and not a.group:
            continue
        print(f"{name:58s} {p:6.2f} {kb:4d}/{len(broken):<3d} {tail(kb, len(broken), p):8.4f} "
              f"{kf:3d}/{len(fixed):<3d}")
    print("\nbroken tasks:")
    for t in broken:
        print(f"  {t:4d} {sorted(tools.get(t, set()))}")
    print("fixed tasks:")
    for t in fixed:
        print(f"  {t:4d} {sorted(tools.get(t, set()))}")


if __name__ == "__main__":
    main()
