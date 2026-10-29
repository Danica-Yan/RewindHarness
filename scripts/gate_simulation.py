#!/usr/bin/env python3
"""Reproduce the gate audit by simulation -- NO API calls.

Task-level flakiness is a mixture fitted to the P0 null-fork data: 80% of
passing tasks re-pass with p = 0.97, 14% with p = 0.7, 6% with p = 0.1.
A HARMLESS candidate has the same per-task p as the incumbent; a BROKEN one
additionally fails `--broken` tasks (p = 0.03). For each safety rule the
simulation reports how often a harmless candidate is rejected (false
rejection) and how often a broken one is caught (power), plus the expected
gate episodes under the rule the gate uses now.

  python scripts/gate_simulation.py               # defaults: pool 188, 2000 runs
"""
from __future__ import annotations

import argparse
import random


def sim(pool: int, runs: int, broken: int, seed: int = 0) -> dict:
    rng = random.Random(seed)
    mix = [(0.80, 0.97), (0.14, 0.70), (0.06, 0.10)]

    def draw_p() -> float:
        u, acc = rng.random(), 0.0
        for w, p in mix:
            acc += w
            if u < acc:
                return p
        return mix[-1][1]

    rules = {"old: drop > 2 pt": 0, "confirm any, k=2": 0, "confirm all, k=2 (used)": 0, "confirm all, k=3": 0}
    out = {"harmless": dict(rules), "broken": dict(rules), "episodes": {"harmless": [], "broken": []}}
    for kind in ("harmless", "broken"):
        for _ in range(runs):
            ps = [draw_p() for _ in range(pool)]
            cand_ps = list(ps)
            if kind == "broken":
                for i in rng.sample([i for i, p in enumerate(ps) if p > 0.9], broken):
                    cand_ps[i] = 0.03
            inc = [rng.random() < p for p in ps]
            cand = [rng.random() < p for p in cand_ps]
            regs = [i for i in range(pool) if inc[i] and not cand[i]]
            imps = [i for i in range(pool) if cand[i] and not inc[i]]
            drop = (sum(inc) - sum(cand)) / pool
            if drop > 0.02:
                out[kind]["old: drop > 2 pt"] += 1
            for name, k, mode in (("confirm any, k=2", 2, "any"), ("confirm all, k=2 (used)", 2, "all"),
                                  ("confirm all, k=3", 3, "all")):
                confirmed = 0
                for i in regs:
                    ci = [rng.random() < cand_ps[i] for _ in range(k)]
                    ii = [rng.random() < ps[i] for _ in range(k)]
                    inc_ok = any(ii) if mode == "any" else all(ii)
                    if not any(ci) and inc_ok:
                        confirmed += 1
                if confirmed > 1:
                    out[kind][name] += 1
            out["episodes"][kind].append(pool + 2 * 2 * (len(regs) + len(imps)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=int, default=188)
    ap.add_argument("--runs", type=int, default=2000)
    ap.add_argument("--broken", type=int, default=5)
    a = ap.parse_args()
    r = sim(a.pool, a.runs, a.broken)
    print(f"pool {a.pool} tasks, {a.runs} simulated gates per row, broken candidate = {a.broken} tasks\n")
    print(f"{'safety rule':28s} {'harmless rejected':>18s} {'broken caught':>14s}")
    for name in r["harmless"]:
        print(f"{name:28s} {r['harmless'][name] / a.runs:>18.1%} {r['broken'][name] / a.runs:>14.1%}")
    for kind in ("harmless", "broken"):
        e = sorted(r["episodes"][kind])
        print(f"\nexpected gate episodes, {kind} candidate (pool + confirmation): "
              f"mean {sum(e) / len(e):.0f}, 90th pct {e[int(0.9 * len(e))]}")
    print("\nWhy no sequential (SPRT) pool: confirmation needs every flipped task, so the pool must be run in\n"
          "full for any candidate that is not clearly harmful; the futility stop (half the pool) already\n"
          "covers the clearly harmful case. The pool is about two thirds of the gate cost above.")


if __name__ == "__main__":
    main()
