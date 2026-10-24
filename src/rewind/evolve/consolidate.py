"""Consolidation: does each learned item still earn its place?

Learned rules, notes, checks and skills only ever accumulate unless
something removes them. For each item in the current version this module
tests the version WITHOUT it on the tasks that item is remembered for
fixing (regression memory, source = the item), forked from step 1 under
both versions. An item is removed only if dropping it costs nothing there
(the version without it passes at least as often) AND the other remembered
fixes still hold; otherwise it is kept, with the numbers as the reason.

Deletion makes the context smaller and removes interactions between
rules -- the "consolidate" half of an accumulating library. Limitation:
the pool is not re-run, so a removal could undo help an item gave silently
elsewhere; the item's own fixes are the evidence that matters most and are
the ones re-checked.
"""
from __future__ import annotations

from typing import Any, Callable

from ..benchmark import Benchmark
from .gate import GatePolicy, efficacy, fisher_greater, regression_check
from .ledger import EditLedger
from .surface import ContextVersion, Edit

KINDS = (("rule", "rules"), ("tool_note", "tool_notes"), ("check", "checks"), ("skill", "skills"))


def items(ctx: ContextVersion) -> list[tuple[str, str]]:
    return [(kind, target) for kind, attr in KINDS for target in getattr(ctx, attr)]


def evidence(ledger: EditLedger, target: str) -> list[tuple[int, str]]:
    """Regression-memory tasks attributed to this item."""
    rows = ledger.db.execute("SELECT task_id, trace, source FROM regressions ORDER BY task_id").fetchall()
    return [(t, tr) for t, tr, src in rows if target in (src or "").split(",")]


def plan(ledger: EditLedger, policy: GatePolicy) -> list[dict]:
    out = []
    suite = ledger.regression_suite()
    for kind, target in items(ledger.current()):
        ev = evidence(ledger, target)
        others = [s for s in suite if s not in ev]
        out.append({"kind": kind, "target": target, "evidence": [t for t, _ in ev],
                    "cost": 2 * policy.efficacy_n * len(ev) + 2 * policy.regression_k * len(others)})
    return out


async def ablate(ledger: EditLedger, policy: GatePolicy, *, bench: Benchmark, provider: Any, user_llm: Any, store: Any,
                 only: str | None = None, log: Callable[[str], None] = print) -> list[dict]:
    results = []
    for kind, target in items(ledger.current()):
        if only and target != only:
            continue
        inc = ledger.current()
        ev = evidence(ledger, target)
        if not ev:
            results.append({"target": target, "removed": False, "episodes": 0,
                            "reason": "no remembered fixes to test it on"})
            continue
        cand = inc.apply(Edit(kind=kind, target=target, op="delete")).model_copy(
            update={"version": ledger.next_version_number(), "parent": inc.version})
        targets = [(t, tr, 0, 0) for t, tr in ev]
        log(f"ablating {kind} {target} on its remembered fixes {[t for t, _ in ev]}")
        with_it, n1 = await efficacy(inc, targets, policy=policy, bench=bench, provider=provider, user_llm=user_llm,
                                     store=store, log=lambda m: log("  [with]" + m))
        without, n2 = await efficacy(cand, targets, policy=policy, bench=bench, provider=provider, user_llm=user_llm,
                                     store=store, log=lambda m: log("  [without]" + m))
        used = n1 + n2
        k_with, k_without, n = with_it.cand_passes, without.cand_passes, with_it.cand_n
        p = fisher_greater(k_with, n, k_without, n)
        data = {"with": k_with, "without": k_without, "n": n, "p_value": p}
        if k_without < k_with:
            removed = False
            reason = f"still earns its place: without it {k_without}/{n} vs {k_with}/{n} with it (p = {p:.3f})"
        else:
            others = [s for s in ledger.regression_suite() if s not in ev]
            chk = None
            if others and policy.regression_k > 0:
                chk = await regression_check(cand, inc, others, k=policy.regression_k, bench=bench, provider=provider,
                                             user_llm=user_llm, store=store, log=log)
                used += chk["episodes"]
                data["regression_check"] = chk
            removed = chk is None or chk["passed"]
            reason = (f"costs nothing on its own fixes ({k_without}/{n} without vs {k_with}/{n} with)"
                      + ("" if chk is None else f"; other remembered fixes {chk['cand_passes']}/{chk['n']} vs "
                                                f"{chk['inc_passes']}/{chk['n']}")
                      + ("" if removed else " -- but other remembered fixes break, kept"))
        e = Edit(kind=kind, target=target, op="delete", rationale="consolidation by ablation")
        round_no = ledger.bump_round()
        ledger.record(round_no, inc.version, e, "ablation", removed, reason, data, cluster=f"ablate:{target}")
        if removed:
            ledger.commit(cand, [e], ledger.rows_path(inc.version), data)
            log(f"  removed -> v{cand.version}")
        log(f"  {'REMOVED' if removed else 'kept'}: {reason}")
        results.append({"target": target, "removed": removed, "reason": reason, "episodes": used,
                        "version": cand.version if removed else None})
    return results
