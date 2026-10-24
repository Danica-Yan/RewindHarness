"""Turn a gate verdict into feedback the proposer can act on.

"pool drop -0.037 exceeds 0.02" tells the proposer nothing about what to do
differently. The feedback built here says where the damage landed and shows
it: which tools the broken tasks concentrate on compared with the pool (with
a one-sided binomial tail, so a coincidence is labelled as one), and one
broken task's agent actions before (passed, incumbent) and after (failed,
candidate). Everything is computed from results already on disk.
"""
from __future__ import annotations

import json
from math import comb
from typing import Any

from pydantic import BaseModel

from ..benchmark import Benchmark


class ToolConcentration(BaseModel):
    tool: str
    pool_share: float
    broken: int
    n_broken: int
    p_value: float           # P(X >= broken) if breakage were random over the pool
    fixed: int


def binom_tail(k: int, n: int, p: float) -> float:
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def task_tools(split: str, task_id: int, *, bench: Benchmark) -> set[str]:
    return {a.name for a in bench.task(split, task_id).actions if a.name in bench.write_tools}


def concentration(pool: list[int], broken: list[int], fixed: list[int], split: str, *,
                  bench: Benchmark) -> list[ToolConcentration]:
    """Per mutating tool: how concentrated the broken tasks are on tasks that
    use it, most surprising first."""
    tools = {t: task_tools(split, t, bench=bench) for t in set(pool) | set(broken) | set(fixed)}
    out = []
    for tool in sorted(bench.write_tools):
        p = sum(1 for t in pool if tool in tools[t]) / len(pool) if pool else 0.0
        kb = sum(1 for t in broken if tool in tools[t])
        kf = sum(1 for t in fixed if tool in tools[t])
        if kb or kf:
            out.append(ToolConcentration(tool=tool, pool_share=p, broken=kb, n_broken=len(broken),
                                         p_value=binom_tail(kb, len(broken), p) if broken else 1.0,
                                         fixed=kf))
    out.sort(key=lambda c: (c.p_value, -c.broken))
    return out


def agent_actions(traj: list[dict], max_items: int = 10) -> list[str]:
    """The agent's tool calls in order, as compact `name(args)` lines."""
    out = []
    for m in traj:
        if m.get("role") != "assistant":
            continue
        for c in (m.get("tool_calls") or [])[:1]:
            fn = c.get("function") or {}
            args = fn.get("arguments") or "{}"
            try:
                args = json.dumps(json.loads(args), ensure_ascii=False, separators=(",", ":"))
            except (json.JSONDecodeError, TypeError):
                pass
            out.append(f"{fn.get('name')}({args[:160]})")
    return out[:max_items]


def _first_row(rows: list[dict], task_id: int) -> dict | None:
    return next((r for r in sorted(rows, key=lambda r: r["trial"]) if r["task_id"] == task_id), None)


def gate_feedback(verdict: dict, incumbent_rows: list[dict], candidate_rows: list[dict],
                  split: str, *, bench: Benchmark) -> dict[str, Any]:
    """Structured + text feedback for a rejected gate verdict."""
    broken, fixed = verdict.get("regressions", []), verdict.get("improvements", [])
    pool = sorted({r["task_id"] for r in candidate_rows})
    conc = concentration(pool, broken, fixed, split, bench=bench) if pool else []
    lines = []
    eff = verdict.get("efficacy")
    if eff:
        lines.append(f"It DID fix its held-out targets: {eff['cand_passes']}/{eff['cand_n']} vs "
                     f"{eff['null_passes']}/{eff['null_n']} without it (p = {eff['p_value']:.3f}).")
    lines.append(f"But on {verdict.get('n_tasks', 0)} unseen tasks it fixed {len(fixed)} and broke "
                 f"{len(broken)} (pass rate {verdict.get('delta', 0):+.3f}).")
    hot = [c for c in conc if c.broken and c.p_value < 0.05]
    if hot:
        c = hot[0]
        lines.append(f"The broken tasks concentrate on `{c.tool}`: {c.broken}/{c.n_broken} use it vs "
                     f"{c.pool_share:.0%} of all tasks (chance {c.p_value:.3f}). The rule changed "
                     f"behaviour there beyond the situation it was meant for -- narrow its trigger.")
    example = None
    for t in broken:
        before, after = _first_row(incumbent_rows, t), _first_row(candidate_rows, t)
        if before and after:
            example = {"task_id": t, "goal": bench.task(split, t).instruction,
                       "before_passed": agent_actions(before.get("traj", [])),
                       "after_failed": agent_actions(after.get("traj", []))}
            break
    if example:
        lines.append(f"Example, a task it broke -- goal: {example['goal'][:300]}")
        lines.append("  before (passed): " + (" ; ".join(example["before_passed"]) or "no tool calls"))
        lines.append("  with the rule (failed): " + (" ; ".join(example["after_failed"])
                                                    or "no tool calls -- the agent declined or stopped"))
    return {"text": "\n".join(lines), "concentration": [c.model_dump() for c in conc],
            "example": example}
