"""Failure signatures and the "only optimise what is common" rule.

A signature is deterministic and readable -- no embeddings -- so a cluster
can be explained in one line and recomputed exactly:

    category | tool | detail

* category: the existing tau-bench taxonomy (`classify_taubench`)
* tool:     the first mutating tool where agent and ground truth disagree
* detail:   for action failures, WHICH argument keys differed on that tool
            (rewind traces carry the agent's real arguments, so this is now
            visible -- run.py results only had tool names); otherwise the
            normalised error text or the coarse action shape

Clusters are counted in DISTINCT TASKS (several trials of one task are one
piece of evidence), and a cluster is eligible only if the Wilson lower bound
of its share of failed tasks clears a floor -- so 3 tasks out of 34 does
not pass for a 9% problem. Everything here is free: it reads results files.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pydantic import BaseModel

from ..env.taubench_adapter import MUTATING_TOOLS
from ..env.taubench_dialogue import load_task
from ..flywheel.taubench_failures import action_shape, classify_taubench
from .stats import wilson_interval


class FailureSignature(BaseModel):
    category: str
    tool: str | None = None
    detail: str | None = None

    @property
    def key(self) -> str:
        return f"{self.category}|{self.tool or '-'}|{self.detail or '-'}"

    @property
    def coarse_key(self) -> str:
        return f"{self.category}|{self.tool or '-'}"


def _agent_calls(traj: list[dict]) -> list[tuple[str, dict]]:
    out = []
    for m in traj:
        if m.get("role") != "assistant":
            continue
        for c in (m.get("tool_calls") or [])[:1]:
            fn = c.get("function") or {}
            if fn.get("name") in MUTATING_TOOLS:
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                out.append((fn["name"], args))
    return out


_NUM = re.compile(r"#?W?\d+")


def _norm_error(msg: str) -> str:
    return _NUM.sub("<n>", msg.replace("Error:", "").strip())[:80]


def signature(row: dict, split: str) -> FailureSignature | None:
    rec = classify_taubench(row)
    if rec is None:
        return None
    if rec.category == "TOOL_RUNTIME_ERROR":
        errs = [m for m in row.get("traj", []) if m.get("role") == "tool"
                and str(m.get("content", "")).startswith("Error:")]
        # the tool that errored: the assistant call right before the error
        tool = None
        traj = row.get("traj", [])
        for i, m in enumerate(traj):
            if m is errs[-1] and i > 0:
                calls = traj[i - 1].get("tool_calls") or []
                tool = (calls[0].get("function") or {}).get("name") if calls else None
        return FailureSignature(category=rec.category, tool=tool,
                                detail=_norm_error(str(errs[-1]["content"])))
    if rec.category == "WRONG_OUTPUT":
        return FailureSignature(category=rec.category, detail="missing required output")
    # action failures: find the first mutating tool where agent and truth differ
    task = load_task(split, row["task_id"])
    gt = [(a.name, a.kwargs) for a in task.actions if a.name in MUTATING_TOOLS]
    agent = _agent_calls(row.get("traj", []))
    gt_names, ag_names = [n for n, _ in gt], [n for n, _ in agent]
    if gt_names == ag_names:
        for (gn, ga), (_, aa) in zip(gt, agent):
            diff = sorted(k for k in set(ga) | set(aa) if ga.get(k) != aa.get(k))
            if diff:
                return FailureSignature(category=rec.category, tool=gn,
                                        detail="wrong args: " + ",".join(diff))
        return FailureSignature(category=rec.category, detail="same calls, state still differs")
    missing = [n for n in gt_names if n not in ag_names]
    tool = missing[0] if missing else next((n for n in ag_names if n not in gt_names), None)
    return FailureSignature(category=rec.category, tool=tool, detail=action_shape(rec))


class ClusterStats(BaseModel):
    key: str
    level: str                     # "fine" | "coarse"
    tasks: list[int]
    n_failed_tasks: int
    share: float
    share_lb: float


class PrevalencePolicy(BaseModel):
    min_distinct_tasks: int = 4    # enough to split A/B for screening
    min_share_lb: float = 0.05     # Wilson lower bound of share of failed tasks


def clusters(rows: list[dict], split: str, *, only_tasks: set[int] | None = None
             ) -> tuple[list[ClusterStats], dict[int, FailureSignature]]:
    """Fine and coarse clusters over failed rows (one signature per task: its
    first failed trial). `only_tasks` restricts members (e.g. to stable
    failures) while the denominator stays all failed tasks."""
    sigs: dict[int, FailureSignature] = {}
    for r in sorted(rows, key=lambda r: (r["task_id"], r["trial"])):
        if r["reward"] >= 1.0 or r["task_id"] in sigs:
            continue
        s = signature(r, split)
        if s is not None:
            sigs[r["task_id"]] = s
    n_failed = len(sigs)
    out: list[ClusterStats] = []
    for level, keyf in (("fine", lambda s: s.key), ("coarse", lambda s: s.coarse_key)):
        groups: dict[str, list[int]] = defaultdict(list)
        for t, s in sigs.items():
            if only_tasks is None or t in only_tasks:
                groups[keyf(s)].append(t)
        for k, ts in groups.items():
            lo, _ = wilson_interval(len(ts), n_failed)
            out.append(ClusterStats(key=k, level=level, tasks=sorted(ts), n_failed_tasks=n_failed,
                                    share=len(ts) / n_failed if n_failed else 0.0, share_lb=lo))
    out.sort(key=lambda c: (-len(c.tasks), c.level != "fine", c.key))
    return out, sigs


def eligible(c: ClusterStats, policy: PrevalencePolicy) -> tuple[bool, str]:
    if len(c.tasks) < policy.min_distinct_tasks:
        return False, f"{len(c.tasks)} tasks < {policy.min_distinct_tasks}"
    if c.share_lb < policy.min_share_lb:
        return False, f"share lower bound {c.share_lb:.3f} < {policy.min_share_lb}"
    return True, "ok"


def pick_cluster(stats: list[ClusterStats], policy: PrevalencePolicy,
                 exclude: set[str] = frozenset()) -> ClusterStats | None:
    """Largest eligible cluster; a fine cluster wins over its coarse parent of
    equal size (more specific evidence for the proposer)."""
    for c in stats:
        if c.key not in exclude and eligible(c, policy)[0]:
            return c
    return None


def describe(stats: list[ClusterStats], policy: PrevalencePolicy, top: int = 12) -> str:
    lines = []
    for c in stats[:top]:
        ok, why = eligible(c, policy)
        lines.append(f"  {'*' if ok else ' '} [{c.level:6s}] {len(c.tasks):3d} tasks  "
                     f"share {c.share:.2f} (lb {c.share_lb:.2f})  {c.key}  {'' if ok else '(' + why + ')'}")
    return "\n".join(lines)


def evidence_rows(rows: list[dict], tasks: list[int]) -> list[dict]:
    """First failed row per task, in task order."""
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: (r["task_id"], r["trial"])):
        if r["task_id"] in tasks and r["reward"] < 1.0 and r["task_id"] not in seen:
            seen.add(r["task_id"])
            out.append(r)
    return out
