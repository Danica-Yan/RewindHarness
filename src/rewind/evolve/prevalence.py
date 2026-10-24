"""Failure signatures and the "only optimise what is common" rule.

A signature is deterministic and readable -- no embeddings -- so a cluster
can be explained in one line and recomputed exactly:

    category | tool | detail

* category: the benchmark's failure taxonomy
* tool:     the first mutating tool where agent and ground truth disagree
* detail:   e.g. WHICH argument keys differed on that tool, the normalised
            error text, or the coarse action shape

The signature itself comes from the benchmark (`Benchmark.failure_signature`):
reading a failed row is format-specific (tau-bench: `classify_taubench` over
its OpenAI-format `traj`), clustering and the prevalence rule are not.

Clusters are counted in DISTINCT TASKS (several trials of one task are one
piece of evidence), and a cluster is eligible only if the Wilson lower bound
of its share of failed tasks clears a floor -- so 3 tasks out of 34 does
not pass for a 9% problem. Everything here is free: it reads results files.
"""
from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel

from ..benchmark import Benchmark, FailureSignature, ResultRow
from .stats import wilson_interval


def signature(row: ResultRow, split: str, *, bench: Benchmark) -> FailureSignature | None:
    return bench.failure_signature(row, split)


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


def clusters(rows: list[ResultRow], split: str, *, bench: Benchmark, only_tasks: set[int] | None = None
             ) -> tuple[list[ClusterStats], dict[int, FailureSignature]]:
    """Fine and coarse clusters over failed rows (one signature per task: its
    first failed trial). `only_tasks` restricts members (e.g. to stable
    failures) while the denominator stays all failed tasks."""
    sigs: dict[int, FailureSignature] = {}
    for r in sorted(rows, key=lambda r: (r["task_id"], r["trial"])):
        if r["reward"] >= 1.0 or r["task_id"] in sigs:
            continue
        s = signature(r, split, bench=bench)
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


def evidence_rows(rows: list[ResultRow], tasks: list[int]) -> list[ResultRow]:
    """First failed row per task, in task order."""
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: (r["task_id"], r["trial"])):
        if r["task_id"] in tasks and r["reward"] < 1.0 and r["task_id"] not in seen:
            seen.add(r["task_id"])
            out.append(r)
    return out
