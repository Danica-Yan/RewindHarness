"""W8's data flywheel: fixed 60/20/20 split, round-by-round candidate fixes
validated on dev and gated before merging, heldout reported after each
merge (plan W8 items 1-3). Runs against the deterministic
`HeuristicRetailProvider` (see that module's docstring for why: fully offline
and reproducible, no LLM API key needed) over the task suite in
`tasks.py`, so every number below is real and reproducible, not
illustrative.

Two independent, named bugs are available to fix (`bug_ignores_ownership`,
`bug_ignores_status`) — but they are not symmetric, and that asymmetry is
itself worth reporting rather than smoothing over:

  - Round 1 fixes `bug_ignores_ownership`. The `cancel_order` tool has no
    ownership check of its own (see env.py), so this bug actually produces
    a *wrong final state* (`REASONING_ERROR`) — fixing it is a genuine
    correctness fix and moves the task pass rate.
  - Round 2 fixes `bug_ignores_status`. The tool *does* refuse a non-pending
    cancel on its own, so this bug never produces a wrong final state (the
    tool's own defense-in-depth already saves it) — fixing it doesn't move
    the pass rate at all (already at ceiling after round 1). What it does
    move is a second, real metric: how often a run only passed *because* a
    lower layer caught a mistake the model shouldn't have made in the first
    place (`defensive_saves` — a `TOOL_RUNTIME_ERROR` on an otherwise-passing
    run). A flywheel that only tracks pass rate would call round 2 a no-op;
    tracking this second metric is what makes it visible as real progress.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..env import ToyRetailEnv, build_registry
from ..loop import AgentConfig, AgentRunner
from ..snapshot import SnapshotStore
from ..trace import TraceWriter
from .failures import FailureRecord, classify, cluster, top_clusters
from .heuristic_provider import HeuristicRetailProvider
from .tasks import TaskSpec, grade_task

# Which bug flag a given (category, tool) cluster maps to — this is the
# stand-in for "a human reads the top cluster and writes a fix"; here the
# mapping is known upfront because the bugs were constructed, but the shape
# (look up a fix by cluster key) is what a real flywheel would do against a
# human- or LLM-authored fix library.
CANDIDATE_FIXES: dict[tuple[str, str | None], str] = {
    ("REASONING_ERROR", "cancel_order"): "bug_ignores_ownership",
    ("TOOL_RUNTIME_ERROR", "cancel_order"): "bug_ignores_status",
}


@dataclass
class SplitRunResult:
    total: int
    passed: int
    failures: list[FailureRecord]
    defensive_saves: int   # passed AND a TOOL_RUNTIME_ERROR is in the trace

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0


async def run_split(tasks: list[TaskSpec], bug_flags: dict[str, bool], snaps: SnapshotStore,
                    registry, cfg: AgentConfig) -> SplitRunResult:
    failures: list[FailureRecord] = []
    passed = 0
    defensive_saves = 0
    for t in tasks:
        env = ToyRetailEnv(t.initial_state())
        provider = HeuristicRetailProvider(t.target_order, t.requester, **bug_flags)
        runner = AgentRunner(provider, registry, cfg, snaps)
        trace = TraceWriter(t.task_id)
        result = await runner.run(env, t.task_text, trace)
        ok = grade_task(env, t)
        if ok:
            passed += 1
            if any(e.type == "tool_call" and e.data.get("error_class") == "TOOL_RUNTIME_ERROR"
                  for e in trace.events):
                defensive_saves += 1
        else:
            rec = classify(trace.events, result, ok)
            if rec is not None:
                failures.append(rec)
    return SplitRunResult(len(tasks), passed, failures, defensive_saves)


@dataclass
class RoundReport:
    round_no: int
    metric: str                              # "pass_rate" or "defensive_saves"
    fix: str | None
    target_cluster: tuple[str, str | None] | None
    dev_before: float
    dev_after: float | None
    gate_passed: bool
    heldout_after: float | None = None
    note: str = ""


@dataclass
class FlywheelReport:
    bug_flags_by_round: list[dict[str, bool]]
    rounds: list[RoundReport]
    baseline: SplitRunResult
    final: SplitRunResult


def _all_off(bug_flags: dict[str, bool]) -> bool:
    return not any(bug_flags.values())


async def run_flywheel(split: dict[str, list[TaskSpec]], n_rounds: int = 2) -> FlywheelReport:
    snaps = SnapshotStore()
    registry = build_registry()
    cfg = AgentConfig(system_prompt="You are a retail support agent.")
    bug_flags: dict[str, bool] = {"bug_ignores_ownership": True, "bug_ignores_status": True}
    flags_history = [dict(bug_flags)]

    baseline = await run_split(split["train"], bug_flags, snaps, registry, cfg)
    dev_baseline = await run_split(split["dev"], bug_flags, snaps, registry, cfg)

    rounds: list[RoundReport] = []
    dev_current = dev_baseline
    for round_no in range(1, n_rounds + 1):
        train_now = await run_split(split["train"], bug_flags, snaps, registry, cfg)
        clusters = top_clusters(train_now.failures, n=5)
        if not clusters:
            # No task-level failures left to target by pass rate; fall back to
            # a defensive-reliance signal so a real remaining bug (one the
            # tool's own guard is quietly covering for) still gets picked.
            candidate_key = ("TOOL_RUNTIME_ERROR", "cancel_order") if bug_flags["bug_ignores_status"] else None
        else:
            candidate_key = clusters[0][0]

        fix = CANDIDATE_FIXES.get(candidate_key) if candidate_key else None
        if fix is None or not bug_flags.get(fix, False):
            rounds.append(RoundReport(round_no, "pass_rate", None, candidate_key,
                                      dev_current.pass_rate, None, False,
                                      note="no known fix for the top cluster (or already applied)"))
            continue

        candidate_flags = dict(bug_flags)
        candidate_flags[fix] = False
        dev_candidate = await run_split(split["dev"], candidate_flags, snaps, registry, cfg)

        if fix == "bug_ignores_ownership":
            metric = "pass_rate"
            improved = dev_candidate.pass_rate > dev_current.pass_rate
            no_regression = dev_candidate.pass_rate >= dev_current.pass_rate
            gate_passed = improved and no_regression
        else:
            metric = "defensive_saves"
            improved = dev_candidate.defensive_saves < dev_current.defensive_saves
            no_regression = dev_candidate.pass_rate >= dev_current.pass_rate
            gate_passed = improved and no_regression

        report = RoundReport(
            round_no, metric, fix, candidate_key,
            dev_current.pass_rate if metric == "pass_rate" else dev_current.defensive_saves,
            dev_candidate.pass_rate if metric == "pass_rate" else dev_candidate.defensive_saves,
            gate_passed)

        if gate_passed:
            bug_flags = candidate_flags
            flags_history.append(dict(bug_flags))
            heldout = await run_split(split["heldout"], bug_flags, snaps, registry, cfg)
            report.heldout_after = heldout.pass_rate if metric == "pass_rate" else heldout.defensive_saves
            dev_current = dev_candidate
            if metric == "defensive_saves":
                report.note = ("pass rate does not move here — the tool's own guard already "
                               "prevented a wrong final state; this fix removes reliance on that "
                               "guard, measured separately (see module docstring)")
        else:
            report.note = "gate failed: candidate did not clear the bar on dev, fix not merged"
        rounds.append(report)

    final = await run_split(split["heldout"], bug_flags, snaps, registry, cfg)
    return FlywheelReport(flags_history, rounds, baseline, final)
