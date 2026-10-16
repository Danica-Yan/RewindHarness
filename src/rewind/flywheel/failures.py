"""Failure taxonomy classification, simple clustering, and auto-paired
correction patches — plan section 4.6 (taxonomy table) and the W7 items
that turn the harness's per-run signals into a batch-level pipeline:
"classify the full failure batch by category" and "auto-pair `patch_for_step`".

`classify()` deliberately reads only what the harness already puts in the
trace (`RunResult.status`/`reason`, `tool_call`/`llm_call` event error
classes) — no separate logging path to keep in sync. Two categories from
the plan's table, CONTEXT_OVERFLOW and POLICY_VIOLATION, have no mechanism
in this harness yet (no compaction, no policy guard), so `classify()` can
never produce them; see README for that as an explicit, honest gap rather
than a silently-unreachable code path.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from ..events import Event
from ..fork import ForkPatch
from ..loop import RunResult

# The plan's 4.6 table, minus CONTEXT_OVERFLOW/POLICY_VIOLATION (see module
# docstring) and WORKER_CRASH (that signal lives in the checkpoint/worker
# layer, not in a single AgentRunner trace — see `classify_worker_crash`).
CATEGORIES = (
    "LLM_API_ERROR", "OUTPUT_MALFORMED", "TOOL_ARG_INVALID", "HALLUCINATED_TOOL",
    "TOOL_RUNTIME_ERROR", "LOOP", "BUDGET_EXCEEDED", "WORKER_CRASH", "REASONING_ERROR",
)

# status/reason values loop.py actually raises RunAbort with -> taxonomy category.
_STATUS_MAP = {
    "malformed": "OUTPUT_MALFORMED",
    "provider_error": "LLM_API_ERROR",
    "loop": "LOOP",
    "budget_exceeded": "BUDGET_EXCEEDED",
    "fenced_out": "WORKER_CRASH",
}


@dataclass
class FailureRecord:
    run_id: str
    category: str
    root_cause_tool: str | None
    error_message: str | None
    step: int | None


def _last_tool_call_with(events: list[Event], predicate) -> Event | None:
    for e in reversed(events):
        if e.type == "tool_call" and predicate(e.data):
            return e
    return None


def classify(events: list[Event], result: RunResult, graded_ok: bool) -> FailureRecord | None:
    """`None` means this run wasn't a failure at all (completed + graded_ok).
    Otherwise returns the best single category this trace's own signals
    support — ties are broken toward the more specific tool-level signal
    over the coarser run-level status, since that's more actionable."""
    if result.status == "completed" and graded_ok:
        return None

    if result.status == "invalid_args":
        # The abort itself doesn't say which flavor tripped max_invalid_args;
        # the last matching tool_call does.
        bad = _last_tool_call_with(events, lambda d: d.get("error_class") in
                                   ("TOOL_ARG_INVALID", "HALLUCINATED_TOOL"))
        cat = bad.data["error_class"] if bad else "TOOL_ARG_INVALID"
        return FailureRecord(events[0].run_id, cat, bad.data["name"] if bad else None,
                             result.reason, bad.step if bad else None)

    if result.status in _STATUS_MAP:
        cat = _STATUS_MAP[result.status]
        tool_ev = _last_tool_call_with(events, lambda d: True) if cat == "LOOP" else None
        return FailureRecord(events[0].run_id, cat, tool_ev.data["name"] if tool_ev else None,
                             result.reason, tool_ev.step if tool_ev else result.steps)

    # status == "completed" but not graded_ok: either a tool genuinely broke
    # (TOOL_RUNTIME_ERROR is visible in the trace even though it didn't abort
    # the run — the loop treats it as a recoverable error message and keeps
    # going) or the model's tool calls all *succeeded* yet the outcome is
    # still wrong, which is exactly what 4.7's attribution is for.
    broken = _last_tool_call_with(events, lambda d: d.get("error_class") == "TOOL_RUNTIME_ERROR")
    if broken is not None:
        return FailureRecord(events[0].run_id, "TOOL_RUNTIME_ERROR", broken.data["name"],
                             broken.data["content"], broken.step)
    last_mutating = _last_tool_call_with(events, lambda d: d.get("ok"))
    return FailureRecord(events[0].run_id, "REASONING_ERROR",
                         last_mutating.data["name"] if last_mutating else None,
                         "tool calls succeeded but the final graded state is wrong",
                         last_mutating.step if last_mutating else result.steps)


def classify_worker_crash(run_id: str) -> FailureRecord:
    """WORKER_CRASH doesn't come from a single AgentRunner trace — it's a
    property of the checkpoint/coordinator layer (a lease expired and a
    different worker took over), already exhaustively tested against both
    sqlite (`test_checkpoint.py`) and real Postgres/real `kill -9`
    (`test_dist_pg.py`, `test_chaos.py`). This is a thin constructor so a
    batch report can list one representative record for the category
    without re-deriving it from trace signals that don't carry it."""
    return FailureRecord(run_id, "WORKER_CRASH", None,
                         "lease expired; a different worker claimed and completed the job", None)


def cluster(records: list[FailureRecord]) -> dict[tuple[str, str | None], list[FailureRecord]]:
    """Plan 4.6/W7 asks for clustering on (failure category, root-cause tool,
    error-message embedding), then says explicitly not to over-engineer it.
    Grouping by (category, tool) alone already produces clean, actionable
    clusters for this harness's failure modes — a distinct cluster per
    "what broke, on which tool" pair — so that's what this does; a text
    embedding sub-clustering *within* a bucket would need a real embedding
    model and a large enough corpus of free-text error messages to be worth
    it, neither of which this toy env's failure modes provide (see README).
    """
    buckets: dict[tuple[str, str | None], list[FailureRecord]] = defaultdict(list)
    for r in records:
        buckets[(r.category, r.root_cause_tool)].append(r)
    return dict(buckets)


def top_clusters(records: list[FailureRecord], n: int = 10) -> list[tuple[tuple[str, str | None], int]]:
    counts = Counter((r.category, r.root_cause_tool) for r in records)
    return counts.most_common(n)


def category_distribution(records: list[FailureRecord]) -> Counter:
    return Counter(r.category for r in records)


def auto_patch_for_step(success_events: list[Event], step: int) -> ForkPatch | None:
    """W7's "auto-paired patch_for_step": instead of a human hand-writing the
    correction at each candidate step (as the original `attribute-demo`
    CLI command does), pull the actual action taken at that same step
    number from a known-*successful* trace of the same task. Returns None
    if that trace doesn't have a (real, non-patched) llm_call at exactly
    that step — callers fall back to a manually supplied patch or a
    strong-model-generated one, per the plan's step 1.
    """
    for e in success_events:
        if e.type == "llm_call" and e.step == step and not e.data.get("patched"):
            resp = e.data.get("response")
            if resp is None:      # this was an error attempt (retried), not the real action
                continue
            return ForkPatch(
                text=resp.get("text"),
                tool_calls=[{"name": c["name"], "args": c["args"]} for c in resp.get("tool_calls", [])])
    return None
