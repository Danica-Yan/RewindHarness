"""Chaos-round invariant checker (plan W6). After N rounds of real faults —
process kills, network toxics — these are the properties that must hold no
matter what failed or when, checked against real artifacts (the trace file
on disk, the real Postgres `jobs` table) rather than anything the test
harness kept track of in memory:

  1. no job is stuck LEASED with a long-expired lease — a worker that
     "took" a job and never gets reclaimed is stuck, whatever caused it;
  2. each run's trace `seq` is contiguous with no gaps and no duplicates —
     a torn write that wasn't correctly truncated on resume shows up here
     as a gap; a duplicate append would show up as a repeat;
  3. the final env hash of a chaos run matches the hash from an
     uninterrupted control run of the same task — chaos changed *when*
     things happened, never *what* the final state is.

Mutating-side-effect-exactly-once (the fourth invariant W6 asks for) is
checked differently: it needs an observable that survives a real process
boundary, so `tests/test_chaos.py` bakes a call counter directly into the
env's own state (the same trick `checkpoint.py`'s tests use, just persisted
where a second OS process can see it) rather than a Python closure variable,
and asserts on that — there's no generic version of it to put here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from ..trace import load_trace


@dataclass
class InvariantReport:
    violations: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def add(self, msg: str) -> None:
        self.violations.append(msg)

    def merge(self, other: "InvariantReport") -> None:
        self.violations.extend(other.violations)


def check_seq_contiguous(trace_path: str | Path) -> list[str]:
    events = load_trace(trace_path)
    seqs = sorted(e.seq for e in events)
    problems: list[str] = []
    if not seqs:
        return problems
    if seqs[0] != 1:
        problems.append(f"{trace_path}: seq does not start at 1 (starts at {seqs[0]})")
    if len(set(seqs)) != len(seqs):
        problems.append(f"{trace_path}: duplicate seq values present")
    for a, b in zip(seqs, seqs[1:]):
        if b not in (a, a + 1):        # (a, a) already reported above as a duplicate
            problems.append(f"{trace_path}: seq gap between {a} and {b}")
    return problems


async def check_no_stuck_leases(conn, grace_seconds: float = 5.0) -> list[str]:
    """A job counts as *stuck* only once its lease has been expired for at
    least `grace_seconds` — a lease that expired a moment ago simply hasn't
    been reclaimed *yet*, which is normal; this checks that reclaiming
    actually happens, not that it happens with zero latency."""
    rows = await conn.fetch(
        "SELECT job_id, lease_owner, status, lease_expires_at FROM jobs "
        "WHERE status='LEASED' AND lease_expires_at < now() - ($1 || ' seconds')::interval",
        str(grace_seconds))
    return [f"job {r['job_id']} stuck LEASED under owner {r['lease_owner']!r} "
            f"(lease expired at {r['lease_expires_at']})" for r in rows]


def check_final_hash_matches(expected: str, actual: str, label: str = "run") -> list[str]:
    if expected != actual:
        return [f"{label}: final env hash mismatch under chaos "
                f"(control={expected}, chaos={actual})"]
    return []


async def full_report(conn, trace_paths: list[str | Path], expected_hash: str | None = None,
                      actual_hash: str | None = None, grace_seconds: float = 5.0) -> InvariantReport:
    """Convenience: run every check that applies and return one report."""
    report = InvariantReport()
    for p in trace_paths:
        for v in check_seq_contiguous(p):
            report.add(v)
    for v in await check_no_stuck_leases(conn, grace_seconds=grace_seconds):
        report.add(v)
    if expected_hash is not None and actual_hash is not None:
        for v in check_final_hash_matches(expected_hash, actual_hash):
            report.add(v)
    return report


def wait_until(predicate, timeout_s: float = 10.0, interval_s: float = 0.05):
    """Poll `predicate()` (sync callable returning truthy/falsy) until it's
    true or `timeout_s` elapses; returns the last (falsy) value on timeout
    rather than raising, so callers decide what a timeout means for them."""
    deadline = time.monotonic() + timeout_s
    result = predicate()
    while not result and time.monotonic() < deadline:
        time.sleep(interval_s)
        result = predicate()
    return result
