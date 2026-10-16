"""Counterfactual root-cause attribution (plan section 4.7).

Given a failed trace and, for some subset of its steps, a proposed
"corrective action", find the *earliest* step where applying that
correction and continuing live actually flips the outcome to success.
That step is the root cause: everything before it didn't matter, and the
correction there is enough on its own.

Two layers, kept apart on purpose:

- `bisect_attribute` is the pure algorithm. It knows nothing about agents,
  forks or environments — just "here is an oracle that tells me the flip
  rate at candidate i", and it does O(log N) oracle calls plus a small
  neighborhood check, exactly like a bisection over a monotonic predicate.
  This is what's actually unit-testable without spinning up a runner.
- `flip_rate_via_fork` wires that oracle to `fork.fork_run`: it forks the
  base trace at a step, applies a patch, continues live for `n_samples`,
  and reports the fraction that graded successfully.

The monotonicity assumption ("intervening earlier is at least as likely to
flip the outcome as intervening later") is only approximate — a trace can
have more than one independent mistake. `bisect_attribute` does not trust
it blindly: after finding a candidate via bisection, it linearly re-checks
a small window of *earlier* candidates, and if one of those also flips the
outcome, it reports that earlier one instead and flags `monotonic=False` so
callers know the assumption was violated on this trace.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..fork import ForkPatch, fork_run
from ..loop import AgentRunner
from ..snapshot import SnapshotStore
from ..trace import TraceWriter

FlipRateFn = Callable[[int], Awaitable[float]]


@dataclass
class AttributionResult:
    root_step: int | None                # None: not even the last candidate flips the outcome
    flip_rates: dict[int, float] = field(default_factory=dict)   # step -> measured flip rate
    monotonic: bool = True                # False: an earlier candidate also flipped it (see module docstring)
    oracle_calls: int = 0                 # how many flip_rate evaluations this attribution cost


async def bisect_attribute(candidates: list[int], flip_rate: FlipRateFn,
                           threshold: float = 0.66, verify_neighborhood: int = 2
                           ) -> AttributionResult:
    """`candidates` is the list of step numbers a correction is defined for,
    in ascending order (not necessarily every step of the trace). Returns the
    earliest one whose flip rate clears `threshold`, or None if even the
    last candidate doesn't reach it."""
    rates: dict[int, float] = {}
    calls = 0

    async def rate(idx: int) -> float:
        nonlocal calls
        step = candidates[idx]
        if step not in rates:
            rates[step] = await flip_rate(step)
            calls += 1
        return rates[step]

    if not candidates:
        return AttributionResult(None, rates, True, calls)

    hi = len(candidates) - 1
    if await rate(hi) < threshold:
        return AttributionResult(None, rates, True, calls)

    lo = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if await rate(mid) >= threshold:
            hi = mid
        else:
            lo = mid + 1
    root_idx = lo

    monotonic = True
    for i in range(max(0, root_idx - verify_neighborhood), root_idx):
        if await rate(i) >= threshold:
            root_idx = i          # an earlier real root cause exists; bisection's assumption broke
            monotonic = False
            break

    return AttributionResult(candidates[root_idx], rates, monotonic, calls)


async def flip_rate_via_fork(base_events: list, snapshots: SnapshotStore, registry, cfg,
                             env_factory: Callable[[dict], Any], grade_fn: Callable[[Any], bool],
                             step: int, patch: ForkPatch,
                             provider_factory: Callable[[int, int], Any],
                             n_samples: int = 3) -> float:
    """Fork the base run at `step` with `patch` applied, continue live
    `n_samples` times (a fresh provider per sample, since live continuation
    is where the real randomness would come from against a real API), and
    return the fraction that graded successfully."""
    successes = 0
    for i in range(n_samples):
        runner = AgentRunner(provider_factory(step, i), registry, cfg, snapshots)
        trace = TraceWriter(f"attr-{base_events[0].run_id}-s{step}-{i}")
        holder: dict[str, Any] = {}

        def factory(state: dict, holder=holder) -> Any:
            holder["env"] = env_factory(state)
            return holder["env"]

        await fork_run(base_events, snapshots, at_step=step, runner=runner,
                       env_factory=factory, trace=trace, patch=patch)
        if grade_fn(holder["env"]):
            successes += 1
    return successes / n_samples
