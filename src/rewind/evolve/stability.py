"""Stability filter: is a failure reproducible, or was the parent unlucky?

P0's null-fork calibration showed failed tau-bench episodes split cleanly:
some tasks fail from any fork point almost every time (the model really
cannot do it), others pass most re-runs (the recorded failure was noise).
Only the first kind is worth an edit -- writing rules for the second kind
"fixes" randomness and makes fork screening report false positives.

For each failed episode: fork from the start (step 1: same first user line,
everything after it live) `n` times with NO patch and count passes.

  stable   passes <= stable_max     -> goes to clustering / the proposer
  flaky    passes >= flaky_min      -> counted, never optimised
  unclear  in between               -> counted, never optimised

The null-fork passes double as the per-task baseline that fork screening
compares a candidate edit against.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel

from ..benchmark import Benchmark, ResultRow, row_ctx_version, row_trace
from ..experiment import fork_episode
from ..snapshot import SnapshotStore
from ..trace import load_trace

Label = Literal["stable", "flaky", "unclear"]


class StabilityRecord(BaseModel):
    task_id: int
    trial: int
    trace: str
    ctx_version: int
    n: int
    passes: int
    label: Label
    null_status: list[str] = []

    @property
    def null_rate(self) -> float:
        return self.passes / self.n if self.n else 0.0


def label_for(passes: int, n: int, stable_max: int = 1, flaky_min: int = 3) -> Label:
    if passes <= stable_max:
        return "stable"
    if passes >= flaky_min:
        return "flaky"
    return "unclear"


def _key(task_id: int, trial: int, version: int) -> str:
    return f"{version}:{task_id}:{trial}"


def load_records(path: str | Path) -> dict[str, StabilityRecord]:
    p = Path(path)
    if not p.exists():
        return {}
    return {_key(r["task_id"], r["trial"], r["ctx_version"]): StabilityRecord(**r)
            for r in json.loads(p.read_text(encoding="utf-8"))}


def save_records(path: str | Path, recs: dict[str, StabilityRecord]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps([r.model_dump() for r in recs.values()], indent=1), encoding="utf-8")
    tmp.replace(p)


def pending(rows: list[ResultRow], out: str | Path) -> list[tuple[ResultRow, int]]:
    recs = load_records(out)
    todo = []
    for r in rows:
        if r["reward"] >= 1.0 or not row_trace(r):
            continue
        v = row_ctx_version(r)
        if _key(r["task_id"], r["trial"], v) not in recs:
            todo.append((r, v))
    return todo


async def assess_failures(rows: list[ResultRow], out: str | Path, *, store: SnapshotStore, bench: Benchmark, provider: Any,
                          user_llm: Any, registry: Any, n: int = 5, stable_max: int = 1,
                          flaky_min: int = 3, concurrency: int = 4,
                          log: Callable[[str], None] = print) -> dict[str, StabilityRecord]:
    """Null-fork every failed row in `rows` (resumable via `out`). Pass only
    the failures worth assessing (e.g. those in a prevalent cluster) to save
    API calls. `registry` must be the one the rows were run with (it carries
    tool-note edits); the system prompt is taken from each trace."""
    recs = load_records(out)
    todo = []
    for r in rows:
        if r["reward"] >= 1.0 or not row_trace(r):
            continue
        v = row_ctx_version(r)
        if _key(r["task_id"], r["trial"], v) not in recs:
            todo.append((r, v))
    log(f"stability: {len(todo)} failed episodes to assess ({len(recs)} cached), {n} null forks each")
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def one(row: ResultRow, version: int) -> None:
        trace = row_trace(row)
        events = load_trace(trace)

        # Sequential with early stopping (API budget): the only label that
        # matters downstream is "stable", so stop as soon as passes exceed
        # stable_max -- a flaky task usually costs 2 forks, not n.
        passes, done, status = 0, 0, []
        try:
            for i in range(n):
                async with sem:
                    ep = await fork_episode(bench, events, 1, provider=provider, user_llm=user_llm,
                                            store=store, registry=registry,
                                            run_id=f"{events[0].run_id}~null1-{i}")
                done += 1
                passes += ep.reward.reward >= 1.0
                status.append(ep.result.status)
                if passes > stable_max:
                    break
        except Exception as e:
            log(f"[error] stability task {row['task_id']}: {type(e).__name__}: {e}")
            return
        label = label_for(passes, n, stable_max, flaky_min) if done == n else (
            "flaky" if passes >= flaky_min else "unclear")
        rec = StabilityRecord(task_id=row["task_id"], trial=row["trial"], trace=trace,
                              ctx_version=version, n=done, passes=passes, label=label,
                              null_status=status)
        async with lock:
            recs[_key(rec.task_id, rec.trial, version)] = rec
            save_records(out, recs)
        log(f"  task {rec.task_id:3d}: null forks {passes}/{n} pass -> {rec.label}")

    await asyncio.gather(*(one(r, v) for r, v in todo))
    return recs
