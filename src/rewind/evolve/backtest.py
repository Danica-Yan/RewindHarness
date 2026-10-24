"""Offline backtest of policy checks over recorded episodes (no API calls).

A prompt rule's effect can only be measured by running the agent again. A
check is a deterministic function of (call, DB state, conversation), and a
rewind trace records all three for every call -- so before a check costs any
episodes we can replay it over every recorded episode and ask where it would
have fired:

* in an episode that PASSED   -> it would have interrupted a run that ended
                                 well ("fires on passing", to be ~0)
* in a FAILED episode, on a call the task's gold solution also makes
                              -> it would have stopped a right action
* in a FAILED episode, on a call the gold solution never makes
                              -> a catch: the kind of wrong action the check
                                 is for

The DB state at each call comes from the snapshot the loop stored at the
start of that step; the conversation from the trace's message events. Only
what the check would have *fired on* is measured here: what the agent does
after being blocked is a counterfactual and needs real episodes (the gate).
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..benchmark import Benchmark
from ..policy import PolicySet
from ..snapshot import SnapshotStore
from ..trace import load_trace


@dataclass
class Firing:
    check_id: str
    task_id: int
    trial: int
    passed: bool
    tool: str
    args: dict
    on_gold: bool
    reason: str
    step: int
    trace: str


@dataclass
class CheckStats:
    check_id: str
    guarded_calls_passing: int = 0       # calls on guarded tools in passing episodes
    guarded_calls_failing: int = 0
    fires_passing: int = 0
    fires_failing_gold: int = 0
    fires_failing_off_gold: int = 0
    episodes_passing_hit: set = field(default_factory=set)
    episodes_failing_caught: set = field(default_factory=set)

    @property
    def false_fire_rate(self) -> float:
        return self.fires_passing / self.guarded_calls_passing if self.guarded_calls_passing else 0.0

    def summary(self, n_pass: int, n_fail: int) -> dict:
        return {"check": self.check_id,
                "passing_episodes_hit": len(self.episodes_passing_hit), "passing_episodes": n_pass,
                "fires_on_passing_calls": f"{self.fires_passing}/{self.guarded_calls_passing}",
                "false_fire_rate": round(self.false_fire_rate, 4),
                "fires_on_gold_calls_in_failures": self.fires_failing_gold,
                "catches": self.fires_failing_off_gold,
                "failing_episodes_caught": len(self.episodes_failing_caught), "failing_episodes": n_fail}


@dataclass
class BacktestReport:
    stats: dict[str, CheckStats]
    firings: list[Firing]
    n_pass: int
    n_fail: int
    skipped: list[str]

    def summaries(self) -> list[dict]:
        return [s.summary(self.n_pass, self.n_fail) for s in self.stats.values()]

    def to_json(self, max_examples: int = 20) -> dict:
        ex: dict[str, list] = {}
        for f in self.firings:
            bucket = "passing" if f.passed else ("failing_gold" if f.on_gold else "catch")
            lst = ex.setdefault(f"{f.check_id}/{bucket}", [])
            if len(lst) < max_examples:
                lst.append({"task_id": f.task_id, "trial": f.trial, "tool": f.tool, "args": f.args,
                            "reason": f.reason, "step": f.step, "trace": f.trace})
        return {"summary": self.summaries(), "examples": ex, "skipped": self.skipped}


def _same_call(name: str, args: dict, gold: list[Any]) -> bool:
    for a in gold:
        if a.name == name and (a.kwargs or {}) == (args or {}):
            return True
    return False


def calls_with_context(events: list, store: SnapshotStore):
    """Yield (step, tool, args, call_id, data, history) for each tool call in a
    trace, with the DB as it was when the step started and the messages up to
    (and including) the assistant message that made the call."""
    messages: list[dict] = []
    env_hash = None
    data_cache: dict[str, dict] = {}

    ref_of: dict[str, str | None] = {}

    def db(h: str | None) -> dict:
        # env snapshots are small and change every turn (user messages);
        # the DB they point to changes only on writes -- cache by DB ref
        if h is None:
            return {}
        if h not in ref_of:
            snap = store.get(h)
            ref_of[h] = snap.get("data_ref")
            if ref_of[h] is None:
                data_cache[h] = snap.get("data", {})
                ref_of[h] = h
        ref = ref_of[h]
        if ref not in data_cache:
            data_cache[ref] = store.get(ref)
        return data_cache[ref]

    for e in events:
        d = e.data
        if e.type == "run_start":
            env_hash = d.get("initial_env_hash", env_hash)
        elif e.type in ("step_start", "env_snapshot"):
            env_hash = d.get("env_hash", env_hash)
        elif e.type == "message":
            messages.append(d["message"])
        elif e.type == "tool_call":
            yield e.step, d["name"], d.get("args") or {}, d.get("call_id"), db(env_hash), list(messages)


def backtest(policy: PolicySet, rows: list[dict], *, store: SnapshotStore,
             gold_actions: Callable[[int], list[Any]], is_write: Callable[[str], bool]) -> BacktestReport:
    stats = {c.id: CheckStats(c.id) for c in policy.checks}
    firings: list[Firing] = []
    skipped: list[str] = []
    n_pass = n_fail = 0
    for r in rows:
        trace = ((r.get("info") or {}).get("rewind") or {}).get("trace")
        if not trace or not Path(trace).exists():
            skipped.append(f"task {r.get('task_id')} trial {r.get('trial')}: no trace on disk")
            continue
        try:
            events = load_trace(trace)
            calls = list(calls_with_context(events, store))
        except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
            skipped.append(f"{trace}: {type(e).__name__}: {e}")
            continue
        passed = r["reward"] >= 1.0
        n_pass += passed
        n_fail += not passed
        gold = gold_actions(r["task_id"])
        for step, tool, args, _cid, data, history in calls:
            for c in policy.checks:
                if tool not in c.tools:
                    continue
                st = stats[c.id]
                if passed:
                    st.guarded_calls_passing += 1
                else:
                    st.guarded_calls_failing += 1
            for d in policy.evaluate(data, tool, args, history, is_write):
                st = stats[d.check_id]
                on_gold = _same_call(tool, args, gold)
                ep = (r["task_id"], r["trial"])
                if passed:
                    st.fires_passing += 1
                    st.episodes_passing_hit.add(ep)
                elif on_gold:
                    st.fires_failing_gold += 1
                else:
                    st.fires_failing_off_gold += 1
                    st.episodes_failing_caught.add(ep)
                firings.append(Firing(d.check_id, r["task_id"], r["trial"], passed, tool, args,
                                      on_gold, d.reason, step, trace))
    return BacktestReport(stats, firings, n_pass, n_fail, skipped)


def benchmark_backtest(policy: PolicySet, rows: list[dict], *, store: SnapshotStore, split: str,
                       bench: Benchmark) -> BacktestReport:
    """`backtest` with gold actions and write tools taken from `bench`."""
    return backtest(policy, rows, store=store, gold_actions=lambda t: bench.task(split, t).actions,
                    is_write=lambda n: n in bench.write_tools)
