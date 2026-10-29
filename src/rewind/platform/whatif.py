"""What-if: judge recorded runs under a different access policy, offline.

Every access decision in a rewind trace was a pure function of things the
trace records: the principal (run_start's config), the call (the assistant
message), and whether the conversation had already taken in untrusted
content (the messages before it). So a policy change can be tried on every
recorded run before it ships, with no model call:

    report = what_if(traces, new_access)
    print(report.render_text())

For each recorded call the report compares what happened (the authz event,
or "unenforced" for runs made before access control was on) with what the
new policy says, and sorts the differences:

  tightened     the call would now be blocked, or need a human
  loosened      a call that was blocked would now run (or only need approval)
  reclassified  blocked either way, for a different reason (deny vs hidden)
  rule_changed  same outcome, decided by a different rule

Two views matter most and are reported on their own:

  * executed writes that would now be stopped -- what the new policy would
    have prevented, and what it may break;
  * blocked calls that would now run -- the risk review of a loosening.

Only the FIRST difference in a run is a clean answer. After it, the run
would have gone differently (the agent sees a denial instead of a result),
so later differences are judged on a history that would not have existed;
they are kept but flagged `downstream`. `confirm_first_divergence` replaces
that guess with the real thing: it forks the run at its first divergence,
replays the recorded action under the new policy and continues live -- the
one step that costs model calls.

Quotas are not re-judged: they depend on every other run's usage, which a
set of traces does not capture.
"""
from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..events import Event
from ..trace import load_trace
from .access import AccessControl
from .identity import Principal

Outcome = Literal["allow", "approval", "deny", "hidden", "unknown", "unenforced"]
Change = Literal["same", "rule_changed", "tightened", "loosened", "reclassified"]
_RANK = {"unenforced": 0, "allow": 0, "approval": 1, "deny": 2, "hidden": 2, "unknown": 2}
_NOT_RUN = {"ACCESS_DENIED", "HALLUCINATED_TOOL", "POLICY_BLOCKED", "QUOTA_EXCEEDED", "LOOP_DETECTED",
            "LOOP_KILLED", "TOOL_ARG_INVALID"}


def recorded_outcome(authz: dict) -> Outcome:
    if authz["verdict"] in ("hidden", "unknown"):
        return authz["verdict"]
    if authz.get("effect") == "require_approval":
        return "approval"                       # granted, rejected or still pending: it needed a human
    return authz.get("effect") or authz["verdict"]


def new_outcome(verdict: str) -> Outcome:
    return "approval" if verdict == "require_approval" else verdict     # type: ignore[return-value]


def classify(old: Outcome, new: Outcome, old_reason: str | None, new_reason: str | None) -> Change:
    if old == new:
        if old_reason and new_reason and old_reason != new_reason and old != "unenforced":
            return "rule_changed"
        return "same"
    if _RANK[new] > _RANK[old]:
        return "tightened"
    if _RANK[new] < _RANK[old]:
        return "loosened"
    return "reclassified" if old != "unenforced" else "same"


@dataclass
class CallDiff:
    run_id: str
    step: int
    idx: int
    name: str
    capability: str | None
    risk: str | None
    tenant: str
    agent: str
    old: Outcome
    new: Outcome
    change: Change
    old_reason: str | None
    new_reason: str | None
    tainted: bool
    ran: bool | None            # the tool really executed (None: the run stopped before it, e.g. paused)
    ok: bool | None             # ... and succeeded
    downstream: bool            # an earlier call of this run already changed
    args: dict = field(default_factory=dict)

    @property
    def stopped_write(self) -> bool:
        """A state-changing call that really happened and would now be
        blocked or held for a human."""
        return self.change == "tightened" and bool(self.ok) and self.risk not in (None, "read")


@dataclass
class WhatIfReport:
    new_version: str
    old_versions: list[str]
    runs: int = 0
    calls: int = 0
    transitions: Counter = field(default_factory=Counter)      # (old, new) -> n
    changes: Counter = field(default_factory=Counter)          # change -> n
    diffs: list[CallDiff] = field(default_factory=list)        # every call whose outcome or rule changed
    skipped: dict[str, str] = field(default_factory=dict)      # run_id -> why
    run_outcomes: dict[str, bool] = field(default_factory=dict)

    # ---- views -----------------------------------------------------------------
    def first_divergences(self) -> list[CallDiff]:
        return [d for d in self.diffs if not d.downstream and d.change in ("tightened", "loosened", "reclassified")]

    def tightened(self) -> list[CallDiff]:
        return [d for d in self.diffs if d.change == "tightened"]

    def loosened(self) -> list[CallDiff]:
        return [d for d in self.diffs if d.change == "loosened"]

    def stopped_writes(self) -> list[CallDiff]:
        return [d for d in self.diffs if d.stopped_write]

    def runs_interrupted(self) -> dict[str, list[str]]:
        """Runs the new policy would interrupt (first change tightens), split
        by how they ended, when run outcomes were given."""
        out: dict[str, list[str]] = defaultdict(list)
        for d in self.first_divergences():
            if d.change != "tightened":
                continue
            ok = self.run_outcomes.get(d.run_id)
            out["passed" if ok is True else "failed" if ok is False else "unknown"].append(d.run_id)
        return dict(out)

    def summary(self) -> dict[str, Any]:
        return {
            "new_policy_version": self.new_version, "recorded_policy_versions": self.old_versions,
            "runs": self.runs, "calls": self.calls, "skipped_runs": len(self.skipped),
            "changes": {k: self.changes.get(k, 0) for k in ("tightened", "loosened", "reclassified",
                                                           "rule_changed")},
            "runs_with_a_change": len({d.run_id for d in self.first_divergences()}),
            "executed_writes_now_stopped": len(self.stopped_writes()),
            "blocked_calls_now_running": sum(d.new == "allow" for d in self.loosened()),
            "runs_interrupted": {k: len(v) for k, v in self.runs_interrupted().items()},
            "downstream_changes": sum(d.downstream for d in self.diffs if d.change != "rule_changed"),
        }

    def to_json(self) -> str:
        return json.dumps({"summary": self.summary(),
                           "transitions": [{"old": o, "new": n, "calls": c}
                                           for (o, n), c in sorted(self.transitions.items())],
                           "diffs": [asdict(d) for d in self.diffs], "skipped": self.skipped},
                          indent=1, sort_keys=False, default=str)

    def render_text(self, limit: int = 10) -> str:
        s = self.summary()
        lines = [f"Policy what-if: {s['runs']} runs, {s['calls']} recorded calls "
                 f"(recorded policy {', '.join(s['recorded_policy_versions']) or 'none'} -> {s['new_policy_version']})"]
        if self.skipped:
            lines.append(f"  skipped {len(self.skipped)} runs (no principal)")
        c = s["changes"]
        lines.append(f"  tightened {c['tightened']}, loosened {c['loosened']}, reclassified {c['reclassified']}, "
                     f"same outcome by a different rule {c['rule_changed']}")
        lines.append(f"  runs whose first change is here: {s['runs_with_a_change']}"
                     f"  (later changes in the same run are downstream: {s['downstream_changes']})")
        if s["runs_interrupted"]:
            lines.append("  runs the new policy would interrupt: " +
                         ", ".join(f"{k} {v}" for k, v in sorted(s["runs_interrupted"].items())))
        trans = [f"{o} -> {n}: {k}" for (o, n), k in sorted(self.transitions.items())
                 if classify(o, n, None, None) != "same"]
        if trans:
            lines += ["", "Outcome transitions (calls):"] + [f"  {t}" for t in trans]

        def table(title: str, rows: list[CallDiff]) -> None:
            if not rows:
                return
            lines.extend(["", f"{title} ({len(rows)}):"])
            by = Counter((d.capability or d.name, d.old, d.new, d.new_reason or "") for d in rows)
            for (cap, o, n, why), k in by.most_common(limit):
                lines.append(f"  {k:4d} x {cap}: {o} -> {n}  [{why}]")
            ex = rows[0]
            lines.append(f"       e.g. run {ex.run_id} step {ex.step}: {ex.name}({json.dumps(ex.args)[:120]})")

        table("Executed writes the new policy would block or send for approval", self.stopped_writes())
        table("Blocked calls that would now run", [d for d in self.loosened() if d.new == "allow"])
        table("Blocked calls that would now only need approval", [d for d in self.loosened() if d.new == "approval"])
        table("Other tightened calls", [d for d in self.tightened() if not d.stopped_write])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# re-judging
# ---------------------------------------------------------------------------
PrincipalFor = Callable[[dict], "Principal | None"]


def _recorded_principal(run_start: dict) -> Principal | None:
    p = (run_start.get("cfg") or {}).get("principal")
    return Principal.model_validate(p) if p else None


@dataclass
class RunJudgement:
    run_id: str
    calls: int = 0
    diffs: list[CallDiff] = field(default_factory=list)        # only calls that changed
    transitions: Counter = field(default_factory=Counter)      # (old, new) over ALL calls
    versions: set[str] = field(default_factory=set)            # recorded policy versions
    skipped: str | None = None


def rejudge_run(events: list[Event], access: AccessControl,
                principal_for: PrincipalFor | None = None) -> RunJudgement:
    """Judge every call of one recorded run under `access`. Pure."""
    if not events or events[0].type != "run_start":
        raise ValueError("trace must begin with run_start")
    start = events[0]
    principal = (principal_for or _recorded_principal)(start.data)
    j = RunJudgement(start.run_id)
    if principal is None:
        j.skipped = "no principal recorded (pass principal_for to judge it as someone)"
        return j
    enforced = any(e.type == "authz" and not e.inherited for e in events)
    j.versions = {e.data.get("policy_version") for e in events if e.type == "authz"} - {None, ""}
    ran = {e.data["action_id"]: (e.data.get("error_class") not in _NOT_RUN, e.data["ok"])
           for e in events if e.type == "tool_call" and not e.inherited}
    messages: list[dict] = []
    changed_before = False
    for e in events:
        if e.type == "message":
            messages.append(e.data["message"])
            continue
        if e.inherited or e.type != ("authz" if enforced else "tool_call"):
            continue
        idx = e.data["idx"]
        call = _find_call(messages, e.data["call_id"], idx)
        args = (call or {}).get("args") or e.data.get("args") or {}
        name = e.data["name"]
        v = access.evaluate(principal, name, args, messages)
        new = new_outcome(v.verdict)
        new_reason = v.decision.reason if v.decision else None
        if enforced:
            old, old_reason = recorded_outcome(e.data), e.data.get("reason")
        else:
            old, old_reason = "unenforced", None
        change = classify(old, new, old_reason, new_reason)
        j.calls += 1
        j.transitions[(old, new)] += 1
        if change != "same":
            was_run, ok = ran.get(f"s{e.step}:c{idx}", (None, None))
            j.diffs.append(CallDiff(
                run_id=j.run_id, step=e.step, idx=idx, name=name,
                capability=v.capability.id if v.capability else e.data.get("capability"),
                risk=v.capability.risk if v.capability else None,
                tenant=principal.tenant_id, agent=principal.agent_id, old=old, new=new, change=change,
                old_reason=old_reason, new_reason=new_reason, tainted=bool(v.extra.get("tainted")),
                ran=was_run, ok=ok if was_run else (False if was_run is False else None),
                downstream=changed_before, args=copy.deepcopy(args)))
            if change != "rule_changed":
                changed_before = True
    return j


def _find_call(messages: list[dict], call_id: str, idx: int) -> dict | None:
    for m in reversed(messages):
        if m.get("role") == "assistant":
            calls = m.get("tool_calls") or []
            return next((c for c in calls if c.get("id") == call_id), calls[idx] if idx < len(calls) else None)
    return None


def what_if(traces: Iterable[list[Event] | str | Path], access: AccessControl, *,
            principal_for: PrincipalFor | None = None,
            run_outcomes: dict[str, bool] | None = None) -> WhatIfReport:
    """Judge every recorded call in `traces` (event lists or JSONL paths)
    under `access` -- a new policy engine plus the capability catalog to
    judge with. Pure: touches no store, calls no model."""
    rep = WhatIfReport(new_version=access.version, old_versions=[], run_outcomes=dict(run_outcomes or {}))
    versions: set[str] = set()
    for t in traces:
        events = load_trace(t) if isinstance(t, (str, Path)) else t
        j = rejudge_run(events, access, principal_for)
        rep.runs += 1
        if j.skipped:
            rep.skipped[j.run_id] = j.skipped
            continue
        versions |= j.versions
        rep.calls += j.calls
        rep.transitions.update(j.transitions)
        rep.changes.update(d.change for d in j.diffs)
        rep.changes["same"] += j.calls - len(j.diffs)
        rep.diffs.extend(j.diffs)
    rep.old_versions = sorted(versions)
    return rep


# ---------------------------------------------------------------------------
# confirming a divergence for real
# ---------------------------------------------------------------------------
async def confirm_first_divergence(events: list[Event], diff: CallDiff, *, snapshots, registry, provider,
                                   env_factory, trace, cfg_override: dict | None = None):
    """Fork the run at `diff.step`, replay the action the model chose there
    under `registry` (carrying the NEW policy), and continue live with
    `provider`. The model is not asked again for that step, so the only
    difference from the recorded run is the policy's answer. Returns the
    fork's RunResult; its trace shows what the agent did instead."""
    from ..fork import ForkPatch, fork_run
    from ..loop import AgentConfig, AgentRunner

    if diff.downstream:
        raise ValueError("confirm the first divergence of a run; later ones depend on it")
    assistant = next((e.data["message"] for e in reversed(events)
                      if e.type == "message" and e.step == diff.step
                      and e.data["message"].get("role") == "assistant"), None)
    if assistant is None:
        raise ValueError(f"no recorded action at step {diff.step}")
    patch = ForkPatch(text=assistant.get("content") or None,
                      tool_calls=[{"name": c["name"], "args": c.get("args") or {}}
                                  for c in assistant.get("tool_calls") or []])
    cfg = AgentConfig(**{**events[0].data["cfg"], **(cfg_override or {})})
    runner = AgentRunner(provider, registry, cfg, snapshots, sleep=_no_sleep)
    return await fork_run(events, snapshots, at_step=diff.step, runner=runner, env_factory=env_factory,
                          trace=trace, patch=patch)


async def _no_sleep(_: float) -> None:
    return None
