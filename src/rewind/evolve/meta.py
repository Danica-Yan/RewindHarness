"""Improving the improver: the proposer's own instructions evolve.

The proposer is told how to write edits by a fixed system prompt. On top of
it sits a short list of LESSONS, versioned in the ledger. `reflect` asks an
LLM to read the ledger -- every edit proposed so far, how far it got
(invalid / backtest / screen / gate) and the gate's feedback -- and to
rewrite the lessons so that future proposals fare better.

A new lesson set is adopted only if it wins a fixed external test
(`meta_eval`): on clusters whose A/B split already exists, the proposer
writes candidates with the OLD lessons and with the NEW ones; both batches
go through the same offline validation and the same screening forks
against each task's own null rate. The new lessons are adopted only if
they produce more screened-in candidates (ties broken by valid ones) and
never fewer. Every candidate's outcome is recorded (stage "meta_screen").

What the meta level can NOT change: the gate, its thresholds, the
screening rule, the held-out splits. Lessons are text the proposer reads;
they are validated like edits (no case details, bounded length, no talk of
the tests themselves), and the external test decides.
"""
from __future__ import annotations

import json
import re
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from .ledger import EditLedger
from .surface import leak_reasons

JsonLLM = Callable[[list[dict]], Awaitable[str]]
KEY = "proposer_lessons"
MAX_LESSONS = 8
# Lessons may mention what happened at each stage ("passed screening but harmed other tasks");
# they may not be about beating the tests themselves.
_GAMING = re.compile(r"(p-?values?|fisher|thresholds?|null rates?|bypass|\bgam(e|ing)\b|\btrick\b|"
                     r"pass(es|ing)? the (gate|test|screen(ing)?|check)|to (pass|get through) (the )?(gate|screen))",
                     re.I)

REFLECT_SYSTEM = """You improve the instructions of a PROPOSER: an LLM that reads failures of a retail
customer-service agent and proposes edits to the agent's context (rules, tool notes, policy checks,
skills). You see every edit it proposed so far, how far each got, and the feedback it received.
Stages: invalid (refused offline), backtest (a check that would fire on good runs or catch nothing),
screen (did not fix held-out failures), gate (fixed them but harmed other tasks, or did not).

Write at most 8 short LESSONS (each under 200 characters) that would make its future edits more
likely to actually fix failures without side effects. Lessons are about HOW to write edits -- choice
of kind, scope of the trigger, wording, what kinds of fixes have worked or failed -- never about
specific customers, orders, ids or task numbers, and never about the evaluation procedure itself.
Keep lessons from the current list that the evidence still supports.
Return JSON: {"lessons": ["...", ...], "rationale": "<one paragraph: what the history shows>"}"""


def lessons(ledger: EditLedger) -> list[str]:
    raw = ledger._get(KEY)  # noqa: SLF001
    return json.loads(raw)["lessons"] if raw else []


def history(ledger: EditLedger) -> list[dict]:
    raw = ledger._get(KEY + "_history")  # noqa: SLF001
    return json.loads(raw) if raw else []


def adopt(ledger: EditLedger, new: list[str], why: dict) -> None:
    h = history(ledger)
    h.append({"lessons": new, **why})
    ledger._set(KEY, json.dumps({"lessons": new}, ensure_ascii=False))  # noqa: SLF001
    ledger._set(KEY + "_history", json.dumps(h, ensure_ascii=False))  # noqa: SLF001
    ledger.db.commit()


def validate_lessons(xs: Any) -> tuple[list[str], str | None]:
    """(kept lessons, note). Offending lessons are dropped one by one (the
    note says which and why); the set is refused only if none survive."""
    if not isinstance(xs, list) or not xs:
        return [], "no lessons"
    xs = [" ".join(str(x).split()) for x in xs if str(x).strip()]
    kept, dropped = [], []
    for x in xs:
        if len(x) > 220:
            dropped.append(f"too long: {x[:60]}...")
        elif leak_reasons(x):
            dropped.append(f"case-specific ({', '.join(leak_reasons(x))}): {x[:60]}")
        elif _GAMING.search(x):
            dropped.append(f"about beating the evaluation, not writing edits: {x[:60]}")
        else:
            kept.append(x)
    if len(kept) > MAX_LESSONS:
        dropped += [f"over the limit of {MAX_LESSONS}: {x[:60]}" for x in kept[MAX_LESSONS:]]
        kept = kept[:MAX_LESSONS]
    note = ("dropped " + "; ".join(dropped)) if dropped else None
    return kept, (note if kept or not dropped else "all lessons refused: " + "; ".join(dropped))


def experience(ledger: EditLedger, exclude_clusters: set[str] = frozenset()) -> list[dict]:
    """Every proposed edit and where it stopped, with gate feedback."""
    out = []
    for cluster, ej, stage, acc, reason, dj in ledger.db.execute(
            "SELECT cluster, edit_json, stage, accepted, reason, data_json FROM decisions "
            "WHERE edit_json IS NOT NULL ORDER BY id"):
        if cluster in exclude_clusters or stage in ("ablation", "meta", "meta_screen"):
            continue
        e = json.loads(ej)
        data = json.loads(dj) if dj else {}
        out.append({"cluster": cluster, "kind": e.get("kind"), "target": e.get("target"), "trigger": e.get("trigger"),
                    "text": (e.get("text") or "")[:400], "stage": stage, "accepted": bool(acc),
                    "reason": reason, "feedback": ((data.get("feedback") or {}).get("text") or "")[:600]})
    return out


def reflect_messages(exp: list[dict], current: list[str]) -> list[dict]:
    parts = ["Current lessons:" if current else "Current lessons: (none yet)"]
    parts += [f"- {x}" for x in current]
    parts.append("\nProposal history (oldest first):")
    for i, x in enumerate(exp, 1):
        trig = f" [when {x['trigger']}]" if x.get("trigger") else ""
        parts.append(f"{i}. ({x['cluster']}) {x['kind']} {x.get('target') or ''}{trig}: {x['text']!r}\n   -> {x['stage']} "
                     f"{'PASSED' if x['accepted'] else 'stopped'}: {x['reason']}")
        if x["feedback"]:
            parts.append("   feedback: " + x["feedback"].replace("\n", " "))
    return [{"role": "system", "content": REFLECT_SYSTEM}, {"role": "user", "content": "\n".join(parts)}]


async def reflect(ledger: EditLedger, llm: JsonLLM, exclude_clusters: set[str] = frozenset()
                  ) -> tuple[list[str], str, str | None, str]:
    """(new lessons, rationale, note, raw output). One LLM call. `note` says
    which lessons were dropped; lessons is empty only if none survived."""
    exp = experience(ledger, exclude_clusters)
    if not exp:
        return [], "", "no proposal history to learn from", ""
    text = await llm(reflect_messages(exp, lessons(ledger)))
    try:
        d = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except (json.JSONDecodeError, ValueError):
        return [], "", "unparseable reflection", text
    new, note = validate_lessons(d.get("lessons"))
    return new, str(d.get("rationale", "")), note, text


class ArmResult(BaseModel):
    proposed: int = 0
    valid: int = 0
    screened_in: int = 0
    forks: int = 0
    edits: list[dict] = []


def meta_clusters(ledger: EditLedger, cfg: Any) -> list[dict]:
    """Every cluster with a recorded A/B split and stability records (any
    version). Both arms are screened against the same B null rates, so any
    bias from an older version's rates is shared and cancels in the
    comparison. Clusters without an accepted edit come first."""
    import re

    from .loop import _paths
    from .stability import load_records
    stab = load_records(_paths(cfg)["stability"])
    out = []
    for cluster, v, reason, dj in ledger.db.execute(
            "SELECT cluster, version_from, reason, data_json FROM decisions WHERE stage='stability' "
            "AND accepted=1 ORDER BY id"):
        data = json.loads(dj) if dj else {}
        a_ids, b_ids = data.get("A"), data.get("B")
        if a_ids is None:
            m = re.search(r"A=\[([\d, ]*)\] B=\[([\d, ]*)\]", reason or "")
            if not m:
                continue
            a_ids = [int(x) for x in m.group(1).split(",") if x.strip()]
            b_ids = [int(x) for x in m.group(2).split(",") if x.strip()]
        rec = {x.task_id: x for x in stab.values() if x.ctx_version == v}
        if not all(t in rec for t in a_ids + b_ids) or any(c["cluster"] == cluster for c in out):
            continue
        solved = ledger.db.execute("SELECT 1 FROM decisions WHERE stage='gate' AND accepted=1 AND cluster=?",
                                   (cluster,)).fetchone() is not None
        out.append({"cluster": cluster, "A": [rec[t] for t in a_ids], "B": [rec[t] for t in b_ids],
                    "solved": solved})
    return sorted(out, key=lambda c: c["solved"])


def meta_cost(cfg: Any, clusters: list[dict]) -> int:
    return sum(2 * cfg.n_candidates * len(c["B"]) * cfg.screen.n for c in clusters)


async def meta_eval(cfg: Any, ledger: EditLedger, new: list[str], clusters: list[dict], *, llm: JsonLLM,
                    tool_names: list[str], provider: Any, user_llm: Any, store: Any,
                    log: Callable[[str], None] = print) -> dict:
    """Old vs new lessons on the same clusters, judged by the fixed offline
    validation + screening. Adopts the new lessons only if they win."""
    from .loop import backtest_edit
    from .prevalence import evidence_rows
    from .proposer import LLMProposer
    from .runner import load_rows, version_setup
    from .screen import screen
    from .surface import validate_edit
    inc = ledger.current()
    rows = load_rows(ledger.rows_path(inc.version))
    base_rows = load_rows(ledger.rows_path(0))
    _, reg = version_setup(inc)
    tools = set(reg._tools)  # noqa: SLF001
    arms = {"old": ArmResult(), "new": ArmResult()}
    round_no = ledger.bump_round()
    for c in clusters:
        a_rows = evidence_rows(base_rows, [r.task_id for r in c["A"]])
        b_tasks = [(r.task_id, r.trace, r.null_rate) for r in c["B"]]
        # both arms see the same rejection history: nothing from this
        # evaluation is recorded until both have proposed
        rejected = ledger.rejected_for_prompt(c["cluster"])
        pending: list[tuple] = []
        for arm, ls in (("old", lessons(ledger)), ("new", new)):
            res = arms[arm]
            prop = LLMProposer(llm, tool_names, lessons=ls)
            edits = await prop.propose(c["cluster"], a_rows, cfg.split, inc, rejected, cfg.n_candidates)
            res.proposed += len(edits)
            for e in edits:
                why = validate_edit(e, inc, known_tools=tools, max_tokens=cfg.gate.max_ctx_tokens)
                if why is None and e.kind == "check" and e.op != "delete":
                    ok, why2, _ = backtest_edit(cfg, e, rows, store, {r.task_id for r in c["A"] + c["B"]})
                    why = None if ok else why2
                if why is not None:
                    res.edits.append({"cluster": c["cluster"], "target": e.target, "stage": "invalid", "why": why})
                    pending.append((e, "meta_screen", False, f"[{arm} lessons] invalid: {why}", None))
                    continue
                res.valid += 1
                cand = inc.apply(e, known_tools=tools)
                ccfg, creg = version_setup(cand)
                sr = await screen([e.edit_id], b_tasks, system_prompt=ccfg.system_prompt, registry=creg,
                                  store=store, provider=provider, user_llm=user_llm, policy=cfg.screen, log=log)
                res.forks += sr.forks
                res.screened_in += sr.passed
                pending.append((e, "meta_screen", sr.passed, f"[{arm} lessons] {sr.reason}", sr.model_dump()))
                res.edits.append({"cluster": c["cluster"], "target": e.target, "stage": "screen",
                                  "passed": sr.passed, "why": sr.reason})
            log(f"  {c['cluster']} [{arm}]: {len(edits)} proposed, cumulative valid {res.valid}, "
                f"screened in {res.screened_in}")
        for e, stage, ok, why, data in pending:
            ledger.record(round_no, inc.version, e, stage, ok, why, data, cluster=c["cluster"])
    o, n = arms["old"], arms["new"]
    win = (n.screened_in > o.screened_in) or (n.screened_in == o.screened_in and n.valid > o.valid)
    reason = (f"new lessons: {n.screened_in} screened in / {n.valid} valid of {n.proposed}; "
              f"old: {o.screened_in} / {o.valid} of {o.proposed}")
    data = {"old": o.model_dump(), "new": n.model_dump(), "lessons": new}
    ledger.record(round_no, inc.version, None, "meta", win, reason, data, cluster="meta")
    if win:
        adopt(ledger, new, {"round": round_no, "reason": reason})
    return {"adopted": win, "reason": reason, "episodes": o.forks + n.forks}
