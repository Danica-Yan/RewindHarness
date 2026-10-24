"""Skill induction and skill usage.

Induction learns from SUCCESS, not failure: passing train episodes are
grouped by the kind of write their task needs (one group per mutating tool),
and one LLM call per group abstracts a few of them into a general procedure
(in the spirit of Agent Workflow Memory). The result is a list of `skill`
edits that go through the same validation and gate as any other edit.

Usage is read back from results: an episode "loaded" a skill if the agent
called `load_skill` with its name. Comparing pass rates of episodes that
loaded a skill with those of the same task type that did not is the
attribution a prompt rule cannot give (but it is observational -- the agent
chooses when to load, so the comparison is a diagnostic, not the gate).
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from ..env.taubench_adapter import MUTATING_TOOLS
from ..env.taubench_dialogue import load_task
from ..skills import LOAD_SKILL
from .proposer import _excerpt
from .surface import ContextVersion, Edit, EditError, parse_skill

JsonLLM = Callable[[list[dict]], Awaitable[str]]

INDUCE_SYSTEM = """You write reusable procedures ("skills") for a retail customer-service agent.
You see several conversations in which the agent handled the SAME kind of request successfully.
Abstract them into ONE general step-by-step procedure that would handle any request of this kind
for any customer and order.

Requirements:
- Steps are short imperative sentences, in order, 3-10 of them. Include the checks and the moment
  to ask the user for explicit confirmation, and how to call the write tool (e.g. once, with all items).
- No order ids, item ids, user ids, emails, names, amounts or other case details.
- The agent's policy is given below. Include EVERY requirement of it that applies to this kind of
  request (status checks, reminders, allowed values, payment constraints), in the step where it
  applies -- the conversations may not show all of them.
- Never add a requirement the policy does not state, and do not narrow one it does (e.g. a balance
  check the policy only demands for gift cards must not be applied to every payment method).
- The agent ALSO follows the learned rules and tool notes listed below (they were proven by tests).
  A skill must never contradict them; where one applies, follow it in the relevant step.
- "description" is ONE line of at most 200 characters saying when to use the skill (the agent
  decides from it whether to load it).
Return JSON: {"name": "<snake_case>", "description": "...", "steps": ["...", ...],
"tools": ["<tool names the procedure uses>"]}"""

CONTRAST_ADDENDUM = """
You ALSO see failed conversations of the same kind of request, with the actions that would have
been correct. The agent already has the policy above in its instructions, so a skill that only
restates it adds nothing. Find what the successful conversations did that the failed ones did NOT
(how they found the right item or variant, what they asked before acting, how they handled an
unusual request), and make those concrete practices the core of the steps. Keep policy steps only
where a failure violated them."""


class SkillGroup(BaseModel):
    tool: str                       # the write tool that defines the group
    passing: list[int]              # row indices of passing episodes whose task uses `tool`
    examples: list[int]             # row indices chosen as evidence
    task_ids: list[int]
    failing_examples: list[int] = []   # contrast mode: A-half failures shown to the inducer


def failure_half(task_id: int) -> str:
    """Fixed A/B split of tasks for skill work: only failures of A tasks are
    ever shown to the inducer (contrast mode), so B failures always stay
    available as gate targets (the gate also excludes every shown task).
    Deterministic, so `plan`, `induce` and `gate` agree without state."""
    return "A" if hashlib.sha256(f"skill-split-{task_id}".encode()).digest()[0] % 2 == 0 else "B"


def task_writes(split: str, task_id: int) -> set[str]:
    return {a.name for a in load_task(split, task_id).actions if a.name in MUTATING_TOOLS}


def skill_groups(rows: list[dict], split: str, *, min_passing: int = 5, k: int = 4,
                 skip_tools: set[str] = frozenset(), contrast: bool = False,
                 fail_k: int = 2) -> list[SkillGroup]:
    """One group per write tool with at least `min_passing` passing episodes.
    Examples: episodes whose task needs ONLY that write first (cleanest
    signal), then the shortest conversations, one per task. With `contrast`,
    also up to `fail_k` failed episodes of the type from A-half tasks."""
    by_tool: dict[str, list[int]] = defaultdict(list)
    writes = {}
    for i, r in enumerate(rows):
        if r["reward"] < 1.0:
            continue
        w = writes.setdefault(r["task_id"], task_writes(split, r["task_id"]))
        for t in w:
            by_tool[t].append(i)
    out = []
    for tool in sorted(by_tool):
        idx = by_tool[tool]
        if tool in skip_tools or len({rows[i]["task_id"] for i in idx}) < min_passing:
            continue
        ranked = sorted(idx, key=lambda i: (len(writes[rows[i]["task_id"]]) > 1, len(rows[i].get("traj", []))))
        ex, seen = [], set()
        for i in ranked:
            if rows[i]["task_id"] not in seen:
                ex.append(i)
                seen.add(rows[i]["task_id"])
            if len(ex) == k:
                break
        fails: list[int] = []
        if contrast:
            seen_f: set[int] = set()
            for i, rr in enumerate(rows):
                t = rr["task_id"]
                if rr["reward"] >= 1.0 or t in seen_f or failure_half(t) != "A":
                    continue
                if tool in writes.setdefault(t, task_writes(split, t)):
                    fails.append(i)
                    seen_f.add(t)
                if len(fails) == fail_k:
                    break
        out.append(SkillGroup(tool=tool, passing=idx, examples=ex,
                              task_ids=sorted({rows[i]["task_id"] for i in idx}), failing_examples=fails))
    return out


def induce_messages(group: SkillGroup, rows: list[dict], split: str, ctx: ContextVersion,
                    tool_names: list[str], policy: str | None = None) -> list[dict]:
    if policy is None:
        from ..env.taubench_dialogue import wiki
        policy = wiki()
    parts = [f"Kind of request: tasks that need `{group.tool}`.", f"Tools: {', '.join(tool_names)}", "",
             "=== The agent's policy ===", policy.strip(), ""]
    if ctx.rules or ctx.tool_notes:
        parts.append("=== Learned rules and tool notes the agent also follows (never contradict) ===")
        parts += [f"- {v}" for v in ctx.rules.values()]
        parts += [f"- note on {k}: {v}" for k, v in ctx.tool_notes.items()]
        parts.append("")
    if ctx.skills:
        parts.append("Skills that already exist (do not duplicate):")
        parts += [f"- {k}: {v['description']}" for k, v in ctx.skills.items()]
        parts.append("")
    for n, i in enumerate(group.examples, 1):
        r = rows[i]
        parts.append(f"=== Successful conversation {n} ===")
        parts.append(f"Customer goal: {load_task(split, r['task_id']).instruction}")
        parts += _excerpt(r.get("traj", []), max_msgs=24, max_chars=220)
        parts.append("")
    for n, i in enumerate(group.failing_examples, 1):
        r = rows[i]
        task = load_task(split, r["task_id"])
        gold = [f"{a.name}({json.dumps(a.kwargs, ensure_ascii=False)})" for a in task.actions
                if a.name != "respond"]
        parts.append(f"=== FAILED conversation {n} (same kind of request) ===")
        parts.append(f"Customer goal: {task.instruction}")
        parts.append("Correct actions would have been: " + "; ".join(gold))
        parts += _excerpt(r.get("traj", []), max_msgs=24, max_chars=220)
        parts.append("")
    system = INDUCE_SYSTEM + (CONTRAST_ADDENDUM if group.failing_examples else "")
    return [{"role": "system", "content": system}, {"role": "user", "content": "\n".join(parts)}]


def parse_skill_edit(text: str, group: SkillGroup, rows: list[dict]) -> Edit | None:
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s < 0 or e <= s:
            return None
        try:
            d = json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(d, dict) or not d.get("name"):
        return None
    body = {k: d.get(k) for k in ("description", "steps", "tools")}
    return Edit(kind="skill", target=str(d["name"]).strip(), op="add",
                text=json.dumps(body, ensure_ascii=False),
                rationale=f"induced from {len(group.examples)} successful"
                          + (f" and {len(group.failing_examples)} failed" if group.failing_examples else "")
                          + f" `{group.tool}` episodes",
                source_cluster=f"skill:{group.tool}",
                evidence_tasks=[rows[i]["task_id"] for i in group.examples + group.failing_examples])


async def induce(groups: list[SkillGroup], rows: list[dict], split: str, ctx: ContextVersion,
                 llm: JsonLLM, tool_names: list[str], policy: str | None = None
                 ) -> list[tuple[SkillGroup, Edit | None, str]]:
    """One LLM call per group. Returns (group, edit or None, raw text)."""
    out = []
    for g in groups:
        text = await llm(induce_messages(g, rows, split, ctx, tool_names, policy))
        out.append((g, parse_skill_edit(text, g, rows), text))
    return out


def library_version(base: ContextVersion, edits: list[Edit], known_tools: set[str]) -> ContextVersion:
    """`base` plus all induced skills as ONE candidate version (skipping ones
    that fail to apply, e.g. duplicate names)."""
    v = base
    for e in edits:
        try:
            parse_skill(e)
            v = v.apply(e, known_tools=known_tools)
        except EditError:
            continue
    return v.model_copy(update={"version": base.version + 1, "parent": base.version})


# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------
def loaded_skills(traj: list[dict]) -> list[str]:
    out = []
    for m in traj:
        if m.get("role") != "assistant":
            continue
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or c
            if fn.get("name") == LOAD_SKILL:
                args = fn.get("arguments") or fn.get("args") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                if args.get("name"):
                    out.append(args["name"])
    return out


class SkillUsage(BaseModel):
    skill: str
    covers_tool: str | None         # the write tool it was induced for, if known
    relevant: int                   # episodes whose task needs that tool
    loaded: int                     # ... of which loaded the skill
    pass_loaded: float | None
    pass_not_loaded: float | None
    loaded_elsewhere: int           # loaded in episodes whose task does not need the tool


def skill_usage(rows: list[dict], split: str, covers: dict[str, str]) -> list[SkillUsage]:
    """`covers`: skill name -> the write tool it is for (from the edit's
    source_cluster "skill:<tool>"). Observational: agents choose when to load."""
    out = []
    names = set(covers) | {s for r in rows for s in loaded_skills(r.get("traj", []))}
    for name in sorted(names):
        tool = covers.get(name)
        rel = [r for r in rows if tool and tool in task_writes(split, r["task_id"])]
        ld = [r for r in rel if name in loaded_skills(r.get("traj", []))]
        nl = [r for r in rel if name not in loaded_skills(r.get("traj", []))]
        elsewhere = sum(1 for r in rows if name in loaded_skills(r.get("traj", [])) and r not in rel)

        def rate(xs: list[dict]) -> float | None:
            return sum(x["reward"] >= 1.0 for x in xs) / len(xs) if xs else None
        out.append(SkillUsage(skill=name, covers_tool=tool, relevant=len(rel), loaded=len(ld),
                              pass_loaded=rate(ld), pass_not_loaded=rate(nl), loaded_elsewhere=elsewhere))
    return out


def covers_from_ledger(ledger: Any) -> dict[str, str]:
    """skill name -> write tool, from the committed edits that added skills."""
    out = {}
    for _v, edits in ledger.lineage(ledger.current().version):
        for e in edits:
            if e.get("kind") == "skill" and str(e.get("source_cluster", "")).startswith("skill:"):
                out[e["target"]] = e["source_cluster"].split(":", 1)[1]
    return out
