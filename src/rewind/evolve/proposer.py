"""Proposer: read a failure cluster's evidence, propose context edits.

One LLM call per cluster returns up to `n` candidate edits as JSON (one call,
several candidates -- API budget). The prompt shows, for each evidence
episode (tasks in the cluster's "A" half only; the "B" half is held out for
screening): the user's goal, the ground-truth mutating actions (allowed: these
are TRAIN tasks), and a compact excerpt of what the agent did. It also shows
the current learned rules and edits already rejected, with why, so the same
idea is not proposed twice.

Every proposal is then checked offline (`surface.validate_edit`): case-
specific text (ids, emails, task numbers), unknown tools, budget overflow
are refused before any evaluation is spent on them.
"""
from __future__ import annotations

import json
import os
from typing import Awaitable, Callable, Protocol

from ..env.taubench_dialogue import load_task
from .surface import ContextVersion, Edit

JsonLLM = Callable[[list[dict]], Awaitable[str]]

SYSTEM = """You improve a retail customer-service agent by editing its context, never its code.
You see failures that share one pattern. Propose GENERAL policy fixes that would prevent this
pattern on unseen customers and orders.

You may propose two kinds of edits:
- "tool_note": a note on ONE tool's description (target = the exact tool name). The agent sees it only
  when it considers that tool. PREFER THIS whenever the failure is about how one tool is used.
- "rule": a policy line in the agent's instructions (target = a short snake_case id), only for
  behaviour that spans several tools or the whole conversation.
Ops: "add", "replace" (an existing rule id / tool note), "delete".

Every edit has a "trigger": the specific situation it applies to ("the customer asks to change an
item's options to an empty set"). It is written as "When <trigger>: <text>". Make the trigger as
NARROW as the evidence allows -- rules that change behaviour beyond their situation break other tasks.

Hard requirements:
- No order ids, item ids, user ids, emails, names or task numbers. Describe the situation, not the case.
- One policy per edit, at most 2 sentences. Do not restate existing rules.
- No unconditional "never" / "always" / "do not" outside the trigger's situation.
- Prefer the smallest change that explains ALL the shown failures.
- If a previous edit for this pattern was rejected with feedback, address that feedback explicitly.
Return JSON: {"edits": [{"kind": ..., "target": ..., "op": ..., "trigger": ..., "text": ...,
"rationale": ...}]}"""


class Proposer(Protocol):
    async def propose(self, cluster_key: str, evidence: list[dict], split: str,
                      ctx: ContextVersion, rejected: list[dict], n: int) -> list[Edit]: ...


def _excerpt(traj: list[dict], max_msgs: int = 14, max_chars: int = 300) -> list[str]:
    """Agent-side conversation, system prompt dropped, long tool outputs cut."""
    out = []
    for m in traj[1:]:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            fn = (m["tool_calls"][0].get("function") or {})
            out.append(f"AGENT CALL {fn.get('name')}({fn.get('arguments')})")
        elif role == "assistant":
            out.append(f"AGENT: {(m.get('content') or '')[:max_chars]}")
        elif role == "tool":
            out.append(f"TOOL: {str(m.get('content'))[:max_chars]}")
        elif role == "user":
            out.append(f"USER: {(m.get('content') or '')[:max_chars]}")
    if len(out) > max_msgs:  # keep the end, where action failures happen
        out = out[:2] + ["..."] + out[-(max_msgs - 3):]
    return out


def build_messages(cluster_key: str, evidence: list[dict], split: str, ctx: ContextVersion,
                   rejected: list[dict], n: int, tool_names: list[str]) -> list[dict]:
    parts = [f"Failure pattern: {cluster_key}", f"Tools: {', '.join(tool_names)}", ""]
    if ctx.rules or ctx.tool_notes:
        parts.append("Current learned rules:")
        parts += [f"- [{k}] {v}" for k, v in ctx.rules.items()]
        parts += [f"- [tool_note {k}] {v}" for k, v in ctx.tool_notes.items()]
        parts.append("")
    if rejected:
        parts.append("Already tried and rejected (do not repeat):")
        for r in rejected:
            trig = f" [when {r['trigger']}]" if r.get("trigger") else ""
            parts.append(f"- {r['kind']} {r['target']}{trig}: {r.get('text')!r} -> {r['reason']}")
            if r.get("feedback"):
                parts += ["    " + line for line in r["feedback"].splitlines()]
        parts.append("")
    for i, row in enumerate(evidence, 1):
        task = load_task(split, row["task_id"])
        gt = [f"{a.name}({json.dumps(a.kwargs, ensure_ascii=False)})" for a in task.actions
              if a.name != "respond"]
        parts.append(f"=== Failure {i} ===")
        parts.append(f"Customer goal: {task.instruction}")
        parts.append("Correct actions: " + "; ".join(gt))
        parts += _excerpt(row.get("traj", []))
        parts.append("")
    parts.append(f"Propose up to {n} alternative edits (different ideas, not variants of one).")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(parts)}]


def parse_edits(text: str, cluster_key: str, evidence_tasks: list[int]) -> list[Edit]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s < 0 or e <= s:
            return []
        try:
            data = json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            return []
    out = []
    for d in (data.get("edits") or []) if isinstance(data, dict) else []:
        try:
            out.append(Edit(kind=d["kind"], target=str(d["target"]).strip(), op=d.get("op", "add"),
                            text=d.get("text"), trigger=d.get("trigger"),
                            rationale=d.get("rationale", ""),
                            source_cluster=cluster_key, evidence_tasks=evidence_tasks))
        except Exception:  # malformed item: skip it, keep the rest
            continue
    return out


class LLMProposer:
    def __init__(self, llm: JsonLLM, tool_names: list[str]) -> None:
        self.llm = llm
        self.tool_names = tool_names
        self.calls = 0

    async def propose(self, cluster_key, evidence, split, ctx, rejected, n=3) -> list[Edit]:
        msgs = build_messages(cluster_key, evidence, split, ctx, rejected, n, self.tool_names)
        self.last_messages = msgs
        self.calls += 1
        text = await self.llm(msgs)
        return parse_edits(text, cluster_key, [r["task_id"] for r in evidence])[:n]


def deepseek_json_llm(model: str = "deepseek-chat", temperature: float = 0.7,
                      max_tokens: int = 1500) -> JsonLLM:
    """One chat call in JSON mode. Temperature > 0 on purpose: the n
    candidates should be different ideas."""
    import openai
    client = openai.AsyncOpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                                base_url="https://api.deepseek.com")

    async def call(messages: list[dict]) -> str:
        resp = await client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
            response_format={"type": "json_object"})
        return resp.choices[0].message.content or ""
    return call
