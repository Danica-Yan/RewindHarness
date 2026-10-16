"""A deterministic, rule-based stand-in for an LLM (plan W8's flywheel
rounds need *some* policy to iteratively improve — this module keeps
that loop fully offline and reproducible without any LLM API key, so the
"model" here is a small heuristic decision function with two independent,
named bugs that can each be toggled off, standing in for what a real round
would be: "the prompt/tool now tells the model to check X before acting").

This is a simulation of the flywheel *mechanics* (split, run, classify,
cluster, pick a fix, gate on dev, merge, report heldout), not a claim that
prompting a real LLM works this way — that honesty is stated plainly here
and in the README, not buried.

`HeuristicRetailProvider` always looks the target order up with `get_order`
first (not `list_orders` — that only returns orders *for the given user*,
which would hide an other-user's order from the model entirely and make
the ownership bug below unobservable; `get_order` returns any order by id,
same as a naive real implementation would likely reach for when the task
names one specific order). Then:

  - if `bug_ignores_ownership` is True, cancels the named order without
    checking whether it belongs to the requester (the tool itself has no
    ownership check — see env.py — so this *succeeds*, just on the wrong
    grounds: `REASONING_ERROR` on "ownership_trap" tasks);
  - if `bug_ignores_status` is True, cancels the named order without
    checking it's actually `pending` (the tool *does* enforce this, so the
    call fails: `TOOL_RUNTIME_ERROR` on "shipped_trap" tasks — but because
    the tool's own guard already prevents the wrong final state, this bug
    shows up in the trace classification without changing the task's
    pass/fail grade; see README for why that makes it a *visibility* fix
    rather than a *correctness* fix, and what that implies for the
    flywheel's round-2 numbers);
  - with both bugs off, it correctly refuses on either trap and only
    cancels the "normal" case.
Both default True (the "un-fixed" baseline the flywheel's round 0 measures).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from ..providers import LLMResponse, ToolCall


@dataclass
class HeuristicRetailProvider:
    target_order: str            # which order this task's text names (see tasks.py)
    requester: str
    bug_ignores_ownership: bool = True
    bug_ignores_status: bool = True

    async def complete(self, messages: list[dict], tools: list[dict], params: dict) -> LLMResponse:
        n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
        if n_assistant == 0:
            return LLMResponse(tool_calls=[ToolCall(name="get_order", args={"order_id": self.target_order})],
                               usage={"input_tokens": 10, "output_tokens": 5})

        last_tool_msg = next((m for m in reversed(messages) if m.get("role") == "tool"), None)
        last_assistant = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
        last_calls = (last_assistant or {}).get("tool_calls") or []
        last_call_name = last_calls[0]["name"] if last_calls else None

        if last_call_name == "get_order" and n_assistant == 1:
            if last_tool_msg is None or last_tool_msg.get("is_error"):
                return LLMResponse(text="I could not find that order.",
                                   usage={"input_tokens": 10, "output_tokens": 5})
            target = json.loads(last_tool_msg["content"])
            owned = target["user_id"] == self.requester
            pending = target["status"] == "pending"
            if not owned and not self.bug_ignores_ownership:
                return LLMResponse(text="That order does not belong to you, so I can't cancel it.",
                                   usage={"input_tokens": 10, "output_tokens": 5})
            if not pending and not self.bug_ignores_status:
                return LLMResponse(text=f"That order is {target['status']} and can no longer be cancelled.",
                                   usage={"input_tokens": 10, "output_tokens": 5})
            return LLMResponse(tool_calls=[ToolCall(name="cancel_order",
                                                    args={"order_id": self.target_order,
                                                          "reason": "ordered_by_mistake"})],
                               usage={"input_tokens": 10, "output_tokens": 5})

        # After the cancel_order attempt (whether it actually succeeded or the
        # tool itself refused it — e.g. a shipped order when bug_ignores_status
        # is still True): wrap up either way.
        ok = last_tool_msg is not None and not last_tool_msg.get("is_error")
        text = "Done, your order has been cancelled." if ok else "Sorry, I could not cancel that order."
        return LLMResponse(text=text, usage={"input_tokens": 10, "output_tokens": 5})
