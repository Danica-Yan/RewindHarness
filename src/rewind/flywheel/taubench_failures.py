"""Failure-mode classification for real tau-bench `EnvRunResult` records
(W1-followup, after the real DeepSeek baseline run -- see docs/DEVLOG.zh-CN.md "真实模型接入：
DeepSeek"). `flywheel/failures.py` classifies failures from *this repo's own*
`AgentRunner` trace format (`Event`/`RunResult`); it can't read tau-bench's
output, which is a different project's own `EnvRunResult` shape
(`task_id`/`reward`/`info`/`traj`/`trial` -- see `tau_bench/types.py`) produced
by *their* `run.py`, not by anything in this repo. This module is the
tau-bench-side counterpart: same spirit (turn per-run signals into a
batch-level failure distribution), different input format, because the two
harnesses don't share a trace schema and reimplementing tau-bench's protocol
just to reuse `failures.py` would be the same scope mistake
`taubench_adapter.py`'s module docstring already declined to make.

Unlike `eval_attribution.py` (which had no real failure corpus and was
explicit about using synthetic ground truth instead), this module exists
*because* a real corpus now exists: 34 real failed tasks out of 115, from
`deepseek-chat` as both agent and user simulator on the retail test split,
1 trial, reward < 1.0. Every category below was picked by looking at what
that real file actually contains, not designed in the abstract first -- see
`scripts/taubench_failure_report.py` for the real distribution it produces.

What tau-bench's own output *does* and *doesn't* tell us, which shapes the
taxonomy:

- `traj` is the full OpenAI-format message list. A tool-role message whose
  `content` starts with `"Error: "` is tau-bench's own convention for a
  failed tool call (see `taubench_adapter.py`'s `_wrap_tool_fn` for the same
  convention on this repo's side) -- a concrete, mechanical signal that's
  always worth checking first, regardless of how the task ultimately graded.
- `info["reward_info"]["info"]` is tau-bench's own `RewardActionInfo`
  (`r_actions`/`gt_data_hash`) for tasks with no required output text, OR
  `RewardOutputInfo` (`r_outputs`/`outputs`) for tasks that also require a
  specific substring in the agent's final response -- `calculate_reward`
  (tau_bench/envs/base.py) *overwrites* the action info with the output info
  when the task has outputs, so for an outputs-bearing task we cannot see
  `r_actions` directly. What we *can* see: if `r_outputs == 1.0` but the
  overall `reward` is still `0.0`, the required text WAS said, so the
  mismatch must be in the actions -- a real, distinguishable failure shape
  (said the right thing, state ended up wrong) worth its own category rather
  than collapsing it into "wrong output".
- `info["reward_info"]["actions"]` is always present (computed once, before
  the info variant is chosen) -- the ground-truth non-`respond` actions.
  Comparing their tool names against the agent's own mutating tool calls in
  `traj` (filtered through `taubench_adapter.MUTATING_TOOLS`, the same
  allowlist the adapter itself uses) gives a cheap, real signal for *why* a
  WRONG_ACTION failure happened: never attempted a mutating action at all,
  attempted a different one, or attempted the right one(s) but evidently
  with different args (names match, hash still differs).
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..env.taubench_adapter import MUTATING_TOOLS

CATEGORIES = (
    "TOOL_RUNTIME_ERROR",                   # a tool call came back "Error: ..." and was never recovered
    "WRONG_OUTPUT",                         # required response text missing (r_outputs < 1.0)
    "ACTION_MISMATCH_DESPITE_RIGHT_OUTPUT",  # said the right thing; final state still wrong
    "WRONG_ACTION",                         # action-only task (no required output text); state wrong
    "EPISODE_NOT_FINISHED",                 # hit the step/time budget before the conversation ended
    "UNKNOWN",                              # reward < 1.0 but none of the above signals fired
)


@dataclass
class TauFailureRecord:
    task_id: int
    trial: int
    category: str
    reward: float
    last_tool_error: str | None
    gt_mutating_tools: list[str] = field(default_factory=list)
    agent_mutating_tools: list[str] = field(default_factory=list)
    missing_outputs: list[str] = field(default_factory=list)
    last_tool_error_tool: str | None = None
    recovered_tool_errors: list[str] = field(default_factory=list)  # errors the agent got past


# A failed call counts as recovered if a later call to the same tool -- or to
# a tool that serves the same purpose -- succeeded. Explicit and auditable:
# in tau-bench retail the two user lookups are interchangeable (email not on
# file -> agent asks for name + zip), and nothing else is.
RECOVERY_EQUIVALENTS: dict[str, set[str]] = {
    "find_user_id_by_email": {"find_user_id_by_name_zip"},
    "find_user_id_by_name_zip": {"find_user_id_by_email"},
}


@dataclass
class ToolResult:
    index: int
    tool: str | None
    content: str
    is_error: bool


def _tool_results(traj: list[dict[str, Any]]) -> list[ToolResult]:
    """Every tool result in order, with the tool's name resolved from the
    message itself (run.py records it), else its tool_call_id, else the
    assistant call right before it (rewind's own OpenAI-format export)."""
    by_id: dict[str, str] = {}
    last_call: str | None = None
    out = []
    for i, m in enumerate(traj):
        role = m.get("role")
        if role == "assistant":
            calls = m.get("tool_calls") or []
            for c in calls:
                name = (c.get("function") or {}).get("name")
                if c.get("id") and name:
                    by_id[c["id"]] = name
            last_call = (calls[0].get("function") or {}).get("name") if calls else None
        elif role == "tool":
            name = m.get("name") or by_id.get(m.get("tool_call_id") or "") or last_call
            content = str(m.get("content", ""))
            out.append(ToolResult(i, name, content, content.startswith("Error:")))
    return out


def tool_errors(traj: list[dict[str, Any]]) -> tuple[list[ToolResult], list[ToolResult]]:
    """(unrecovered, recovered) tool errors. An error is recovered if a later
    call to the same tool, or an equivalent one, succeeded."""
    results = _tool_results(traj)
    unrecovered, recovered = [], []
    for j, r in enumerate(results):
        if not r.is_error:
            continue
        ok_tools = {r.tool} | RECOVERY_EQUIVALENTS.get(r.tool or "", set())
        if any(not later.is_error and later.tool in ok_tools for later in results[j + 1:]):
            recovered.append(r)
        else:
            unrecovered.append(r)
    return unrecovered, recovered


def _tool_error_messages(traj: list[dict[str, Any]]) -> list[str]:
    """Unrecovered tool errors only (kept for callers of the old helper)."""
    return [r.content for r in tool_errors(traj)[0]]


def _agent_mutating_tool_calls(traj: list[dict[str, Any]]) -> list[str]:
    """Tool names the agent itself called that are in `MUTATING_TOOLS` --
    `ToolCallingAgent.solve` (tau_bench/agents/tool_calling_agent.py) keeps
    only the first tool call per assistant turn, so one name per turn."""
    names = []
    for m in traj:
        if m.get("role") != "assistant":
            continue
        for call in (m.get("tool_calls") or [])[:1]:
            name = (call.get("function") or {}).get("name")
            if name in MUTATING_TOOLS:
                names.append(name)
    return names


def classify_taubench(result: dict[str, Any]) -> TauFailureRecord | None:
    """`result` is one `EnvRunResult.model_dump()` dict, as tau-bench's own
    `run.py` writes them to its results json. `None` means this task
    graded as a full success (`reward >= 1.0`), not a failure."""
    if result["reward"] >= 1.0:
        return None

    traj = result.get("traj", [])
    unrecovered, recovered = tool_errors(traj)
    reward_info = (result.get("info") or {}).get("reward_info") or {}
    info = reward_info.get("info") or {}
    gt_actions = [a["name"] for a in reward_info.get("actions", []) if a["name"] in MUTATING_TOOLS]
    agent_actions = _agent_mutating_tool_calls(traj)

    # Only an error the agent never got past explains the failure; a call that
    # failed and was then retried successfully falls through to the outcome
    # checks below (it is still recorded in `recovered_tool_errors`).
    rewind_status = ((result.get("info") or {}).get("rewind") or {}).get("status")
    if unrecovered:
        category = "TOOL_RUNTIME_ERROR"
    elif not reward_info or (rewind_status is not None and rewind_status != "completed"):
        # run.py never computes a reward when an episode runs out of steps;
        # rewind grades it anyway but records why the loop stopped
        category = "EPISODE_NOT_FINISHED"
    elif "r_outputs" in info and info["r_outputs"] < 1.0:
        category = "WRONG_OUTPUT"
    elif "r_outputs" in info:  # r_outputs == 1.0 but overall reward < 1.0
        category = "ACTION_MISMATCH_DESPITE_RIGHT_OUTPUT"
    elif "r_actions" in info:
        category = "WRONG_ACTION"
    else:
        category = "UNKNOWN"

    missing = [k for k, found in info.get("outputs", {}).items() if not found]

    return TauFailureRecord(
        task_id=result["task_id"], trial=result.get("trial", 0), category=category,
        reward=result["reward"], last_tool_error=unrecovered[-1].content if unrecovered else None,
        gt_mutating_tools=gt_actions, agent_mutating_tools=agent_actions, missing_outputs=missing,
        last_tool_error_tool=unrecovered[-1].tool if unrecovered else None,
        recovered_tool_errors=[r.content for r in recovered])


def summarize(results: list[dict[str, Any]]) -> tuple[list[TauFailureRecord], Counter]:
    records = [r for r in (classify_taubench(x) for x in results) if r is not None]
    return records, Counter(r.category for r in records)


def action_shape(record: TauFailureRecord) -> str:
    """Within WRONG_ACTION / ACTION_MISMATCH_DESPITE_RIGHT_OUTPUT, the
    cheapest real signal for *why*: compare tool-name sequences only (not
    args -- those aren't visible without replaying, which is exactly the
    scope `taubench_adapter.py` already draws a line around for the user
    simulator; this stays within what's visible in the results file)."""
    gt, agent = record.gt_mutating_tools, record.agent_mutating_tools
    if not agent and gt:
        return "never attempted a mutating action"
    if agent and not gt:
        return "took a mutating action the task didn't call for"
    if set(agent) == set(gt) and agent != gt and len(agent) == len(gt):
        return "right tools, different order"
    if set(gt) - set(agent):
        return "missing at least one required mutating action"
    if set(agent) - set(gt):
        return "took an extra, unrequired mutating action"
    if agent == gt:
        return "same tool-name sequence -- mismatch is in the arguments, not which tools were called"
    return "unclear from tool names alone"


def load_results(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
