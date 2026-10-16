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
    "TOOL_RUNTIME_ERROR",                   # a tool call in the trajectory came back "Error: ..."
    "WRONG_OUTPUT",                         # required response text missing (r_outputs < 1.0)
    "ACTION_MISMATCH_DESPITE_RIGHT_OUTPUT",  # said the right thing; final state still wrong
    "WRONG_ACTION",                         # action-only task (no required output text); state wrong
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


def _tool_error_messages(traj: list[dict[str, Any]]) -> list[str]:
    return [str(m.get("content", "")) for m in traj
            if m.get("role") == "tool" and str(m.get("content", "")).startswith("Error:")]


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
    tool_errors = _tool_error_messages(traj)
    reward_info = (result.get("info") or {}).get("reward_info") or {}
    info = reward_info.get("info") or {}
    gt_actions = [a["name"] for a in reward_info.get("actions", []) if a["name"] in MUTATING_TOOLS]
    agent_actions = _agent_mutating_tool_calls(traj)

    if tool_errors:
        category = "TOOL_RUNTIME_ERROR"
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
        reward=result["reward"], last_tool_error=tool_errors[-1] if tool_errors else None,
        gt_mutating_tools=gt_actions, agent_mutating_tools=agent_actions, missing_outputs=missing)


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
