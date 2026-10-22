"""SFT data collected as a by-product (training itself is deferred: no GPU).

Only train-split episodes that passed the grader under a context version in
the ledger (v0 = the original agent, or a gate-accepted version) are kept.
Test episodes are rejected by type, so later evaluation is never
contaminated. Records are the agent-side conversation in OpenAI chat format,
ready for a chat SFT trainer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class TrajectoryRecord(BaseModel):
    task_id: int
    trial: int
    split: Literal["train"]          # test/dev episodes cannot be represented
    ctx_version: int
    source: Literal["live", "fork"] = "live"
    had_tool_error: bool
    messages: list[dict]
    reward: float


def collect(rows: list[dict], split: str, ctx_version: int) -> list[TrajectoryRecord]:
    if split != "train":
        return []
    out = []
    for r in rows:
        if r["reward"] < 1.0 or (r["info"].get("rewind") or {}).get("ctx_version", 0) != ctx_version:
            continue
        traj = r.get("traj") or []
        out.append(TrajectoryRecord(
            task_id=r["task_id"], trial=r["trial"], split="train", ctx_version=ctx_version,
            had_tool_error=any(m.get("role") == "tool" and str(m.get("content", "")).startswith("Error:")
                               for m in traj),
            messages=traj, reward=r["reward"]))
    return out


def append_jsonl(path: str | Path, recs: list[TrajectoryRecord]) -> int:
    """Append records not already present (by task, trial, version); returns how many were added."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                seen.add((d["task_id"], d["trial"], d["ctx_version"]))
    new = [r for r in recs if (r.task_id, r.trial, r.ctx_version) not in seen]
    with p.open("a", encoding="utf-8") as f:
        for r in new:
            f.write(r.model_dump_json() + "\n")
    return len(new)
