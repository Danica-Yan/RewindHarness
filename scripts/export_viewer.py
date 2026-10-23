#!/usr/bin/env python3
"""Bundle an Evolve run into one JSON file for viewer/index.html (no API calls).

Contents: every context version (rules, tool notes, the edits that made it),
every decision with its numbers and feedback, and -- for each gate decision --
the episodes of the tasks that flipped, under the incumbent AND under the
candidate, so the viewer can show them side by side. Episodes are compacted:
the system prompt is replaced by its size plus its learned-rules section,
long tool outputs are cut.

Usage:
  python scripts/export_viewer.py --run-dir runs/p1 --out runs/p1/viewer.json
  # then open viewer/index.html in a browser and load runs/p1/viewer.json
  # extra episodes (any rewind trace, e.g. a fork): --trace path/to/run.jsonl
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rewind.trace import load_trace

LEARNED = "## Additional policies (learned)"


def compact_episode(trace_path: str, max_chars: int = 1500) -> dict | None:
    p = Path(trace_path)
    if not p.exists():
        return None
    events = load_trace(p)
    start = events[0].data if events else {}
    items = []
    for e in events:
        d = e.data
        base = {"step": e.step, "inherited": e.inherited}
        if e.type == "message":
            m = d["message"]
            role = m.get("role")
            if role == "system":
                content = m.get("content") or ""
                learned = content.split(LEARNED, 1)[1].strip() if LEARNED in content else ""
                items.append({**base, "kind": "system", "chars": len(content), "learned": learned})
            elif role == "assistant":
                if m.get("content"):
                    items.append({**base, "kind": "agent", "text": m["content"][:max_chars]})
                for c in m.get("tool_calls") or []:
                    items.append({**base, "kind": "call", "name": c.get("name"),
                                  "args": json.dumps(c.get("args", {}), ensure_ascii=False)[:max_chars]})
            elif role == "tool":
                content = str(m.get("content", ""))
                items.append({**base, "kind": "result", "text": content[:max_chars],
                              "error": content.startswith(("Error", "ERROR")) or bool(m.get("is_error"))})
            elif role == "user":
                items.append({**base, "kind": "user", "text": (m.get("content") or "")[:max_chars]})
        elif e.type == "user_turn" and d.get("done"):
            items.append({**base, "kind": "stop"})
        elif e.type == "guard":
            text = d.get("guard")
            if text == "policy_check":
                text = f"policy {d.get('check')}: {d.get('action')} -- {d.get('reason')}"
            items.append({**base, "kind": "guard", "text": text})
        elif e.type == "llm_call" and d.get("error"):
            items.append({**base, "kind": "guard", "text": f"LLM error: {d['error'].get('class')}"})
        elif e.type == "run_end":
            items.append({**base, "kind": "end", "status": d.get("status"), "reason": d.get("reason")})
    return {"run_id": events[0].run_id if events else p.stem, "path": str(p),
            "parent_run_id": start.get("parent_run_id"), "fork_step": start.get("fork_step"),
            "items": items}


def rows_by_task(path: str | None) -> dict[int, dict]:
    if not path or not Path(path).exists():
        return {}
    out: dict[int, dict] = {}
    for r in json.loads(Path(path).read_text(encoding="utf-8")):
        if r["task_id"] not in out or r["trial"] < out[r["task_id"]]["trial"]:
            out[r["task_id"]] = r
    return out


def row_summary(r: dict | None) -> dict | None:
    if r is None:
        return None
    task = (r.get("info") or {}).get("task") or {}
    return {"task_id": r["task_id"], "reward": r["reward"], "goal": task.get("instruction"),
            "trace": ((r.get("info") or {}).get("rewind") or {}).get("trace")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="runs/p1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ledger", default=None, help="default: <run-dir>/ledger.sqlite")
    ap.add_argument("--max-pairs", type=int, default=8, help="flipped tasks per gate decision")
    ap.add_argument("--trace", action="append", default=[], help="extra trace file(s) to include")
    a = ap.parse_args()
    run = Path(a.run_dir)
    db = sqlite3.connect(str(a.ledger or run / "ledger.sqlite"))

    versions = []
    for v, parent, cj, ej, rows_path, vj in db.execute(
            "SELECT version, parent, ctx_json, edits_json, rows_path, verdict_json FROM versions ORDER BY version"):
        ctx = json.loads(cj)
        versions.append({"version": v, "parent": parent, "rules": ctx.get("rules", {}),
                         "tool_notes": ctx.get("tool_notes", {}), "checks": ctx.get("checks", {}),
                         "edits": json.loads(ej),
                         "rows_path": rows_path})
    rows_path = {v["version"]: v["rows_path"] for v in versions}

    decisions, episodes = [], {}
    for did, rnd, vf, edit_id, cluster, ej, stage, acc, reason, dj in db.execute(
            "SELECT id, round, version_from, edit_id, cluster, edit_json, stage, accepted, reason, data_json "
            "FROM decisions ORDER BY id"):
        data = json.loads(dj) if dj else {}
        d = {"id": did, "round": rnd, "version_from": vf, "stage": stage, "accepted": bool(acc),
             "reason": reason, "cluster": cluster, "edit": json.loads(ej) if ej else None}
        if stage == "screen":
            d["screen"] = {"tasks": data.get("tasks", []), "forks": data.get("forks")}
        if stage == "backtest":
            d["backtest"] = {k: data.get(k) for k in ("fires_on_passing_calls", "false_fire_rate", "catches",
                                                       "cluster_caught", "examples")}
        if stage == "stability":
            d["split"] = {"A": data.get("A"), "B": data.get("B")}
        if stage == "gate":
            keep = ("efficacy", "n_tasks", "delta", "ci", "incumbent_pass1", "candidate_pass1",
                    "regressions", "improvements", "confirmed_regressions", "confirmed_fixes",
                    "episodes", "confirm_episodes", "stopped_early", "regate_of", "regated")
            d["gate"] = {k: data.get(k) for k in keep if k in data}
            d["feedback"] = (data.get("feedback") or {}).get("text")
            d["concentration"] = (data.get("feedback") or {}).get("concentration")
            inc_rows = rows_by_task(rows_path.get(vf))
            cand_rows = rows_by_task(data.get("rows_path"))
            pairs = []
            flips = [("broke", t) for t in data.get("regressions", [])] + \
                    [("fixed", t) for t in data.get("improvements", [])]
            for kind, t in flips[: a.max_pairs]:
                before, after = row_summary(inc_rows.get(t)), row_summary(cand_rows.get(t))
                for side in (before, after):
                    if side and side["trace"] and side["trace"] not in episodes:
                        ep = compact_episode(side["trace"])
                        if ep:
                            episodes[side["trace"]] = ep
                pairs.append({"kind": kind, "task_id": t, "before": before, "after": after,
                              "confirmed": t in (data.get("confirmed_regressions") or [])
                              + (data.get("confirmed_fixes") or [])})
            d["pairs"] = pairs
        decisions.append(d)

    for t in a.trace:
        ep = compact_episode(t)
        if ep:
            episodes[t] = ep

    out = Path(a.out or run / "viewer.json")
    out.write_text(json.dumps({"run_dir": str(run), "versions": versions, "decisions": decisions,
                               "episodes": episodes}, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB): {len(versions)} versions, "
          f"{len(decisions)} decisions, {len(episodes)} episodes")


if __name__ == "__main__":
    main()
