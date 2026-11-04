"""Find tau-bench result jsons on your machine and tell you which split each one is.

Usage (from intent_funnel/):  python scripts/find_results.py ~/path/to/RewindHarness [more dirs...]

A file counts as a tau-bench result if it is a JSON list of objects with "task_id", "reward" and "traj".
The split is decided by matching info.task.instruction against data/taubench/tasks_{train,test}.json
(the file name doesn't record the split); without that field it falls back to the task_id range.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "node_modules", ".venv", "venv", "__pycache__", "site-packages", ".cache"}


def instr_index():
    idx = {}
    for split in ["train", "test", "dev"]:
        p = ROOT / f"data/taubench/tasks_{split}.json"
        if p.exists():
            for t in json.loads(p.read_text()):
                idx.setdefault(t["instruction"].strip(), set()).add(split)
    return idx


def walk(d):
    for p in Path(d).expanduser().rglob("*.json"):
        if not any(part in SKIP for part in p.parts) and p.stat().st_size > 2000:
            yield p


def inspect(p, idx):
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    if isinstance(data, dict):  # some runners wrap results: {"results": [...]}
        data = next((v for v in data.values() if isinstance(v, list)), None)
    if not (isinstance(data, list) and data and isinstance(data[0], dict)
            and {"task_id", "reward", "traj"} <= set(data[0])):
        return None
    splits, n_traj = set(), 0
    for r in data:
        n_traj += bool(r.get("traj"))
        task = (r.get("info") or {}).get("task") or {}
        ins = (task.get("instruction") or "").strip()
        if ins in idx:
            splits |= idx[ins]
    if not splits:
        ids = [r["task_id"] for r in data]
        splits = {"train (guess: task_id > 114)"} if max(ids) > 114 else {"unknown (task_id <= 114: test or train)"}
    trials = sorted({r.get("trial", 0) for r in data})
    passed = sum(1 for r in data if r.get("reward", 0) >= 1)
    return {"split": "/".join(sorted(splits)), "rows": len(data), "with_traj": n_traj,
            "trials": trials, "pass": f"{passed}/{len(data)}"}


def main():
    dirs = sys.argv[1:] or ["."]
    idx = instr_index()
    found = []
    for d in dirs:
        for p in walk(d):
            info = inspect(p, idx)
            if info:
                found.append((p, info))
    if not found:
        print("No tau-bench result files found. Try a wider directory, e.g. your home folder.")
        return
    for p, i in sorted(found, key=lambda x: (x[1]["split"], str(x[0]))):
        print(f"[{i['split']:>5}] rows={i['rows']:<4} trials={i['trials']} pass={i['pass']:<8} traj={i['with_traj']:<4} {p}")


if __name__ == "__main__":
    main()
