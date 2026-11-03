"""Statically parse tau-bench retail data (tool schemas, tasks) via `ast` — no code from the repo is executed.

Usage: python -I parse_taubench.py <tau-bench repo root> <out dir>
Outputs: tools.json (OpenAI-style schemas), wiki.md copy, tasks_{train,test,dev}.json
"""
import ast
import json
import sys
from pathlib import Path


def _get_info_dict(path: Path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_info":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return):
                    return ast.literal_eval(sub.value)
    return None


def _tasks(path: Path):
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Task":
            kw = {k.arg: k.value for k in node.keywords}
            actions = []
            for a in kw["actions"].elts:
                akw = {k.arg: k.value for k in a.keywords}
                actions.append({"name": ast.literal_eval(akw["name"]),
                                "kwargs": ast.literal_eval(akw["kwargs"])})
            out.append({
                "user_id": ast.literal_eval(kw["user_id"]),
                "instruction": ast.literal_eval(kw["instruction"]),
                "actions": actions,
                "outputs": ast.literal_eval(kw["outputs"]) if "outputs" in kw else [],
            })
    return out


def main(repo: str, out: str):
    root = Path(repo) / "tau_bench" / "envs" / "retail"
    outd = Path(out)
    outd.mkdir(parents=True, exist_ok=True)
    tools = []
    for p in sorted((root / "tools").glob("*.py")):
        if p.name == "__init__.py":
            continue
        info = _get_info_dict(p)
        if info:
            tools.append(info)
    (outd / "tools.json").write_text(json.dumps(tools, indent=1))
    (outd / "wiki.md").write_text((root / "wiki.md").read_text())
    for split in ["train", "test", "dev"]:
        tasks = _tasks(root / f"tasks_{split}.py")
        for i, t in enumerate(tasks):
            t["task_id"] = i
        (outd / f"tasks_{split}.json").write_text(json.dumps(tasks, indent=1))
        print(split, len(tasks))
    print("tools", len(tools))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
