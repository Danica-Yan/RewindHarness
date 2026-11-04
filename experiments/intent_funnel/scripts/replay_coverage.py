"""Offline end-to-end proxy: replay real trajectories through a router with cumulative routing and check,
at every write call, whether that tool was available in the route at that moment. No API calls.

  route_t = route_{t-1} ∪ classify(user_turn_t); a fallback at any turn = full prompt from then on.

Reports:
  write_coverage  : fraction of write calls whose tool was in the route (or route was full)  <- the key number
  task_coverage   : fraction of trajectories where every write was covered
  step_token_ratio: mean static-prompt tokens per user turn vs. always-full prompt

Usage: python replay_coverage.py --router rules|tfidf_sent [--threshold 0.05 0.9] <x.replay.jsonl>
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from intent_funnel.budget import PromptBudget  # noqa: E402
from intent_funnel.router import KeywordRouter, decide  # noqa: E402
from intent_funnel.taxonomy import tools_for  # noqa: E402


def replay(traj, route_fn, budget):
    route, full = set(), False
    per_turn_routes = []
    for text in traj["user_turns"]:
        r = route_fn(text)
        if r is None:
            full = True
        else:
            route |= set(r)
        per_turn_routes.append(None if full else set(route))
    covered = []
    for w in traj["writes"]:
        t = w["after_user_turn"]
        rt = per_turn_routes[t] if 0 <= t < len(per_turn_routes) else None
        covered.append(rt is None or w["tool"] in tools_for(sorted(rt)))
    costs = [budget.cost(None if rt is None else sorted(rt)) / budget.full for rt in per_turn_routes]
    return covered, costs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", default="rules", choices=["rules", "tfidf_sent"])
    ap.add_argument("--threshold", type=float, nargs=2, default=[0.05, 0.9], metavar=("T_INCLUDE", "GATE"))
    ap.add_argument("replay")
    a = ap.parse_args()
    budget = PromptBudget(ROOT / "data/taubench")
    if a.router == "rules":
        kr = KeywordRouter()
        route_fn = kr.route
    else:
        from intent_funnel.data import load, make_train
        from intent_funnel.router import SentenceUnion, TfidfLRRouter
        rows = load()
        tr = make_train(rows)
        m = SentenceUnion(TfidfLRRouter().fit([r["text"] for r in tr], [r["labels"] for r in tr]))
        route_fn = lambda t: decide(m.predict_proba([t]), m.classes, *a.threshold)[0]  # noqa: E731
    trajs = [json.loads(l) for l in open(a.replay) if l.strip()]
    allw, tasks_ok, allc = [], [], []
    for tj in trajs:
        cov, costs = replay(tj, route_fn, budget)
        allw += cov
        tasks_ok.append(all(cov))
        allc += costs
    out = {"router": a.router, "n_trajectories": len(trajs), "n_writes": len(allw),
           "write_coverage": sum(allw) / max(len(allw), 1), "task_coverage": sum(tasks_ok) / max(len(tasks_ok), 1),
           "step_token_ratio": sum(allc) / max(len(allc), 1)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
