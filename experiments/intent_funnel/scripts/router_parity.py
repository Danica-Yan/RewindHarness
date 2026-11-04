"""Parity check: drive the live TauRouter (the object rewind calls) through replays turn by turn, the way
AgentRunner does, and compare write coverage / static tokens with the offline evaluation.
Usage (from experiments/intent_funnel): python scripts/router_parity.py [replay.jsonl ...]"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from intent_funnel.budget import PromptBudget
from intent_funnel.data import load_replays
from intent_funnel.rewind_router import build_router

B = PromptBudget(Path(__file__).resolve().parents[1] / "data/taubench")
router = build_router("tfidf")
wiki = B.wiki


def tokens(u):
    if u is None or u.full:
        return B.full
    return B.ntok(u.prompt) + sum(B.tool_tok[t] for t in u.tools if t in B.tool_tok)


for path in sys.argv[1:] or ["data/processed/agentctx/tau_traj_test.replay.jsonl"]:
    R = load_replays(path)
    cov, cost, fell = [], [], 0
    for rp in R:
        units, full, per = [], False, []
        for j, ut in enumerate(rp["user_turns"]):
            u = router.update(units, full, ut, rp["agent_before"][j], wiki)
            units, full = u.units, u.full
            per.append(u)
        fell += full
        for w in rp["writes"]:
            u = per[w["after_user_turn"]]
            cov.append(u.full or w["tool"] in u.tools)
        cost += [tokens(u) / B.full for u in per]
    print(f"{Path(path).name}: write coverage {np.mean(cov):.2%}  static tokens {np.mean(cost):.3f}  "
          f"fell back {fell / len(R):.1%}  ({len(R)} trajectories)")
