"""Re-evaluate a router from saved per-turn probabilities (finetune.py writes seed*_probs.npz) — no GPU.

Usage: python scripts/eval_saved_probs.py out/qwen/seed0_probs.npz [--no-groups]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import intent_funnel.router as router_mod  # noqa: E402
from intent_funnel.budget import PromptBudget  # noqa: E402
from intent_funnel.data import load_replays  # noqa: E402
from intent_funnel.evaluate import replay_crossfit  # noqa: E402


class Saved:
    def __init__(self, P, classes):
        self.P, self.classes = P, list(classes)

    def predict_proba(self, texts):
        assert len(texts) == len(self.P), (len(texts), len(self.P))
        return self.P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--no-groups", dest="groups", action="store_false")
    a = ap.parse_args()
    z = np.load(a.npz)
    replays = load_replays()
    assert list(z["replay_lens"]) == [len(r["user_turns"]) for r in replays], "replay file differs from the one used on Kaggle"
    router_mod.GROUP_EXPANSION = a.groups
    pooled, picks, curve = replay_crossfit(Saved(z["replay"], z["classes"]), replays, PromptBudget(ROOT / "data/taubench"))
    print(json.dumps({"groups": a.groups, "picks": picks, **pooled}, indent=1))
    pts = sorted((v["step_token"], v["write_coverage"], k) for k, v in curve.items())
    front, best = [], -1
    for t, w, k in pts:
        if w > best:
            front.append((round(t, 3), round(w, 4), k))
            best = w
    print("pareto (token, write_cov, t_include|gate):", front)


if __name__ == "__main__":
    main()
