"""CPU baselines: keyword rules and TF-IDF + one-vs-rest LR, plus training-set ablations.

  A_full_whole  : external + tau train, whole-text inference
  A_full_sent   : same model, sentence-split + union at inference
  B_no_tau_sent : external only                    -> value of in-domain data
  C_no_abcd_sent: no ABCD                          -> generalisation to human-written ABCD

Usage (repo root): python scripts/run_baselines.py [--extra data/processed/tau_traj_train.jsonl ...]
Writes out/baselines.json and out/baselines.md
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from intent_funnel.budget import PromptBudget  # noqa: E402
from intent_funnel.data import eval_sets, load, load_replays, make_train  # noqa: E402
from intent_funnel.evaluate import evaluate, headline, latency, to_markdown  # noqa: E402
import intent_funnel.router as router_mod  # noqa: E402
from intent_funnel.router import RulesAsProba, SentenceUnion, TfidfLRRouter  # noqa: E402

NO_TRAJ = lambda r: r["source"] != "tau_traj"  # noqa: E731
CONFIGS = [
    # (name, training filter, sentence-union at inference, route-group expansion)
    ("A_no_traj_sent", NO_TRAJ, True, True),                         # external + tau templates only
    ("D_with_traj_whole_nogroups", lambda r: True, False, False),    # ablation: no sibling groups
    ("D_with_traj_whole", lambda r: True, False, True),              # + real trajectory turns
    ("D_with_traj_sent", lambda r: True, True, True),
    ("E_traj_only_whole", lambda r: r["source"] == "tau_traj", False, True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="extra jsonl files (e.g. trajectory turns)")
    a = ap.parse_args()
    budget = PromptBudget(ROOT / "data/taubench")
    rows = load(extra=a.extra)
    test = eval_sets(rows, "test")
    replays = load_replays()
    out = {"full_prompt_tokens": budget.full, "n_test_replays": len(replays), "models": {}}
    md = [f"# Baselines (CPU)\n\nFull prompt = {budget.full} DeepSeek tokens (wiki + 16 tool schemas). "
          "Per-set: safe = needed write tools ⊆ route or fallback; risk = misroutes among routed; "
          "token = routed/full prompt.\n"]
    router_mod.GROUP_EXPANSION = True
    rules = RulesAsProba()
    res = evaluate(rules, test, budget, replays=replays)
    out["models"]["rules"] = res
    md.append(to_markdown("Keyword rules (no hit -> continue)", res))
    for name, keep, sent, groups in CONFIGS:
        router_mod.GROUP_EXPANSION = groups
        tr = make_train(rows, keep)
        t0 = time.time()
        model = TfidfLRRouter().fit([r["text"] for r in tr], [r["labels"] for r in tr])
        fit_s = time.time() - t0
        if sent:
            model = SentenceUnion(model)
        res = evaluate(model, test, budget, replays=replays)
        res.update({"n_train": len(tr), "fit_seconds": round(fit_s, 1)})
        if name == "D_with_traj_sent":
            res["latency"] = latency(model, [r["text"] for r in test.get("tau_traj", test["abcd"])])
        out["models"][name] = res
        md.append(to_markdown(f"TF-IDF + LR — {name} (n_train={len(tr)})", res))
        print(name, "|", headline(res))
    (ROOT / "out").mkdir(exist_ok=True)
    (ROOT / "out/baselines.json").write_text(json.dumps(out, indent=1))
    (ROOT / "out/baselines.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
