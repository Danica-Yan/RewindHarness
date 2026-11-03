"""Held-out validation with thresholds frozen in advance.

  freeze : pick each router's (t_include, gate) on the DeepSeek test replay
           (cheapest setting with write_coverage >= 99%, else the highest coverage) and write
           out/frozen_thresholds.json. Run this BEFORE looking at any held-out numbers.
  eval   : apply the frozen thresholds unchanged to held-out replays (gpt-4o / Sonnet user simulators).

Routers: rules, TF-IDF (all data / trajectories only), Qwen seeds from saved probabilities,
Qwen 3-seed average, and the hybrid max(p_qwen, p_tfidf_traj). Route groups on for all.
Qwen probabilities for a held-out set come from Kaggle:
  python scripts/finetune.py --eval-adapter <adapter> --replay data/heldout/<set>.replay.jsonl --replay-only --out ...
and are expected at out/qwen/seed{s}_probs_<set>.npz.

Usage: python scripts/validate_heldout.py freeze
       python scripts/validate_heldout.py eval gpt-4o_retail sonnet-35-new_retail
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import intent_funnel.router as rm  # noqa: E402
from intent_funnel.budget import PromptBudget  # noqa: E402
from intent_funnel.data import load, load_replays, make_train  # noqa: E402
from intent_funnel.evaluate import GATES, INCLUDES, replay_metrics, replay_probs  # noqa: E402
from intent_funnel.router import RulesAsProba, TfidfLRRouter  # noqa: E402

rm.GROUP_EXPANSION = True
FROZEN = ROOT / "out/frozen_thresholds.json"
TARGET = 0.99
DEV_REPLAY = "data/processed/tau_traj_test.replay.jsonl"


def split(P, replays):
    out, k = [], 0
    for rp in replays:
        n = len(rp["user_turns"])
        out.append(P[k:k + n])
        k += n
    return out


def cpu_models():
    rows = load()
    models = {"rules": RulesAsProba()}
    for name, keep in [("tfidf_all", lambda r: True), ("tfidf_traj", lambda r: r["source"] == "tau_traj")]:
        tr = make_train(rows, keep, seed=0)
        models[name] = TfidfLRRouter().fit([r["text"] for r in tr], [r["labels"] for r in tr])
    return models


def all_probs(replays, tag, models):
    """-> {router: list of per-trajectory prob matrices}, classes. tag picks the saved Qwen npz files."""
    probs, classes = {}, None
    for name, m in models.items():
        P = np.vstack(replay_probs(m, replays))
        classes = classes or [c for c in m.classes]
        probs[name] = (P, list(m.classes))
    qwen = []
    for s in range(3):
        f = ROOT / f"out/qwen/seed{s}_probs_{tag}.npz"
        if f.exists():
            z = np.load(f)
            assert list(z["replay_lens"]) == [len(r["user_turns"]) for r in replays], f"{f} does not match replays"
            probs[f"qwen_seed{s}"] = (z["replay"], list(z["classes"]))
            qwen.append(f"qwen_seed{s}")
    if qwen:
        cls = probs[qwen[0]][1]
        T = probs["tfidf_traj"][0][:, [probs["tfidf_traj"][1].index(c) for c in cls]]
        avg = np.mean([probs[q][0] for q in qwen], 0)
        if len(qwen) == 3:
            probs["qwen_avg3"] = (avg, cls)
            probs["hybrid_avg3"] = (np.maximum(avg, T), cls)
        for q in qwen:
            probs[f"hybrid_{q}"] = (np.maximum(probs[q][0], T), cls)
    return probs


def freeze():
    budget = PromptBudget(ROOT / "data/taubench")
    replays = load_replays(DEV_REPLAY)
    probs = all_probs(replays, "alldata", cpu_models())
    frozen = {}
    for name, (P, cls) in probs.items():
        Ps = split(P, replays)
        grid = {(ti, g): replay_metrics(replays, Ps, cls, budget, ti, g) for ti in INCLUDES for g in GATES}
        ok = [k for k, v in grid.items() if v["write_coverage"] >= TARGET]
        best = min(ok, key=lambda k: grid[k]["step_token"]) if ok else max(grid, key=lambda k: grid[k]["write_coverage"])
        frozen[name] = {"t_include": best[0], "gate": best[1], "dev": grid[best]}
        d = grid[best]
        print(f"{name:<20} t_include={best[0]:<5} gate={best[1]:<4} dev cov {d['write_coverage']:.1%} tokens {d['step_token']:.3f}")
    FROZEN.write_text(json.dumps({"target_write_coverage": TARGET, "dev": DEV_REPLAY, "routers": frozen}, indent=1))
    print("wrote", FROZEN)


def evaluate(tags):
    budget = PromptBudget(ROOT / "data/taubench")
    frozen = json.loads(FROZEN.read_text())["routers"]
    models = cpu_models()
    report = {}
    for tag in tags:
        replays = load_replays(f"data/heldout/{tag}.replay.jsonl")
        probs = all_probs(replays, tag, models)
        print(f"\n== {tag}: {len(replays)} trajectories, {sum(len(r['writes']) for r in replays)} writes")
        report[tag] = {}
        for name, f in frozen.items():
            if name not in probs:
                print(f"{name:<20} (no saved probabilities for this set yet)")
                continue
            P, cls = probs[name]
            m = replay_metrics(replays, split(P, replays), cls, budget, f["t_include"], f["gate"])
            report[tag][name] = m
            print(f"{name:<20} cov {m['write_coverage']:.1%} (dev {f['dev']['write_coverage']:.1%})  task {m['task_coverage']:.1%}  "
                  f"tokens {m['step_token']:.3f} (dev {f['dev']['step_token']:.3f})  fell back {m['traj_fell_back']:.1%}")
    (ROOT / "out/heldout_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    {"freeze": lambda: freeze(), "eval": lambda: evaluate(sys.argv[2:])}[sys.argv[1]]()
