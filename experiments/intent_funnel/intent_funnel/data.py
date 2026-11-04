"""Shared data loading / training-set construction for every router (baselines, fine-tuned, DeepSeek)."""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = ["data/processed/external.jsonl", "data/processed/tau_instr.jsonl",
                 "data/processed/tau_traj_train.jsonl", "data/processed/tau_traj_test.jsonl"]
TEST_REPLAY = "data/processed/tau_traj_test.replay.jsonl"

OOS_CAP = 3000            # external OOS rows kept for training (~15k raw would swamp every long input)
TAU_NATURAL_UPSAMPLE = 5  # applies to natural in-domain rows (trajectory turns / paraphrases)


def load(files=DEFAULT_FILES, extra=()):
    rows = []
    for p in list(files) + list(extra):
        p = Path(p)
        p = p if p.is_absolute() else ROOT / p
        if not p.exists():  # trajectory files are optional
            continue
        rows += [json.loads(l) for l in open(p) if l.strip()]
    return [r for r in rows if r["source"] != "banking77_ood-oos"]  # identical to clinc oos_test


EVAL_SETS = {
    "tau_natural": lambda r: r["source"] == "tau_instr" and r["meta"].get("style") == "natural",
    "tau_terse": lambda r: r["source"] == "tau_instr" and r["meta"].get("style") == "terse",
    "tau_traj": lambda r: r["source"] == "tau_traj",
    "abcd": lambda r: r["source"] == "abcd",
    "bitext": lambda r: r["source"] == "bitext",
    "clinc_oos": lambda r: r["source"] == "clinc_oos",
    "banking_id_oos": lambda r: r["source"] == "banking77_id-oos",
}


def load_replays(path=TEST_REPLAY):
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def eval_sets(rows, split="test"):
    out = {k: [r for r in rows if r["split"] == split and f(r)] for k, f in EVAL_SETS.items()}
    return {k: v for k, v in out.items() if v}


def is_natural_in_domain(r):
    return (r["source"] == "tau_traj") or (r["source"] == "tau_instr" and r["meta"].get("style") == "natural") \
        or r["source"] == "tau_paraphrase"


def make_train(rows, keep=lambda r: True, seed=0):
    rnd = random.Random(seed)
    tr = [r for r in rows if r["split"] == "train" and keep(r) and r["labels"] and not r["meta"].get("dirty")]
    oos = [r for r in tr if r["labels"] == ["out_of_scope"]]
    rest = [r for r in tr if r["labels"] != ["out_of_scope"]]
    rnd.shuffle(oos)
    up = [r for r in rest if is_natural_in_domain(r)]
    out = rest + oos[:OOS_CAP] + up * (TAU_NATURAL_UPSAMPLE - 1)
    rnd.shuffle(out)
    return out
