"""Sample a hand-audit sheet from weakly labelled trajectory turns (stratified by label), with context.

Usage: python scripts/make_audit.py data/processed/tau_traj_train.jsonl data/processed/tau_traj_train.replay.jsonl out/audit_train.csv
Fill columns `verdict` (Y = label right, N = wrong) and `correct_labels` (only when N; '+'-separated,
from: cancel modify_items modify_order_address modify_payment modify_user_address return exchange
info_query transfer out_of_scope continue). Then: python scripts/apply_audit.py ...
"""
import csv
import json
import random
import sys
from collections import defaultdict

rows_p, replay_p, out_p = sys.argv[1:4]
N_INTENT, N_CONT, SEED = 150, 50, 11
rows = [json.loads(l) for l in open(rows_p)]
replays = [json.loads(l) for l in open(replay_p)]
# rows are emitted replay by replay, turn by turn -> rebuild the mapping
it = iter(rows)
ctx = {}
for k, rp in enumerate(replays):
    for j in range(len(rp["user_turns"])):
        r = next(it)
        nxt_write = next((w["tool"] for w in rp["writes"] if w["after_user_turn"] >= j), "")
        ctx[id(r)] = (k, j, rp["user_turns"][j - 1] if j > 0 else "", rp["user_turns"][j + 1] if j + 1 < len(rp["user_turns"]) else "", nxt_write)
rnd = random.Random(SEED)
by = defaultdict(list)
for r in rows:
    by["+".join(r["labels"])].append(r)
intent_keys = [k for k in by if k != "continue"]
total = sum(len(by[k]) for k in intent_keys)
sample = []
for k in intent_keys:  # proportional, at least 3 per label when available
    n = max(min(3, len(by[k])), round(N_INTENT * len(by[k]) / total))
    sample += rnd.sample(by[k], min(n, len(by[k])))
sample = sample[:N_INTENT] + rnd.sample(by["continue"], N_CONT)
rnd.shuffle(sample)
with open(out_p, "w", newline="", encoding="utf-8-sig") as f:  # BOM so Excel/Numbers open UTF-8 cleanly
    w = csv.writer(f)
    w.writerow(["id", "auto_label", "text", "prev_user_turn", "next_user_turn", "next_write_tool", "verdict", "correct_labels", "note"])
    for r in sample:
        k, j, prev, nxt, tool = ctx[id(r)]
        w.writerow([f"{k}:{j}", "+".join(r["labels"]), r["text"], prev, nxt, tool, "", "", ""])
print("wrote", out_p, len(sample), "rows")
