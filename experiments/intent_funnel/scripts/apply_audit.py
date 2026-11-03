"""Apply a filled audit sheet: ground truth = auto_label if verdict Y, else correct_labels.

Reports accuracy of (a) the weak labels as shown in the sheet (the rule version that produced it) and
(b) the CURRENT weak labels in the rows file (after rule fixes), both against the audited truth,
then writes the truth into the rows file (meta.audited = true).

Usage: python scripts/apply_audit.py out/audit_train.csv data/processed/tau_traj_train.jsonl data/processed/tau_traj_train.replay.jsonl
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, Path(__file__).resolve().parents[1].as_posix())
from intent_funnel.taxonomy import INTENTS  # noqa: E402

audit_p, rows_p, replay_p = sys.argv[1:4]
rows = [json.loads(l) for l in open(rows_p)]
replays = [json.loads(l) for l in open(replay_p)]
pos, i = {}, 0
for k, rp in enumerate(replays):
    for j in range(len(rp["user_turns"])):
        pos[f"{k}:{j}"] = i
        i += 1
norm = lambda s: frozenset(x.strip() for x in s.split("+") if x.strip())  # noqa: E731
n = sheet_ok = cur_ok = 0
per = Counter(), Counter()
cur_errors = []
for a in csv.DictReader(open(audit_p, encoding="utf-8-sig")):
    v = (a["verdict"] or "").strip().upper()
    if v not in ("Y", "N"):
        print("unfilled:", a["id"])
        continue
    truth = norm(a["auto_label"]) if v == "Y" else norm(a["correct_labels"])
    if not truth or not truth <= set(INTENTS):
        print("skip", a["id"], "invalid labels:", a["correct_labels"])
        continue
    r = rows[pos[a["id"]]]
    n += 1
    sheet_ok += norm(a["auto_label"]) == truth
    cur_ok += frozenset(r["labels"]) == truth
    per[0][a["auto_label"]] += 1
    per[1][a["auto_label"]] += norm(a["auto_label"]) == truth
    if frozenset(r["labels"]) != truth:
        cur_errors.append((a["id"], r["labels"], sorted(truth), r["text"][:90]))
    r["labels"] = sorted(truth)
    r["meta"]["audited"] = True
print(f"audited rows: {n}")
print(f"weak labels in sheet (rule at audit time) correct: {sheet_ok}/{n} = {sheet_ok / max(n, 1):.1%}")
print(f"current weak labels (after rule fixes)    correct: {cur_ok}/{n} = {cur_ok / max(n, 1):.1%}")
for k in sorted(per[0]):
    print(f"  {k:<36} {per[1][k]}/{per[0][k]}")
print("remaining disagreements (current rule vs truth):")
for e in cur_errors:
    print("  ", e)
Path(rows_p).write_text("".join(json.dumps(r) + "\n" for r in rows))
