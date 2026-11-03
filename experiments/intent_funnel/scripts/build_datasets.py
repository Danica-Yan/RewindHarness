"""Build unified jsonl datasets: {"text", "labels": [..], "source", "split", "meta"}.

Usage: python -I build_datasets.py <ext dir with cloned repos> <out dir>
"""
import csv
import gzip
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from intent_funnel.label_maps import BITEXT_INTENT, abcd_label  # noqa: E402

random.seed(13)


def opening_request(turns, min_words=6, max_turns=2):
    """ABCD: first customer turns joined until the request is long enough (first turn is often just 'Hi')."""
    cust = [t for sp, t in turns if sp == "customer"]
    out = []
    for t in cust[:max_turns]:
        out.append(t.strip())
        if len(" ".join(out).split()) >= min_words:
            break
    return " ".join(out)


def abcd(ext):
    d = json.load(gzip.open(ext / "abcd/data/abcd_v1.1.json.gz"))
    rows = []
    for split in ["train", "dev", "test"]:
        for c in d[split]:
            s = c["scenario"]
            lab = abcd_label(s["flow"], s["subflow"])
            if lab is None:
                continue
            rows.append({"text": opening_request(c["original"]), "labels": [lab], "source": "abcd",
                         "split": {"dev": "valid"}.get(split, split),
                         "meta": {"flow": s["flow"], "subflow": s["subflow"], "convo_id": c["convo_id"]}})
    return rows


def bitext(ext):
    p = ext / "customer-support-llm-chatbot-training-dataset/data/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    by = defaultdict(list)
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            lab = BITEXT_INTENT.get(r["intent"])
            if lab is None:
                continue
            by[r["intent"]].append({"text": r["instruction"], "labels": [lab], "source": "bitext",
                                    "meta": {"intent": r["intent"], "flags": r["flags"]}})
    rows = []
    for intent, rs in by.items():  # stratified 70/10/20 per original intent
        random.shuffle(rs)
        n = len(rs)
        for i, r in enumerate(rs):
            r["split"] = "train" if i < 0.7 * n else ("valid" if i < 0.8 * n else "test")
            rows.append(r)
    return rows


def clinc_oos(ext):
    d = json.load(open(ext / "oos-eval/data/data_full.json"))
    rows = []
    for k, split in [("oos_train", "train"), ("oos_val", "valid"), ("oos_test", "test")]:
        for text, _ in d[k]:
            rows.append({"text": text, "labels": ["out_of_scope"], "source": "clinc_oos", "split": split,
                         "meta": {}})
    return rows


def banking_oos(ext):
    base = ext / "Few-Shot-Intent-Detection/Datasets/BANKING77-OOS"
    rows = []
    for kind, splits in [("id-oos", ["train", "valid", "test"]), ("ood-oos", ["valid", "test"])]:
        for split in splits:
            for line in (base / kind / split / "seq.in").read_text().splitlines():
                if line.strip():
                    rows.append({"text": line.strip(), "labels": ["out_of_scope"],
                                 "source": f"banking77_{kind}", "split": split, "meta": {}})
    return rows


def main(ext, out):
    ext, out = Path(ext), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    all_rows = abcd(ext) + bitext(ext) + clinc_oos(ext) + banking_oos(ext)
    with open(out / "external.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    stats = Counter((r["source"], r["split"], r["labels"][0]) for r in all_rows)
    table = defaultdict(Counter)
    for (src, split, lab), c in stats.items():
        table[f"{src}/{split}"][lab] += c
    for k in sorted(table):
        print(k, dict(table[k]))
    print("total", len(all_rows))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
