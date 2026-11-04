"""tau-bench retail -> pseudo-utterance dataset (free proxy until real trajectories are labeled).

Each task instruction is a 2nd-person brief for the user simulator ("You are X in 19122. You want to ...").
We rewrite it to 1st person and drop persona/identity sentences, keeping the request sentences.
Labels = multi-label intents from gold actions. Train tasks -> train/valid, test tasks -> test.

This is a PROXY: instructions are longer and more explicit than real opening turns, so expect it to be
easier than real per-turn routing. Real per-turn data comes from scripts/extract_trajectories.py.

Usage: python -I build_tau.py data/taubench data/processed/tau_instr.jsonl
"""
import json
import random
import zlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from intent_funnel.taxonomy import intents_from_actions  # noqa: E402

PERSONA = re.compile(r"^(you are (?!(?:now|not|going|willing|happy|ok|fine))[A-Z]?[a-z]+ [A-Z]?[a-z]+.*?(in|zip)|your (name|email|zip|user id)|you (are|seem|tend|like|prefer) .*(person|polite|rude|detail|impatient|direct|confident|private|shy|logical|insecure|messy|curious|patient|cautious|organized|outgoing|busy|independent|optimistic|pessimistic|flexible|creative|relaxing|dependent|sad|happy|angry|reserved|lazy|rigid))", re.I)
SUBS = [
    (r"\byou're\b", "I'm"), (r"\byou've\b", "I've"), (r"\byou'd\b", "I'd"), (r"\byou'll\b", "I'll"),
    (r"\byou are\b", "I am"), (r"\byou were\b", "I was"), (r"\byou have\b", "I have"),
    (r"\byourself\b", "myself"), (r"\byours\b", "mine"), (r"\byour\b", "my"), (r"\byou\b", "I"),
]


def first_person(instr: str) -> str:
    sents = re.split(r"(?<=[.!?])\s+", instr.strip())
    keep = [s for s in sents if not PERSONA.search(s)]
    txt = " ".join(keep) or instr
    for pat, rep in SUBS:
        txt = re.sub(pat, rep, txt, flags=re.I)
    txt = re.sub(r"\bI (wants|wishes|needs|thinks|likes|prefers|wonders|says|asks)\b",
                 lambda m: "I " + m.group(1)[:-1], txt)
    return txt


TERSE = re.compile(r"(^|[.!?]\s+)(For #W\d+|Cancel order #W|Return #W|Exchange #W|Modify #W)", re.I)


def style_of(instr: str) -> str:
    """'terse' = annotator shorthand that names the action verbatim (trivially easy); else 'natural'."""
    return "terse" if TERSE.search(instr.strip()) or ("{'" in instr and "via" in instr) else "natural"


def main(data_dir, out_path):
    d = Path(data_dir)
    random.seed(7)
    rows = []
    for src_split in ["train", "test"]:
        tasks = json.loads((d / f"tasks_{src_split}.json").read_text())
        for t in tasks:
            split = src_split
            if src_split == "train":  # group split by user so valid users are unseen in train
                split = "valid" if (zlib.crc32(t["user_id"].encode()) % 100) < 25 else "train"
            rows.append({"text": first_person(t["instruction"]), "labels": intents_from_actions(t["actions"]),
                         "source": "tau_instr", "split": split,
                         "meta": {"task_split": src_split, "task_id": t["task_id"], "user_id": t["user_id"],
                                  "style": style_of(t["instruction"])}})
    Path(out_path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(len(rows), "rows")
    for r in rows[:3]:
        print(r["labels"], "|", r["text"][:220])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
