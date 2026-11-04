"""Turn real tau-bench result files into (1) a replay fixture and (2) per-user-turn weak labels.
Labels are always derived from the replay record, so you can re-label without the raw results:

  python extract_trajectories.py --split test out.jsonl results1.json [results2.json ...]
  python extract_trajectories.py --split test --from-replay out.jsonl out.replay.jsonl   # re-label only

Replay record: {"task_id", "reward", "user_turns": [...], "writes": [{"after_user_turn", "tool"}]}

Weak-labelling rule (audit a sample by hand before trusting it):
  A "segment" = user turns since the previous write call. For each write in a segment, every
  non-identity turn of that segment that *mentions* the write's intent (keyword match) gets that
  intent. If no turn mentions it, the first non-identity, non-confirmation turn gets it.
  Turn 0 additionally gets the intent of any later write it mentions (users often list everything up front).
  Every other turn (identity details, confirmations, vague openers, chit-chat) gets ["continue"]:
  no new intent -> the router keeps the current route (neither adds tools nor falls back).
  Trajectories without any write: first turn = ["info_query"], the rest = ["continue"].
Only reward==1 trajectories are used by default (their writes are then known-correct).
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from intent_funnel.taxonomy import ACTION_TO_INTENT  # noqa: E402
from intent_funnel.textclean import clean  # noqa: E402

CONFIRM = re.compile(r"^\W*(yes|yeah|yep|sure|ok(ay)?|confirm(ed)?|correct|go ahead|please (do|proceed)|"
                     r"that'?s (right|correct|fine)|proceed|sounds good|do it)\b", re.I)
STOP = re.compile(r"###STOP###")
IDENTITY = re.compile(r"(@\w+\.\w+|\bzip\b|\b\d{5}\b|\bmy name is\b|^\W*(it'?s|i'?m) [A-Z][a-z]+ [A-Z][a-z]+\W*$|"
                      r"^\W*#?W\d{7}\W*$)", re.I)
MENTION = {
    "cancel": r"\bcancel",
    "exchange": r"\bexchang|\bswap|\breplace|\bswitch\b|\bdifferent (one|model|version|color|size)",
    "return": r"\breturn|\brefund|\bsend (it|them|.*) back",
    "modify_items": r"\b(change|modify|modif|switch|swap|update|different|instead)\b",
    "modify_order_address": r"\baddress|\bmoved\b|\bship(ping)? to\b",
    "modify_user_address": r"\baddress|\bmoved\b",
    "modify_payment": r"\bpay|\bcard\b|\bpaypal|\bgift card",
    "transfer": r"\b(human|agent|representative|real person|someone|supervisor)\b",
}
MENTION = {k: re.compile(v, re.I) for k, v in MENTION.items()}
# turn-0 lookback is credited for writes far away, so it needs stronger evidence: a generic
# "update/change" must refer to items, not to an address or payment ("update my address for order X")
STRICT0 = dict(MENTION)
STRICT0["modify_items"] = re.compile(r"\b(change|modify|switch|swap|update)\b[^.?!]*\b(item|items|size|color|colou?r|"
                                     r"option|version|model|product|one)\b|\bmodify (the|my) (?!address|payment)", re.I)


def tool_calls(msg):
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", tc)
        yield fn.get("name")


def to_replay(traj, task_id, reward):
    """user_turns[j] is the j-th user message; agent_before[j] is the agent's last natural-language message
    before it (what the agent just said/proposed, available to a router at that moment)."""
    user_turns, agent_before, writes = [], [], []
    last_agent = ""
    for m in traj:
        role = m.get("role")
        if role == "user":
            text = (m.get("content") or "").strip()
            if not STOP.search(text):
                user_turns.append(text)
                agent_before.append(last_agent)
        elif role == "assistant":
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                last_agent = content.strip()
            for name in tool_calls(m):
                if name in ACTION_TO_INTENT:
                    writes.append({"after_user_turn": len(user_turns) - 1, "tool": name})
    return {"task_id": task_id, "reward": reward, "user_turns": user_turns, "agent_before": agent_before,
            "writes": writes}


def is_confirm(t):
    return bool(CONFIRM.match(t)) and len(t.split()) <= 8


def is_identity(t):
    return bool(IDENTITY.search(t)) and len(t.split()) <= 15


def label_replay(rp, split):
    cleaned = [clean(t) for t in rp["user_turns"]]
    turns = [c[0] for c in cleaned]
    labels = [[] for _ in turns]
    if not rp["writes"]:
        if turns:
            labels[0] = ["info_query"]
    ends_seen = set()
    for w in rp["writes"]:
        end = w["after_user_turn"]
        if end < 0:
            continue
        intent = ACTION_TO_INTENT[w["tool"]]
        # segment start = after the last write that happened at an EARLIER user turn
        # (several writes after the same turn, e.g. "proceed with both", share one segment)
        start = max([e for e in ends_seen if e < end], default=-1) + 1
        seg = list(range(start, end + 1)) or [end]
        cand = [j for j in seg if not is_identity(turns[j])]
        hits = [j for j in cand if MENTION.get(intent) and MENTION[intent].search(turns[j])]
        if not hits:
            first = next((j for j in cand if not is_confirm(turns[j])), cand[0] if cand else seg[0])
            hits = [first]
        # opening turns often list every request up front: credit turn 0 for later writes it mentions
        if seg[0] > 0 and turns and STRICT0.get(intent) and STRICT0[intent].search(turns[0]) \
                and not is_identity(turns[0]):
            hits.append(0)
        for j in hits:
            if intent not in labels[j]:
                labels[j].append(intent)
        ends_seen.add(end)
    rows = []
    for j, t in enumerate(turns):
        rows.append({"text": t, "labels": labels[j] or ["continue"], "source": "tau_traj", "split": split,
                     "meta": {"task_id": rp["task_id"], "turn": j, "reward": rp["reward"],
                              "auto": "mention" if labels[j] else "continue",
                              **({"dirty": cleaned[j][1]} if cleaned[j][1] else {})}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "valid", "test"])
    ap.add_argument("--include-failed", action="store_true")
    ap.add_argument("--from-replay", action="store_true", help="inputs are .replay.jsonl files")
    ap.add_argument("out")
    ap.add_argument("inputs", nargs="+")
    a = ap.parse_args()
    replays, seen = [], set()
    for p in a.inputs:
        if a.from_replay:
            replays += [json.loads(l) for l in open(p) if l.strip()]
            continue
        for r in json.loads(Path(p).read_text()):
            key = (r.get("task_id"), r.get("trial", 0), p)
            if key in seen or (r.get("reward", 0) < 1 and not a.include_failed):
                continue
            seen.add(key)
            replays.append(to_replay(r.get("traj", []), r.get("task_id"), r.get("reward")))
    rows = [row for rp in replays for row in label_replay(rp, a.split)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x) + "\n" for x in rows))
    if not a.from_replay:
        out.with_suffix(".replay.jsonl").write_text("".join(json.dumps(x) + "\n" for x in replays))
    n_cont = sum(1 for x in rows if x["labels"] == ["continue"])
    print(f"{len(replays)} trajectories -> {len(rows)} user-turn rows ({n_cont} 'continue')")


if __name__ == "__main__":
    main()
