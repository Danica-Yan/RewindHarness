"""What the loop did and what each stage was worth, from the ledger alone.

* funnel   -- every proposed edit and the stage that stopped it
* cost     -- episodes spent per stage (stability null forks, screening
              forks, gate episodes) and per accepted edit
* value    -- what each filter saved: episodes the next, more expensive
              stage would have spent on the candidates it stopped, and
              failures it kept away from the proposer
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .ledger import EditLedger


def _decisions(ledger: EditLedger) -> list[dict]:
    out = []
    for did, rnd, stage, acc, reason, dj, ej, cluster in ledger.db.execute(
            "SELECT id, round, stage, accepted, reason, data_json, edit_json, cluster FROM decisions ORDER BY id"):
        out.append({"id": did, "round": rnd, "stage": stage, "accepted": bool(acc), "reason": reason,
                    "data": json.loads(dj) if dj else {}, "edit": json.loads(ej) if ej else None,
                    "cluster": cluster})
    return out


def build(ledger: EditLedger, stability_path: str | Path) -> dict:
    from .stability import load_records
    ds = _decisions(ledger)
    edits = [d for d in ds if d["edit"] and d["stage"] not in ("ablation", "meta", "meta_screen")]
    overturned = ledger.overturned()

    # final stage reached by each distinct edit
    final: dict[str, dict] = {}
    for d in edits:
        final[d["edit"]["edit_id"]] = d
    funnel = Counter()
    for d in final.values():
        st = d["stage"]
        funnel[f"{st}: {'passed' if d['accepted'] else 'stopped'}"] += 1

    stab = load_records(stability_path)
    labels = Counter(r.label for r in stab.values())
    stab_forks = sum(r.n for r in stab.values())
    screen_forks = sum(d["data"].get("forks", 0) for d in ds if d["stage"] == "screen")
    meta_forks = sum(d["data"].get("forks", 0) for d in ds if d["stage"] == "meta_screen")
    gates = [d for d in ds if d["stage"] == "gate"]
    gate_eps = sum(d["data"].get("episodes", 0) for d in gates if d["data"].get("regate_of") is None) + \
        sum(d["data"].get("confirm_episodes", 0) for d in gates if d["data"].get("regate_of") is not None)
    abl_eps = sum(d["data"].get("n", 0) * 2 for d in ds if d["stage"] == "ablation")
    accepted = [d for d in gates if d["accepted"]]
    total = stab_forks + screen_forks + gate_eps + meta_forks + abl_eps

    full_gates = [d["data"]["episodes"] for d in gates if d["data"].get("episodes") and d["data"].get("n_tasks")]
    avg_gate = statistics.mean(full_gates) if full_gates else None
    screened_out = [d for d in final.values() if d["stage"] == "screen" and not d["accepted"]]
    invalid = [d for d in final.values() if d["stage"] in ("invalid", "backtest") and not d["accepted"]]
    screen_per_edit = statistics.mean([d["data"]["forks"] for d in ds if d["stage"] == "screen"
                                       and d["data"].get("forks")] or [0])
    value = []
    value.append({"stage": "stability filter (null forks)",
                  "effect": f"{labels.get('flaky', 0)} flaky + {labels.get('unclear', 0)} unclear of "
                            f"{sum(labels.values())} assessed failures never reached the proposer "
                            f"({labels.get('stable', 0)} stable did)",
                  "cost": f"{stab_forks} forks"})
    value.append({"stage": "offline validation + backtest",
                  "effect": f"{len(invalid)} candidates refused before any episode"
                            + (f" (would have cost ~{round(len(invalid) * screen_per_edit)} screening forks)"
                               if screen_per_edit else ""),
                  "cost": "0 episodes"})
    value.append({"stage": "screening",
                  "effect": f"{len(screened_out)} candidates stopped before the gate"
                            + (f" (would have cost ~{round(len(screened_out) * avg_gate)} gate episodes)"
                               if avg_gate else ""),
                  "cost": f"{screen_forks} forks"})
    value.append({"stage": "flip confirmation (gate audit)",
                  "effect": f"{len(overturned)} wrong rejection(s) overturned"
                            + (", turning into accepted edits" if overturned else ""),
                  "cost": f"{sum(d['data'].get('confirm_episodes', 0) for d in gates)} re-runs"})
    return {
        "funnel": dict(sorted(funnel.items())), "proposed": len(final), "accepted": len(accepted),
        "episodes": {"stability": stab_forks, "screening": screen_forks, "gate": gate_eps,
                     "meta_eval": meta_forks, "ablation": abl_eps, "total": total},
        "per_accepted_edit": round(total / len(accepted)) if accepted else None,
        "value": value, "rounds": ledger.round_no(),
    }


def to_markdown(r: dict) -> str:
    lines = ["## Evolve run report", "",
             f"{r['proposed']} edits proposed, {r['accepted']} accepted over {r['rounds']} round(s); "
             f"{r['episodes']['total']} episodes in total"
             + (f", **{r['per_accepted_edit']} per accepted edit**." if r["per_accepted_edit"] else "."), "",
             "| Where proposals stopped | Count |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in r["funnel"].items()]
    lines += ["", "| Episodes by stage | Count |", "| --- | --- |"]
    lines += [f"| {k} | {v} |" for k, v in r["episodes"].items()]
    lines += ["", "| Stage | What it did | Cost |", "| --- | --- | --- |"]
    lines += [f"| {v['stage']} | {v['effect']} | {v['cost']} |" for v in r["value"]]
    return "\n".join(lines) + "\n"


def report(ledger: EditLedger, cfg: Any) -> str:
    from .loop import _paths
    return to_markdown(build(ledger, _paths(cfg)["stability"]))
