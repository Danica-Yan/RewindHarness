"""End-to-end report: control vs routed tau-bench runs made with scripts/run_taubench_rewind.py.

Reads each results json (tau-bench run.py format with info.rewind.trace paths) and the JSONL traces:
  pass^1 (pooled over trials), per-episode LLM input / cached / output tokens, and routing stats
  (fallbacks, escalations, route violations, routed steps).
Billed cost needs prices, which change, so pass them in (USD per 1M tokens):
  --price-miss  input tokens not served from cache   --price-hit  cached input tokens   --price-out  output

Usage:
  python scripts/router_e2e_report.py --control ../../runs/p0/test_rewind.json \
      --treatment ../../runs/router/test_tfidf.json [--price-miss X --price-hit Y --price-out Z]
Only (task, trial) pairs present in BOTH files are compared when --paired is set.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def load(path):
    rows = json.loads(Path(path).read_text())
    base = Path(path).parent
    out = {}
    for r in rows:
        rw = r.get("info", {}).get("rewind", {})
        tp = rw.get("trace")
        if tp and not Path(tp).is_absolute() and not Path(tp).exists():
            tp = str(base / Path(tp).name)
        out[(r["task_id"], r["trial"])] = (r["reward"], tp)
    return out


def trace_stats(tp):
    s = Counter()
    if not tp or not Path(tp).exists():
        s["missing_trace"] = 1
        return s
    for line in open(tp):
        e = json.loads(line)
        if e.get("inherited"):
            continue
        if e["type"] == "llm_call" and "response" in e["data"]:
            u = e["data"]["response"].get("usage") or {}
            s["calls"] += 1
            s["input"] += u.get("input_tokens", 0)
            s["cached"] += u.get("cached_tokens", 0)
            s["output"] += u.get("output_tokens", 0)
        elif e["type"] == "route":
            reason = e["data"]["reason"]
            if reason == "user_turn" and e["data"]["full"] and "fallback" in (e["data"].get("detail") or {}):
                s["fallback"] += 1
            elif reason == "escalate":
                s["escalate"] += 1
            elif reason.startswith("route_violation"):
                s["violation"] += 1
    return s


def summarise(name, eps, prices):
    rewards = [r for r, _ in eps.values()]
    st = [trace_stats(tp) for _, tp in eps.values()]
    tot = sum(st, Counter())
    n = len(eps)
    line = {"run": name, "episodes": n, "pass^1": float(np.mean(rewards)) if n else 0.0,
            "llm_calls/ep": tot["calls"] / max(n, 1), "input_tok/ep": tot["input"] / max(n, 1),
            "cached_tok/ep": tot["cached"] / max(n, 1), "output_tok/ep": tot["output"] / max(n, 1),
            "episodes_fell_back": sum(1 for x in st if x["fallback"]) / max(n, 1),
            "episodes_escalated": sum(1 for x in st if x["escalate"]) / max(n, 1),
            "episodes_violation": sum(1 for x in st if x["violation"]) / max(n, 1),
            "missing_traces": tot["missing_trace"]}
    if prices:
        miss, hit, out = prices
        cost = ((tot["input"] - tot["cached"]) * miss + tot["cached"] * hit + tot["output"] * out) / 1e6
        line["usd/ep"] = cost / max(n, 1)
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--treatment", required=True, nargs="+")
    ap.add_argument("--paired", action="store_true", help="compare only (task, trial) pairs present in both")
    ap.add_argument("--price-miss", type=float)
    ap.add_argument("--price-hit", type=float)
    ap.add_argument("--price-out", type=float)
    a = ap.parse_args()
    prices = (a.price_miss, a.price_hit, a.price_out) if a.price_miss is not None else None
    ctrl = load(a.control)
    runs = [("control", ctrl)] + [(Path(t).stem, load(t)) for t in a.treatment]
    if a.paired:
        keys = set(ctrl)
        for _, r in runs[1:]:
            keys &= set(r)
        runs = [(n, {k: v for k, v in r.items() if k in keys}) for n, r in runs]
    lines = [summarise(n, r, prices) for n, r in runs]
    cols = list(lines[0].keys())
    print(" | ".join(cols))
    for L in lines:
        print(" | ".join(f"{L[c]:.3f}" if isinstance(L[c], float) else str(L[c]) for c in cols))
    if len(lines) > 1:
        c = lines[0]
        for L in lines[1:]:
            print(f"\n{L['run']} vs control: pass^1 {L['pass^1'] - c['pass^1']:+.3f}, input tokens/ep "
                  f"{L['input_tok/ep'] / max(c['input_tok/ep'], 1) - 1:+.1%}"
                  + (f", cost/ep {L['usd/ep'] / max(c['usd/ep'], 1e-9) - 1:+.1%}" if prices else ""))
    print("\nFor significance, use the repo's own paired multi-trial + bootstrap tool:\n"
          "  python scripts/taubench_multitrial_compare.py <control.json> <treatment.json>")


if __name__ == "__main__":
    main()
