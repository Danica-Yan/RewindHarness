"""Prefix-cache economics of tool routing, from real rewind traces (control vs routed).

  1. cache granularity (cached token counts are multiples of the cache block)
  2. static prefix size: first-call input tokens, control vs routed
  3. upper bound on the bill share ANY cache-friendly router could save: it removes D static tokens from
     every call, and those tokens would otherwise be cache hits (the full prompt is a stable prefix
     shared across episodes); the conversation part is unchanged
  4. cache busts in routed runs (calls whose cached share collapses) and the route change behind each

Usage: python scripts/cache_bound.py data/e2e_pilot [--price-miss 0.14 --price-hit 0.0028 --price-out 0.28]
Trace files are matched as control_<name>.jsonl / pilot_<name>.jsonl.
"""
import argparse
import glob
import json
from pathlib import Path


def usages(f):
    return [e["data"]["response"]["usage"] for e in map(json.loads, open(f))
            if e["type"] == "llm_call" and "response" in e["data"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--price-miss", type=float, default=0.14)
    ap.add_argument("--price-hit", type=float, default=0.0028)
    ap.add_argument("--price-out", type=float, default=0.28)
    a = ap.parse_args()
    ctrl = sorted(glob.glob(str(Path(a.dir) / "control_*.jsonl")))
    pilot = sorted(glob.glob(str(Path(a.dir) / "pilot_*.jsonl")))
    cached = [u.get("cached_tokens", 0) for f in ctrl + pilot for u in usages(f)]
    block = 128 if all(c % 128 == 0 for c in cached) else (64 if all(c % 64 == 0 for c in cached) else None)
    print(f"cache block size consistent with: {block} tokens")
    c0 = [usages(f)[0]["input_tokens"] for f in ctrl]
    p0 = [usages(f)[0]["input_tokens"] for f in pilot]
    D = sum(c0) / len(c0) - sum(p0) / len(p0)
    print(f"first-call input tokens: control {c0}, routed {p0} -> ~{D:.0f} static tokens removed by routing")
    pm, po = a.price_miss, a.price_out
    for name, ph in [(f"actual prices (hit = miss/{pm / a.price_hit:.0f})", a.price_hit),
                     ("hypothetical hit = miss/10", pm / 10), ("no prefix cache", pm)]:
        bill = save = 0.0
        for f in ctrl:
            us = usages(f)
            bill += sum((u["input_tokens"] - u.get("cached_tokens", 0)) * pm + u.get("cached_tokens", 0) * ph
                        + u["output_tokens"] * po for u in us)
            save += len(us) * D * ph
        print(f"{name:<34} a perfect cache-friendly router saves at most {save / bill:.1%} of the bill")
    for f in pilot:
        ev = [json.loads(l) for l in open(f)]
        us = usages(f)
        routes = {e["step"]: e["data"] for e in ev if e["type"] == "route"}
        prev = set()
        changes = {}
        for s in sorted(routes):
            u = set(routes[s]["units"])
            if u - prev:
                changes[s] = (sorted(u - prev), routes[s].get("detail", {}).get("agent_proposed"))
            prev = u
        busts = [(i + 1, x["input_tokens"] - x.get("cached_tokens", 0)) for i, x in enumerate(us)
                 if i > 0 and x.get("cached_tokens", 0) < 0.7 * x["input_tokens"]]
        for step, missed in busts:
            added, prop = changes.get(step, ("?", None))
            print(f"{Path(f).name}: cache bust at step {step}: {missed} tokens re-billed at miss price; "
                  f"route added {added}" + (f" (agent-proposal rules fired: {prop})" if prop else ""))


if __name__ == "__main__":
    main()
