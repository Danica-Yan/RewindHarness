"""DeepSeek as the router (the "just ask the big model" baseline). Run locally with your key.

Spends API budget, so: responses are cached on disk (re-runs are free), --max-calls caps spend,
and --dry-run prints the exact token count of what would be sent without calling anything.

  python scripts/deepseek_zeroshot.py --dry-run
  DEEPSEEK_API_KEY=... python scripts/deepseek_zeroshot.py --sets tau_natural --max-calls 120

Model name: defaults to "deepseek-chat" — check the current model list in DeepSeek's API docs.
Output: out/deepseek/<model>.json (routing metrics with the same definitions as every other router,
plus measured latency and classification-call token usage).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from intent_funnel.budget import PromptBudget  # noqa: E402
from intent_funnel.data import eval_sets, load  # noqa: E402
from intent_funnel.metrics import route_metrics  # noqa: E402
from intent_funnel.taxonomy import INTENTS  # noqa: E402

SYSTEM = """You route customer messages for an online retail store's support agent.
Return JSON {"intents": [...]} using ONLY these labels (several allowed if the message has several requests):
cancel: cancel a pending (not yet shipped) order
modify_items: change items/options in a pending order
modify_order_address: change the shipping address of a pending order
modify_payment: change the payment method of a pending order
modify_user_address: change the default address on the user's account
return: return delivered items for a refund
exchange: exchange delivered items for other variants
info_query: only asking about orders, products, status or policy
transfer: wants a human agent
out_of_scope: anything this retail store's agent cannot handle
If the message is too vague to tell, return {"intents": ["unclear"]}."""


def call(client, model, text):
    t0 = time.perf_counter()
    r = client.chat.completions.create(model=model, temperature=0, response_format={"type": "json_object"},
                                       messages=[{"role": "system", "content": SYSTEM},
                                                 {"role": "user", "content": text}])
    ms = (time.perf_counter() - t0) * 1000
    u = r.usage
    return {"content": r.choices[0].message.content, "ms": ms,
            "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
            "cached_tokens": getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", None)}


def parse(content):
    try:
        labs = [x for x in json.loads(content).get("intents", []) if x in INTENTS]
    except Exception:
        return None
    return labs or None  # unclear / invalid -> fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--sets", nargs="+", default=["tau_natural"])
    ap.add_argument("--max-calls", type=int, default=150)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    budget = PromptBudget(ROOT / "data/taubench")
    sets = eval_sets(load(), "test")
    cache_p = ROOT / "out/deepseek/cache.jsonl"
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if cache_p.exists():
        for line in open(cache_p):
            x = json.loads(line)
            cache[x["key"]] = x
    key = lambda t: hashlib.sha1(f"{a.model}|{SYSTEM}|{t}".encode()).hexdigest()  # noqa: E731

    todo = [r for s in a.sets for r in sets[s] if key(r["text"]) not in cache]
    est = sum(budget.ntok(SYSTEM) + budget.ntok(r["text"]) + 10 for r in todo)
    print(f"{len(todo)} uncached requests, ~{est} prompt tokens (+~15 completion tokens each)")
    if a.dry_run:
        return
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=a.base_url)
    with open(cache_p, "a") as f:
        for i, r in enumerate(todo[:a.max_calls]):
            res = call(client, a.model, r["text"])
            res["key"] = key(r["text"])
            cache[res["key"]] = res
            f.write(json.dumps(res) + "\n")
            f.flush()
            if i % 20 == 0:
                print(i, res["ms"], res["content"][:80])

    report = {"model": a.model, "sets": {}}
    for s in a.sets:
        rs = [r for r in sets[s] if key(r["text"]) in cache]
        hits = [cache[key(r["text"])] for r in rs]
        routes = [parse(h["content"]) for h in hits]
        m = route_metrics(routes, [r["labels"] for r in rs], budget)
        ms = sorted(h["ms"] for h in hits)
        m.update({"latency_p50_ms": ms[len(ms) // 2] if ms else None,
                  "latency_p95_ms": ms[int(len(ms) * 0.95)] if ms else None,
                  "router_prompt_tokens_mean": sum(h["prompt_tokens"] for h in hits) / max(len(hits), 1),
                  "router_completion_tokens_mean": sum(h["completion_tokens"] for h in hits) / max(len(hits), 1),
                  "n_scored": len(rs), "n_total": len(sets[s])})
        report["sets"][s] = m
        print(s, json.dumps(m, indent=1))
    (ROOT / f"out/deepseek/{a.model}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
