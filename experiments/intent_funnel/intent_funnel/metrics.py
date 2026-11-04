"""Routing metrics.

safe route   : write tools needed by gold ⊆ write tools in route (the agent has every tool it needs),
               or FALLBACK (full prompt is always safe). info_query / transfer / out_of_scope / continue
               need only the shared read-only tools, so any route serves them.
coverage     : fraction of requests that were routed (not fallback)
risk         : 1 - safe rate among routed requests  (the misroute rate that actually hurts)
token_ratio  : mean routed prompt tokens / full prompt tokens (fallback counts as 1.0)
top1 / macro-F1 / ECE: classifier quality on single-label sets (argmax vs gold)
"""
import numpy as np

from .taxonomy import needed_write_tools


def route_metrics(routes, golds, budget=None):
    n = len(routes)
    routed = [r is not None for r in routes]
    safe = [r is None or needed_write_tools(g) <= needed_write_tools(r) for r, g in zip(routes, golds)]
    n_routed = sum(routed)
    risk = (sum(1 for r, s in zip(routed, safe) if r and not s) / n_routed) if n_routed else 0.0
    out = {"n": n, "coverage": n_routed / n if n else 0.0, "risk": risk, "safe_rate": sum(safe) / n if n else 0.0,
           "exact_routed": (sum(1 for r, g in zip(routes, golds) if r is not None and set(r) == set(g)) / n_routed)
           if n_routed else 0.0}
    if budget is not None:
        out["token_ratio"] = float(np.mean([budget.cost(r) / budget.full for r in routes]))
    return out


def ece(conf, correct, bins=10):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def cls_metrics(probs, classes, golds_single):
    from sklearn.metrics import f1_score
    pred = [classes[j] for j in probs.argmax(1)]
    conf = probs.max(1)
    correct = [p == g for p, g in zip(pred, golds_single)]
    labs = sorted(set(golds_single))
    return {"top1": float(np.mean(correct)),
            "macro_f1": float(f1_score(golds_single, pred, labels=labs, average="macro", zero_division=0)),
            "ece": ece(conf, correct)}
