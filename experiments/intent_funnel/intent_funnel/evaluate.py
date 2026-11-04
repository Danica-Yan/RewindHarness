"""One evaluation protocol for every probabilistic router.

Primary metric = REPLAY on real test trajectories (cumulative routing, the way the router is deployed):
  route_t = route_{t-1} ∪ decide(p(user_turn_t)); a fallback at any turn means full prompt from then on.
  write_coverage : fraction of write calls whose tool was in the route at that moment (or route was full)
  task_coverage  : fraction of trajectories with every write covered
  step_token     : mean static-prompt tokens per user turn / full prompt (fallback turns = 1.0)

Thresholds (t_include, gate) are chosen by 2-fold cross-fitting over test *tasks* (all trials of a task
stay in one fold): pick on fold A (cheapest setting with write_coverage >= target), score fold B, swap.
The model never trains on test; only two scalars are fitted, and reported numbers are out-of-fold.
Per-turn and external-set metrics are secondary sanity checks at the fold-A thresholds.
"""
import time
import zlib

import numpy as np

from .metrics import cls_metrics, route_metrics
from .router import decide
from .textclean import clean
from .taxonomy import tools_for

GATES = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
INCLUDES = [0.05, 0.1, 0.15, 0.25, 0.4]
TARGET_WRITE_COVERAGE = 0.95
TARGET_RISK = 0.05  # per-turn sets (used when no replays are given)


def _fold(key):
    return zlib.crc32(str(key).encode()) % 2


# ---------------------------------------------------------------- replay
def replay_probs(model, replays):
    texts = [clean(t)[0] for rp in replays for t in rp["user_turns"]]
    P = model.predict_proba(texts) if texts else np.zeros((0, len(model.classes)))
    out, k = [], 0
    for rp in replays:
        n = len(rp["user_turns"])
        out.append(P[k:k + n])
        k += n
    return out


def replay_metrics(replays, probs, classes, budget, t_include, gate):
    cov, task_ok, costs, fallbacks = [], [], [], 0
    for rp, P in zip(replays, probs):
        route, full = set(), False
        per_turn = []
        decisions = decide(P, classes, t_include=t_include, gate=gate) if len(P) else []
        for d in decisions:
            if d is None:
                full = True
            elif not full:
                route |= set(d)
            per_turn.append(None if full else set(route))
        fallbacks += full
        ok = []
        for w in rp["writes"]:
            t = w["after_user_turn"]
            rt = per_turn[t] if 0 <= t < len(per_turn) else None
            ok.append(rt is None or w["tool"] in tools_for(sorted(rt)))
        cov += ok
        task_ok.append(all(ok))
        costs += [budget.cost(None if rt is None else sorted(rt)) / budget.full for rt in per_turn]
    return {"n_traj": len(replays), "n_writes": len(cov), "write_coverage": float(np.mean(cov)) if cov else 1.0,
            "task_coverage": float(np.mean(task_ok)) if task_ok else 1.0,
            "step_token": float(np.mean(costs)) if costs else 1.0,
            "traj_fell_back": fallbacks / max(len(replays), 1)}


def replay_crossfit(model, replays, budget, target=TARGET_WRITE_COVERAGE):
    probs = replay_probs(model, replays)
    folds = [[i for i, rp in enumerate(replays) if _fold(rp["task_id"]) == k] for k in (0, 1)]
    agg, picks = [], []
    for a, b in [(0, 1), (1, 0)]:
        ra, pa = [replays[i] for i in folds[a]], [probs[i] for i in folds[a]]
        grid = {(ti, g): replay_metrics(ra, pa, model.classes, budget, ti, g) for ti in INCLUDES for g in GATES}
        ok = [k for k, v in grid.items() if v["write_coverage"] >= target]
        best = min(ok, key=lambda k: grid[k]["step_token"]) if ok else max(grid, key=lambda k: grid[k]["write_coverage"])
        picks.append(best)
        agg.append((folds[b], best))
    # score each fold with the other fold's pick, then pool
    res_parts = []
    for idx, (ti, g) in agg:
        rr, pp = [replays[i] for i in idx], [probs[i] for i in idx]
        res_parts.append((replay_metrics(rr, pp, model.classes, budget, ti, g), len(rr)))
    n = sum(w for _, w in res_parts) or 1
    pooled = {k: sum(m[k] * w for m, w in res_parts) / n for k in ["write_coverage", "task_coverage", "step_token",
                                                                    "traj_fell_back"]}
    pooled["n_traj"] = n
    curve = {f"{ti}|{g}": replay_metrics(replays, probs, model.classes, budget, ti, g) for ti in INCLUDES for g in GATES}
    return pooled, picks, curve


# ---------------------------------------------------------------- per-turn / single-request sets
def _pick(probs, classes, golds, budget, target_risk):
    grid = {(ti, g): route_metrics(decide(probs, classes, t_include=ti, gate=g), golds, budget)
            for ti in INCLUDES for g in GATES}
    ok = [k for k, v in grid.items() if v["risk"] <= target_risk and v["coverage"] > 0]
    return min(ok, key=lambda k: grid[k]["token_ratio"]) if ok else min(grid, key=lambda k: grid[k]["risk"])


def crossfit(probs, classes, rows, budget, target_risk=TARGET_RISK):
    keyf = lambda r: r["meta"].get("user_id", r["meta"].get("task_id"))  # noqa: E731
    folds = [[i for i, r in enumerate(rows) if _fold(keyf(r)) == k] for k in (0, 1)]
    routes, picks = [None] * len(rows), []
    for a, b in [(0, 1), (1, 0)]:
        ia, ib = folds[a], folds[b]
        if not ia or not ib:
            continue
        best = _pick(probs[ia], classes, [rows[i]["labels"] for i in ia], budget, target_risk)
        picks.append(best)
        for i, rt in zip(ib, decide(probs[ib], classes, t_include=best[0], gate=best[1])):
            routes[i] = rt
    return route_metrics(routes, [r["labels"] for r in rows], budget), picks


def evaluate(model, sets, budget, replays=None, primary="tau_natural"):
    """model: .predict_proba(list[str]) -> (n, C) and .classes. Returns a JSON-able dict."""
    res = {"test": {}}
    if replays:
        pooled, picks, curve = replay_crossfit(model, replays, budget)
        res.update({"primary": "replay", "replay": pooled, "replay_picks": picks, "replay_curve": curve})
        ti, gate = picks[0]
    else:
        prim = primary if primary in sets else next(iter(sets))
        probs = model.predict_proba([r["text"] for r in sets[prim]])
        cf, picks = crossfit(probs, model.classes, sets[prim], budget)
        res.update({"primary": prim, "crossfit": cf, "crossfit_picks": picks})
        ti, gate = picks[0] if picks else (0.25, 0.5)
    res["applied_threshold"] = {"t_include": ti, "gate": gate}
    for k, rs in sets.items():
        p = model.predict_proba([r["text"] for r in rs])
        golds = [r["labels"] for r in rs]
        e = {"at_threshold": route_metrics(decide(p, model.classes, t_include=ti, gate=gate), golds, budget)}
        if all(len(g) == 1 for g in golds):
            e["cls"] = cls_metrics(p, model.classes, [g[0] for g in golds])
        res["test"][k] = e
    return res


def latency(model, texts, n=200):
    lat = []
    for s in texts[:n]:
        t0 = time.perf_counter()
        model.predict_proba([s])
        lat.append((time.perf_counter() - t0) * 1000)
    return {"p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)), "n": len(lat)}


def headline(res):
    if res.get("primary") == "replay":
        r = res["replay"]
        return (f"**replay on {r['n_traj']} real test trajectories (thresholds cross-fit, picks {res['replay_picks']}):** "
                f"write coverage {r['write_coverage']:.1%}, task coverage {r['task_coverage']:.1%}, "
                f"step token {r['step_token']:.2f}, trajectories that fell back {r['traj_fell_back']:.1%}")
    cf = res["crossfit"]
    return (f"**{res['primary']} (thresholds cross-fit, picks {res['crossfit_picks']}):** coverage {cf['coverage']:.1%}, "
            f"risk {cf['risk']:.1%}, token {cf['token_ratio']:.2f}")


def to_markdown(name, res):
    L = [f"## {name}\n", headline(res) + "\n",
         f"Per-set rows use t_include={res['applied_threshold']['t_include']}, gate={res['applied_threshold']['gate']}.\n",
         "| set | n | coverage | risk | token | top1 | macroF1 | ECE |", "|---|---|---|---|---|---|---|---|"]
    f = lambda x: f"{x:.3f}" if isinstance(x, float) else "–"  # noqa: E731
    for k, e in res["test"].items():
        a, c = e["at_threshold"], e.get("cls", {})
        L.append(f"| {k} | {a['n']} | {a['coverage']:.1%} | {a['risk']:.1%} | {a['token_ratio']:.2f} | "
                 f"{f(c.get('top1'))} | {f(c.get('macro_f1'))} | {f(c.get('ece'))} |")
    if "latency" in res:
        L.append(f"\nLatency per request ({res.get('device', 'cpu')}): p50 {res['latency']['p50_ms']:.1f} ms, "
                 f"p95 {res['latency']['p95_ms']:.1f} ms")
    return "\n".join(L) + "\n"
