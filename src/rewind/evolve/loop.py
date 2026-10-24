"""One round of self-improvement.

    incumbent rows ──► clusters (free) ──► stability filter (null forks)
        ──► A/B split ──► proposer (1 LLM call) ──► offline checks
        ──► fork screen on B ──► gate on unseen pool ──► commit / reject

Budget: every phase checks its worst-case episode count against what is
left of `budget_episodes` before spending anything, and `plan()` prints the
worst case for the whole round without any API call. At most one candidate
reaches the gate per round (the gate is the expensive step).
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .gate import GatePolicy, gate
from .ledger import EditLedger
from .prevalence import PrevalencePolicy, clusters, describe, eligible, evidence_rows
from .runner import load_rows, version_setup
from .screen import ScreenPolicy, screen
from .stability import assess_failures
from .surface import parse_check, validate_edit


class BacktestPolicy(BaseModel):
    """Zero-cost filter for `check` edits, run on the incumbent's recorded
    episodes before any screening fork is spent."""
    max_false_fire_rate: float = 0.02   # fires / guarded calls, in episodes that passed
    min_cluster_catches: int = 1        # this cluster's failed episodes with an off-gold fire


class EvolveConfig(BaseModel):
    split: str = "train"
    run_dir: str = "runs/p1"
    prevalence: PrevalencePolicy = Field(default_factory=PrevalencePolicy)
    stability_n: int = 5
    stable_max: int = 1
    screen: ScreenPolicy = Field(default_factory=ScreenPolicy)
    gate: GatePolicy = Field(default_factory=GatePolicy)
    n_candidates: int = 3
    max_clusters_per_round: int = 2
    max_members_per_cluster: int = 10   # stability is assessed on at most this many
    budget_episodes: int = 300          # hard cap on episodes (runs + forks) per round
    concurrency: int = 4
    backtest: BacktestPolicy = Field(default_factory=BacktestPolicy)


class RoundReport(BaseModel):
    round: int
    version_from: int
    version_to: int | None = None
    clusters_considered: list[str] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)
    episodes_used: int = 0
    proposer_calls: int = 0
    committed_edit: dict | None = None
    gate: dict | None = None


class Budget:
    def __init__(self, total: int) -> None:
        self.total, self.used = total, 0

    @property
    def left(self) -> int:
        return self.total - self.used

    def can(self, n: int) -> bool:
        return n <= self.left

    def spend(self, n: int) -> None:
        self.used += n


def _paths(cfg: EvolveConfig) -> dict[str, Path]:
    d = Path(cfg.run_dir)
    return {"stability": d / "stability.json", "gate": d / "gate", "rounds": d / "rounds"}


def _pool(rows: list[dict], shown: set[int]) -> list[int]:
    return sorted({r["task_id"] for r in rows} - shown)


def _dedupe(cs: list) -> list:
    """A fine cluster and its coarse parent with the same members are one cluster."""
    seen, out = set(), []
    for c in cs:
        if tuple(c.tasks) not in seen:
            seen.add(tuple(c.tasks))
            out.append(c)
    return out


def plan(cfg: EvolveConfig, ledger: EditLedger) -> str:
    """Worst-case API use of the next round. No API calls."""
    inc = ledger.current()
    rows = load_rows(ledger.rows_path(inc.version))
    shown = ledger.shown()
    pool_rows = [r for r in rows if r["task_id"] not in shown]
    stats, sigs = clusters(pool_rows, cfg.split)
    elig = _dedupe([c for c in stats if eligible(c, cfg.prevalence)[0]])[: cfg.max_clusters_per_round]
    lines = [f"incumbent v{inc.version}: {len({r['task_id'] for r in rows})} tasks, "
             f"{len(sigs)} failed in the unseen pool ({len(shown)} tasks already shown)",
             "clusters (* = eligible):", describe(stats, cfg.prevalence), ""]
    worst = 0
    for c in elig:
        m = min(len(c.tasks), cfg.max_members_per_cluster)
        st = m * cfg.stability_n
        b = m // 2
        sc = cfg.n_candidates * b * cfg.screen.n
        worst += st + sc
        lines.append(f"cluster {c.key}: stability <= {st} forks, screening <= {sc} forks")
    pool_after = len(_pool(rows, shown)) - (min(len(elig[0].tasks), cfg.max_members_per_cluster) if elig else 0)
    g = pool_after * cfg.gate.trials + 2 * cfg.gate.confirm_k * max(4, round(0.15 * pool_after))
    eff = 0
    if cfg.gate.mode == "targeted" and elig:
        eff = (min(len(elig[0].tasks), cfg.max_members_per_cluster) // 2) * cfg.gate.efficacy_n
        lines.append(f"gate efficacy (held-out targets, runs first): <= {eff} forks")
    worst += g + eff
    lines.append(f"gate pool + flip confirmation (only if efficacy passes): <= {g} episodes "
                 f"({'about half if the candidate is clearly harmful' if cfg.gate.futility_stop else 'no early stop'})")
    if elig:
        m0 = min(len(elig[0].tasks), cfg.max_members_per_cluster)
        first = m0 * cfg.stability_n + cfg.n_candidates * (m0 // 2) * cfg.screen.n + g + \
            (m0 // 2) * (cfg.gate.efficacy_n if cfg.gate.mode == "targeted" else 0)
        lines.append(f"to be sure the FIRST cluster can reach the gate this round: --budget {first}")
    lines.append(f"proposer: <= {len(elig)} LLM calls (small)")
    lines.append(f"worst case this round: {worst} episodes; hard cap budget_episodes = {cfg.budget_episodes}")
    return "\n".join(lines)


async def run_round(cfg: EvolveConfig, ledger: EditLedger, *, provider: Any, user_llm: Any, store: Any,
                    proposer: Any, log: Callable[[str], None] = print) -> RoundReport:
    paths = _paths(cfg)
    budget = Budget(cfg.budget_episodes)
    round_no = ledger.bump_round()
    inc = ledger.current()
    rep = RoundReport(round=round_no, version_from=inc.version)
    rows = load_rows(ledger.rows_path(inc.version))
    shown = ledger.shown()
    pool_rows = [r for r in rows if r["task_id"] not in shown]
    stats, _ = clusters(pool_rows, cfg.split)
    tried = {d["cluster"] for d in ledger.decisions()
             if d["stage"] == "stability" and d["version_from"] == inc.version}
    uniq = _dedupe([c for c in stats if eligible(c, cfg.prevalence)[0] and c.key not in tried])
    _, inc_registry = version_setup(inc)

    def note(msg: str) -> None:
        rep.events.append(msg)
        log(msg)

    if not uniq:
        note("no eligible cluster in the unseen pool -- nothing to optimise this round")
        _save(paths, rep)
        return rep

    for cluster in uniq[: cfg.max_clusters_per_round]:
        rep.clusters_considered.append(cluster.key)
        members = evidence_rows(pool_rows, cluster.tasks)[: cfg.max_members_per_cluster]
        cost = len(members) * cfg.stability_n
        if not budget.can(cost):
            note(f"budget: stability for {cluster.key} needs <= {cost}, {budget.left} left -- stop")
            break
        note(f"cluster {cluster.key}: {len(cluster.tasks)} tasks, assessing {len(members)} for stability")
        recs = await assess_failures(members, paths["stability"], store=store, provider=provider,
                                     user_llm=user_llm, registry=inc_registry, n=cfg.stability_n,
                                     stable_max=cfg.stable_max, concurrency=cfg.concurrency, log=log)
        mine = [recs[k] for k in recs if recs[k].ctx_version == inc.version
                and recs[k].task_id in {r["task_id"] for r in members}]
        budget.spend(sum(r.n for r in mine))
        stable = sorted((r for r in mine if r.label == "stable"), key=lambda r: r.task_id)
        if len(stable) < cfg.prevalence.min_distinct_tasks:
            ledger.record(round_no, inc.version, None, "stability", False,
                          f"{len(stable)} stable of {len(mine)} assessed "
                          f"(need {cfg.prevalence.min_distinct_tasks})", cluster=cluster.key)
            note(f"  only {len(stable)} stable failures -- mostly noise, skip cluster")
            continue
        a = [r for i, r in enumerate(stable) if i % 2 == 0]
        b = [r for i, r in enumerate(stable) if i % 2 == 1]
        ledger.mark_shown([r.task_id for r in a], round_no, "proposer")
        ledger.mark_shown([r.task_id for r in b], round_no, "screen")
        ledger.record(round_no, inc.version, None, "stability", True,
                      f"{len(stable)} stable: A={[r.task_id for r in a]} B={[r.task_id for r in b]}",
                      {"A": [r.task_id for r in a], "B": [r.task_id for r in b]}, cluster=cluster.key)

        stop_round = await _try_edits(cfg, ledger, round_no, inc, rows, pool_rows, cluster.key, a, b,
                                      budget, rep, provider=provider, user_llm=user_llm, store=store,
                                      proposer=proposer, log=log, note=note)
        if rep.gate is not None or stop_round:
            break  # a gated or gate-pending candidate ends the round

    rep.episodes_used = budget.used
    _save(paths, rep)
    return rep



def _gate_reserve(cfg: EvolveConfig, pool: list[int], n_targets: int) -> int:
    """Worst-case-ish gate cost: efficacy + pool + flip confirmation (flips
    budgeted at 15% of the pool, both versions)."""
    eff = n_targets * cfg.gate.efficacy_n if cfg.gate.mode == "targeted" else 0
    conf = 2 * cfg.gate.confirm_k * max(4, round(0.15 * len(pool)))
    return eff + len(pool) * cfg.gate.trials + conf


def _record_gate(ledger: EditLedger, round_no: int, inc: Any, e: Any, v: Any, rows: list[dict],
                 split: str, extra: dict | None = None) -> int:
    from .feedback import gate_feedback
    data = v.model_dump()
    data.update(extra or {})
    did = ledger.record(round_no, inc.version, e, "gate", v.accepted, v.reason, data)
    if not v.accepted and Path(v.rows_path).exists():
        ledger.set_feedback(did, gate_feedback(data, rows, load_rows(v.rows_path), split))
    return did


def backtest_edit(cfg: EvolveConfig, e: Any, rows: list[dict], store: Any,
                  cluster_tasks: set[int]) -> tuple[bool, str, dict]:
    """Replay a proposed check over the incumbent's recorded episodes (no API):
    it must stay quiet where the agent succeeded and fire on at least one of
    this cluster's failures, on a call the gold solution does not make."""
    from ..policy import PolicySet
    from .backtest import taubench_backtest
    chk = parse_check(e).model_copy(update={"mode": "block"})
    rep = taubench_backtest(PolicySet([chk]), rows, store=store, split=cfg.split)
    s = rep.stats[chk.id]
    caught = sorted({t for t, _ in s.episodes_failing_caught} & cluster_tasks)
    data = {**s.summary(rep.n_pass, rep.n_fail), "cluster_caught": caught,
            "skipped": len(rep.skipped), **rep.to_json(max_examples=3)}
    if rep.n_pass + rep.n_fail < max(1, len(rows) // 2):
        return False, f"only {rep.n_pass + rep.n_fail}/{len(rows)} episodes have traces + snapshots", data
    p = cfg.backtest
    why = (f"fires on {s.fires_passing}/{s.guarded_calls_passing} calls in passing episodes "
           f"({s.false_fire_rate:.1%}, max {p.max_false_fire_rate:.0%}); catches {len(caught)} of this "
           f"cluster's failures (need {p.min_cluster_catches}), {s.fires_failing_off_gold} off-gold calls "
           f"overall")
    return s.false_fire_rate <= p.max_false_fire_rate and len(caught) >= p.min_cluster_catches, why, data


async def _try_edits(cfg: EvolveConfig, ledger: EditLedger, round_no: int, inc: Any, rows: list[dict],
                     pool_rows: list[dict], cluster_key: str, a: list, b: list, budget: Budget,
                     rep: RoundReport, *, provider: Any, user_llm: Any, store: Any, proposer: Any,
                     log: Callable[[str], None], note: Callable[[str], None]) -> bool:
    """Propose for one cluster's A half, then check, screen and gate. Returns
    True if the round must stop (a candidate was gated, or one passed
    screening but the budget cannot cover its gate)."""
    _, inc_registry = version_setup(inc)
    tools = set(inc_registry._tools)
    a_rows = evidence_rows(rows, [r.task_id for r in a])
    edits = await proposer.propose(cluster_key, a_rows, cfg.split, inc,
                                   ledger.rejected_for_prompt(cluster_key), cfg.n_candidates)
    rep.proposer_calls += 1
    note(f"  proposer returned {len(edits)} candidate edit(s)")
    b_tasks = [(r.task_id, r.trace, r.null_rate) for r in b]
    for e in edits:
        if e.edit_id in ledger.tried_edit_ids():
            continue
        why = validate_edit(e, inc, known_tools=tools, max_tokens=cfg.gate.max_ctx_tokens)
        if why:
            ledger.record(round_no, inc.version, e, "invalid", False, why, cluster=cluster_key)
            note(f"  [invalid] {e.kind} {e.target}: {why}")
            continue
        if e.kind == "check" and e.op != "delete":
            ok, why, data = backtest_edit(cfg, e, rows, store, {r.task_id for r in a + b})
            ledger.record(round_no, inc.version, e, "backtest", ok, why, data, cluster=cluster_key)
            note(f"  backtest {e.target}: {'PASS' if ok else 'fail'} -- {why}")
            if not ok:
                continue
        cost = len(b_tasks) * cfg.screen.n
        if not budget.can(cost):
            note(f"budget: screening needs <= {cost}, {budget.left} left -- stop")
            return True
        cand = inc.apply(e, known_tools=tools).model_copy(
            update={"version": ledger.next_version_number(), "parent": inc.version})
        cand_cfg, cand_reg = version_setup(cand)
        shown = cand.rules.get(e.target) or cand.tool_notes.get(e.target) or cand.checks.get(e.target) \
            or cand.skills.get(e.target)
        note(f"  screening {e.kind} {e.target}: {shown!r}")
        sr = await screen([e.edit_id], b_tasks, system_prompt=cand_cfg.system_prompt,
                          registry=cand_reg, store=store, provider=provider, user_llm=user_llm,
                          policy=cfg.screen, log=log)
        budget.spend(sr.forks)
        ledger.record(round_no, inc.version, e, "screen", sr.passed, sr.reason, sr.model_dump(),
                      cluster=cluster_key)
        note(f"  screen: {'PASS' if sr.passed else 'fail'} -- {sr.reason}")
        if not sr.passed:
            continue

        pool = _pool(rows, ledger.shown())
        targets = [(r.task_id, r.trace, r.passes, r.n) for r in b]
        cost = _gate_reserve(cfg, pool, len(targets))
        if not budget.can(cost):
            note(f"budget: gate needs <= {cost} episodes, {budget.left} left -- "
                 f"edit passed screening but is NOT gated this round; "
                 f"run `run_evolve.py gate-pending` to gate it without redoing anything")
            return True
        out = _paths(cfg)["gate"] / f"v{cand.version}_{e.edit_id}.json"
        v = await gate(cand, rows, pool, split=cfg.split, out=out, provider=provider,
                       user_llm=user_llm, store=store, policy=cfg.gate, targets=targets,
                       concurrency=cfg.concurrency, log=log, incumbent=inc)
        budget.spend(v.episodes)
        rep.gate = v.model_dump()
        _record_gate(ledger, round_no, inc, e, v, rows, cfg.split)
        note(f"  gate: {'ACCEPT' if v.accepted else 'reject'} -- {v.reason}")
        if v.accepted:
            ledger.commit(cand, [e], str(out), v.model_dump())
            rep.version_to = cand.version
            rep.committed_edit = e.model_dump()
            note(f"  committed v{cand.version}")
        return True  # at most one gate per round
    return False


def retry_candidates(ledger: EditLedger, cfg: EvolveConfig) -> list[dict]:
    """Clusters at the current version whose A/B split exists and that have no
    accepted edit yet, with their attempt count (proposer batches so far)."""
    import re

    from .stability import load_records
    v = ledger.current().version
    stab = load_records(_paths(cfg)["stability"])
    out = []
    for did, cluster, reason, dj in ledger.db.execute(
            "SELECT id, cluster, reason, data_json FROM decisions WHERE stage='stability' "
            "AND accepted=1 AND version_from=? ORDER BY id", (v,)):
        data = json.loads(dj) if dj else {}
        a_ids, b_ids = data.get("A"), data.get("B")
        if a_ids is None:  # recorded before A/B were stored as data
            m = re.search(r"A=\[([\d, ]*)\] B=\[([\d, ]*)\]", reason or "")
            if not m:
                continue
            a_ids = [int(x) for x in m.group(1).split(",") if x.strip()]
            b_ids = [int(x) for x in m.group(2).split(",") if x.strip()]
        rec = {r.task_id: r for r in stab.values() if r.ctx_version == v}
        if not all(t in rec for t in a_ids + b_ids):
            continue
        accepted = ledger.db.execute(
            "SELECT 1 FROM decisions WHERE stage='gate' AND accepted=1 AND cluster=? AND version_from=?",
            (cluster, v)).fetchone()
        attempts = ledger.db.execute(
            "SELECT COUNT(DISTINCT round) FROM decisions WHERE cluster=? AND version_from=? "
            "AND stage IN ('invalid','screen')", (cluster, v)).fetchone()[0]
        if not accepted:
            out.append({"cluster": cluster, "A": [rec[t] for t in a_ids], "B": [rec[t] for t in b_ids],
                        "attempts": attempts})
    return out


async def retry_cluster(cfg: EvolveConfig, ledger: EditLedger, *, provider: Any, user_llm: Any,
                        store: Any, proposer: Any, cluster_key: str | None = None,
                        max_attempts: int = 3, log: Callable[[str], None] = print) -> RoundReport:
    """Another proposal for a cluster whose A/B split already exists: no
    stability re-run, the proposer sees this cluster's rejections WITH their
    feedback. Note: B has now screened more than one candidate, so screening
    on it is mildly optimistic; the gate's efficacy check uses fresh forks
    and the pool never contains A or B."""
    paths = _paths(cfg)
    budget = Budget(cfg.budget_episodes)
    inc = ledger.current()
    rows = load_rows(ledger.rows_path(inc.version))
    backfill_feedback(ledger, cfg)
    cands = [c for c in retry_candidates(ledger, cfg) if cluster_key in (None, c["cluster"])]
    round_no = ledger.bump_round()
    rep = RoundReport(round=round_no, version_from=inc.version)

    def note(msg: str) -> None:
        rep.events.append(msg)
        log(msg)

    if not cands:
        note("no cluster to retry at this version")
        _save(paths, rep)
        return rep
    c = cands[-1] if cluster_key is None else cands[0]
    if c["attempts"] >= max_attempts:
        note(f"cluster {c['cluster']} already had {c['attempts']} attempts (max {max_attempts})")
        _save(paths, rep)
        return rep
    rep.clusters_considered.append(c["cluster"])
    note(f"retry {c['cluster']} (attempt {c['attempts'] + 1}): "
         f"A={[r.task_id for r in c['A']]} B={[r.task_id for r in c['B']]}")
    pool_rows = [r for r in rows if r["task_id"] not in ledger.shown()]
    await _try_edits(cfg, ledger, round_no, inc, rows, pool_rows, c["cluster"], c["A"], c["B"], budget,
                     rep, provider=provider, user_llm=user_llm, store=store, proposer=proposer,
                     log=log, note=note)
    rep.episodes_used = budget.used
    _save(paths, rep)
    return rep


def backfill_feedback(ledger: EditLedger, cfg: EvolveConfig) -> int:
    """Attach feedback to rejected gate decisions recorded before feedback
    existed. Free: reads results already on disk."""
    from .feedback import gate_feedback
    n = 0
    for did, v, data in ledger.gate_rejections_without_feedback():
        if Path(data["rows_path"]).exists():
            inc_rows = load_rows(ledger.rows_path(v))
            ledger.set_feedback(did, gate_feedback(data, inc_rows, load_rows(data["rows_path"]), cfg.split))
            n += 1
    return n


def regate_candidates(ledger: EditLedger) -> list[dict]:
    """Rejected gate decisions at the current version that were decided
    without flip confirmation (before it existed) and passed efficacy."""
    from .surface import Edit
    v = ledger.current().version
    out = []
    for did, rnd, ej, dj in ledger.db.execute(
            "SELECT id, round, edit_json, data_json FROM decisions WHERE stage='gate' AND accepted=0 "
            "AND version_from=? ORDER BY id", (v,)):
        data = json.loads(dj)
        eff = data.get("efficacy")
        if eff and eff.get("passed") and not data.get("confirm_episodes") and not data.get("regated"):
            out.append({"decision_id": did, "round": rnd, "edit": Edit.model_validate_json(ej),
                        "verdict": data})
    return out


async def regate(cfg: EvolveConfig, ledger: EditLedger, *, provider: Any, user_llm: Any, store: Any,
                 decision_id: int | None = None, log: Callable[[str], None] = print) -> list[dict]:
    """Re-decide earlier gate rejections under the current policy. Reuses the
    efficacy result and every pool episode already run (the runner skips
    them); only flip confirmation is new."""
    from .gate import Efficacy
    results = []
    for c in regate_candidates(ledger):
        if decision_id is not None and c["decision_id"] != decision_id:
            continue
        inc = ledger.current()
        rows = load_rows(ledger.rows_path(inc.version))
        old = c["verdict"]
        pool = sorted({r["task_id"] for r in load_rows(old["rows_path"])}) if Path(old["rows_path"]).exists() \
            else _pool(rows, ledger.shown())
        e = c["edit"]
        _, reg = version_setup(inc)
        cand = inc.apply(e, known_tools=set(reg._tools)).model_copy(
            update={"version": ledger.next_version_number(), "parent": inc.version})
        log(f"re-gating {e.kind} {e.target} (decision {c['decision_id']}): was '{old['reason']}'")
        v = await gate(cand, rows, pool, split=cfg.split, out=Path(old["rows_path"]), provider=provider,
                       user_llm=user_llm, store=store, policy=cfg.gate, concurrency=cfg.concurrency,
                       log=log, incumbent=inc, prior_efficacy=Efficacy(**old["efficacy"]))
        _record_gate(ledger, c["round"], inc, e, v, rows, cfg.split,
                     {"regate_of": c["decision_id"]})
        ledger.mark_regated(c["decision_id"])
        log(f"  gate: {'ACCEPT' if v.accepted else 'reject'} -- {v.reason}")
        results.append({"edit": e.model_dump(), "verdict": v.model_dump(), "was": old["reason"]})
        if v.accepted:
            ledger.commit(cand, [e], old["rows_path"], v.model_dump())
            log(f"  committed v{cand.version}")
            break
    return results

def pending_candidates(ledger: EditLedger, cfg: EvolveConfig) -> list[dict]:
    """Edits that passed screening but were never gated (e.g. the round ran
    out of budget), each with its held-out targets rebuilt from the screening
    record and the stability cache -- so gating them redoes nothing."""
    from .stability import load_records
    from .surface import Edit
    rows = ledger.db.execute(
        "SELECT round, version_from, edit_json, cluster, data_json FROM decisions "
        "WHERE stage='screen' AND accepted=1 ORDER BY id").fetchall()
    gated = {r[0] for r in ledger.db.execute(
        "SELECT edit_id FROM decisions WHERE stage='gate'").fetchall()}
    stab = load_records(_paths(cfg)["stability"])
    out = []
    for round_no, v_from, ej, cluster, dj in rows:
        e = Edit.model_validate_json(ej)
        if e.edit_id in gated or v_from != ledger.current().version:
            continue
        targets = []
        for t in json.loads(dj)["tasks"]:
            rec = next((r for r in stab.values()
                        if r.task_id == t["task_id"] and r.ctx_version == v_from), None)
            if rec is not None:
                targets.append((rec.task_id, rec.trace, rec.passes, rec.n))
        out.append({"round": round_no, "version_from": v_from, "edit": e, "cluster": cluster,
                    "targets": targets})
    return out


async def gate_pending(cfg: EvolveConfig, ledger: EditLedger, *, provider: Any, user_llm: Any,
                       store: Any, log: Callable[[str], None] = print) -> list[dict]:
    """Gate every pending candidate (oldest first) until one is accepted."""
    results = []
    budget = Budget(cfg.budget_episodes)
    for p in pending_candidates(ledger, cfg):
        inc = ledger.current()
        rows = load_rows(ledger.rows_path(inc.version))
        pool = _pool(rows, ledger.shown())
        e = p["edit"]
        cost = _gate_reserve(cfg, pool, len(p["targets"]))
        if not budget.can(cost):
            log(f"budget: gate needs <= {cost}, {budget.left} left -- stop")
            break
        _, reg = version_setup(inc)
        cand = inc.apply(e, known_tools=set(reg._tools)).model_copy(
            update={"version": ledger.next_version_number(), "parent": inc.version})
        log(f"gating {e.kind} {e.target}: {e.text!r} on {len(p['targets'])} targets + {len(pool)} pool tasks")
        out = _paths(cfg)["gate"] / f"v{cand.version}_{e.edit_id}.json"
        v = await gate(cand, rows, pool, split=cfg.split, out=out, provider=provider, user_llm=user_llm,
                       store=store, policy=cfg.gate, targets=p["targets"],
                       concurrency=cfg.concurrency, log=log, incumbent=inc)
        budget.spend(v.episodes)
        _record_gate(ledger, p["round"], inc, e, v, rows, cfg.split)
        log(f"  gate: {'ACCEPT' if v.accepted else 'reject'} -- {v.reason}")
        results.append({"edit": e.model_dump(), "verdict": v.model_dump()})
        if v.accepted:
            ledger.commit(cand, [e], str(out), v.model_dump())
            log(f"  committed v{cand.version}")
            break
    return results


def _save(paths: dict[str, Path], rep: RoundReport) -> None:
    paths["rounds"].mkdir(parents=True, exist_ok=True)
    (paths["rounds"] / f"round{rep.round:02d}.json").write_text(
        json.dumps(rep.model_dump(), indent=1, ensure_ascii=False, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# skill library: induced from successes (and optionally contrasted with the
# A half of failures), gated as one candidate version on stable B failures
# ---------------------------------------------------------------------------
def skill_failures(base_rows: list[dict], split: str, tools: set[str]) -> list[dict]:
    """Failed baseline episodes (with a trace) whose task needs one of
    `tools`, one per task."""
    from .skills import task_writes
    out, seen = [], set()
    for r in sorted(base_rows, key=lambda r: (r["task_id"], r["trial"])):
        tr = ((r.get("info") or {}).get("rewind") or {}).get("trace")
        if r["reward"] >= 1.0 or not tr or r["task_id"] in seen:
            continue
        if task_writes(split, r["task_id"]) & tools:
            out.append(r)
            seen.add(r["task_id"])
    return out


def stability_of(rows: list[dict], cfg: EvolveConfig) -> dict[int, Any]:
    """Cached stability record per task for these baseline rows (if any)."""
    from .stability import _key, load_records
    recs = load_records(_paths(cfg)["stability"])
    out = {}
    for r in rows:
        v = ((r.get("info") or {}).get("rewind") or {}).get("ctx_version", 0)
        rec = recs.get(_key(r["task_id"], r["trial"], v))
        if rec is not None:
            out[r["task_id"]] = rec
    return out


def skill_target_plan(cfg: EvolveConfig, base_rows: list[dict], tools: set[str],
                      exclude: set[int]) -> dict:
    """Which failures can be targets: not shown to the inducer, and STABLE
    (fails >= 4 of 5 null forks) by independent stability data -- the
    first library gate used every baseline failure and the incumbent then
    passed 21 of 40 forks on them: mostly unlucky runs, not real failures."""
    fails = [r for r in skill_failures(base_rows, cfg.split, tools) if r["task_id"] not in exclude]
    stab = stability_of(fails, cfg)
    missing = [r for r in fails if r["task_id"] not in stab]
    return {"failures": fails, "stability": stab, "missing": missing,
            "stable": [r for r in fails if r["task_id"] in stab and stab[r["task_id"]].label == "stable"],
            "assess_cost": len(missing) * cfg.stability_n}


def skill_gate_cost(cfg: EvolveConfig, pool: list[int], n_targets: int) -> int:
    """Null forks under the incumbent + candidate forks + pool + confirmation."""
    return n_targets * cfg.gate.efficacy_n + _gate_reserve(cfg, pool, n_targets)


async def gate_skill_library(cfg: EvolveConfig, ledger: EditLedger, edits: list, base_rows: list[dict], *,
                             provider: Any, user_llm: Any, store: Any, max_targets: int = 8,
                             log: Callable[[str], None] = print) -> dict:
    """Gate a set of induced skills as ONE candidate version.

    Targets: failures of the covered task types that the inducer never saw,
    restricted to STABLE ones by the stability filter's null forks under the
    baseline (assessed now if not cached). Efficacy: the incumbent's rate on
    them is measured fresh (forks from step 1 under the incumbent's context)
    -- independent of the stability data used to select them, so selection
    cannot bias the comparison -- and the candidate is forked on the same
    tasks; one-sided Fisher test. Every efficacy fork is saved as a results
    row, so `induce_skills.py usage` can tell whether skills were loaded.
    Safety: the usual pool run + flip confirmation, excluding every task
    shown to the inducer or used as a target."""
    from .gate import Efficacy, efficacy, fisher_greater
    from .skills import library_version
    from .stability import assess_failures
    inc = ledger.current()
    inc_rows = load_rows(ledger.rows_path(inc.version))
    _, reg = version_setup(inc)
    known = set(reg._tools)  # noqa: SLF001
    cand = library_version(inc, edits, known).model_copy(update={"version": ledger.next_version_number()})
    added = [e for e in edits if e.target in cand.skills and e.target not in inc.skills]
    if not added:
        return {"accepted": False, "reason": "no induced skill applies to the current version", "episodes": 0}
    tools = {e.source_cluster.split(":", 1)[1] for e in added if e.source_cluster.startswith("skill:")}
    evidence = {t for e in added for t in e.evidence_tasks}
    tp = skill_target_plan(cfg, base_rows, tools, evidence)
    used = 0
    if tp["missing"]:
        log(f"stability: assessing {len(tp['missing'])} uncached failures under the baseline")
        _, base_reg = version_setup(ledger.version(0))
        before = {k for k in stability_of(tp["missing"], cfg)}
        await assess_failures(tp["missing"], _paths(cfg)["stability"], store=store, provider=provider,
                              user_llm=user_llm, registry=base_reg, n=cfg.stability_n,
                              stable_max=cfg.stable_max, concurrency=cfg.concurrency, log=log)
        new = stability_of(tp["missing"], cfg)
        used += sum(rec.n for t, rec in new.items() if t not in before)
        tp = skill_target_plan(cfg, base_rows, tools, evidence)
    targets = [(r["task_id"], r["info"]["rewind"]["trace"], 0, 0) for r in tp["stable"]][:max_targets]
    pool = _pool(inc_rows, ledger.shown() | evidence | {t[0] for t in targets})
    round_no = ledger.bump_round()
    ledger.mark_shown(sorted(evidence), round_no, "skill_evidence")
    log(f"skill library: {len(added)} skills ({', '.join(e.target for e in added)}); "
        f"{len(tp['failures'])} held-out failures, {len(tp['stable'])} stable -> {len(targets)} targets; "
        f"pool {len(pool)}")
    if not targets:
        return {"accepted": False, "reason": "no stable held-out failure of the covered task types",
                "episodes": used}

    gdir = _paths(cfg)["gate"]
    null, n_null = await efficacy(inc, targets, policy=cfg.gate, provider=provider, user_llm=user_llm,
                                  store=store, log=lambda m: log("  [incumbent]" + m),
                                  rows_out=gdir / f"v{cand.version}_skills_eff_incumbent.json")
    cnd, n_cand = await efficacy(cand, targets, policy=cfg.gate, provider=provider, user_llm=user_llm,
                                 store=store, log=lambda m: log("  [skills]" + m),
                                 rows_out=gdir / f"v{cand.version}_skills_eff_candidate.json")
    used += n_null + n_cand
    p = fisher_greater(cnd.cand_passes, cnd.cand_n, null.cand_passes, null.cand_n)
    lift = cnd.cand_passes / cnd.cand_n - null.cand_passes / null.cand_n
    eff = Efficacy(tasks=[t[0] for t in targets], cand_passes=cnd.cand_passes, cand_n=cnd.cand_n,
                   null_passes=null.cand_passes, null_n=null.cand_n, p_value=p, lift=lift,
                   passed=p < cfg.gate.alpha and lift >= cfg.gate.min_lift)
    log(f"  efficacy: skills {eff.cand_passes}/{eff.cand_n} vs incumbent {eff.null_passes}/{eff.null_n}, "
        f"p = {p:.4f}")
    out = gdir / f"v{cand.version}_skills.json"
    v = await gate(cand, inc_rows, pool, split=cfg.split, out=out, provider=provider, user_llm=user_llm,
                   store=store, policy=cfg.gate, concurrency=cfg.concurrency, log=log, incumbent=inc,
                   prior_efficacy=eff)
    v = v.model_copy(update={"episodes": v.episodes + used})
    data = v.model_dump()
    data["efficacy_rows"] = [str(gdir / f"v{cand.version}_skills_eff_{w}.json") for w in ("incumbent", "candidate")]
    for e in added:
        ledger.record(round_no, inc.version, e, "gate", v.accepted, v.reason, data, cluster=e.source_cluster)
    if v.accepted:
        ledger.commit(cand, added, str(out), data)
        log(f"  committed v{cand.version}")
    return {"accepted": v.accepted, "reason": v.reason, "episodes": v.episodes,
            "version": cand.version if v.accepted else None, "skills": [e.target for e in added],
            "efficacy_rows": data["efficacy_rows"]}
