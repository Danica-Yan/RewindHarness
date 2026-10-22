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
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from .gate import GatePolicy, gate
from .ledger import EditLedger
from .prevalence import PrevalencePolicy, clusters, describe, eligible, evidence_rows
from .runner import load_rows, version_setup
from .screen import ScreenPolicy, screen
from .stability import assess_failures
from .surface import validate_edit


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
    g = pool_after * cfg.gate.trials
    eff = 0
    if cfg.gate.mode == "targeted" and elig:
        eff = (min(len(elig[0].tasks), cfg.max_members_per_cluster) // 2) * cfg.gate.efficacy_n
        lines.append(f"gate efficacy (held-out targets, runs first): <= {eff} forks")
    worst += g + eff
    lines.append(f"gate pool (only if efficacy passes): <= {g} episodes "
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
                      cluster=cluster.key)

        a_rows = evidence_rows(pool_rows, [r.task_id for r in a])
        edits = await proposer.propose(cluster.key, a_rows, cfg.split, inc,
                                       ledger.rejected_for_prompt(), cfg.n_candidates)
        rep.proposer_calls += 1
        tools = set(inc_registry._tools)  # noqa: SLF001
        note(f"  proposer returned {len(edits)} candidate edit(s)")
        b_tasks = [(r.task_id, r.trace, r.null_rate) for r in b]
        stop_round = False
        for e in edits:
            if e.edit_id in ledger.tried_edit_ids():
                continue
            why = validate_edit(e, inc, known_tools=tools, max_tokens=cfg.gate.max_ctx_tokens)
            if why:
                ledger.record(round_no, inc.version, e, "invalid", False, why)
                note(f"  [invalid] {e.kind} {e.target}: {why}")
                continue
            cost = len(b_tasks) * cfg.screen.n
            if not budget.can(cost):
                note(f"budget: screening needs <= {cost}, {budget.left} left -- stop")
                break
            cand = inc.apply(e, known_tools=tools).model_copy(
                update={"version": ledger.next_version_number(), "parent": inc.version})
            cand_cfg, cand_reg = version_setup(cand)
            note(f"  screening {e.kind} {e.target}: {e.text!r}")
            sr = await screen([e.edit_id], b_tasks, system_prompt=cand_cfg.system_prompt,
                              registry=cand_reg, store=store, provider=provider, user_llm=user_llm,
                              policy=cfg.screen, log=log)
            budget.spend(sr.forks)
            ledger.record(round_no, inc.version, e, "screen", sr.passed, sr.reason, sr.model_dump())
            note(f"  screen: {'PASS' if sr.passed else 'fail'} -- {sr.reason}")
            if not sr.passed:
                continue

            pool = _pool(rows, ledger.shown())
            targets = [(r.task_id, r.trace, r.passes, r.n) for r in b]
            cost = len(pool) * cfg.gate.trials + (
                len(targets) * cfg.gate.efficacy_n if cfg.gate.mode == "targeted" else 0)
            if not budget.can(cost):
                note(f"budget: gate needs <= {cost} episodes, {budget.left} left -- "
                     f"edit passed screening but is NOT gated this round; "
                     f"run `run_evolve.py gate-pending` to gate it without redoing anything")
                stop_round = True
                break
            out = paths["gate"] / f"v{cand.version}_{e.edit_id}.json"
            v = await gate(cand, rows, pool, split=cfg.split, out=out, provider=provider,
                           user_llm=user_llm, store=store, policy=cfg.gate, targets=targets,
                           concurrency=cfg.concurrency, log=log)
            budget.spend(v.episodes)
            rep.gate = v.model_dump()
            ledger.record(round_no, inc.version, e, "gate", v.accepted, v.reason, v.model_dump())
            note(f"  gate: {'ACCEPT' if v.accepted else 'reject'} -- {v.reason}")
            if v.accepted:
                ledger.commit(cand, [e], str(out), v.model_dump())
                rep.version_to = cand.version
                rep.committed_edit = e.model_dump()
                note(f"  committed v{cand.version}")
            break  # at most one gate per round
        if rep.gate is not None or stop_round:
            break  # a gated or gate-pending candidate ends the round

    rep.episodes_used = budget.used
    _save(paths, rep)
    return rep


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
        cost = len(pool) * cfg.gate.trials + len(p["targets"]) * cfg.gate.efficacy_n
        if not budget.can(cost):
            log(f"budget: gate needs <= {cost}, {budget.left} left -- stop")
            break
        _, reg = version_setup(inc)
        cand = inc.apply(e, known_tools=set(reg._tools)).model_copy(  # noqa: SLF001
            update={"version": ledger.next_version_number(), "parent": inc.version})
        log(f"gating {e.kind} {e.target}: {e.text!r} on {len(p['targets'])} targets + {len(pool)} pool tasks")
        out = _paths(cfg)["gate"] / f"v{cand.version}_{e.edit_id}.json"
        v = await gate(cand, rows, pool, split=cfg.split, out=out, provider=provider, user_llm=user_llm,
                       store=store, policy=cfg.gate, targets=p["targets"],
                       concurrency=cfg.concurrency, log=log)
        budget.spend(v.episodes)
        ledger.record(p["round"], inc.version, e, "gate", v.accepted, v.reason, v.model_dump())
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
