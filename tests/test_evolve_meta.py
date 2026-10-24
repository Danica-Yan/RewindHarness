"""Regression memory, consolidation by ablation, and the meta level
(proposer lessons). Fake agent / fake LLMs, no API."""
from __future__ import annotations

import asyncio
import json

import pytest

from tests.test_evolve_loop import (  # noqa: F401  (base_run is a fixture)
    _IMPORT_ERROR, HARD, MAGIC, POISON, FakeAgent, FakeProposer, base_run, fake_user, rule, setup,
)

pytestmark = pytest.mark.skipif(_IMPORT_ERROR is not None, reason=f"tau_bench not importable: {_IMPORT_ERROR}")

GOOD = f"Before exchanging, {MAGIC} with the user."


def _commit_good_rule(store, ledger, cfg):
    from rewind.evolve.loop import run_round
    rep = asyncio.run(run_round(cfg, ledger, provider=FakeAgent(), user_llm=fake_user, store=store,
                                proposer=FakeProposer([rule("exchange_confirm", GOOD)]), log=lambda m: None))
    assert rep.version_to == 1
    return rep


def test_accepted_fixes_are_remembered_and_rechecked_by_later_gates(setup):
    from rewind.evolve.gate import Efficacy, gate
    from rewind.evolve.runner import load_rows
    from rewind.evolve.surface import Edit
    store, ledger, cfg = setup
    _commit_good_rule(store, ledger, cfg)
    suite = ledger.regression_suite()
    assert suite and {t for t, _ in suite} <= HARD          # the held-out exchange failures it fixed
    # a later edit that rewrites the rule and drops the key phrase: passes its own (faked)
    # efficacy and a pool of unrelated tasks, but the remembered fixes catch it
    inc = ledger.current()
    bad = inc.apply(Edit(kind="rule", target="exchange_confirm", op="replace", text="Be polite.",
                         trigger="the customer asks to exchange delivered items")).model_copy(update={"version": 9})
    rows = load_rows(ledger.rows_path(inc.version))
    pool = sorted({r["task_id"] for r in rows} - HARD - ledger.shown())[:6]
    eff = Efficacy(tasks=[], cand_passes=5, cand_n=5, null_passes=0, null_n=5, p_value=0.004, lift=1.0, passed=True)
    v = asyncio.run(gate(bad, rows, pool, split="train", out=cfg.run_dir + "/gate/bad.json", provider=FakeAgent(),
                         user_llm=fake_user, store=store, policy=cfg.gate, incumbent=inc, prior_efficacy=eff,
                         regression_suite=suite, log=lambda m: None))
    assert not v.accepted and v.reason.startswith("breaks remembered fixes")
    assert set(v.regression_check["broken"]) == {t for t, _ in suite}
    # rollback forgets the fixes of the versions rolled back
    ledger.rollback(0)
    assert ledger.regression_suite() == []


def test_backfill_for_versions_committed_before_memory_existed(setup):
    from rewind.evolve.loop import backfill_memory
    store, ledger, cfg = setup
    _commit_good_rule(store, ledger, cfg)
    n = len(ledger.regression_suite())
    ledger.db.execute("DELETE FROM regressions")
    assert backfill_memory(ledger, cfg) == n and backfill_memory(ledger, cfg) == 0


def test_consolidation_keeps_a_rule_that_earns_its_place_and_drops_one_that_does_not(setup):
    from rewind.evolve import consolidate as consol
    from rewind.evolve.surface import Edit
    store, ledger, cfg = setup
    cfg.gate.efficacy_n = 2
    cfg.gate.regression_k = 2
    _commit_good_rule(store, ledger, cfg)
    # a no-op rule committed with remembered "fixes" that pass anyway
    inc = ledger.current()
    noop = Edit(kind="rule", target="be_kind", op="add", text="Thank the customer at the end.",
                trigger="the conversation is about to end")
    v2 = inc.apply(noop).model_copy(update={"version": ledger.next_version_number(), "parent": inc.version})
    ledger.commit(v2, [noop], ledger.rows_path(inc.version), {})
    from rewind.evolve.loop import trace_lookup
    traces = trace_lookup(ledger, cfg)
    easy = [t for t in sorted(traces) if t not in HARD][:2]
    ledger.remember([(t, traces[t]) for t in easy], v2.version, "be_kind")

    plan = {i["target"]: i for i in consol.plan(ledger, cfg.gate)}
    assert plan["be_kind"]["evidence"] == easy and plan["exchange_confirm"]["evidence"]
    res = {r["target"]: r for r in asyncio.run(consol.ablate(
        ledger, cfg.gate, provider=FakeAgent(), user_llm=fake_user, store=store, log=lambda m: None))}
    assert not res["exchange_confirm"]["removed"] and "still earns its place" in res["exchange_confirm"]["reason"]
    assert res["be_kind"]["removed"]
    cur = ledger.current()
    assert "be_kind" not in cur.rules and "exchange_confirm" in cur.rules
    # ablation outcomes are not shown to the proposer as rejected ideas
    assert all(r["target"] != "exchange_confirm" for r in ledger.rejected_for_prompt())


def test_lessons_are_validated():
    from rewind.evolve.meta import validate_lessons
    good = "Prefer a tool_note when the failure is about one tool."
    ok, note = validate_lessons([good])
    assert ok == [good] and note is None
    # mentioning what happened at a stage is fine; being about beating the tests is not
    stage = "When a rule passes screening but harms unseen tasks, narrow its trigger to the failing situation."
    kept, note = validate_lessons([stage, "Order #W1234567 needs care.",
                                   "Write edits that pass the gate's Fisher test.", good])
    assert kept == [stage, good]
    assert "case-specific" in note and "beating the evaluation" in note
    none, note = validate_lessons(["Tune wording until the p-value is small."])
    assert none == [] and note.startswith("all lessons refused")
    kept, note = validate_lessons([f"Lesson number {i} about scope." for i in range(10)])
    assert len(kept) == 8 and "over the limit" in note


def test_meta_eval_adopts_lessons_only_when_they_win(setup):
    """The proposer LLM writes a poisonous rule unless its lessons say to
    name the confirmation step; reflect proposes that lesson; meta-eval
    compares old vs new lessons on the recorded cluster and adopts them."""
    from rewind.evolve import meta
    from rewind.evolve.loop import run_round
    store, ledger, cfg = setup
    rep = asyncio.run(run_round(cfg, ledger, provider=FakeAgent(), user_llm=fake_user, store=store,
                                proposer=FakeProposer([rule("poison", f"You must {POISON}.")]), log=lambda m: None))
    assert rep.version_to is None
    clusters = meta.meta_clusters(ledger, cfg)
    assert clusters and clusters[0]["B"]

    async def reflect_llm(messages):
        assert POISON in messages[1]["content"] and "screen stopped" in messages[1]["content"]
        return json.dumps({"lessons": ["For exchange failures, ask the agent to confirm the exchange items."],
                           "rationale": "the only proposal stopped the agent from acting"})
    new, why, note, raw = asyncio.run(meta.reflect(ledger, reflect_llm))
    assert note is None and new and why and raw

    async def proposer_llm(messages):
        system = messages[0]["content"]
        good = "confirm the exchange items" in system.split("Lessons from earlier rounds")[-1] \
            if "Lessons from earlier rounds" in system else False
        text = GOOD if good else f"You must {POISON}."
        return json.dumps({"edits": [{"kind": "rule", "target": "x_good" if good else "x_bad", "op": "add",
                                      "trigger": "the customer asks to exchange delivered items",
                                      "text": text}]})
    res = asyncio.run(meta.meta_eval(cfg, ledger, new, clusters[:1], llm=proposer_llm, tool_names=[],
                                     provider=FakeAgent(), user_llm=fake_user, store=store, log=lambda m: None))
    assert res["adopted"], res
    assert meta.lessons(ledger) == new and meta.history(ledger)[-1]["lessons"] == new
    stages = [d["stage"] for d in ledger.decisions()]
    assert "meta" in stages and "meta_screen" in stages

    # a lesson set that does no better is not adopted
    worse = ["Keep rules short."]
    res2 = asyncio.run(meta.meta_eval(cfg, ledger, worse, clusters[:1], llm=proposer_llm, tool_names=[],
                                      provider=FakeAgent(), user_llm=fake_user, store=store, log=lambda m: None))
    assert not res2["adopted"] and meta.lessons(ledger) == new
