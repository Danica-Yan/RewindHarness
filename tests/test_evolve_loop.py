"""End-to-end self-improvement round, offline (no API).

A fake agent plays the real tau-bench retail tasks with real tools and the
real grader. It solves every task by its ground-truth actions EXCEPT the
"hard" ones (first required mutation is an exchange), which it refuses --
unless its system prompt contains a specific learned rule. So:

* the hard tasks form one stable, prevalent failure cluster
* a proposer that writes the right rule should pass screening AND the gate
* a rule that breaks everything must die in screening
* a rule that fixes the cluster but breaks other tasks must die at the gate
* a too-small budget must stop the round before the gate
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from rewind.env.taubench_adapter import _ensure_importable

try:
    _ensure_importable()
    _IMPORT_ERROR = None
except ImportError as e:
    _IMPORT_ERROR = e

pytestmark = pytest.mark.skipif(_IMPORT_ERROR is not None,
                                reason=f"tau_bench not importable: {_IMPORT_ERROR}")

if _IMPORT_ERROR is None:
    from rewind.env.taubench_adapter import MUTATING_TOOLS, golden_script
    from rewind.env.taubench_dialogue import STOP_TOKEN, load_task
    from rewind.evolve.gate import GatePolicy
    from rewind.evolve.ledger import EditLedger
    from rewind.evolve.loop import EvolveConfig, plan, run_round
    from rewind.evolve.prevalence import PrevalencePolicy
    from rewind.evolve.runner import load_rows, run_batch
    from rewind.evolve.screen import ScreenPolicy
    from rewind.evolve.surface import ContextVersion, Edit, leak_reasons
    from rewind.providers import LLMResponse
    from rewind.snapshot import SnapshotStore

N_TASKS = 60
MAGIC = "confirm the exchange items"
POISON = "never call any tool"
SIDE_EFFECT = "only handle exchanges"
SMALL_SIDE = "ask twice before cancelling"   # breaks a few tasks, reproducibly
NOISY_MARK = "thank the customer warmly"     # NOISY tasks fail once under it, then pass


def _first_mut(i):
    m = [a.name for a in load_task("train", i).actions if a.name in MUTATING_TOOLS]
    return m[0] if m else None


HARD = {i for i in range(N_TASKS)} if _IMPORT_ERROR else \
    {i for i in range(N_TASKS) if _first_mut(i) == "exchange_delivered_order_items"}
CANCEL3 = set() if _IMPORT_ERROR else \
    set(sorted(i for i in range(N_TASKS) if _first_mut(i) == "cancel_pending_order")[:3])


class FakeAgent:
    """`noisy`: tasks that fail their FIRST episode under a prompt containing
    NOISY_MARK and pass afterwards -- a flip that does not reproduce."""
    def __init__(self, noisy: set[int] = frozenset()):
        self.by_instruction = {load_task("train", i).instruction: i for i in range(N_TASKS)}
        self.noisy = set(noisy)
        self.episodes: dict[int, int] = {}

    async def complete(self, messages, tools, params):
        system = messages[0]["content"].lower()
        task_id = self.by_instruction[messages[1]["content"]]
        script = [r for r in golden_script(load_task("train", task_id)) if r.tool_calls]
        k = sum(1 for m in messages if m["role"] == "assistant")
        if k == 0 and NOISY_MARK in system:
            self.episodes[task_id] = self.episodes.get(task_id, 0) + 1
        noisy_fail = task_id in self.noisy and NOISY_MARK in system and self.episodes.get(task_id) == 1
        refuse = (POISON in system
                  or (task_id in HARD and MAGIC not in system)
                  or (SIDE_EFFECT in system and task_id not in HARD)
                  or (SMALL_SIDE in system and task_id in CANCEL3)
                  or noisy_fail)
        if refuse:
            return LLMResponse(text="Sorry, I cannot help with that.")
        return script[k] if k < len(script) else LLMResponse(text="All done.")


async def fake_user(messages):
    if len(messages) == 2:  # reset: answer the opener with the instruction itself
        return re.search(r"Instruction: (.*)\n", messages[0]["content"]).group(1)
    return STOP_TOKEN


class FakeProposer:
    def __init__(self, edits):
        self.edits = edits
        self.calls = 0
        self.seen_tasks = []
        self.seen_rejected = []

    async def propose(self, cluster_key, evidence, split, ctx, rejected, n):
        self.calls += 1
        self.seen_tasks += [r["task_id"] for r in evidence]
        self.seen_rejected.append(rejected)
        return [e.model_copy(update={"source_cluster": cluster_key}) for e in self.edits][:n]


def rule(target, text, trigger="the customer asks to exchange delivered items"):
    return Edit(kind="rule", target=target, op="add", text=text, trigger=trigger)


@pytest.fixture(scope="module")
def base_run(tmp_path_factory):
    """The v0 baseline, run once per module and shared read-only."""
    d = tmp_path_factory.mktemp("base")
    store = SnapshotStore(d / "snaps")
    asyncio.run(run_batch("train", list(range(N_TASKS)), 1, d / "base.json", ctx=ContextVersion(),
                          provider=FakeAgent(), user_llm=fake_user, store=store,
                          concurrency=8, log=lambda m: None))
    return d


@pytest.fixture
def setup(base_run, tmp_path):
    import shutil
    shutil.copy(base_run / "base.json", tmp_path / "base.json")
    return _setup(tmp_path, base_run)


def _setup(tmp_path, base_run=None):
    store = SnapshotStore((base_run or tmp_path) / "snaps")
    base = tmp_path / "base.json"
    if not base.exists():
        asyncio.run(run_batch("train", list(range(N_TASKS)), 1, base, ctx=ContextVersion(),
                              provider=FakeAgent(), user_llm=fake_user, store=store,
                              concurrency=8, log=lambda m: None))
    ledger = EditLedger(tmp_path / "ledger.sqlite")
    ledger.init_base(str(base))
    cfg = EvolveConfig(run_dir=str(tmp_path / "run"),
                       prevalence=PrevalencePolicy(min_distinct_tasks=4, min_share_lb=0.05),
                       stability_n=3, screen=ScreenPolicy(n=2), gate=GatePolicy(trials=1),
                       max_members_per_cluster=6, budget_episodes=200, concurrency=8)
    return store, ledger, cfg


def test_good_rule_is_committed_and_bad_ones_are_stopped_at_screening(setup):
    store, ledger, cfg = setup
    base_rows = load_rows(ledger.rows_path(0))
    assert {r["task_id"] for r in base_rows if r["reward"] < 1} == HARD

    text = plan(cfg, ledger)
    assert "exchange_delivered_order_items" in text and "worst case" in text

    prop = FakeProposer([rule("poison", f"You must {POISON}."),
                         rule("leaky", "For #W1234567 do an exchange."),
                         rule("exchange_confirm", f"Before exchanging, {MAGIC} with the user.")])
    rep = asyncio.run(run_round(cfg, ledger, provider=FakeAgent(), user_llm=fake_user, store=store,
                                proposer=prop, log=lambda m: None))
    stages = [(d["stage"], d["accepted"]) for d in ledger.decisions()]
    assert ("invalid", 0) in stages                        # leaky id refused offline
    assert stages.count(("screen", 0)) == 1                 # poison died in screening
    assert ("screen", 1) in stages and ("gate", 1) in stages
    assert rep.version_to == 1 and ledger.current().rules == {
        "exchange_confirm": "When the customer asks to exchange delivered items: "
                            f"Before exchanging, {MAGIC} with the user."}
    # the gate pool never contains a task the proposer saw or screening used
    gate_rows = load_rows(ledger.rows_path(1))
    assert not {r["task_id"] for r in gate_rows} & ledger.shown()
    assert set(prop.seen_tasks) <= ledger.shown()
    assert rep.gate["improvements"] and not rep.gate["regressions"]
    assert rep.episodes_used <= cfg.budget_episodes

    # next round: the fixed cluster is gone from the new incumbent's failures
    rep2 = asyncio.run(run_round(cfg, ledger, provider=FakeAgent(), user_llm=fake_user, store=store,
                                 proposer=FakeProposer([]), log=lambda m: None))
    assert rep2.version_to is None and "no eligible cluster" in rep2.events[0]


def test_rule_with_side_effects_is_rejected_by_the_gate(setup):
    store, ledger, cfg = setup
    prop = FakeProposer([rule("exchanges_only", f"{MAGIC.capitalize()}; {SIDE_EFFECT}.")])
    rep = asyncio.run(run_round(cfg, ledger, provider=FakeAgent(), user_llm=fake_user, store=store,
                                proposer=prop, log=lambda m: None))
    assert rep.version_to is None and ledger.current().version == 0
    assert rep.gate is not None and not rep.gate["accepted"]
    assert rep.gate["regressions"]
    # rejected ideas are fed back to the proposer next time
    assert ledger.rejected_for_prompt()[0]["target"] == "exchanges_only"


def test_budget_cap_stops_before_the_gate(setup):
    store, ledger, cfg = setup
    cfg = cfg.model_copy(update={"budget_episodes": 40})
    prop = FakeProposer([rule("exchange_confirm", f"Before exchanging, {MAGIC} with the user.")])
    rep = asyncio.run(run_round(cfg, ledger, provider=FakeAgent(), user_llm=fake_user, store=store,
                                proposer=prop, log=lambda m: None))
    assert rep.gate is None and rep.version_to is None
    assert any("NOT gated" in e for e in rep.events)
    assert rep.episodes_used <= 40

    # gate-pending finishes the job: no stability, screening or proposer redone
    from rewind.evolve.loop import gate_pending, pending_candidates
    pend = pending_candidates(ledger, cfg)
    assert len(pend) == 1 and len(pend[0]["targets"]) == 3
    shown_before = ledger.shown()
    res = asyncio.run(gate_pending(cfg.model_copy(update={"budget_episodes": 200}), ledger,
                                   provider=FakeAgent(), user_llm=fake_user, store=store,
                                   log=lambda m: None))
    assert res[0]["verdict"]["accepted"] and ledger.current().version == 1
    assert ledger.shown() == shown_before and prop.calls == 1
    assert pending_candidates(ledger, cfg) == []


def test_leak_checks_and_edit_application():
    assert leak_reasons("refund order #W2378156") == ["order id"]
    assert leak_reasons("email yusuf@example.com") == ["email"]
    assert leak_reasons("Confirm item choice before exchanging.") == []
    v1 = ContextVersion().apply(rule("r1", "Always confirm."))
    assert v1.version == 1 and "Always confirm." in v1.render_system_prompt("WIKI")
    assert v1.rules["r1"].startswith("When the customer asks to exchange")
    v2 = v1.apply(Edit(kind="rule", target="r1", op="delete"))
    assert v2.render_system_prompt("WIKI") == "WIKI"


def test_fisher_one_sided():
    from rewind.evolve.gate import fisher_greater
    assert fisher_greater(15, 15, 0, 9) < 1e-5
    assert fisher_greater(1, 5, 1, 5) > 0.5
    assert abs(fisher_greater(3, 5, 0, 5) - 0.0833) < 1e-3  # textbook value


def test_targeted_gate_skips_the_pool_when_there_is_no_effect(setup, tmp_path):
    """A no-op rule fails efficacy on the held-out targets and never spends
    the ~pool-sized run -- the expensive part of the gate."""
    from rewind.evolve.gate import gate
    from rewind.evolve.runner import load_rows as lr
    store, ledger, cfg = setup
    rows = lr(ledger.rows_path(0))
    hard_rows = [r for r in rows if r["task_id"] in HARD][:3]
    targets = [(r["task_id"], r["info"]["rewind"]["trace"], 0, 5) for r in hard_rows]
    noop = ContextVersion().apply(rule("noop", "Be polite to the customer."))
    out = tmp_path / "noop_gate.json"
    v = asyncio.run(gate(noop, rows, [r["task_id"] for r in rows if r["task_id"] not in HARD],
                         split="train", out=out, provider=FakeAgent(), user_llm=fake_user, store=store,
                         policy=GatePolicy(efficacy_n=5), targets=targets, log=lambda m: None))
    assert not v.accepted and "no proven effect" in v.reason
    assert v.episodes == 15 and not out.exists()


# --- scoped rules ---------------------------------------------------------------
def test_blanket_rules_and_single_tool_rules_are_refused_offline():
    from rewind.evolve.surface import validate_edit
    tools = {"modify_pending_order_items", "exchange_delivered_order_items"}
    blanket = Edit(kind="rule", target="b", op="add", text="Do not offer alternative variants.")
    assert "trigger" in validate_edit(blanket, ContextVersion(), known_tools=tools, max_tokens=1500)
    one_tool = rule("t", "Pass the same id to modify_pending_order_items twice.",
                    trigger="the customer wants the item unchanged")
    assert "tool_note on modify_pending_order_items" in validate_edit(
        one_tool, ContextVersion(), known_tools=tools, max_tokens=1500)
    note = Edit(kind="tool_note", target="modify_pending_order_items", op="add",
                text="Keep the current item if no new option is requested.",
                trigger="the requested options are empty")
    assert validate_edit(note, ContextVersion(), known_tools=tools, max_tokens=1500) is None
    v = ContextVersion().apply(note, known_tools=tools)
    assert v.tool_notes["modify_pending_order_items"].startswith("When the requested options are empty:")


# --- flip confirmation ---------------------------------------------------------------
def _gate_direct(setup, cand_text, *, noisy=frozenset(), policy=None):
    from rewind.evolve.gate import Efficacy, gate
    store, ledger, cfg = setup
    rows = load_rows(ledger.rows_path(0))
    pool = sorted(t for t in range(N_TASKS) if t not in set(sorted(HARD)[:6]))  # A/B stand-ins removed
    cand = ContextVersion().apply(rule("r", cand_text))
    eff = Efficacy(tasks=[1], cand_passes=5, cand_n=5, null_passes=0, null_n=5, p_value=0.004,
                   lift=1.0, passed=True)
    out = Path(cfg.run_dir) / "gate" / "direct.json"
    v = asyncio.run(gate(cand, rows, pool, split="train", out=out, provider=FakeAgent(noisy),
                         user_llm=fake_user, store=store, policy=policy or GatePolicy(),
                         incumbent=ContextVersion(), prior_efficacy=eff, log=lambda m: None))
    return v, ledger, cfg, rows


def test_noise_flips_are_not_confirmed_and_do_not_block(setup):
    """Three tasks 'break' once under the candidate by chance; re-runs pass,
    so nothing is confirmed and the (helpful) edit is accepted."""
    noisy = set(sorted(set(range(N_TASKS)) - HARD)[:3])
    v, *_ = _gate_direct(setup, f"Before exchanging, {MAGIC} with the user; {NOISY_MARK}.", noisy=noisy)
    assert set(v.regressions) == noisy
    assert v.confirmed_regressions == [] and v.accepted
    assert v.confirm_episodes == (len(v.regressions) + len(v.improvements)) * 2 * 2


def test_small_real_breakage_is_confirmed_even_when_the_pool_looks_fine(setup):
    """Fixes the hard tasks AND reproducibly breaks 3 cancellations: the pool
    CI alone is positive, the confirmed regressions reject it."""
    v, *_ = _gate_direct(setup, f"Before exchanging, {MAGIC} with the user; {SMALL_SIDE}.")
    assert v.ci[1] > 0 and v.delta > 0
    assert set(v.confirmed_regressions) == CANCEL3 and not v.accepted
    assert "confirmed regressions" in v.reason


def test_old_margin_rule_rejects_noise_and_regate_overturns_it(setup):
    """Recorded the old way (drop margin, no confirmation) a noisy-but-harmless
    edit is rejected; regate reuses its episodes, confirms nothing, accepts."""
    from rewind.evolve.loop import _record_gate, regate, regate_candidates
    store, ledger, cfg = setup
    noisy = set(sorted(set(range(N_TASKS)) - HARD)[:2])
    harmless = f"Before exchanging, {NOISY_MARK}."   # fixes nothing; 2 tasks flip once by chance
    old = GatePolicy(confirm_k=0, max_drop=0.02)
    v_old, ledger, cfg, rows = _gate_direct(setup, harmless, noisy=noisy, policy=old)
    assert not v_old.accepted and "exceeds" in v_old.reason and v_old.ci[1] >= 0
    e = rule("r", harmless)
    _record_gate(ledger, 1, ContextVersion(), e, v_old, rows, "train")
    assert len(regate_candidates(ledger)) == 1
    agent = FakeAgent(noisy)
    agent.episodes = {t: 1 for t in noisy}  # their one noisy episode already happened
    res = asyncio.run(regate(cfg, ledger, provider=agent, user_llm=fake_user, store=store,
                             log=lambda m: None))
    assert res and res[0]["verdict"]["accepted"]
    assert res[0]["verdict"]["confirmed_regressions"] == []
    assert ledger.current().version == 1 and regate_candidates(ledger) == []


# --- rejection feedback + retry -------------------------------------------------------
def test_rejection_feedback_reaches_the_proposer_on_retry(setup):
    from rewind.evolve.loop import retry_candidates, retry_cluster
    store, ledger, cfg = setup
    bad = FakeProposer([rule("exchanges_only", f"{MAGIC.capitalize()}; {SIDE_EFFECT}.")])
    rep = asyncio.run(run_round(cfg, ledger, provider=FakeAgent(), user_llm=fake_user, store=store,
                                proposer=bad, log=lambda m: None))
    assert rep.gate is not None and not rep.gate["accepted"]
    fb = ledger.rejected_for_prompt(rep.clusters_considered[0])[0]["feedback"]
    assert "broke" in fb and "before (passed)" in fb
    cands = retry_candidates(ledger, cfg)
    assert len(cands) == 1 and cands[0]["attempts"] == 1
    shown_before = ledger.shown()
    good = FakeProposer([rule("exchange_confirm", f"Before exchanging, {MAGIC} with the user.")])
    rep2 = asyncio.run(retry_cluster(cfg, ledger, provider=FakeAgent(), user_llm=fake_user, store=store,
                                     proposer=good, log=lambda m: None))
    seen = good.seen_rejected[0]
    assert seen[0]["target"] == "exchanges_only" and "broke" in seen[0]["feedback"]
    assert rep2.version_to == 1 and ledger.shown() == shown_before  # no new tasks spent
