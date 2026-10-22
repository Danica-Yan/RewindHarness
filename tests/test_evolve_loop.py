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
import json
import re

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


def _first_mut(i):
    m = [a.name for a in load_task("train", i).actions if a.name in MUTATING_TOOLS]
    return m[0] if m else None


HARD = {i for i in range(N_TASKS)} if _IMPORT_ERROR else \
    {i for i in range(N_TASKS) if _first_mut(i) == "exchange_delivered_order_items"}


class FakeAgent:
    def __init__(self):
        self.by_instruction = {load_task("train", i).instruction: i for i in range(N_TASKS)}

    async def complete(self, messages, tools, params):
        system = messages[0]["content"].lower()
        task_id = self.by_instruction[messages[1]["content"]]
        script = [r for r in golden_script(load_task("train", task_id)) if r.tool_calls]
        k = sum(1 for m in messages if m["role"] == "assistant")
        refuse = (POISON in system
                  or (task_id in HARD and MAGIC not in system)
                  or (SIDE_EFFECT in system and task_id not in HARD))
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

    async def propose(self, cluster_key, evidence, split, ctx, rejected, n):
        self.calls += 1
        self.seen_tasks += [r["task_id"] for r in evidence]
        return [e.model_copy(update={"source_cluster": cluster_key}) for e in self.edits][:n]


def rule(target, text):
    return Edit(kind="rule", target=target, op="add", text=text)


def _setup(tmp_path):
    store = SnapshotStore(tmp_path / "snaps")
    base = tmp_path / "base.json"
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


def test_good_rule_is_committed_and_bad_ones_are_stopped_at_screening(tmp_path):
    store, ledger, cfg = _setup(tmp_path)
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
        "exchange_confirm": f"Before exchanging, {MAGIC} with the user."}
    # the gate pool never contains a task the proposer saw or screening used
    gate_rows = load_rows(ledger.rows_path(1))
    assert not {r["task_id"] for r in gate_rows} & ledger.shown()
    assert set(prop.seen_tasks) <= ledger.shown()
    assert rep.gate["improvements"] and not rep.gate["regressions"]
    assert rep.episodes_used <= cfg.budget_episodes
    # SFT file: only train successes, tagged with the version they ran under
    sft = [json.loads(line) for line in
           (tmp_path / "run" / "sft" / "train_success.jsonl").read_text().splitlines()]
    assert sft and all(d["split"] == "train" and d["reward"] == 1.0 for d in sft)
    assert {d["ctx_version"] for d in sft} == {0, 1}

    # next round: the fixed cluster is gone from the new incumbent's failures
    rep2 = asyncio.run(run_round(cfg, ledger, provider=FakeAgent(), user_llm=fake_user, store=store,
                                 proposer=FakeProposer([]), log=lambda m: None))
    assert rep2.version_to is None and "no eligible cluster" in rep2.events[0]


def test_rule_with_side_effects_is_rejected_by_the_gate(tmp_path):
    store, ledger, cfg = _setup(tmp_path)
    prop = FakeProposer([rule("exchanges_only", f"{MAGIC.capitalize()}; {SIDE_EFFECT}.")])
    rep = asyncio.run(run_round(cfg, ledger, provider=FakeAgent(), user_llm=fake_user, store=store,
                                proposer=prop, log=lambda m: None))
    assert rep.version_to is None and ledger.current().version == 0
    assert rep.gate is not None and not rep.gate["accepted"]
    assert rep.gate["regressions"]
    # rejected ideas are fed back to the proposer next time
    assert ledger.rejected_for_prompt()[0]["target"] == "exchanges_only"


def test_budget_cap_stops_before_the_gate(tmp_path):
    store, ledger, cfg = _setup(tmp_path)
    cfg = cfg.model_copy(update={"budget_episodes": 40})
    prop = FakeProposer([rule("exchange_confirm", f"Before exchanging, {MAGIC} with the user.")])
    rep = asyncio.run(run_round(cfg, ledger, provider=FakeAgent(), user_llm=fake_user, store=store,
                                proposer=prop, log=lambda m: None))
    assert rep.gate is None and rep.version_to is None
    assert any("NOT gated" in e for e in rep.events)
    assert rep.episodes_used <= 40


def test_leak_checks_and_edit_application():
    assert leak_reasons("refund order #W2378156") == ["order id"]
    assert leak_reasons("email yusuf@example.com") == ["email"]
    assert leak_reasons("Confirm item choice before exchanging.") == []
    v1 = ContextVersion().apply(rule("r1", "Always confirm."))
    assert v1.version == 1 and "Always confirm." in v1.render_system_prompt("WIKI")
    v2 = v1.apply(Edit(kind="rule", target="r1", op="delete"))
    assert v2.render_system_prompt("WIKI") == "WIKI"


def test_fisher_one_sided():
    from rewind.evolve.gate import fisher_greater
    assert fisher_greater(15, 15, 0, 9) < 1e-5
    assert fisher_greater(1, 5, 1, 5) > 0.5
    assert abs(fisher_greater(3, 5, 0, 5) - 0.0833) < 1e-3  # textbook value


def test_targeted_gate_skips_the_pool_when_there_is_no_effect(tmp_path):
    """A no-op rule fails efficacy on the held-out targets and never spends
    the ~pool-sized run -- the expensive part of the gate."""
    from rewind.evolve.gate import gate
    from rewind.evolve.runner import load_rows as lr
    store, ledger, cfg = _setup(tmp_path)
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
