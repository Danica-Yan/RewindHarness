"""Skills: the surface (versioned, rendered as an index + load_skill tool),
loading inside real tau-bench episodes, induction from successes, usage
stats, and gating a library as one version. No API: scripted / fake agents.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from rewind.evolve.surface import ContextVersion, Edit, EditError, validate_edit
from rewind.skills import LOAD_SKILL, MAX_SKILLS, SKILLS_HEADER, Skill
from rewind.tools import ToolRegistry

EXCHANGE = {"description": "The customer wants to exchange items of a delivered order for other variants.",
            "steps": ["Authenticate the customer by email, or by name and zip code.",
                      "Look up the order and check that its status is delivered.",
                      "Ask for ALL items to exchange at once, and look up each product's variants.",
                      "List the exchange and get an explicit yes.",
                      "Call exchange_delivered_order_items once with every item."],
            "tools": ["find_user_id_by_email", "get_order_details", "get_product_details",
                      "exchange_delivered_order_items"]}
TOOLS = {"find_user_id_by_email", "find_user_id_by_name_zip", "get_order_details", "get_product_details",
         "exchange_delivered_order_items", "cancel_pending_order"}
OK = dict(known_tools=TOOLS, max_tokens=2000)


def skill_edit(name="exchange_items", op="add", **over) -> Edit:
    return Edit(kind="skill", target=name, op=op, text=json.dumps({**EXCHANGE, **over}),
                source_cluster="skill:exchange_delivered_order_items")


def test_skill_model_validation():
    Skill(name="abc", **EXCHANGE)
    with pytest.raises(ValueError, match="snake_case"):
        Skill(name="Exchange Items", **EXCHANGE)
    with pytest.raises(ValueError, match="2-12 steps"):
        Skill(name="abc", **{**EXCHANGE, "steps": ["only one"]})
    with pytest.raises(ValueError, match="one line"):
        Skill(name="abc", **{**EXCHANGE, "description": "short"})


def test_skill_edits_render_an_index_and_a_load_tool():
    v0 = ContextVersion(rules={"r": "When x: y."})
    v1 = v0.apply(skill_edit(), known_tools=TOOLS)
    prompt = v1.render_system_prompt("WIKI")
    assert prompt.startswith("WIKI") and SKILLS_HEADER in prompt and "When x: y." in prompt
    assert "- exchange_items: The customer wants to exchange" in prompt
    assert "Call exchange_delivered_order_items once" not in prompt      # steps load on demand
    reg = v1.patch_registry(ToolRegistry())
    tool = reg._tools[LOAD_SKILL]  # noqa: SLF001
    assert tool.input_schema["properties"]["name"]["enum"] == ["exchange_items"] and not tool.mutating
    out = reg.execute(None, LOAD_SKILL, {"name": "exchange_items"}, "a1")
    assert out.ok and "5. Call exchange_delivered_order_items once with every item." in out.content
    bad = reg.execute(None, LOAD_SKILL, {"name": "nope"}, "a2")
    assert not bad.ok and "available: exchange_items" in bad.content
    # versions: replace, delete, carry-over of other surfaces, unchanged fingerprint without skills
    v2 = v1.apply(skill_edit(op="replace", steps=["Step one here.", "Step two here."]), known_tools=TOOLS)
    assert v2.skills["exchange_items"]["steps"] == ["Step one here.", "Step two here."] and v2.rules == v0.rules
    v3 = v2.apply(Edit(kind="rule", target="r2", op="add", text="z", trigger="the customer asks a"),
                  known_tools=TOOLS)
    assert v3.skills == v2.skills                                        # rule edits keep skills
    assert v2.apply(Edit(kind="skill", target="exchange_items", op="delete")).skills == {}
    assert ContextVersion(rules={"r": "x"}).fingerprint() == ContextVersion(rules={"r": "x"}).fingerprint()
    assert v1.fingerprint() != v0.fingerprint()
    assert ContextVersion.model_validate_json(v1.model_dump_json()).skills == v1.skills


def test_skill_edit_validation():
    v0 = ContextVersion()
    assert validate_edit(skill_edit(), v0, **OK) is None
    assert "unknown tool" in validate_edit(skill_edit(tools=["teleport"]), v0, **OK)
    assert "case-specific" in validate_edit(
        skill_edit(steps=["Look up order #W1234567.", "Then exchange it."]), v0, **OK)
    assert "not JSON" in validate_edit(Edit(kind="skill", target="abc", op="add", text="steps: ..."), v0, **OK)
    full = ContextVersion(skills={f"s{i:02d}x": Skill(name=f"s{i:02d}x", **EXCHANGE).model_dump()
                                  for i in range(MAX_SKILLS)})
    with pytest.raises(EditError, match="library is full"):
        full.apply(skill_edit(), known_tools=TOOLS)


# ---------------------------------------------------------------------------
# with tau-bench (scripted)
# ---------------------------------------------------------------------------
try:
    from rewind.env.taubench_adapter import _ensure_importable
    _ensure_importable()
    _TAU = None
except ImportError as e:  # pragma: no cover
    _TAU = e
needs_tau = pytest.mark.skipif(_TAU is not None, reason=f"tau_bench not importable: {_TAU}")


@needs_tau
def test_agent_loads_a_skill_mid_episode_and_reward_is_unaffected(tmp_path):
    from rewind.env.taubench_dialogue import (
        STOP_TOKEN, build_dialogue_registry, faithful_config, result_record, run_episode, wiki,
    )
    from rewind.evolve.skills import loaded_skills, skill_usage
    from rewind.providers import LLMResponse, ScriptedProvider, ToolCall
    from rewind.snapshot import SnapshotStore
    from tests.test_policy import _two_write_task
    from tests.test_taubench_dialogue import golden_dialogue, scripted_user
    idx = _two_write_task()
    v = ContextVersion().apply(skill_edit(), known_tools=TOOLS)
    script, _ = golden_dialogue("train", idx)
    script = [script[0], LLMResponse(tool_calls=[ToolCall(id="", name=LOAD_SKILL, args={"name": "exchange_items"})]),
              *script[1:]]
    ep = asyncio.run(run_episode(
        "train", idx, provider=ScriptedProvider(script),
        user_llm=scripted_user(["hi", "Thanks, please go ahead.", STOP_TOKEN]), store=SnapshotStore(),
        cfg=faithful_config(system_prompt=v.render_system_prompt(wiki())),
        registry=v.patch_registry(build_dialogue_registry())))
    assert ep.reward.reward == 1.0                       # a read-only extra call changes nothing
    call = next(e.data for e in ep.events if e.type == "tool_call" and e.data["name"] == LOAD_SKILL)
    assert call["ok"] and "1. Authenticate the customer" in call["content"]
    row = result_record(ep.env, ep.result, 0, None, ep.events)
    assert loaded_skills(row["traj"]) == ["exchange_items"]
    [u] = skill_usage([row, dict(row, trial=1, reward=0.0, traj=[])], "train",
                      {"exchange_items": "exchange_delivered_order_items"})
    assert (u.relevant, u.loaded, u.pass_loaded, u.pass_not_loaded) == (2, 1, 1.0, 0.0)


@needs_tau
def test_groups_and_induction_from_passing_episodes():
    from rewind.env.taubench_dialogue import load_task
    from rewind.evolve.skills import induce, library_version, skill_groups, task_writes
    rows = []
    for t in range(40):
        rows.append({"task_id": t, "trial": 0, "reward": 1.0 if t % 3 else 0.0,
                     "traj": [{"role": "system", "content": "w"}] * (1 + t % 5), "info": {}})
    groups = skill_groups(rows, "train", min_passing=3, k=2)
    assert groups
    for g in groups:
        assert all(rows[i]["reward"] == 1.0 and g.tool in task_writes("train", rows[i]["task_id"])
                   for i in g.passing)
        assert len(g.examples) <= 2 and len({rows[i]["task_id"] for i in g.examples}) == len(g.examples)
        assert all(load_task("train", t) for t in g.task_ids)

    async def llm(messages):
        assert "Successful conversation 1" in messages[1]["content"]
        assert "=== The agent's policy ===" in messages[1]["content"]
        assert "Exchange or modify order tools can only be called once" in messages[1]["content"]
        tool = messages[1]["content"].split("`")[1]
        return json.dumps({"name": f"do_{tool}"[:40], "description": f"The customer needs {tool} done for them.",
                           "steps": ["Authenticate the customer first.", f"Call {tool} after a yes."],
                           "tools": [tool]})
    res = asyncio.run(induce(groups, rows, "train", ContextVersion(), llm, sorted(TOOLS)))
    edits = [e for _, e, _ in res]
    assert all(e and e.kind == "skill" and e.source_cluster.startswith("skill:") for e in edits)
    from rewind.env.taubench_adapter import MUTATING_TOOLS
    lib = library_version(ContextVersion(), edits, set(MUTATING_TOOLS) | TOOLS)
    assert len(lib.skills) == len(edits) and lib.version == 1


@needs_tau
def test_skill_library_gate_end_to_end(tmp_path):
    """Fake agent: refuses exchanges unless its prompt says MAGIC. A skill
    whose description says it fixes them; the library is gated as one
    version against the incumbent's fresh null rate and committed."""
    from tests.test_evolve_loop import MAGIC, FakeAgent, _setup, fake_user
    from rewind.evolve.loop import gate_skill_library
    from rewind.evolve.runner import load_rows
    store, ledger, cfg = _setup(tmp_path)
    cfg = cfg.model_copy(update={"budget_episodes": 1000})
    cfg.gate.efficacy_n = 2
    base_rows = load_rows(ledger.rows_path(0))
    good = skill_edit(description=f"Exchanges of delivered items: always {MAGIC} with the user first.")
    good = good.model_copy(update={"evidence_tasks": [1, 2]})
    res = asyncio.run(gate_skill_library(cfg, ledger, [good], base_rows, provider=FakeAgent(),
                                         user_llm=fake_user, store=store, log=lambda m: None))
    assert res["accepted"], res
    # targets were stability-checked under the baseline first, and every efficacy fork was saved
    from rewind.evolve.stability import load_records
    assert load_records(tmp_path / "run" / "stability.json")
    inc_rows, cand_rows = (load_rows(p) for p in res["efficacy_rows"])
    assert len(inc_rows) == len(cand_rows) > 0 and all("traj" in r for r in cand_rows)
    assert sum(r["reward"] for r in cand_rows) > sum(r["reward"] for r in inc_rows)
    cur = ledger.current()
    assert cur.version == 1 and list(cur.skills) == ["exchange_items"]
    assert {1, 2} <= ledger.shown()
    gate = [d for d in ledger.decisions() if d["stage"] == "gate"]
    assert gate and gate[0]["accepted"] and "vs null 0/" in gate[0]["reason"]

    # a skill that changes nothing has no effect on the targets -> rejected, nothing committed
    useless = skill_edit(name="exchange_items_b", description="Exchanges of delivered items, done carefully.")
    store2, ledger2, cfg2 = _setup(tmp_path / "b")
    cfg2.gate.efficacy_n = 2
    res2 = asyncio.run(gate_skill_library(cfg2, ledger2, [useless], load_rows(ledger2.rows_path(0)),
                                          provider=FakeAgent(), user_llm=fake_user, store=store2,
                                          log=lambda m: None))
    assert not res2["accepted"] and "no proven effect" in res2["reason"]
    assert ledger2.current().version == 0


@needs_tau
def test_contrast_groups_show_only_a_half_failures_and_targets_must_be_stable(tmp_path):
    from rewind.evolve.loop import EvolveConfig, skill_target_plan
    from rewind.evolve.skills import failure_half, induce_messages, parse_skill_edit, skill_groups
    from rewind.evolve.stability import StabilityRecord, _key, save_records
    assert failure_half(7) == failure_half(7) and {failure_half(t) for t in range(40)} == {"A", "B"}
    rows = [{"task_id": t, "trial": 0, "reward": 1.0 if t % 3 else 0.0, "traj": [],
             "info": {"rewind": {"trace": f"/x/{t}.jsonl", "ctx_version": 0}}} for t in range(60)]
    plain = skill_groups(rows, "train", min_passing=3, k=2)
    assert all(not g.failing_examples for g in plain)
    groups = skill_groups(rows, "train", min_passing=3, k=2, contrast=True)
    shown = [rows[i]["task_id"] for g in groups for i in g.failing_examples]
    assert shown and all(failure_half(t) == "A" and rows[t]["reward"] == 0.0 for t in shown)
    g = next(g for g in groups if g.failing_examples)
    msgs = induce_messages(g, rows, "train", ContextVersion(), sorted(TOOLS), policy="POLICY")
    assert "FAILED conversation 1" in msgs[1]["content"] and "Correct actions would have been" in msgs[1]["content"]
    assert "restates it adds nothing" in msgs[0]["content"]
    e = parse_skill_edit(json.dumps({"name": "abc", **EXCHANGE}), g, rows)
    assert set(e.evidence_tasks) >= {rows[i]["task_id"] for i in g.failing_examples}

    # target plan: unseen failures only, and only those the stability data calls stable
    cfg = EvolveConfig(run_dir=str(tmp_path))
    tools = {g.tool}
    fails = [r for r in rows if r["reward"] == 0.0]
    recs = {}
    for k, r in enumerate(fails):
        passes = 0 if k % 2 == 0 else 4
        recs[_key(r["task_id"], 0, 0)] = StabilityRecord(task_id=r["task_id"], trial=0, trace="t", ctx_version=0,
                                                          n=5, passes=passes, label="stable" if passes <= 1 else "flaky")
    save_records(tmp_path / "stability.json", recs)
    tp = skill_target_plan(cfg, rows, tools, exclude=set(shown))
    assert tp["failures"] and not {r["task_id"] for r in tp["failures"]} & set(shown)
    assert all(tp["stability"][r["task_id"]].label == "stable" for r in tp["stable"])
    assert len(tp["stable"]) < len(tp["failures"]) and not tp["missing"]
