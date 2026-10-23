"""Policy checks (rule -> code): condition library, the check surface in
ContextVersion, enforcement in the agent loop, and the offline backtest.

The condition / surface tests need nothing; the loop and backtest tests run
scripted tau-bench episodes (no API) and skip when tau_bench is missing.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from rewind.evolve.surface import ContextVersion, Edit, EditError, validate_edit
from rewind.policy import Check, PolicySet, turns

ROOT = Path(__file__).resolve().parents[1]
RETAIL_CHECKS = json.loads((ROOT / "data/policy_checks/taubench_retail.json").read_text())
WRITES = {"cancel_pending_order", "modify_pending_order_items", "modify_user_address"}


def is_write(name: str) -> bool:
    return name in WRITES


def agent(text):
    return {"role": "assistant", "content": text, "tool_calls": []}


def user(text):
    return {"role": "user", "content": text}


_n = iter(range(10_000))


def call(name, args, result="ok", error=False):
    cid = f"c{next(_n)}"
    return [{"role": "assistant", "content": "", "tool_calls": [{"id": cid, "name": name, "args": args}]},
            {"role": "tool", "tool_call_id": cid, "content": result, "is_error": error}]


def pending(name, args):
    """The assistant message making the call being checked (no result yet)."""
    return [{"role": "assistant", "content": "", "tool_calls": [{"id": "now", "name": name, "args": args}]}]


def check(cid, **kw):
    return next(Check.model_validate({**c, **kw}) for c in RETAIL_CHECKS if c["id"] == cid)


def fired(chk: Check, history, tool, args, data=None) -> bool:
    return bool(PolicySet([chk]).evaluate(data or {}, tool, args, history, is_write))


SYS = [{"role": "system", "content": "wiki"}]
CANCEL = ("cancel_pending_order", {"order_id": "#W1", "reason": "no longer needed"})


# ---------------------------------------------------------------------------
# conditions
# ---------------------------------------------------------------------------
def test_confirmation_needs_a_yes_that_answers_the_agent():
    c = check("confirm_before_write")
    opening_yes = SYS + [user("Cancel #W1, yes I'm sure.")] + pending(*CANCEL)
    assert fired(c, opening_yes, *CANCEL)                       # nothing was proposed yet
    asked = SYS + [user("Cancel #W1."), agent("Cancel #W1 for 'no longer needed'. Confirm?")]
    assert not fired(c, asked + [user("Yes, please.")] + pending(*CANCEL), *CANCEL)
    assert fired(c, asked + [user("No, wait.")] + pending(*CANCEL), *CANCEL)
    assert fired(c, asked + [user("What would the refund be?")] + pending(*CANCEL), *CANCEL)
    # reads between the yes and the write are fine
    h = asked + [user("yes")] + call("get_order_details", {"order_id": "#W1"}) + pending(*CANCEL)
    assert not fired(c, h, *CANCEL)


def test_one_confirmation_covers_one_successful_write():
    c = check("confirm_before_write")
    base = SYS + [user("hi"), agent("Shall I cancel #W1?"), user("yes")]
    used = base + call(*CANCEL)
    assert fired(c, used + pending("modify_user_address", {"user_id": "u"}), "modify_user_address", {"user_id": "u"})
    failed = base + call(*CANCEL, result="Error: non-pending", error=True)   # a failed write uses nothing
    assert not fired(c, failed + pending(*CANCEL), *CANCEL)
    two = check("confirm_before_write", params={"max_writes": 2})
    assert not fired(two, used + pending(*CANCEL), *CANCEL)


def test_text_sent_with_a_tool_call_is_not_something_the_user_saw():
    c = check("confirm_before_write")
    hidden = [{"role": "assistant", "content": "I'll cancel it, ok?",
               "tool_calls": [{"id": "x", "name": "get_order_details", "args": {}}]},
              {"role": "tool", "tool_call_id": "x", "content": "{}", "is_error": False}]
    # the user's "ok" answers the opening exchange, not the hidden text
    h = SYS + [user("hi"), agent("What can I do?"), user("ok cancel W1")] + hidden + pending(*CANCEL)
    t = [x.kind for x in turns(h)]
    assert t.count("agent") == 1
    assert not fired(c, h, *CANCEL)     # the last user line answers the agent and says ok
    h2 = SYS + [user("cancel W1")] + hidden + pending(*CANCEL)
    assert fired(c, h2, *CANCEL)


def test_authenticated_owner():
    c = check("act_for_authenticated_user")
    data = {"orders": {"#W1": {"user_id": "ann_1"}, "#W2": {"user_id": "bob_2"}}}
    auth = SYS + [user("hi")] + call("find_user_id_by_email", {"email": "a@x.com"}, result="ann_1")
    args = {"order_id": "#W1", "reason": "no longer needed"}
    assert fired(c, SYS + [user("hi")] + pending("cancel_pending_order", args), "cancel_pending_order", args, data)
    assert not fired(c, auth + pending("cancel_pending_order", args), "cancel_pending_order", args, data)
    other = {"order_id": "#W2", "reason": "no longer needed"}
    assert fired(c, auth + pending("cancel_pending_order", other), "cancel_pending_order", other, data)
    assert fired(c, auth, "modify_user_address", {"user_id": "bob_2"}, data)
    assert not fired(c, auth, "modify_user_address", {"user_id": "ann_1"}, data)
    # a failed lookup authenticates nobody; a second user is refused
    failed = SYS + call("find_user_id_by_email", {"email": "b@x.com"}, result="Error: user not found", error=True)
    assert fired(c, failed, "cancel_pending_order", args, data)
    two = auth + call("find_user_id_by_name_zip", {}, result="bob_2")
    assert fired(c, two, "cancel_pending_order", args, data)
    # unknown order: let the tool report it
    assert not fired(c, auth, "cancel_pending_order", {"order_id": "#W9"}, data)


def test_called_before_and_arg_conditions():
    c = check("check_order_before_write")
    looked = SYS + call("get_order_details", {"order_id": "#W1"})
    assert not fired(c, looked, *CANCEL)
    assert fired(c, SYS + call("get_order_details", {"order_id": "#W2"}), *CANCEL)
    assert fired(c, SYS + call("get_order_details", {"order_id": "#W1"}, error=True), *CANCEL)

    same = Check(id="same_len", tools=["modify_pending_order_items"], condition="args_same_length",
                 params={"a": "item_ids", "b": "new_item_ids"}, message="m")
    assert fired(same, SYS, "modify_pending_order_items", {"item_ids": ["1", "2"], "new_item_ids": ["3"]})
    assert not fired(same, SYS, "modify_pending_order_items", {"item_ids": ["1"], "new_item_ids": ["3"]})
    st = Check(id="pending_only", tools=["cancel_pending_order"], condition="state_field_in",
               params={"collection": "orders", "key_arg": "order_id", "field": "status", "values": ["pending"]},
               message="m")
    assert fired(st, SYS, *CANCEL, data={"orders": {"#W1": {"status": "delivered"}}})
    assert not fired(st, SYS, *CANCEL, data={"orders": {"#W1": {"status": "pending"}}})


def test_block_budget_and_observe_mode():
    c = check("confirm_before_write", max_blocks=1)
    blocked = [{"role": "tool", "tool_call_id": "x", "is_error": True,
                "content": f"Error: {c.tag} get a yes first"}]
    h = SYS + [user("cancel")] + blocked + pending(*CANCEL)
    [d] = PolicySet([c]).evaluate({}, *CANCEL, h, is_write)
    assert d.action == "allow_over_budget"
    [d] = PolicySet([check("confirm_before_write", mode="observe")]).evaluate(
        {}, *CANCEL, SYS + [user("cancel")], is_write)
    assert d.action == "observe" and d.message.startswith("[policy:confirm_before_write]")


def test_check_validation():
    with pytest.raises(ValueError, match="unknown condition"):
        Check(id="abc", tools=["t"], condition="eval_python", message="m")
    with pytest.raises(ValueError, match="needs params"):
        Check(id="abc", tools=["t"], condition="args_same_length", params={"a": "x"}, message="m")
    with pytest.raises(ValueError, match="does not take"):
        Check(id="abc", tools=["t"], condition="arg_in", params={"arg": "x", "values": [], "code": 1},
              message="m")
    with pytest.raises(ValueError, match="snake_case"):
        Check(id="Bad Id", tools=["t"], condition="arg_in", params={"arg": "x", "values": []}, message="m")


# ---------------------------------------------------------------------------
# the check surface
# ---------------------------------------------------------------------------
def check_edit(cid="confirm_before_write", op="add", **over) -> Edit:
    spec = {k: v for k, v in next(c for c in RETAIL_CHECKS if c["id"] == cid).items() if k != "id"}
    spec.update(over)
    return Edit(kind="check", target=cid, op=op, text=json.dumps(spec))


TOOLS = {"cancel_pending_order", "exchange_delivered_order_items", "modify_pending_order_address",
         "modify_pending_order_items", "modify_pending_order_payment", "modify_user_address",
         "return_delivered_order_items", "get_order_details", "get_user_details",
         "find_user_id_by_email", "find_user_id_by_name_zip"}


def test_check_edits_version_and_patch_the_registry():
    from rewind.tools import ToolRegistry
    v0 = ContextVersion()
    v1 = v0.apply(check_edit(), known_tools=TOOLS)
    assert list(v1.checks) == ["confirm_before_write"] and v1.version == 1
    assert v1.token_size() == 0                     # a check adds nothing to the prompt
    assert v1.fingerprint() != v0.fingerprint()
    assert v0.fingerprint() == ContextVersion().fingerprint()
    reg = v1.patch_registry(ToolRegistry())
    assert [c.id for c in reg.policy.checks] == ["confirm_before_write"]
    v2 = v1.apply(Edit(kind="check", target="confirm_before_write", op="delete"), known_tools=TOOLS)
    assert not v2.checks
    with pytest.raises(EditError, match="already exists"):
        v1.apply(check_edit(), known_tools=TOOLS)
    # round trip through the ledger's JSON
    assert ContextVersion.model_validate_json(v1.model_dump_json()).checks == v1.checks


def test_validate_check_edit():
    v0 = ContextVersion()
    ok = dict(known_tools=TOOLS, max_tokens=2000)
    assert validate_edit(check_edit(), v0, **ok) is None
    assert "not JSON" in validate_edit(Edit(kind="check", target="x_y", op="add", text="{nope"), v0, **ok)
    assert "unknown tool" in validate_edit(check_edit(tools=["refund_everything"]), v0, **ok)
    assert "unknown condition" in validate_edit(check_edit(condition="run_code"), v0, **ok)
    assert "case-specific" in validate_edit(check_edit(message="Never cancel #W1234567."), v0, **ok)


# ---------------------------------------------------------------------------
# enforcement in the loop + backtest (scripted tau-bench episodes)
# ---------------------------------------------------------------------------
try:
    from rewind.env.taubench_adapter import _ensure_importable
    _ensure_importable()
    _TAU = None
except ImportError as e:  # pragma: no cover
    _TAU = e
needs_tau = pytest.mark.skipif(_TAU is not None, reason=f"tau_bench not importable: {_TAU}")


def _two_write_task():
    from rewind.env.taubench_adapter import MUTATING_TOOLS
    from rewind.env.taubench_dialogue import load_task, n_tasks
    for i in range(n_tasks("train")):
        t = load_task("train", i)
        if sum(a.name in MUTATING_TOOLS for a in t.actions) >= 2 and not t.outputs:
            return i
    raise LookupError


def _episode(idx, checks, tmp_path, name, replies=("Thanks, please go ahead.",)):
    from rewind.env.taubench_dialogue import build_dialogue_registry, run_episode
    from rewind.providers import ScriptedProvider
    from rewind.snapshot import SnapshotStore
    from tests.test_taubench_dialogue import (
        golden_dialogue,
        scripted_user,
    )
    script, _ = golden_dialogue("train", idx)
    store = SnapshotStore(tmp_path / "snaps")
    reg = ContextVersion(checks={c.id: c.model_dump() for c in checks}).patch_registry(build_dialogue_registry())
    from rewind.env.taubench_dialogue import STOP_TOKEN
    user = scripted_user(["I need help.", *replies, STOP_TOKEN, STOP_TOKEN, STOP_TOKEN])
    trace = tmp_path / f"{name}.jsonl"
    ep = asyncio.run(run_episode("train", idx, provider=ScriptedProvider(script), user_llm=user,
                                 store=store, registry=reg, run_id=name, trace_path=str(trace)))
    return ep, store, trace


@needs_tau
def test_loop_blocks_unconfirmed_second_write_and_records_it(tmp_path):
    idx = _two_write_task()
    free, _, _ = _episode(idx, [], tmp_path, "free")
    assert free.reward.reward == 1.0
    ep, _, _ = _episode(idx, [check("confirm_before_write")], tmp_path, "guarded")
    calls = [e.data for e in ep.events if e.type == "tool_call"]
    writes = [c for c in calls if c["name"] in ("exchange_delivered_order_items", "cancel_pending_order")]
    assert writes[0]["ok"] and not writes[1]["ok"]
    assert writes[1]["error_class"] == "POLICY_BLOCKED"
    guards = [e.data for e in ep.events if e.type == "guard" and e.data["guard"] == "policy_check"]
    assert [(g["check"], g["action"]) for g in guards] == [("confirm_before_write", "block")]
    tool_msgs = [e.data["message"] for e in ep.events if e.type == "message" and e.data["message"]["role"] == "tool"]
    assert any(m["content"].startswith("Error: [policy:confirm_before_write]") for m in tool_msgs)
    assert ep.reward.reward == 0.0            # the blocked write never ran: the DB differs from gold


@needs_tau
def test_observe_mode_never_changes_the_episode(tmp_path):
    idx = _two_write_task()
    ep, _, _ = _episode(idx, [check("confirm_before_write", mode="observe")], tmp_path, "obs")
    assert ep.reward.reward == 1.0
    guards = [e.data for e in ep.events if e.type == "guard" and e.data["guard"] == "policy_check"]
    assert [g["action"] for g in guards] == ["observe"]


@needs_tau
def test_fork_and_replay_keep_the_policy(tmp_path):
    from rewind.env.taubench_dialogue import fork_episode, replay_episode
    from rewind.providers import ScriptedProvider
    from rewind.trace import load_trace
    idx = _two_write_task()
    chk = [check("confirm_before_write")]
    ep, store, trace = _episode(idx, chk, tmp_path, "parent")
    reg = ep_registry(chk)
    # replay under the same policy reproduces the block byte for byte
    assert asyncio.run(replay_episode(load_trace(trace), store, registry=reg)).ok
    # ...and without it the replay diverges at the blocked call (the tool would run)
    from rewind.env.taubench_dialogue import build_dialogue_registry
    assert not asyncio.run(replay_episode(load_trace(trace), store, registry=build_dialogue_registry())).ok
    # a fork from step 1 under the policy blocks the same call
    from rewind.env.taubench_dialogue import STOP_TOKEN
    from tests.test_taubench_dialogue import (
        golden_dialogue,
        scripted_user,
    )
    script, _ = golden_dialogue("train", idx)
    f = asyncio.run(fork_episode(load_trace(trace), 1, provider=ScriptedProvider(script),
                                 user_llm=scripted_user(["Thanks, please go ahead.", STOP_TOKEN, STOP_TOKEN]),
                                 store=store, registry=reg))
    assert any(e.type == "tool_call" and e.data["error_class"] == "POLICY_BLOCKED" for e in f.events)


def ep_registry(checks):
    from rewind.env.taubench_dialogue import build_dialogue_registry
    return ContextVersion(checks={c.id: c.model_dump() for c in checks}).patch_registry(build_dialogue_registry())


@needs_tau
def test_backtest_counts_fires_on_passing_and_catches_on_failing(tmp_path):
    from rewind.env.taubench_dialogue import result_record
    from rewind.evolve.backtest import taubench_backtest
    from rewind.snapshot import SnapshotStore
    idx = _two_write_task()
    # recorded WITHOUT any policy (as all real runs so far)
    ep, store, trace = _episode(idx, [], tmp_path, "rec")
    row = result_record(ep.env, ep.result, 0, str(trace), ep.events)
    assert row["reward"] == 1.0
    failing = dict(row, reward=0.0, trial=1)

    pol = PolicySet([check("confirm_before_write"), check("act_for_authenticated_user")])
    rep = taubench_backtest(pol, [row, failing], store=SnapshotStore(tmp_path / "snaps"), split="train")
    s = {x["check"]: x for x in rep.summaries()}
    # the scripted agent wrote twice on one "go ahead": live, the second write was blocked;
    # offline the same firing shows up as a fire on a passing episode
    assert s["confirm_before_write"]["fires_on_passing_calls"] == "1/2"
    # on the copy marked failed, that call is a gold action -> not a catch
    assert s["confirm_before_write"]["fires_on_gold_calls_in_failures"] == 1
    assert s["confirm_before_write"]["catches"] == 0
    # the golden script never looks the user up: every guarded call fires
    assert s["act_for_authenticated_user"]["passing_episodes_hit"] == 1
    js = rep.to_json()
    assert "confirm_before_write/passing" in js["examples"] and not rep.skipped
    # same verdict as the live loop on the same call
    live, _, _ = _episode(idx, [check("confirm_before_write")], tmp_path, "live")
    blocked = [e.data["name"] for e in live.events if e.type == "tool_call" and e.data["error_class"] == "POLICY_BLOCKED"]
    offline = [f.tool for f in rep.firings if f.check_id == "confirm_before_write" and f.passed]
    assert blocked == offline


@needs_tau
def test_backtest_catches_an_off_gold_write(tmp_path):
    """A failed episode whose agent cancelled a different order with no
    confirmation: the check fires on a call the gold solution never makes."""
    from rewind.env.taubench_dialogue import (
        STOP_TOKEN,
        build_dialogue_registry,
        load_task,
        result_record,
        run_episode,
    )
    from rewind.evolve.backtest import taubench_backtest
    from rewind.providers import LLMResponse, ScriptedProvider, ToolCall
    from rewind.snapshot import SnapshotStore
    from tests.test_taubench_dialogue import scripted_user
    idx = _two_write_task()
    store = SnapshotStore(tmp_path / "snaps")
    from rewind.env.taubench_dialogue import TauBenchDialogueEnv  # noqa: F401
    data_orders = None
    # pick some pending order to cancel that the gold never touches
    from rewind.env.taubench_adapter import TauBenchRetailEnv
    data_orders = TauBenchRetailEnv().data["orders"]
    gold_orders = {a.kwargs.get("order_id") for a in load_task("train", idx).actions}
    oid = next(k for k, v in data_orders.items() if v["status"] == "pending" and k not in gold_orders)
    script = [LLMResponse(text="Hello!"),
              LLMResponse(tool_calls=[ToolCall(id="", name="cancel_pending_order",
                                               args={"order_id": oid, "reason": "no longer needed"})]),
              LLMResponse(text="Done.")]
    trace = tmp_path / "bad.jsonl"
    ep = asyncio.run(run_episode("train", idx, provider=ScriptedProvider(script),
                                 user_llm=scripted_user(["hi", "what do you mean?", STOP_TOKEN]),
                                 store=store, registry=build_dialogue_registry(), trace_path=str(trace)))
    row = result_record(ep.env, ep.result, 0, str(trace), ep.events)
    assert row["reward"] == 0.0
    rep = taubench_backtest(PolicySet([check("confirm_before_write")]), [row], store=store, split="train")
    [s] = rep.summaries()
    assert s["catches"] == 1 and s["failing_episodes_caught"] == 1

    # the Evolve stage built on it: quiet on passing runs + a catch in the cluster -> pass
    from rewind.evolve.loop import EvolveConfig, backtest_edit
    spec = {k: v for k, v in RETAIL_CHECKS[0].items() if k != "id"}
    e = Edit(kind="check", target="confirm_before_write", op="add", text=json.dumps(spec))
    ok, why, data = backtest_edit(EvolveConfig(), e, [row], store, {idx})
    assert ok and data["cluster_caught"] == [idx], why
    ok, why, _ = backtest_edit(EvolveConfig(), e, [row], store, {idx + 1})
    assert not ok and "catches 0 of this cluster" in why
