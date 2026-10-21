"""Real multi-turn tau-bench episodes inside rewind's loop (P0).

Offline: the agent is a ScriptedProvider replaying a task's ground-truth
actions, and the user simulator is a scripted reply list -- so these tests
check the plumbing (dialogue turns, full tau-bench reward, snapshot dedup,
replay, fork) without an API key. Needs tau_bench importable (same skip rule
as test_taubench_adapter.py).
"""
from __future__ import annotations

import asyncio

import pytest

from rewind.env.taubench_adapter import _ensure_importable, golden_script

try:
    _ensure_importable()
    _IMPORT_ERROR = None
except ImportError as e:
    _IMPORT_ERROR = e

from rewind.fork import ForkPatch
from rewind.providers import LLMResponse, ScriptedProvider, ToolCall
from rewind.snapshot import SnapshotStore

pytestmark = pytest.mark.skipif(_IMPORT_ERROR is not None,
                                reason=f"tau_bench not importable: {_IMPORT_ERROR}")

if _IMPORT_ERROR is None:
    from rewind.env.taubench_dialogue import (
        STOP_TOKEN, TauBenchDialogueEnv, TauUserSim, faithful_config, fork_episode, grade,
        load_task, n_tasks, replay_episode, result_record, run_episode,
    )


def _run(coro):
    return asyncio.run(coro)


def scripted_user(replies: list[str]):
    it = iter(replies)
    seen: list[list[dict]] = []

    async def call(messages):
        seen.append(messages)
        return next(it)
    call.seen = seen  # type: ignore[attr-defined]
    return call


def _first_task(split: str, *, with_outputs: bool, min_mutations: int = 1) -> int:
    from rewind.env.taubench_adapter import MUTATING_TOOLS
    for i in range(n_tasks(split)):
        t = load_task(split, i)
        muts = sum(a.name in MUTATING_TOOLS for a in t.actions)
        if bool(t.outputs) == with_outputs and muts >= min_mutations:
            return i
    raise LookupError("no such task")


def golden_dialogue(split: str, idx: int, final_text: str = "All done."):
    """Agent: greet, then the ground-truth tool calls, then a closing text
    turn. User: answers the greeting, then ends the conversation."""
    tool_turns = [r for r in golden_script(load_task(split, idx)) if r.tool_calls]
    script = [LLMResponse(text="Sure, let me help with that."), *tool_turns,
              LLMResponse(text=final_text)]
    return script, ["Thanks, please go ahead.", STOP_TOKEN]


def test_golden_dialogue_episode_gets_full_reward_and_records_user_turns():
    idx = _first_task("train", with_outputs=False)
    script, replies = golden_dialogue("train", idx)
    store = SnapshotStore()
    user = scripted_user(["I need help with my order.", *replies])
    ep = _run(run_episode("train", idx, provider=ScriptedProvider(script), user_llm=user,
                          store=store))
    assert ep.result.status == "completed"
    assert ep.reward.reward == 1.0 and ep.reward.r_actions
    turns = [e for e in ep.events if e.type == "user_turn"]
    assert [t.data["done"] for t in turns] == [False, True]
    # the simulator saw tau-bench's own prompt and role convention
    first_view = user.seen[0]
    assert first_view[0]["role"] == "system" and "You are a user interacting with an agent" \
        in first_view[0]["content"]
    assert first_view[1] == {"role": "user", "content": "Hi! How can I help you today?"}
    # the opening user line is the agent's first user message
    msgs = [e.data["message"] for e in ep.events if e.type == "message"]
    assert msgs[0]["role"] == "system" and msgs[1] == {"role": "user",
                                                        "content": "I need help with my order."}


def test_outputs_task_needs_the_required_text_in_what_the_agent_said():
    idx = _first_task("test", with_outputs=True, min_mutations=0)
    task = load_task("test", idx)
    said_all = "Here you go: " + " and ".join(task.outputs)
    for final, expect in ((said_all, 1.0), ("Done, anything else?", 0.0)):
        script, replies = golden_dialogue("test", idx, final_text=final)
        ep = _run(run_episode("test", idx, provider=ScriptedProvider(script),
                              user_llm=scripted_user(["hello", *replies]), store=SnapshotStore()))
        assert ep.reward.r_actions
        assert ep.reward.reward == expect
        assert ep.reward.r_outputs == expect


def test_db_is_snapshotted_once_per_distinct_content():
    idx = _first_task("train", with_outputs=False, min_mutations=2)
    script, replies = golden_dialogue("train", idx)
    store = SnapshotStore()
    ep = _run(run_episode("train", idx, provider=ScriptedProvider(script),
                          user_llm=scripted_user(["hi", *replies]), store=store))
    step_snaps = [store.get(e.data["env_hash"]) for e in ep.events if e.type == "step_start"]
    data_refs = {s["data_ref"] for s in step_snaps}
    n_mut = sum(1 for r in script if r.tool_calls
                and r.tool_calls[0].name.startswith(("modify", "cancel", "exchange", "return")))
    # one DB copy before any mutation + at most one per mutation, not one per step
    assert len(data_refs) <= n_mut + 1 < len(step_snaps)
    assert all("data" not in s for s in step_snaps)  # the DB itself is never inlined


def test_replay_is_deterministic_without_any_model_or_user_calls():
    idx = _first_task("train", with_outputs=False)
    script, replies = golden_dialogue("train", idx)
    store = SnapshotStore()
    ep = _run(run_episode("train", idx, provider=ScriptedProvider(script),
                          user_llm=scripted_user(["hi", *replies]), store=store))
    rep = _run(replay_episode(ep.events, store))
    assert rep.ok, rep


def test_null_fork_restores_dialogue_state_and_reaches_the_same_outcome():
    idx = _first_task("train", with_outputs=False, min_mutations=1)
    script, replies = golden_dialogue("train", idx)
    store = SnapshotStore()
    ep = _run(run_episode("train", idx, provider=ScriptedProvider(script),
                          user_llm=scripted_user(["hi", *replies]), store=store))
    k = 3  # after the greeting turn and one tool call
    snap = store.get(next(e for e in ep.events if e.type == "step_start" and e.step == k)
                     .data["env_hash"])
    # the restored simulator holds the conversation so far (system, opener, 2 turns, 1 reply)
    assert len(snap["user"]) == 5 and snap["user"][-1]["content"] == "Thanks, please go ahead."
    rest = script[k - 1:]
    fk = _run(fork_episode(ep.events, k, provider=ScriptedProvider(rest),
                           user_llm=scripted_user([STOP_TOKEN]), store=store))
    assert fk.reward.reward == ep.reward.reward == 1.0
    assert [e for e in fk.events if e.inherited]  # prefix copied, flagged


def test_counterfactual_fork_with_a_wrong_mutation_flips_the_outcome():
    idx = _first_task("train", with_outputs=False, min_mutations=1)
    script, replies = golden_dialogue("train", idx)
    store = SnapshotStore()
    ep = _run(run_episode("train", idx, provider=ScriptedProvider(script),
                          user_llm=scripted_user(["hi", *replies]), store=store))
    from rewind.env.taubench_adapter import MUTATING_TOOLS
    mut_i = next(i for i, r in enumerate(script) if r.tool_calls
                 and r.tool_calls[0].name in MUTATING_TOOLS)
    step = mut_i + 1
    # replace the first mutation by a harmless read: the DB never reaches the goal state
    patch = ForkPatch(tool_calls=[{"name": "list_all_product_types", "args": {}}])
    rest = [r for r in script[mut_i + 1:]]
    fk = _run(fork_episode(ep.events, step, provider=ScriptedProvider(rest),
                           user_llm=scripted_user([STOP_TOKEN]), store=store, patch=patch))
    assert ep.reward.reward == 1.0 and fk.reward.reward == 0.0


def test_transfer_to_human_ends_the_episode_and_single_tool_call_truncates():
    idx = _first_task("train", with_outputs=False)
    script = [LLMResponse(tool_calls=[
        ToolCall(id="a", name="transfer_to_human_agents", args={"summary": "user wants a human"}),
        ToolCall(id="b", name="list_all_product_types", args={})])]
    ep = _run(run_episode("train", idx, provider=ScriptedProvider(script),
                          user_llm=scripted_user(["get me a human"]), store=SnapshotStore()))
    assert ep.env.terminated and ep.result.status == "completed"
    assert any(e.type == "guard" and e.data.get("guard") == "truncate_tool_calls"
               for e in ep.events)
    tool_msgs = [e.data["message"] for e in ep.events
                 if e.type == "message" and e.data["message"]["role"] == "tool"]
    assert len(tool_msgs) == 1


def test_tool_errors_use_tau_bench_format_and_result_row_matches_run_py_shape():
    from rewind.flywheel.taubench_failures import classify_taubench
    idx = _first_task("train", with_outputs=False)
    script = [LLMResponse(tool_calls=[ToolCall(id="a", name="get_order_details",
                                               args={"order_id": "#W0000000"})]),
              LLMResponse(text="I could not find that order.")]
    ep = _run(run_episode("train", idx, provider=ScriptedProvider(script),
                          user_llm=scripted_user(["hi", STOP_TOKEN]), store=SnapshotStore()))
    tool_msg = next(e.data["message"] for e in ep.events
                    if e.type == "message" and e.data["message"]["role"] == "tool")
    assert tool_msg["content"].startswith("Error: ")
    row = result_record(ep.env, ep.result, 0, None, ep.events)
    assert row["reward"] == 0.0 and row["traj"][0]["role"] == "system"
    rec = classify_taubench(row)
    assert rec is not None and rec.category == "TOOL_RUNTIME_ERROR"


def test_faithful_config_disables_rewind_only_aborts():
    cfg = faithful_config()
    assert cfg.single_tool_call and cfg.max_steps == 30
    assert cfg.loop_kill_at > 100 and cfg.max_invalid_args > 100
    assert cfg.system_prompt.startswith("# Retail agent policy")
