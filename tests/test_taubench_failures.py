"""Unit tests for flywheel/taubench_failures.py's classifier, using small
hand-built `EnvRunResult`-shaped dicts (the schema tau_bench/types.py
defines) -- one per category, each minimal but shaped exactly like a real
record, not a real file. The real distribution (34 real failures from the
uploaded DeepSeek test-split run) is reported by
scripts/taubench_failure_report.py and recorded in README.md -- this file
only checks the classification logic itself is correct, the way
test_eval_attribution.py checks bisect_attribute without needing a live LLM.
"""
from __future__ import annotations

from rewind.env.taubench_failures import action_shape, classify_taubench, summarize


def _tool_msg(content: str, name: str = "get_order_details") -> dict:
    return {"role": "tool", "name": name, "content": content}


def _assistant_call(name: str, args: dict | None = None) -> dict:
    return {"role": "assistant", "content": None,
            "tool_calls": [{"function": {"name": name, "arguments": "{}"}}]}


def _result(task_id: int, reward: float, traj: list[dict], info: dict,
           actions: list[dict] | None = None) -> dict:
    return {"task_id": task_id, "reward": reward, "trial": 0,
            "info": {"reward_info": {"reward": reward, "info": info, "actions": actions or []}},
            "traj": traj}


def test_success_is_not_a_failure():
    r = _result(0, 1.0, [], {"r_actions": 1.0, "gt_data_hash": "x"})
    assert classify_taubench(r) is None


def test_tool_runtime_error_detected_even_with_action_info():
    traj = [_assistant_call("cancel_pending_order"), _tool_msg("Error: order not found")]
    r = _result(1, 0.0, traj, {"r_actions": 0.0, "gt_data_hash": "x"})
    rec = classify_taubench(r)
    assert rec.category == "TOOL_RUNTIME_ERROR"
    assert rec.last_tool_error == "Error: order not found"


def test_wrong_output_when_required_text_missing():
    r = _result(2, 0.0, [], {"r_outputs": 0.0, "outputs": {"42.00": False, "order confirmed": True}})
    rec = classify_taubench(r)
    assert rec.category == "WRONG_OUTPUT"
    assert rec.missing_outputs == ["42.00"]


def test_action_mismatch_despite_right_output():
    r = _result(3, 0.0, [], {"r_outputs": 1.0, "outputs": {"42.00": True}})
    rec = classify_taubench(r)
    assert rec.category == "ACTION_MISMATCH_DESPITE_RIGHT_OUTPUT"


def test_wrong_action_for_action_only_task():
    r = _result(4, 0.0, [], {"r_actions": 0.0, "gt_data_hash": "x"})
    rec = classify_taubench(r)
    assert rec.category == "WRONG_ACTION"


def test_unknown_when_no_recognizable_signal():
    r = _result(5, 0.0, [], {})
    rec = classify_taubench(r)
    assert rec.category == "UNKNOWN"


def test_summarize_distribution():
    results = [
        _result(0, 1.0, [], {"r_actions": 1.0, "gt_data_hash": "x"}),
        _result(1, 0.0, [], {"r_actions": 0.0, "gt_data_hash": "x"}),
        _result(2, 0.0, [], {"r_actions": 0.0, "gt_data_hash": "x"}),
    ]
    records, dist = summarize(results)
    assert len(records) == 2
    assert dist["WRONG_ACTION"] == 2


def test_action_shape_never_attempted():
    traj = []
    actions = [{"name": "cancel_pending_order", "kwargs": {}}]
    r = _result(6, 0.0, traj, {"r_actions": 0.0, "gt_data_hash": "x"}, actions=actions)
    rec = classify_taubench(r)
    assert action_shape(rec) == "never attempted a mutating action"


def test_action_shape_same_tools_different_args():
    traj = [_assistant_call("cancel_pending_order")]
    actions = [{"name": "cancel_pending_order", "kwargs": {}}]
    r = _result(7, 0.0, traj, {"r_actions": 0.0, "gt_data_hash": "x"}, actions=actions)
    rec = classify_taubench(r)
    assert "arguments" in action_shape(rec)


def test_action_shape_extra_unrequired_action():
    traj = [_assistant_call("cancel_pending_order")]
    r = _result(8, 0.0, traj, {"r_actions": 0.0, "gt_data_hash": "x"}, actions=[])
    rec = classify_taubench(r)
    assert action_shape(rec) == "took a mutating action the task didn't call for"


# --- recovered vs unrecovered tool errors ------------------------------------
_ACTION_INFO = {"r_actions": False, "gt_data_hash": "x"}


def test_error_retried_successfully_is_not_a_tool_error():
    """Call fails, same tool then succeeds, task still fails: the failure is
    explained by the outcome, not by the error the agent got past."""
    traj = [_assistant_call("get_order_details"), _tool_msg("Error: order not found"),
            _assistant_call("get_order_details"), _tool_msg('{"order_id": "#W1"}')]
    rec = classify_taubench(_result(1, 0.0, traj, _ACTION_INFO))
    assert rec.category == "WRONG_ACTION"
    assert rec.last_tool_error is None
    assert rec.recovered_tool_errors == ["Error: order not found"]


def test_error_recovered_through_an_equivalent_tool():
    """Email lookup fails, name+zip lookup succeeds: the user was found."""
    traj = [_assistant_call("find_user_id_by_email"),
            _tool_msg("Error: user not found", name="find_user_id_by_email"),
            _assistant_call("find_user_id_by_name_zip"),
            _tool_msg("yusuf_rossi_9620", name="find_user_id_by_name_zip")]
    rec = classify_taubench(_result(2, 0.0, traj, _ACTION_INFO))
    assert rec.category == "WRONG_ACTION" and len(rec.recovered_tool_errors) == 1


def test_error_followed_only_by_other_tools_stays_a_tool_error():
    """The exchange failed and was never redone; reading other data later
    does not count as recovering it."""
    traj = [_assistant_call("exchange_delivered_order_items"),
            _tool_msg("Error: non-delivered order cannot be exchanged",
                      name="exchange_delivered_order_items"),
            _assistant_call("get_order_details"), _tool_msg('{"order_id": "#W1"}')]
    rec = classify_taubench(_result(3, 0.0, traj, _ACTION_INFO))
    assert rec.category == "TOOL_RUNTIME_ERROR"
    assert rec.last_tool_error_tool == "exchange_delivered_order_items"


def test_tool_name_resolved_from_tool_call_id_when_message_has_no_name():
    """rewind's own export (to_openai_messages) has no `name` on tool messages."""
    traj = [{"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "cancel_pending_order", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "Error: order not found"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c2", "function": {"name": "cancel_pending_order", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c2", "content": '{"status": "cancelled"}'}]
    rec = classify_taubench(_result(4, 0.0, traj, _ACTION_INFO))
    assert rec.category == "WRONG_ACTION" and rec.recovered_tool_errors


def test_episode_that_ran_out_of_steps_has_its_own_category():
    traj = [_assistant_call("return_delivered_order_items"),
            _tool_msg("Error: non-delivered order cannot be returned", name="return_delivered_order_items"),
            _assistant_call("return_delivered_order_items"),
            _tool_msg('{"status": "return requested"}', name="return_delivered_order_items")]
    row = {"task_id": 5, "reward": 0.0, "trial": 0, "info": {"reward_info": None}, "traj": traj}
    assert classify_taubench(row).category == "EPISODE_NOT_FINISHED"
    row2 = {"task_id": 6, "reward": 0.0, "trial": 0, "traj": [],
            "info": {"reward_info": {"reward": 0.0, "info": _ACTION_INFO, "actions": []},
                     "rewind": {"status": "budget_exceeded"}}}
    assert classify_taubench(row2).category == "EPISODE_NOT_FINISHED"
