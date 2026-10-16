from rewind.providers import to_anthropic_messages


def test_conversion_merges_tool_results_and_repair():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "name": "get_order", "args": {"order_id": "O1"}},
            {"id": "b", "name": "list_orders", "args": {"user_id": "u1"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "{}", "is_error": False},
        {"role": "tool", "tool_call_id": "b", "content": "ERROR", "is_error": True},
        {"role": "user", "content": "repair"},
    ]
    system, out = to_anthropic_messages(msgs)
    assert system == "sys"
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert [b["type"] for b in out[1]["content"]] == ["tool_use", "tool_use"]
    last = out[2]["content"]
    assert [b["type"] for b in last] == ["tool_result", "tool_result", "text"]
    assert last[1]["is_error"] is True
