"""Message/tool conversion for DeepSeekProvider (OpenAI-compatible API).
No network: DeepSeekProvider itself needs a real key, same as
AnthropicProvider (see test_anthropic_convert.py) -- only the pure
conversion functions are unit-tested here."""
import json

from rewind.providers import to_openai_messages, to_openai_tools


def test_conversion_round_trips_tool_calls_and_results():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "name": "get_order", "args": {"order_id": "O1"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "{}", "is_error": False},
    ]
    out = to_openai_messages(msgs)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
    assistant = out[2]
    assert assistant["tool_calls"][0]["function"]["name"] == "get_order"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"order_id": "O1"}
    tool_msg = out[3]
    assert tool_msg == {"role": "tool", "tool_call_id": "a", "content": "{}"}  # is_error dropped


def test_conversion_omits_tool_calls_key_for_a_plain_final_answer():
    msgs = [{"role": "assistant", "content": "done"}]
    out = to_openai_messages(msgs)
    assert out == [{"role": "assistant", "content": "done"}]


def test_tool_schema_conversion_wraps_as_openai_function_shape():
    tools = [{"name": "get_order", "description": "get it", "input_schema": {"type": "object"}}]
    out = to_openai_tools(tools)
    assert out == [{"type": "function", "function": {
        "name": "get_order", "description": "get it", "parameters": {"type": "object"}}}]
