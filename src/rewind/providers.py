"""LLM provider abstraction + test doubles + Anthropic/DeepSeek adapters.

`AnthropicProvider`/`DeepSeekProvider` are not covered by the offline tests
(they need a key); the message conversion each relies on is unit-tested.
"""
from __future__ import annotations

import json
import os
from typing import Any, Protocol

from pydantic import BaseModel, Field

from .util import canonical_json


class ToolCall(BaseModel):
    id: str = ""
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    model: str | None = None
    stop_reason: str | None = None


class ProviderError(Exception):
    def __init__(self, message: str, retryable: bool = True, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class MalformedOutput(Exception):
    """The model produced output that could not be parsed into text/tool calls."""


class Provider(Protocol):
    async def complete(self, messages: list[dict], tools: list[dict],
                       params: dict[str, Any]) -> LLMResponse: ...


class ScriptedProvider:
    """Returns pre-scripted responses in order; Exception items are raised."""

    def __init__(self, script: list[LLMResponse | Exception]) -> None:
        self.script = list(script)
        self.calls = 0

    async def complete(self, messages, tools, params) -> LLMResponse:
        if self.calls >= len(self.script):
            raise AssertionError("ScriptedProvider exhausted")
        item = self.script[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item.model_copy(deep=True)


class FaultInjectingProvider:
    """Wraps a provider and fails on chosen attempt indexes (0-based) or always."""

    def __init__(self, inner: Provider, fail_calls: dict[int, Exception] | None = None,
                 fail_always: Exception | None = None) -> None:
        self.inner = inner
        self.fail_calls = fail_calls or {}
        self.fail_always = fail_always
        self.attempts = 0

    async def complete(self, messages, tools, params) -> LLMResponse:
        i = self.attempts
        self.attempts += 1
        if self.fail_always is not None:
            raise self.fail_always
        if i in self.fail_calls:
            raise self.fail_calls[i]
        return await self.inner.complete(messages, tools, params)


def to_anthropic_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Internal message format -> (system, Anthropic messages). Consecutive
    same-role messages are merged; tool results become user tool_result blocks."""
    system_parts: list[str] = []
    out: list[dict] = []

    def push(role: str, blocks: list[dict]) -> None:
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    for m in messages:
        role = m["role"]
        if role == "system":
            system_parts.append(m["content"])
        elif role == "user":
            push("user", [{"type": "text", "text": m["content"]}])
        elif role == "assistant":
            blocks: list[dict] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for c in m.get("tool_calls", []):
                blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"],
                               "input": c["args"]})
            if not blocks:
                blocks.append({"type": "text", "text": "(no output)"})
            push("assistant", blocks)
        elif role == "tool":
            push("user", [{"type": "tool_result", "tool_use_id": m["tool_call_id"],
                           "content": m["content"], "is_error": bool(m.get("is_error"))}])
    return ("\n\n".join(system_parts) or None), out


def to_openai_messages(messages: list[dict]) -> list[dict]:
    """Internal message format -> OpenAI-style chat messages (DeepSeek's API
    is OpenAI-compatible, so this is shared, not Anthropic-specific). Unlike
    `to_anthropic_messages` there's no system/user merging to do -- OpenAI's
    format already matches rewind's shape closely; this just renames
    `tool_calls[].args` -> `arguments` (JSON-encoded, per the API) and drops
    the internal-only `is_error` flag (the error is already encoded in the
    tool message's `content` text by loop.py's `ERROR[...]:` prefix)."""
    out: list[dict] = []
    for m in messages:
        role = m["role"]
        if role in ("system", "user"):
            out.append({"role": role, "content": m["content"]})
        elif role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": m.get("content") or None}
            calls = m.get("tool_calls") or []
            if calls:
                entry["tool_calls"] = [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": canonical_json(c["args"])}}
                    for c in calls]
            out.append(entry)
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
    return out


def to_openai_tools(tools: list[dict]) -> list[dict]:
    """rewind's tool schema ({name, description, input_schema}) -> OpenAI
    function-calling shape ({"type": "function", "function": {...}})."""
    return [{"type": "function", "function": {
        "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in tools]


class AnthropicProvider:
    def __init__(self, model: str | None = None, max_tokens: int = 1024, client=None) -> None:
        import anthropic  # optional dependency

        self._anthropic = anthropic
        self.client = client or anthropic.AsyncAnthropic()
        self.model = model or os.environ.get("REWIND_MODEL", "claude-sonnet-5")
        self.max_tokens = max_tokens

    async def complete(self, messages, tools, params) -> LLMResponse:
        a = self._anthropic
        system, msgs = to_anthropic_messages(messages)
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=params.get("max_tokens") or self.max_tokens,
            messages=msgs,
            tools=tools,
        )
        if system:
            kwargs["system"] = system
        if params.get("temperature") is not None:  # set cfg.temperature=None if a model rejects it
            kwargs["temperature"] = params["temperature"]
        try:
            resp = await self.client.messages.create(**kwargs)
        except a.RateLimitError as e:
            raise ProviderError(str(e), retryable=True, status=429) from e
        except (a.APITimeoutError, a.APIConnectionError) as e:
            raise ProviderError(str(e), retryable=True) from e
        except a.APIStatusError as e:
            raise ProviderError(str(e), retryable=e.status_code >= 500, status=e.status_code) from e

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for b in resp.content:
            if b.type == "text":
                text_parts.append(b.text)
            elif b.type == "tool_use":
                calls.append(ToolCall(id=b.id, name=b.name, args=dict(b.input)))
        if resp.stop_reason == "max_tokens" and calls:
            raise MalformedOutput("output truncated by max_tokens while emitting a tool call")
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        cached = getattr(resp.usage, "cache_read_input_tokens", None)
        if cached:
            usage["cached_tokens"] = cached
        return LLMResponse(text="".join(text_parts) or None, tool_calls=calls, usage=usage,
                           model=resp.model, stop_reason=resp.stop_reason)


class DeepSeekProvider:
    """DeepSeek's chat API is OpenAI-compatible (same request/response
    shape, different base_url), so this reuses the `openai` SDK rather than
    a bespoke HTTP client -- the officially documented integration path.
    `deepseek-chat` (DeepSeek-V3) supports tool calling; `deepseek-reasoner`
    (R1) does not as of this writing, so it isn't a sane default here."""

    def __init__(self, model: str | None = None, max_tokens: int = 1024, client=None,
                base_url: str = "https://api.deepseek.com") -> None:
        import openai  # optional dependency, same lazy-import pattern as AnthropicProvider

        self._openai = openai
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if client is None and not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set (and no client was passed in)")
        self.client = client or openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model or os.environ.get("REWIND_MODEL", "deepseek-chat")
        self.max_tokens = max_tokens

    async def complete(self, messages, tools, params) -> LLMResponse:
        o = self._openai
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=params.get("max_tokens") or self.max_tokens,
            messages=to_openai_messages(messages),
        )
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
        if params.get("temperature") is not None:
            kwargs["temperature"] = params["temperature"]
        try:
            resp = await self.client.chat.completions.create(**kwargs)
        except o.RateLimitError as e:
            raise ProviderError(str(e), retryable=True, status=429) from e
        except (o.APITimeoutError, o.APIConnectionError) as e:
            raise ProviderError(str(e), retryable=True) from e
        except o.APIStatusError as e:
            raise ProviderError(str(e), retryable=e.status_code >= 500, status=e.status_code) from e

        choice = resp.choices[0]
        msg = choice.message
        calls: list[ToolCall] = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError as e:
                raise MalformedOutput(f"tool call arguments were not valid JSON: {e}") from e
            calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))
        if choice.finish_reason == "length" and calls:
            raise MalformedOutput("output truncated by max_tokens while emitting a tool call")
        usage = {}
        if resp.usage is not None:
            usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
            cached = getattr(getattr(resp.usage, "prompt_tokens_details", None), "cached_tokens", None)
            if cached:
                usage["cached_tokens"] = cached
        return LLMResponse(text=msg.content or None, tool_calls=calls, usage=usage,
                           model=resp.model, stop_reason=choice.finish_reason)
