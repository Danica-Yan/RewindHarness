import asyncio

import pytest

from rewind import AgentConfig, AgentRunner, LLMResponse, SnapshotStore, ToolCall, TraceWriter
from rewind.env import ToyRetailEnv, build_registry


def call(name, **args):
    return LLMResponse(tool_calls=[ToolCall(name=name, args=args)],
                       usage={"input_tokens": 10, "output_tokens": 5})


def text(t):
    return LLMResponse(text=t, usage={"input_tokens": 10, "output_tokens": 5})


class Sleeps:
    def __init__(self):
        self.calls = []

    async def __call__(self, d):
        self.calls.append(d)


@pytest.fixture
def make():
    """Factory: returns (runner, env, trace, snaps, registry) for a provider."""
    def _make(provider, cfg=None, fallback=None, sleep=None):
        snaps = SnapshotStore()
        registry = build_registry()
        runner = AgentRunner(provider, registry, cfg or AgentConfig(system_prompt="sys"),
                             snaps, fallback=fallback, sleep=sleep or Sleeps())
        return runner, ToyRetailEnv(), TraceWriter("t-1"), snaps, registry
    return _make


def run(coro):
    return asyncio.run(coro)
