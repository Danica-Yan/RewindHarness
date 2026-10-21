"""Plumbing test for the P0 scripts with a fake model and a fake user (no
API calls): results file in run.py format, per-episode traces, shared
snapshot store, resume, and the null-fork calibration reading all of it."""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
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

from rewind.providers import LLMResponse  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class FakeModel:
    """Greets once, then says goodbye."""
    def __init__(self, *a, **k):
        pass

    async def complete(self, messages, tools, params):
        n = sum(1 for m in messages if m["role"] == "assistant")
        return LLMResponse(text="How can I help?" if n == 0 else "Goodbye.")


def fake_user_llm(*a, **k):
    async def call(messages):
        agent_turns = sum(1 for m in messages if m["role"] == "user")
        return "I have a question." if agent_turns < 3 else "###STOP###"
    return call


def test_runner_then_calibration(tmp_path, monkeypatch):
    run = _load("run_taubench_rewind")
    monkeypatch.setattr(run, "DeepSeekProvider", FakeModel)
    monkeypatch.setattr(run, "litellm_user_llm", fake_user_llm)
    out = tmp_path / "res.json"
    args = argparse.Namespace(split="test", task_ids="0-2", num_trials=2, concurrency=2,
                              out=str(out), trace_dir=None, store_dir=str(tmp_path / "snaps"),
                              model="x", user_model="x", user_provider="x")
    asyncio.run(run.main_async(args))
    rows = json.loads(out.read_text())
    assert sorted((r["task_id"], r["trial"]) for r in rows) == \
        sorted((t, k) for t in (0, 1, 2) for k in (0, 1))
    assert all(Path(r["info"]["rewind"]["trace"]).exists() for r in rows)
    assert rows[0]["traj"][0]["role"] == "system"
    # resume: nothing left to do, file unchanged in size
    before = out.read_text()
    asyncio.run(run.main_async(args))
    assert out.read_text() == before

    cal = _load("taubench_fork_calibration")
    monkeypatch.setattr(cal, "DeepSeekProvider", FakeModel)
    monkeypatch.setattr(cal, "litellm_user_llm", fake_user_llm)
    cout = tmp_path / "cal.json"
    cargs = argparse.Namespace(results=str(out), store_dir=str(tmp_path / "snaps"), reruns=str(out),
                               fractions="0,0.5", n=2, max_episodes=3, concurrency=2, seed=0,
                               out=str(cout), model="x", user_model="x", user_provider="x")
    asyncio.run(cal.main_async(cargs))
    crow = json.loads(cout.read_text())
    assert len(crow) == 3 * 2 * 2
    # deterministic fake model + user: a null fork always agrees with its parent
    assert all(r["reward"] == r["parent_reward"] for r in crow)
