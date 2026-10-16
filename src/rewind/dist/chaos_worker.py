"""Runnable as a real OS process (`python -m rewind.dist.chaos_worker ...`) —
this is the piece that makes W6's process-kill chaos test genuine rather
than simulated: `tests/test_chaos.py` launches this as a *subprocess* via
`subprocess.Popen`, waits for observable progress, then sends it a real
`SIGKILL`. Nothing about the crash is faked or hooked from inside the
process, unlike `SimulatedCrash` in `worker.py` (which models "died at this
exact point" for exhaustive in-process testing) — here the OS actually ends
the process.

Everything this needs to resume correctly from a *different* process lives
outside process memory, by construction: `PgCoordinator` (real Postgres),
`SnapshotStore(root=...)` (files on disk, not the default in-memory dict),
and `TraceWriter`'s trace file. A second invocation pointed at the same
Postgres DSN and the same `--snapshot-dir`/`--trace-dir` picks up exactly
where the first one's last durable checkpoint left off.

The mutating tool call count is folded into `env.state` itself (not a
Python-side counter) precisely so it survives the process boundary: the
parent test process can load the final snapshot after both workers are done
and assert on `state["_cancel_calls"]` directly.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from ..env import ToyRetailEnv, build_registry
from ..loop import AgentConfig
from ..providers import LLMResponse, ToolCall
from ..snapshot import SnapshotStore
from ..worker import run_worker_step
from .pg_coordinator import PgCoordinator


def _call(name: str, **args) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name=name, args=args)],
                       usage={"input_tokens": 10, "output_tokens": 5})


def _text(t: str) -> LLMResponse:
    return LLMResponse(text=t, usage={"input_tokens": 10, "output_tokens": 5})


# Same 3-step scenario `test_checkpoint.py` uses (list -> cancel -> done), so
# this script and the pytest chaos test agree on what "correct" looks like.
SCRIPT = [
    _call("list_orders", user_id="u1"),
    _call("cancel_order", order_id="O1", reason="ordered_by_mistake"),
    _text("done"),
]


class DelayedScriptedProvider:
    """Like `ScriptedProvider`, but sleeps before returning each response —
    real wall-clock time, since this runs as its own process and the parent
    needs a wide-enough window to land a real SIGKILL mid-step rather than
    racing a return that happens in microseconds."""

    def __init__(self, script: list[LLMResponse], delay_s: float) -> None:
        self.script = list(script)
        self.delay_s = delay_s
        self.i = 0

    async def complete(self, messages, tools, params) -> LLMResponse:
        await asyncio.sleep(self.delay_s)
        resp = self.script[self.i]
        self.i += 1
        return resp


def counting_registry():
    """Same idea as `test_checkpoint.py`'s `counting_registry`, but the
    counter lives in `env.state` (survives snapshot/restore across a real
    process boundary) instead of a Python closure (would not)."""
    reg = build_registry()
    orig = reg._tools["cancel_order"].fn

    def wrapped(env, a):
        env.state["_cancel_calls"] = env.state.get("_cancel_calls", 0) + 1
        return orig(env, a)

    reg._tools["cancel_order"].fn = wrapped
    return reg


async def main_async(args: argparse.Namespace) -> int:
    coord = await PgCoordinator.connect(args.dsn)
    snaps = SnapshotStore(root=args.snapshot_dir)
    blobs = SnapshotStore(root=args.snapshot_dir)  # same content-addressed space is fine
    registry = counting_registry()
    provider = DelayedScriptedProvider(SCRIPT, delay_s=args.step_delay)

    # A PID file this exact worker writes right before it starts doing any
    # real work, so the parent test can wait for "the child is alive and
    # about to claim" without a fixed sleep (flaky) or reading stdout eagerly
    # (buffering races) — it just polls for this file.
    if args.pid_file:
        with open(args.pid_file, "w") as f:
            f.write(str(os.getpid()))

    result = await run_worker_step(
        coord, snaps, blobs, registry, provider, ToyRetailEnv.restore, args.trace_dir,
        owner=args.owner, lease_seconds=args.lease_seconds)
    await coord.close()

    if result is None:
        print(json.dumps({"claimed": False}))
        return 0
    res, env = result
    print(json.dumps({
        "claimed": True, "status": res.status, "steps": res.steps,
        "env_hash": res.env_hash, "cancel_calls": env.state.get("_cancel_calls", 0),
    }))
    return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=True)
    p.add_argument("--snapshot-dir", required=True)
    p.add_argument("--trace-dir", required=True)
    p.add_argument("--owner", required=True)
    p.add_argument("--lease-seconds", type=float, default=5.0)
    p.add_argument("--step-delay", type=float, default=0.0,
                   help="Real sleep before each scripted LLM response, widening the "
                        "window for the parent to deliver a real SIGKILL mid-step.")
    p.add_argument("--pid-file", default=None)
    args = p.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
