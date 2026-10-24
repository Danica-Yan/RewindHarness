"""CLI.  `rewind demo` runs fully offline; `run` / `fork` need the anthropic extra
and ANTHROPIC_API_KEY (not exercised by the offline tests)."""
from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from pathlib import Path

from .coordination import SqliteCoordinator
from .env import ToyRetailEnv, build_registry, grade
from .flywheel import bisect_attribute, flip_rate_via_fork
from .flywheel.eval_attribution import evaluate_attribution_accuracy
from .fork import ForkPatch, fork_run
from .loop import AgentConfig, AgentRunner
from .providers import LLMResponse, ScriptedProvider, ToolCall
from .replay import replay_run
from .snapshot import SnapshotStore
from .trace import TraceWriter, load_trace
from .worker import SimulatedCrash, run_worker_step, submit_job

SYSTEM = ("You are a retail support agent. Look up orders with the tools before acting. "
          "Only pending orders can be cancelled.")
TASK = "I'm user u1. I ordered headphones by mistake, please cancel that order."


def _call(name: str, **args) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name=name, args=args)],
                       usage={"input_tokens": 100, "output_tokens": 20})


def _text(t: str) -> LLMResponse:
    return LLMResponse(text=t, usage={"input_tokens": 120, "output_tokens": 15})


def _summary(events) -> str:
    return " ".join(f"{e.type}@{e.step}" for e in events if e.type in ("llm_call", "tool_call", "guard"))


async def demo(out: Path) -> None:
    snaps = SnapshotStore(out / "snapshots")
    registry = build_registry()
    cfg = AgentConfig(system_prompt=SYSTEM)

    # 1. A run where the (scripted) model picks the wrong order at step 2 and gives up.
    bad_script = [
        _call("list_orders", user_id="u1"),
        _call("cancel_order", order_id="O2", reason="ordered_by_mistake"),  # O2 is shipped
        _text("Sorry, I could not cancel your order."),
    ]
    trace = TraceWriter("demo-base", out / "demo-base" / "trace.jsonl")
    env = ToyRetailEnv()
    res = await AgentRunner(ScriptedProvider(bad_script), registry, cfg, snaps).run(env, TASK, trace)
    trace.close()
    print(f"[run]    status={res.status} steps={res.steps} graded_ok={grade(env)}")

    # 2. Replay from the trace on disk: proves the harness is deterministic.
    events = load_trace(out / "demo-base" / "trace.jsonl")
    rep = await replay_run(events, snaps, registry, ToyRetailEnv.restore)
    print(f"[replay] ok={rep.ok} events={rep.n_events}")

    # 3. Fork at step 2 with the corrected action: counterfactual says step 2 is the root cause.
    child_trace = TraceWriter("demo-fork", out / "demo-fork" / "trace.jsonl")
    runner = AgentRunner(ScriptedProvider([_text("Done, order O1 is cancelled.")]),
                         registry, cfg, snaps)
    patch = ForkPatch(tool_calls=[{"name": "cancel_order",
                                   "args": {"order_id": "O1", "reason": "ordered_by_mistake"}}])
    # fork_run restores its own env from the snapshot; grab it via a factory closure
    holder: dict = {}

    def factory(state):
        holder["env"] = ToyRetailEnv.restore(state)
        return holder["env"]

    res2 = await fork_run(events, snaps, at_step=2, runner=runner, env_factory=factory,
                          trace=child_trace, patch=patch)
    child_trace.close()
    print(f"[fork]   status={res2.status} graded_ok={grade(holder['env'])} "
          f"(parent=demo-base, fork_step=2)")
    print("traces written to", out)


async def ckpt_demo(out: Path) -> None:
    """Submit a job, run a worker that crashes mid-step-2 (after the mutating
    tool call actually applied, before the step is confirmed), then run a
    second worker that resumes and finishes correctly with no double-cancel."""
    coord = SqliteCoordinator()
    snaps, blobs = SnapshotStore(), SnapshotStore()
    registry = build_registry()
    cfg = AgentConfig(system_prompt=SYSTEM)
    script = [_call("list_orders", user_id="u1"),
             _call("cancel_order", order_id="O1", reason="ordered_by_mistake"),
             _text("Done, your order is cancelled.")]

    job_id = await submit_job(coord, snaps, "ckpt-demo", TASK, cfg, ToyRetailEnv().state)
    print(f"[submit] job={job_id}")

    def crash_after_step2_tool(tag: str, step: int) -> None:
        if (tag, step) == ("after_tool_call", 2):
            raise SimulatedCrash("worker A killed right after cancel_order applied")

    try:
        await run_worker_step(coord, snaps, blobs, registry, ScriptedProvider(list(script)),
                              ToyRetailEnv.restore, out, owner="worker-A",
                              debug_hook=crash_after_step2_tool)
    except SimulatedCrash as e:
        ckpt = await coord.latest_checkpoint("ckpt-demo")
        print(f"[crash]  {e} (latest durable checkpoint: step={ckpt.step} kind={ckpt.kind})")

    job = await coord.get(job_id)
    print(f"[job]    status={job.status} (still LEASED to the dead worker until its lease expires)")

    # In production a second worker slot would just try to claim; here we force
    # the issue by pretending time passed so worker A's lease has expired.
    coord.db.execute("UPDATE jobs SET lease_expires_at=0 WHERE job_id=?", (job_id,))
    coord.db.commit()

    result, env = await run_worker_step(coord, snaps, blobs, registry, ScriptedProvider(script[2:]),
                                        ToyRetailEnv.restore, out, owner="worker-B")
    print(f"[resume] status={result.status} graded_ok={grade(env)} "
          f"O1_status={env.state['orders']['O1']['status']}")
    print(f"[job]    status={(await coord.get(job_id)).status}")


async def attribute_demo(out: Path) -> None:
    """Take the same failing trace as `demo`, propose a (no-op) correction at
    step 1 and the real fix at step 2, and let bisection find that step 2 —
    not step 1 — is the root cause, using O(log N) forks instead of trying
    every step."""
    snaps = SnapshotStore()
    registry = build_registry()
    cfg = AgentConfig(system_prompt=SYSTEM)
    bad_script = [
        _call("list_orders", user_id="u1"),
        _call("cancel_order", order_id="O2", reason="ordered_by_mistake"),  # O2 is shipped
        _text("Sorry, I could not cancel your order."),
    ]
    trace = TraceWriter("attr-base", out / "attr-base" / "trace.jsonl")
    env = ToyRetailEnv()
    await AgentRunner(ScriptedProvider(list(bad_script)), registry, cfg, snaps).run(env, TASK, trace)
    print(f"[base]   graded_ok={grade(env)} (this is the failure we're attributing)")

    def patch_for(step: int) -> ForkPatch | None:
        if step == 1:
            return ForkPatch(tool_calls=[{"name": "list_orders", "args": {"user_id": "u1"}}])
        if step == 2:
            return ForkPatch(tool_calls=[{"name": "cancel_order",
                                          "args": {"order_id": "O1", "reason": "ordered_by_mistake"}}])
        return None

    def provider_for(step: int, sample_idx: int):
        # A faithful re-ask of the model after a no-op "fix" at step 1 would
        # just repeat the original mistake at step 2; the real fix at step 2
        # needs nothing more than wrapping up.
        if step == 1:
            return ScriptedProvider([_call("cancel_order", order_id="O2",
                                           reason="ordered_by_mistake"), _text("Sorry.")])
        return ScriptedProvider([_text("Done.")])

    calls = {"n": 0}

    async def flip_rate(step: int) -> float:
        calls["n"] += 1
        rate = await flip_rate_via_fork(trace.events, snaps, registry, cfg, ToyRetailEnv.restore,
                                        grade, step, patch_for(step), provider_for, n_samples=2)
        print(f"[probe]  step={step} patch={patch_for(step).tool_calls[0]['name']} flip_rate={rate}")
        return rate

    res = await bisect_attribute([1, 2], flip_rate, threshold=0.66)
    print(f"[result] root_step={res.root_step} monotonic={res.monotonic} "
         f"oracle_calls={calls['n']} (checked {len(res.flip_rates)}/2 candidates)")


def _live_provider(provider_name: str, model: str | None):
    """`--provider anthropic` (default) or `--provider deepseek`. Both are
    optional dependencies (`pip install -e ".[anthropic]"` /
    `pip install -e ".[deepseek]"`), imported lazily so picking one doesn't
    require the other to be installed."""
    if provider_name == "deepseek":
        from .providers import DeepSeekProvider
        return DeepSeekProvider(model=model)
    from .providers import AnthropicProvider
    return AnthropicProvider(model=model)


async def run_live(task: str, out: Path, model: str | None, provider_name: str = "anthropic") -> None:
    snaps = SnapshotStore(out / "snapshots")
    run_id = "run-" + uuid.uuid4().hex[:8]
    trace = TraceWriter(run_id, out / run_id / "trace.jsonl")
    env = ToyRetailEnv()
    runner = AgentRunner(_live_provider(provider_name, model), build_registry(),
                         AgentConfig(system_prompt=SYSTEM), snaps)
    res = await runner.run(env, task, trace)
    trace.close()
    print(json.dumps(res.__dict__, indent=2, ensure_ascii=False), "\ngraded_ok =", grade(env))


async def replay_cmd(trace_path: Path, snap_dir: Path) -> None:
    rep = await replay_run(load_trace(trace_path), SnapshotStore(snap_dir), build_registry(),
                           ToyRetailEnv.restore)
    print(json.dumps(rep.__dict__, indent=2, ensure_ascii=False, default=str))


async def fork_cmd(trace_path: Path, snap_dir: Path, at: int, patch_json: str | None,
                   out: Path, model: str | None, provider_name: str = "anthropic") -> None:
    events = load_trace(trace_path)
    snaps = SnapshotStore(snap_dir)
    run_id = f"fork-{events[0].run_id}-s{at}-" + uuid.uuid4().hex[:4]
    trace = TraceWriter(run_id, out / run_id / "trace.jsonl")
    cfg = AgentConfig(**events[0].data["cfg"])
    runner = AgentRunner(_live_provider(provider_name, model), build_registry(), cfg, snaps)
    patch = ForkPatch.model_validate_json(patch_json) if patch_json else None
    holder: dict = {}

    def factory(state):
        holder["env"] = ToyRetailEnv.restore(state)
        return holder["env"]

    res = await fork_run(events, snaps, at_step=at, runner=runner, env_factory=factory,
                         trace=trace, patch=patch)
    trace.close()
    print(json.dumps(res.__dict__, indent=2, ensure_ascii=False), "\ngraded_ok =", grade(holder["env"]))


async def attribute_eval(root_positions: list[int]) -> None:
    """W7: attribution accuracy against synthetic ground truth, using
    `auto_patch_for_step` (no hand-written patches) — see
    `flywheel/eval_attribution.py`'s module docstring for why "synthetic"
    rather than the plan's "30-50 hand-labeled real failures" (no real
    failure corpus exists yet)."""
    rows, metrics = await evaluate_attribution_accuracy(root_positions)
    for r in rows:
        mark = "OK" if r.predicted == r.root_position else "MISS"
        print(f"[{mark}] true_root={r.root_position} predicted={r.predicted} "
             f"oracle_calls={r.oracle_calls} monotonic={r.monotonic}")
    print(f"\nn={metrics['n']} exact_match_accuracy={metrics['exact_match_accuracy']:.1%} "
         f"within_1_step_accuracy={metrics['within_1_step_accuracy']:.1%} "
         f"avg_oracle_calls={metrics['avg_oracle_calls']:.1f}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="rewind")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo"); d.add_argument("--out", default="runs")
    ck = sub.add_parser("ckpt-demo"); ck.add_argument("--out", default="runs/ckpt-demo")
    at = sub.add_parser("attribute-demo"); at.add_argument("--out", default="runs/attribute-demo")
    r = sub.add_parser("run"); r.add_argument("--task", default=TASK)
    r.add_argument("--out", default="runs"); r.add_argument("--model")
    r.add_argument("--provider", choices=["anthropic", "deepseek"], default="anthropic")
    rp = sub.add_parser("replay"); rp.add_argument("trace"); rp.add_argument("--snapshots", default="runs/snapshots")
    f = sub.add_parser("fork"); f.add_argument("trace"); f.add_argument("--at", type=int, required=True)
    f.add_argument("--patch", help='JSON, e.g. {"tool_calls":[{"name":"get_order","args":{"order_id":"O1"}}]}')
    f.add_argument("--snapshots", default="runs/snapshots"); f.add_argument("--out", default="runs")
    f.add_argument("--model")
    f.add_argument("--provider", choices=["anthropic", "deepseek"], default="anthropic")
    ae = sub.add_parser("attribute-eval")
    ae.add_argument("--root-positions", default="1,2,3,4,5")
    a = p.parse_args(argv)
    if a.cmd == "demo":
        asyncio.run(demo(Path(a.out)))
    elif a.cmd == "ckpt-demo":
        asyncio.run(ckpt_demo(Path(a.out)))
    elif a.cmd == "attribute-demo":
        asyncio.run(attribute_demo(Path(a.out)))
    elif a.cmd == "run":
        asyncio.run(run_live(a.task, Path(a.out), a.model, a.provider))
    elif a.cmd == "replay":
        asyncio.run(replay_cmd(Path(a.trace), Path(a.snapshots)))
    elif a.cmd == "fork":
        asyncio.run(fork_cmd(Path(a.trace), Path(a.snapshots), a.at, a.patch, Path(a.out), a.model, a.provider))
    elif a.cmd == "attribute-eval":
        asyncio.run(attribute_eval([int(x) for x in a.root_positions.split(",")]))


if __name__ == "__main__":
    main()
