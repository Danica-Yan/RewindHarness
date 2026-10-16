#!/usr/bin/env python3
"""W7's "run the full failure batch and report the category distribution": generates a batch of runs
covering every failure category this harness can actually produce (see
`flywheel/failures.py`'s module docstring for the two the plan lists that
have no mechanism here yet — CONTEXT_OVERFLOW, POLICY_VIOLATION — and are
therefore honestly absent from this report rather than faked), classifies
each with `classify()`, and prints the category distribution and top
clusters — the batch-level view the data flywheel (W8) would consume.

This is a synthetic batch (scripted providers constructing each failure
deliberately), not a sample drawn from real production traffic — there is
no real traffic yet (see README, W1 is still "write the τ-bench adapter").
Run it to see the taxonomy exercised end to end:

  python scripts/failure_batch.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rewind.env import ToyRetailEnv, build_registry, grade  # noqa: E402
from rewind.flywheel.failures import (  # noqa: E402
    category_distribution, classify, classify_worker_crash, cluster, top_clusters)
from rewind.loop import AgentConfig, AgentRunner  # noqa: E402
from rewind.providers import LLMResponse, MalformedOutput, ProviderError, ScriptedProvider, ToolCall  # noqa: E402
from rewind.snapshot import SnapshotStore  # noqa: E402
from rewind.trace import TraceWriter  # noqa: E402

SYS = "You are a retail support agent."
TASK = "I'm u1, cancel my headphones order."


def _call(name: str, **args) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name=name, args=args)],
                       usage={"input_tokens": 10, "output_tokens": 5})


def _text(t: str) -> LLMResponse:
    return LLMResponse(text=t, usage={"input_tokens": 10, "output_tokens": 5})


async def _no_sleep(_delay: float) -> None:
    pass   # batch generation shouldn't pay real backoff delays for injected 429s etc.


def _scenarios() -> list[tuple[str, list, AgentConfig | None]]:
    """(label, script, cfg_override) — cfg_override lets a scenario need a
    tighter budget (e.g. max_steps) without affecting the others."""
    scenarios: list[tuple[str, list, AgentConfig | None]] = []

    # LLM_API_ERROR: every attempt 429s, retries exhaust.
    for i in range(3):
        scenarios.append((f"llm_api_error_{i}",
                          [ProviderError(f"rate limited #{i}", retryable=True, status=429)] * 5, None))

    # OUTPUT_MALFORMED: fails the repair-prompt retry too.
    for i in range(2):
        scenarios.append((f"output_malformed_{i}", [MalformedOutput(), MalformedOutput()], None))

    # TOOL_ARG_INVALID: invalid `reason` enum value, repeated past max_invalid_args.
    # Each call varies order_id so the harness's own loop detector (identical
    # (tool, args) hash repeated) doesn't kick in and substitute a
    # LOOP_DETECTED outcome for what should be a validation failure.
    for i in range(3):
        calls = [_call("cancel_order", order_id=oid, reason="not_a_real_reason")
                for oid in ["O1", "O2", "O3"]]
        scenarios.append((f"tool_arg_invalid_{i}", calls, None))

    # HALLUCINATED_TOOL: a tool name not in the registry, same anti-loop-detection care.
    for i in range(2):
        calls = [_call("delete_order", order_id=oid) for oid in ["O1", "O2", "O3"]]
        scenarios.append((f"hallucinated_tool_{i}", calls, None))

    # TOOL_RUNTIME_ERROR: valid args, but the order is shipped — ToolError at runtime.
    for i in range(3):
        scenarios.append((f"tool_runtime_error_{i}",
                          [_call("cancel_order", order_id="O2", reason="ordered_by_mistake"),
                           _text("Sorry, that order can't be cancelled.")], None))

    # LOOP: identical (tool, args) repeated to loop_kill_at.
    for i in range(2):
        same = _call("get_order", order_id="O1")
        scenarios.append((f"loop_{i}", [same] * 6, None))

    # BUDGET_EXCEEDED: max_steps too small for the script.
    for i in range(2):
        script = [_call("get_order", order_id=x) for x in ["O1", "O2", "O3"]] + [_text("done")]
        scenarios.append((f"budget_exceeded_{i}", script, AgentConfig(system_prompt=SYS, max_steps=2)))

    # REASONING_ERROR: every tool call succeeds; wrong order gets cancelled.
    for i in range(4):
        scenarios.append((f"reasoning_error_{i}",
                          [_call("list_orders", user_id="u1"),
                           _call("cancel_order", order_id="O3", reason="ordered_by_mistake"),
                           _text("Done.")], None))

    return scenarios


async def run_batch() -> list:
    snaps = SnapshotStore()
    registry = build_registry()
    records = []
    for label, script, cfg_override in _scenarios():
        cfg = cfg_override or AgentConfig(system_prompt=SYS)
        env = ToyRetailEnv()
        trace = TraceWriter(f"batch-{label}")
        runner = AgentRunner(ScriptedProvider(list(script)), registry, cfg, snaps, sleep=_no_sleep)
        result = await runner.run(env, TASK, trace)
        rec = classify(trace.events, result, grade(env))
        if rec is not None:
            records.append(rec)
        else:
            print(f"  (skipped {label}: graded ok, not a failure)")

    # WORKER_CRASH doesn't come out of a single AgentRunner trace (see
    # `classify_worker_crash`'s docstring) — one representative record so the
    # distribution/report below covers the full taxonomy this repo produces.
    records.append(classify_worker_crash("batch-worker-crash-0"))
    records.append(classify_worker_crash("batch-worker-crash-1"))
    return records


def main() -> None:
    records = asyncio.run(run_batch())

    print(f"\n{len(records)} failure samples classified.\n")
    dist = category_distribution(records)
    print(f"{'category':<20} {'count':>6} {'share':>7}")
    for cat, n in dist.most_common():
        print(f"{cat:<20} {n:>6} {n / len(records):>6.1%}")

    print("\ntop clusters (category, root_cause_tool):")
    for (cat, tool), n in top_clusters(records, n=15):
        print(f"  {n:>3}x  {cat:<20} tool={tool}")

    buckets = cluster(records)
    print(f"\n{len(buckets)} distinct (category, tool) clusters from {len(records)} samples.")


if __name__ == "__main__":
    main()
