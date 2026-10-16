"""Synthetic-ground-truth accuracy evaluation for the counterfactual
attribution algorithm (plan 4.7's validation step, generalized).

The plan asks for "hand-label the root-cause step of 30-50 failed trajectories and report attribution accuracy" — hand
label 30-50 real failed trajectories with their true root-cause step and
report exact-match / ±1-step accuracy. This repo has no real failure corpus
yet (still the toy retail env, not τ-bench — see README/plan W1), so there
is nothing to hand-label. What *is* available honestly: a way to construct
failing trajectories whose root-cause step is known BY CONSTRUCTION (we
choose exactly which step's action is wrong), across several different
step positions, and check whether `bisect_attribute` finds it. That's a
real accuracy number against real ground truth — synthetic ground truth,
not human-labeled — and it's exactly what should be reported honestly
instead of dressing up a toy-env number as if it came from the human
labeling the plan describes.

Each scenario: the task is "cancel O1 for u1". The model's *wrong* script
spends `root_position - 1` steps on harmless reads (`list_orders`) and then,
at step `root_position`, cancels the wrong order (`O3` — pending, so the
call succeeds; it's just the wrong target) instead of `O1`, then gives up.
The *reference* (successful) trace has the identical shape, except its
action at `root_position` cancels `O1` instead of `O3` — so
`auto_patch_for_step` (the W7 auto-pairing item) can pull each candidate
step's correction straight from that reference trace, exactly the way the
plan's step 1 describes ("prefer the action at the matching position in a successful trajectory of the same task"), no
hand-written patches. `root_position` itself is the one candidate that
actually flips the outcome; every earlier candidate is a no-op read, so
correcting it changes nothing and the continuation must faithfully redo
the original mistake for the test to be meaningful (see `_provider_for`).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..env import ToyRetailEnv, build_registry, grade
from ..fork import ForkPatch
from ..loop import AgentConfig, AgentRunner
from ..providers import LLMResponse, ScriptedProvider, ToolCall
from ..snapshot import SnapshotStore
from ..trace import TraceWriter
from .attribute import bisect_attribute, flip_rate_via_fork
from .failures import auto_patch_for_step

SYS = "You are a retail support agent."
TASK = "I'm u1, cancel my headphones order."
CORRECT_ORDER = "O1"
DISTRACTOR_ORDER = "O3"          # pending, wrong owner — the call succeeds, just wrong


def _call(name: str, **args) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(name=name, args=args)],
                       usage={"input_tokens": 10, "output_tokens": 5})


def _text(t: str) -> LLMResponse:
    return LLMResponse(text=t, usage={"input_tokens": 10, "output_tokens": 5})


def _read_step(order_id: str) -> LLMResponse:
    return _call("get_order", order_id=order_id)


def _order_cycle(i: int) -> str:
    # Distinct args per read so consecutive "harmless" reads never repeat
    # identical (tool, args) — the harness's own loop detector would
    # otherwise kick in after 3 identical calls and start substituting a
    # LOOP_DETECTED warning message, which would be a confound this
    # scenario generator doesn't want.
    return ["O1", "O2", "O3"][i % 3]


def _cancel(order_id: str) -> LLMResponse:
    return _call("cancel_order", order_id=order_id, reason="ordered_by_mistake")


@dataclass
class Scenario:
    root_position: int          # ground truth: the one step whose correction flips the outcome
    bad_script: list[LLMResponse]
    good_script: list[LLMResponse]
    candidates: list[int]


def make_scenario(root_position: int) -> Scenario:
    """`root_position` >= 1. Steps 1..root_position-1 are harmless reads;
    step root_position is the mutating call (wrong in `bad_script`, right in
    `good_script`); the final step is a closing text message."""
    reads = [_read_step(_order_cycle(i)) for i in range(root_position - 1)]
    bad_script = reads + [_cancel(DISTRACTOR_ORDER), _text("Sorry, could not find your order.")]
    good_script = reads + [_cancel(CORRECT_ORDER), _text("Done.")]
    return Scenario(root_position, bad_script, good_script, list(range(1, root_position + 1)))


def _provider_for(scenario: Scenario, step: int):
    """Continuation after a patch at `step`: if `step` is the true root, the
    patch already fixed everything — just wrap up. Otherwise the patch was a
    no-op read, so the continuation must faithfully redo the rest of the
    original (wrong) script, or a trivial re-decision would make even a
    no-op "correction" look like it flipped the outcome — which would make
    the accuracy number meaningless."""
    if step == scenario.root_position:
        return ScriptedProvider([_text("Done.")])
    tail_reads = [_read_step(_order_cycle(i)) for i in range(step, scenario.root_position - 1)]
    tail = tail_reads + [_cancel(DISTRACTOR_ORDER), _text("Sorry, could not find your order.")]
    return ScriptedProvider(tail)


async def run_base_and_reference(scenario: Scenario, snaps: SnapshotStore, registry, cfg
                                 ) -> tuple[list, list]:
    """Runs both the failing trace and its same-shape successful reference,
    returning their event lists (the reference is what `auto_patch_for_step`
    reads from — nothing here hand-writes a patch)."""
    bad_trace = TraceWriter(f"eval-bad-{scenario.root_position}")
    bad_env = ToyRetailEnv()
    await AgentRunner(ScriptedProvider(list(scenario.bad_script)), registry, cfg, snaps).run(
        bad_env, TASK, bad_trace)
    assert not grade(bad_env), "scenario construction bug: the 'bad' script graded as correct"

    good_trace = TraceWriter(f"eval-good-{scenario.root_position}")
    good_env = ToyRetailEnv()
    await AgentRunner(ScriptedProvider(list(scenario.good_script)), registry, cfg, snaps).run(
        good_env, TASK, good_trace)
    assert grade(good_env), "scenario construction bug: the 'good' script did not grade as correct"

    return bad_trace.events, good_trace.events


@dataclass
class AttributionEvalRow:
    root_position: int
    predicted: int | None
    oracle_calls: int
    monotonic: bool


async def evaluate_one(root_position: int, snaps: SnapshotStore, registry, cfg,
                       n_samples: int = 2) -> AttributionEvalRow:
    scenario = make_scenario(root_position)
    bad_events, good_events = await run_base_and_reference(scenario, snaps, registry, cfg)

    def patch_for(step: int) -> ForkPatch:
        patch = auto_patch_for_step(good_events, step)
        assert patch is not None, f"auto_patch_for_step found nothing at step {step}"
        return patch

    async def flip_rate(step: int) -> float:
        return await flip_rate_via_fork(bad_events, snaps, registry, cfg, ToyRetailEnv.restore,
                                        grade, step, patch_for(step),
                                        lambda s, i: _provider_for(scenario, s), n_samples=n_samples)

    res = await bisect_attribute(scenario.candidates, flip_rate, threshold=0.66)
    return AttributionEvalRow(root_position, res.root_step, res.oracle_calls, res.monotonic)


async def evaluate_attribution_accuracy(root_positions: list[int] | None = None
                                        ) -> tuple[list[AttributionEvalRow], dict[str, float]]:
    root_positions = root_positions or [1, 2, 3, 4, 5]
    snaps = SnapshotStore()
    registry = build_registry()
    cfg = AgentConfig(system_prompt=SYS)
    rows = [await evaluate_one(rp, snaps, registry, cfg) for rp in root_positions]

    n = len(rows)
    exact = sum(1 for r in rows if r.predicted == r.root_position)
    within_1 = sum(1 for r in rows if r.predicted is not None and abs(r.predicted - r.root_position) <= 1)
    metrics = {
        "n": n,
        "exact_match_accuracy": exact / n if n else 0.0,
        "within_1_step_accuracy": within_1 / n if n else 0.0,
        "avg_oracle_calls": sum(r.oracle_calls for r in rows) / n if n else 0.0,
    }
    return rows, metrics
