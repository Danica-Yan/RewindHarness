"""StepCheckpointer: adapts a `CoordinatorStore` to the loop's `Checkpointer` port.

This module is the only place where the kernel's view of durability
("save this step") meets the coordination layer's view ("token-gated write
to a job's checkpoint log"). It depends on the `CoordinatorStore` Protocol,
never on a concrete backend, and on `state` -- not on the agent runner.
The backends live in coordination/sqlite.py and dist/pg_coordinator.py.

Checkpoint kinds per step, matching the harness's commit order
(LLM result -> execute tools -> confirm step):

  "intent": written right after the model picks an action for `step`, before
            any tool executes. Captures the action itself.
  "tool":   written after each tool call that runs on its own (or once
            after a concurrent read-only batch), so a resume does not
            re-run a side effect that already happened.
  "step":   written after all of that step's tools have executed and the env
            has been updated. Captures the resulting env hash.

Resume logic (see worker.py): if the latest checkpoint is a "step", start a
*fresh* LLM call for step+1. If it's an "intent", the tools for that step
never (durably) finished, so resume replays the *same recorded action*
(AgentRunner's `forced` parameter — the same mechanism fork() uses for a
counterfactual action) rather than asking the model again. That keeps a
step's tool calls effectively-once even though the step-level LLM decision
is retried at the process level: whatever state existed only in the crashed
worker's memory is discarded, and the exact same action is redone against
state restored from the last durable checkpoint.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .coordination.store import CoordinatorStore
from .coordination.types import Checkpoint, CheckpointKind
from .providers import LLMResponse
from .snapshot import SnapshotStore
from .state import LoopState, RunAbort, StepStart
from .trace import TraceWriter


class StepCheckpointer:
    """Adapts a CoordinatorStore + two content-addressed stores to loop.py's
    `Checkpointer` protocol. `snapshots` holds env states (shared with
    replay/fork); `blobs` holds the (messages, loop_state, action) bundle —
    it can be the same SnapshotStore instance, since both are just generic
    content-addressed dict storage."""

    def __init__(self, coordinator: CoordinatorStore, job_id: str, run_id: str, fencing_token: int,
                 snapshots: SnapshotStore, blobs: SnapshotStore) -> None:
        self.coordinator = coordinator
        self.job_id = job_id
        self.run_id = run_id
        self.fencing_token = fencing_token
        self.snapshots = snapshots
        self.blobs = blobs

    async def save(self, kind: CheckpointKind, step: int, trace: TraceWriter, env: Any, messages: list[dict],
                   state: LoopState, action: LLMResponse | None = None, *,
                   point: StepStart | None = None) -> None:
        # env: always the CURRENT env (it carries the idempotency ledger).
        # Mid-step kinds restore the step's start for everything else.
        env_hash = self.snapshots.put(env.snapshot())
        bundle = {"messages": messages, "loop_state": asdict(state),
                  "action": action.model_dump() if action is not None else None}
        if point is not None:
            bundle["step_env_hash"] = point.env_hash
        bundle_ref = self.blobs.put(bundle)
        seq = point.seq if point is not None else trace.last_seq
        ckpt = Checkpoint(self.run_id, step, kind, seq, self.fencing_token, env_hash, bundle_ref)
        ok = await self.coordinator.write_checkpoint(self.job_id, ckpt)
        if not ok:
            raise RunAbort("fenced_out",
                           f"job {self.job_id} step {step}: fencing token superseded, "
                           "another worker has already taken over this run")
