"""The port every coordination backend implements.

worker.py and checkpoint.py are written against this Protocol only. The
contract is async because the production backend (Postgres) does network
I/O; the sqlite backend implements the same async signatures, so there is
exactly one calling convention and no sync/async bridging helper.

Every write is fencing-token gated: a write carrying a token that is no
longer the job's current one must be rejected (return False), never applied.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import Checkpoint, Job, JobSpec


@runtime_checkable
class CoordinatorStore(Protocol):
    async def submit(self, spec: JobSpec, priority: int = 0) -> str:
        """Enqueue a job; returns its job_id."""

    async def claim(self, owner: str, lease_seconds: float = 30) -> Job | None:
        """Atomically lease the next runnable job (PENDING, or LEASED with an
        expired lease), bumping its fencing token. None if nothing is runnable."""

    async def heartbeat(self, job_id: str, fencing_token: int, lease_seconds: float = 30) -> bool:
        """Extend a lease. False if the token is stale."""

    async def get(self, job_id: str) -> Job | None: ...

    async def write_checkpoint(self, job_id: str, ckpt: Checkpoint) -> bool:
        """Token-gated checkpoint write. False if `ckpt.fencing_token` is stale."""

    async def latest_checkpoint(self, run_id: str) -> Checkpoint | None:
        """Highest step; within it the most-progressed kind (step > tool > intent)."""

    async def complete(self, job_id: str, fencing_token: int, status: str,
                       result: dict | None = None) -> bool:
        """Token-gated completion. False if the token is stale."""
