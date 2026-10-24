"""Domain model of the coordination layer: jobs, leases, checkpoints.

Backend-neutral. Both `SqliteCoordinator` (coordination/sqlite.py) and
`PgCoordinator` (dist/pg_coordinator.py) produce and consume these types;
neither backend owns them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

# Commit order within one step (see checkpoint.py for the resume rules):
#   intent -> tool (one per durably applied call, or one per read-only batch) -> step
CheckpointKind = Literal["intent", "tool", "step"]


class StaleFencingToken(Exception):
    """Raised when a checkpoint or completion write is rejected because a newer
    worker has already taken over this job (this worker is a zombie)."""


@dataclass
class JobSpec:
    run_id: str
    task: str
    cfg: dict[str, Any]
    initial_env_ref: str


@dataclass
class Job:
    job_id: str
    run_id: str
    spec: JobSpec
    status: str
    attempt: int
    fencing_token: int


@dataclass
class Checkpoint:
    run_id: str
    step: int
    kind: CheckpointKind
    seq: int
    fencing_token: int
    env_hash: str
    bundle_ref: str
