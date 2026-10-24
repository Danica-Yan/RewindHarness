"""Coordination layer: job queue, leases/fencing, checkpoint log.

    types.py   backend-neutral domain model (JobSpec, Job, Checkpoint, ...)
    store.py   the `CoordinatorStore` port (async) that worker.py and
               checkpoint.py depend on
    sqlite.py  offline backend; the Postgres backend lives in
               dist/pg_coordinator.py and implements the same port
"""
from .sqlite import Coordinator, SqliteCoordinator
from .store import CoordinatorStore
from .types import Checkpoint, CheckpointKind, Job, JobSpec, StaleFencingToken

__all__ = ["Checkpoint", "CheckpointKind", "Coordinator", "CoordinatorStore", "Job", "JobSpec",
           "SqliteCoordinator", "StaleFencingToken"]
