"""sqlite implementation of `CoordinatorStore`: job queue with lease +
fencing, and a checkpoint log.

Mirrors the Postgres backend (dist/pg_coordinator.py) but runs on sqlite so
the whole thing works offline with zero services. The protocol -- atomic
claim, monotonic fencing token, token-gated checkpoint writes -- is the
same; concurrency here is process-local (sqlite + a single connection), so
what's being demonstrated is the *protocol*, not multi-host throughput.

Methods are `async` to satisfy the `CoordinatorStore` port. sqlite calls on
a local file/in-memory DB return in microseconds, so they run inline on the
event loop rather than being pushed to a thread.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid

from .types import Checkpoint, Job, JobSpec

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  spec TEXT NOT NULL,                 -- JSON: task, cfg, initial_env_ref
  status TEXT NOT NULL DEFAULT 'PENDING',   -- PENDING / LEASED / DONE / FAILED
  priority INTEGER NOT NULL DEFAULT 0,
  attempt INTEGER NOT NULL DEFAULT 0,
  lease_owner TEXT,
  lease_expires_at REAL,
  fencing_token INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  result TEXT
);
CREATE TABLE IF NOT EXISTS checkpoints (
  run_id TEXT NOT NULL,
  step INTEGER NOT NULL,
  kind TEXT NOT NULL,
  seq INTEGER NOT NULL,               -- trace seq at write time (for truncating torn writes)
  fencing_token INTEGER NOT NULL,
  env_hash TEXT NOT NULL,
  bundle_ref TEXT NOT NULL,           -- ref into a SnapshotStore: {messages, loop_state, action}
  written_at REAL NOT NULL,
  PRIMARY KEY (run_id, step, kind)
);
"""


class SqliteCoordinator:
    def __init__(self, path: str = ":memory:", clock=None) -> None:
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()
        self.clock = clock

    def _now(self) -> float:
        return self.clock.perf() if self.clock else time.time()

    # ---- submit / claim / heartbeat -------------------------------------------
    async def submit(self, spec: JobSpec, priority: int = 0) -> str:
        job_id = "job_" + uuid.uuid4().hex[:10]
        with self.db:
            self.db.execute(
                "INSERT INTO jobs (job_id, run_id, spec, priority, created_at) VALUES (?,?,?,?,?)",
                (job_id, spec.run_id, json.dumps(spec.__dict__), priority, self._now()))
        return job_id

    async def claim(self, owner: str, lease_seconds: float = 30) -> Job | None:
        """Atomically claim the oldest PENDING job, or one whose lease expired
        (a crashed worker's job). Bumping fencing_token on every claim is what
        fences out whichever worker held the job before."""
        now = self._now()
        with self.db:
            row = self.db.execute(
                "SELECT job_id FROM jobs WHERE status='PENDING' "
                "   OR (status='LEASED' AND lease_expires_at < ?) "
                "ORDER BY priority DESC, created_at LIMIT 1", (now,)).fetchone()
            if row is None:
                return None
            job_id = row["job_id"]
            self.db.execute(
                "UPDATE jobs SET status='LEASED', lease_owner=?, lease_expires_at=?, "
                "   fencing_token=fencing_token+1, attempt=attempt+1 WHERE job_id=?",
                (owner, now + lease_seconds, job_id))
            return self._get(job_id)

    async def heartbeat(self, job_id: str, fencing_token: int, lease_seconds: float = 30) -> bool:
        now = self._now()
        with self.db:
            cur = self.db.execute(
                "UPDATE jobs SET lease_expires_at=? WHERE job_id=? AND fencing_token=? AND status='LEASED'",
                (now + lease_seconds, job_id, fencing_token))
            return cur.rowcount > 0

    async def get(self, job_id: str) -> Job | None:
        return self._get(job_id)

    def _get(self, job_id: str) -> Job | None:
        row = self.db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        return Job(row["job_id"], row["run_id"], JobSpec(**json.loads(row["spec"])),
                   row["status"], row["attempt"], row["fencing_token"])

    # ---- checkpoints ------------------------------------------------------------
    async def write_checkpoint(self, job_id: str, ckpt: Checkpoint) -> bool:
        """Token-gated write: rejected (returns False) if `ckpt.fencing_token`
        is no longer the job's current token, i.e. another worker took over."""
        with self.db:
            cur = self.db.execute(
                "INSERT OR REPLACE INTO checkpoints "
                "  (run_id, step, kind, seq, fencing_token, env_hash, bundle_ref, written_at) "
                "SELECT ?,?,?,?,?,?,?,? WHERE (SELECT fencing_token FROM jobs WHERE job_id=?) = ?",
                (ckpt.run_id, ckpt.step, ckpt.kind, ckpt.seq, ckpt.fencing_token, ckpt.env_hash,
                 ckpt.bundle_ref, self._now(), job_id, ckpt.fencing_token))
            return cur.rowcount > 0

    async def latest_checkpoint(self, run_id: str) -> Checkpoint | None:
        # Within the highest step number, prefer the most-progressed kind:
        # step (fully confirmed) > tool (some calls durably applied) > intent.
        row = self.db.execute(
            "SELECT * FROM checkpoints WHERE run_id=? ORDER BY step DESC, "
            "  CASE kind WHEN 'step' THEN 2 WHEN 'tool' THEN 1 ELSE 0 END DESC LIMIT 1",
            (run_id,)).fetchone()
        if row is None:
            return None
        return Checkpoint(row["run_id"], row["step"], row["kind"], row["seq"],
                          row["fencing_token"], row["env_hash"], row["bundle_ref"])

    # ---- completion ---------------------------------------------------------------
    async def complete(self, job_id: str, fencing_token: int, status: str,
                       result: dict | None = None) -> bool:
        with self.db:
            cur = self.db.execute(
                "UPDATE jobs SET status=?, result=? WHERE job_id=? AND fencing_token=? AND status='LEASED'",
                (status, json.dumps(result or {}), job_id, fencing_token))
            return cur.rowcount > 0


# Historical name, kept so existing callers (`from rewind import Coordinator`)
# keep working.
Coordinator = SqliteCoordinator
