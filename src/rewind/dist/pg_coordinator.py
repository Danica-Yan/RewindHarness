"""Postgres-backed Coordinator: the real-infra twin of `checkpoint.Coordinator`.

Same job-queue-with-lease-and-fencing-plus-checkpoint-log protocol as the
sqlite version (see `checkpoint.py`'s module docstring for *why* it's built
the way it is — intent/tool/step checkpoint granularity, fencing tokens,
etc.), same method names, but implemented against real Postgres with
`asyncpg`, so worker.py's `run_worker_step` runs unmodified against either
backend (see `_await.maybe_await`).

The one thing sqlite genuinely could not give us: `claim()` here uses
`SELECT ... FOR UPDATE SKIP LOCKED`, real row-level locking that is safe
under actual concurrent connections (multiple coroutines, multiple
processes, multiple machines) — not just "safe within one Python process,"
which is all a single sqlite connection could ever demonstrate.

Needs `pip install -e ".[dist]"` and a reachable Postgres. Connect with
`PgCoordinator.connect(dsn)`; `tests/test_dist_pg.py` looks for
`REWIND_PG_DSN` (falling back to `postgresql://rewind:rewind@127.0.0.1:5432/rewind_test`)
and skips itself with a clear reason if it can't connect.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

import asyncpg

from ..checkpoint import Checkpoint, Job, JobSpec

DEFAULT_DSN = "postgresql://rewind:rewind@127.0.0.1:5432/rewind_test"

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id text PRIMARY KEY,
  run_id text NOT NULL,
  spec text NOT NULL,
  status text NOT NULL DEFAULT 'PENDING',
  priority integer NOT NULL DEFAULT 0,
  attempt integer NOT NULL DEFAULT 0,
  lease_owner text,
  lease_expires_at timestamptz,
  fencing_token bigint NOT NULL DEFAULT 0,
  created_at timestamptz NOT NULL DEFAULT now(),
  result text
);
CREATE TABLE IF NOT EXISTS checkpoints (
  run_id text NOT NULL,
  step integer NOT NULL,
  kind text NOT NULL,
  seq integer NOT NULL,
  fencing_token bigint NOT NULL,
  env_hash text NOT NULL,
  bundle_ref text NOT NULL,
  written_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (run_id, step, kind)
);
CREATE INDEX IF NOT EXISTS checkpoints_run_id_idx ON checkpoints (run_id, step DESC);
-- claim()'s WHERE clause only ever cares about PENDING/LEASED rows, but
-- without an index Postgres has nothing to scan except the whole table --
-- invisible on a test-sized jobs table, but every claim becomes an O(n)
-- sequential scan once COMPLETED/FAILED history piles up in a real
-- deployment. `now()` isn't allowed in an index predicate (not immutable),
-- so this can't also encode the "lease_expires_at < now()" half of the
-- WHERE clause -- but it still prunes the index down to just the active
-- (PENDING or LEASED) rows, which is the subset that actually matters: the
-- lease-expiry check then runs as a cheap filter over that small set
-- instead of a scan over every job the table has ever held.
CREATE INDEX IF NOT EXISTS jobs_claimable_idx ON jobs (priority DESC, created_at)
  WHERE status IN ('PENDING', 'LEASED');
"""


def _rowcount(status: str) -> int:
    # asyncpg's execute() returns e.g. "INSERT 0 1" / "UPDATE 1"; last token is the count.
    return int(status.rsplit(" ", 1)[-1])


class PgCoordinator:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    @classmethod
    async def connect(cls, dsn: str | None = None) -> "PgCoordinator":
        pool = await asyncpg.create_pool(dsn or DEFAULT_DSN, min_size=1, max_size=10)
        coord = cls(pool)
        await coord.ensure_schema()
        return coord

    async def ensure_schema(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(SCHEMA)

    async def close(self) -> None:
        await self.pool.close()

    # ---- submit / claim / heartbeat -------------------------------------------
    async def submit(self, spec: JobSpec, priority: int = 0) -> str:
        job_id = "job_" + uuid.uuid4().hex[:10]
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO jobs (job_id, run_id, spec, priority) VALUES ($1,$2,$3,$4)",
                job_id, spec.run_id, json.dumps(spec.__dict__), priority)
        return job_id

    async def claim(self, owner: str, lease_seconds: float = 30) -> Job | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE jobs SET status='LEASED', lease_owner=$1,
                       lease_expires_at = now() + ($2 || ' seconds')::interval,
                       fencing_token = fencing_token + 1, attempt = attempt + 1
                WHERE job_id = (
                  SELECT job_id FROM jobs
                  WHERE status='PENDING' OR (status='LEASED' AND lease_expires_at < now())
                  ORDER BY priority DESC, created_at
                  FOR UPDATE SKIP LOCKED LIMIT 1)
                RETURNING *
                """, owner, str(lease_seconds))
            return self._row_to_job(row) if row else None

    async def heartbeat(self, job_id: str, fencing_token: int, lease_seconds: float = 30) -> bool:
        async with self.pool.acquire() as conn:
            status = await conn.execute(
                "UPDATE jobs SET lease_expires_at = now() + ($1 || ' seconds')::interval "
                "WHERE job_id=$2 AND fencing_token=$3 AND status='LEASED'",
                str(lease_seconds), job_id, fencing_token)
            return _rowcount(status) > 0

    async def get(self, job_id: str) -> Job | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM jobs WHERE job_id=$1", job_id)
            return self._row_to_job(row) if row else None

    @staticmethod
    def _row_to_job(row: asyncpg.Record) -> Job:
        return Job(row["job_id"], row["run_id"], JobSpec(**json.loads(row["spec"])),
                   row["status"], row["attempt"], row["fencing_token"])

    # ---- checkpoints ------------------------------------------------------------
    async def write_checkpoint(self, job_id: str, ckpt: Checkpoint) -> bool:
        async with self.pool.acquire() as conn:
            status = await conn.execute(
                """
                INSERT INTO checkpoints (run_id, step, kind, seq, fencing_token, env_hash, bundle_ref)
                SELECT $1, $2, $3, $4, $5, $6, $7
                WHERE (SELECT fencing_token FROM jobs WHERE job_id=$8) = $5
                ON CONFLICT (run_id, step, kind) DO UPDATE SET
                  seq = EXCLUDED.seq, fencing_token = EXCLUDED.fencing_token,
                  env_hash = EXCLUDED.env_hash, bundle_ref = EXCLUDED.bundle_ref,
                  written_at = now()
                """, ckpt.run_id, ckpt.step, ckpt.kind, ckpt.seq, ckpt.fencing_token,
                ckpt.env_hash, ckpt.bundle_ref, job_id)
            # An INSERT whose SELECT source is empty (stale token) reports "INSERT 0 0"
            # even with ON CONFLICT present, since there was no candidate row to conflict on.
            return _rowcount(status) > 0

    async def latest_checkpoint(self, run_id: str) -> Checkpoint | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM checkpoints WHERE run_id=$1 ORDER BY step DESC, "
                "  CASE kind WHEN 'step' THEN 2 WHEN 'tool' THEN 1 ELSE 0 END DESC LIMIT 1",
                run_id)
            if row is None:
                return None
            return Checkpoint(row["run_id"], row["step"], row["kind"], row["seq"],
                              row["fencing_token"], row["env_hash"], row["bundle_ref"])

    # ---- completion ---------------------------------------------------------------
    async def complete(self, job_id: str, fencing_token: int, status: str,
                       result: dict[str, Any] | None = None) -> bool:
        async with self.pool.acquire() as conn:
            st = await conn.execute(
                "UPDATE jobs SET status=$1, result=$2 WHERE job_id=$3 AND fencing_token=$4 "
                "  AND status='LEASED'",
                status, json.dumps(result or {}), job_id, fencing_token)
            return _rowcount(st) > 0
