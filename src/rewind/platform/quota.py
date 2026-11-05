"""Usage entitlement: how MUCH a principal may use a capability.

Access control answers "may this agent call send_email at all?". Quotas
answer "how many times, per tenant / user / agent / run, per window?" and
they are stateful, shared across every worker, so they live in a store.

  QuotaRule     data, like access rules: which capabilities, counted per
                what (tenant, user, agent or run), a limit, an optional fixed
                window, a cost per call
  QuotaMeter    turns one call into the buckets it must be charged to
  QuotaStore    charges a set of buckets ALL-OR-NOTHING: either every
                matching rule has room and all are incremented, or none is

Three properties the stores guarantee, and the tests pin down:

  * No over-spend under concurrency. Postgres does the check and the
    increment in one conditional UPDATE (`used + cost <= limit`), so N
    workers racing for the last unit cannot all win it.
  * Exactly-once charging. A charge is keyed by (run, action_id). A worker
    that crashed after charging and is resumed replays the same recorded
    action with the same action_id -- and gets the recorded result back
    instead of being charged again (the same idea as `env.applied`).
  * Deterministic replay. A quota outcome is external state, like an LLM
    response: the trace records it, and `replay_run` serves the recorded
    outcome instead of touching the store (see `RecordedQuota`).

Charging happens after access control allowed the call and before it runs
(attempt-based: a call that then fails at the tool is still counted).
"""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .capability import CapabilityDescriptor
from .identity import Principal

Per = Literal["tenant", "user", "agent", "run"]


class QuotaRule(BaseModel):
    id: str
    capabilities: list[str]                       # globs over capability ids
    per: Per
    limit: int = Field(gt=0)
    window_seconds: int | None = Field(default=None, gt=0)   # None: never resets
    cost: int = Field(default=1, gt=0)
    tenants: list[str] = Field(default_factory=list)          # empty = every tenant
    description: str = ""

    @field_validator("capabilities")
    @classmethod
    def _caps(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("a quota rule must name at least one capability glob")
        return v

    def matches(self, p: Principal, cap: CapabilityDescriptor) -> bool:
        return ((not self.tenants or p.tenant_id in self.tenants)
                and any(fnmatchcase(cap.id, c) for c in self.capabilities))

    def subject(self, p: Principal, run_id: str) -> str:
        # tenant-qualified, so two tenants' "u1" never share a bucket
        return {"tenant": p.tenant_id, "user": f"{p.tenant_id}/{p.user_id}",
                "agent": f"{p.tenant_id}/{p.agent_id}", "run": run_id}[self.per]


@dataclass
class Charge:
    key: str            # bucket identity: rule, subject, window
    rule_id: str
    limit: int
    cost: int
    reset_at: float | None


@dataclass
class QuotaResult:
    ok: bool
    charged: list[dict] = field(default_factory=list)   # [{rule, used, limit}] after charging
    exceeded: str | None = None                          # first rule without room
    retry_after: float | None = None                     # seconds, None = no window
    deduplicated: bool = False                           # this action was already charged

    def to_event(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_event(cls, d: dict) -> QuotaResult:
        return cls(**d)


class QuotaMeter:
    """Rules + store + clock. `now` returns epoch seconds (inject for tests)."""

    def __init__(self, rules: list[QuotaRule], store: Any, now: Callable[[], float] | None = None) -> None:
        ids = [r.id for r in rules]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate quota rule ids")
        import time
        self.rules = list(rules)
        self.store = store
        self.now = now or time.time

    def charges(self, p: Principal, cap: CapabilityDescriptor, run_id: str) -> list[Charge]:
        t = self.now()
        out = []
        for r in self.rules:
            if not r.matches(p, cap):
                continue
            if r.window_seconds:
                start = int(t // r.window_seconds) * r.window_seconds
                key, reset = f"{r.id}|{r.subject(p, run_id)}|{start}", float(start + r.window_seconds)
            else:
                key, reset = f"{r.id}|{r.subject(p, run_id)}|-", None
            out.append(Charge(key, r.id, r.limit, r.cost, reset))
        return out

    def charge(self, p: Principal, cap: CapabilityDescriptor, run_id: str, action_id: str):
        """QuotaResult, or an awaitable of one for an async store. A call no
        rule covers is free and touches no store."""
        items = self.charges(p, cap, run_id)
        if not items:
            return QuotaResult(True)
        return self.store.charge(f"{run_id}:{action_id}", items, self.now())


def _exceeded(item: Charge, used: int, now: float) -> QuotaResult:
    return QuotaResult(False, [{"rule": item.rule_id, "used": used, "limit": item.limit}],
                       exceeded=item.rule_id,
                       retry_after=None if item.reset_at is None else max(0.0, item.reset_at - now))


class InMemoryQuotaStore:
    """Single-process store (a lock makes it thread-safe). Same contract as
    PgQuotaStore; used by the offline tests and single-process demos."""

    def __init__(self) -> None:
        self.used: dict[str, int] = {}
        self._charges: dict[str, QuotaResult] = {}
        self._lock = threading.Lock()

    def charge(self, charge_id: str, items: list[Charge], now: float) -> QuotaResult:
        with self._lock:
            if charge_id in self._charges:
                prev = self._charges[charge_id]
                return QuotaResult(**{**asdict(prev), "deduplicated": True})
            for it in items:
                used = self.used.get(it.key, 0)
                if used + it.cost > it.limit:
                    res = _exceeded(it, used, now)
                    self._charges[charge_id] = res
                    return res
            charged = []
            for it in items:
                self.used[it.key] = self.used.get(it.key, 0) + it.cost
                charged.append({"rule": it.rule_id, "used": self.used[it.key], "limit": it.limit})
            res = QuotaResult(True, charged)
            self._charges[charge_id] = res
            return res


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------
PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS quota_buckets (
  key text PRIMARY KEY,
  used bigint NOT NULL DEFAULT 0,
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS quota_charges (
  charge_id text PRIMARY KEY,
  result jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);
"""


class PgQuotaStore:
    """Postgres-backed store shared by every worker. Needs the [dist] extra.

    One transaction per charge:
      1. claim the charge id (INSERT .. ON CONFLICT DO NOTHING). If another
         transaction holds it, this INSERT waits for that one to finish and
         then conflicts -- the recorded result is returned: exactly once.
      2. inside a savepoint, for each bucket in key order (a fixed lock order,
         so two charges over the same buckets cannot deadlock):
           UPDATE .. SET used = used + cost WHERE key = .. AND used + cost <= limit
         If any bucket has no room, roll back to the savepoint: all or nothing.
      3. store the result on the charge row, commit.

    Old windows' bucket rows are never read again; deleting rows whose key's
    window ended long ago is a periodic job, not part of the hot path.
    """

    def __init__(self, pool: Any) -> None:
        self.pool = pool

    @classmethod
    async def connect(cls, dsn: str) -> PgQuotaStore:
        import asyncpg
        store = cls(await asyncpg.create_pool(dsn, min_size=1, max_size=20))
        async with store.pool.acquire() as conn:
            await conn.execute(PG_SCHEMA)
        return store

    async def close(self) -> None:
        await self.pool.close()

    async def charge(self, charge_id: str, items: list[Charge], now: float) -> QuotaResult:
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                won = await conn.fetchval(
                    "INSERT INTO quota_charges (charge_id) VALUES ($1) "
                    "ON CONFLICT (charge_id) DO NOTHING RETURNING charge_id", charge_id)
                if won is None:
                    prev = await conn.fetchval("SELECT result FROM quota_charges WHERE charge_id=$1",
                                               charge_id)
                    return QuotaResult(**{**json.loads(prev), "deduplicated": True})
                result = await self._apply(conn, items, now)
                await conn.execute("UPDATE quota_charges SET result=$2 WHERE charge_id=$1",
                                   charge_id, json.dumps(result.to_event()))
                return result

    @staticmethod
    async def _apply(conn: Any, items: list[Charge], now: float) -> QuotaResult:
        ordered = sorted(items, key=lambda i: i.key)
        charged: dict[str, dict] = {}
        try:
            async with conn.transaction():                     # savepoint
                for it in ordered:
                    await conn.execute("INSERT INTO quota_buckets (key) VALUES ($1) "
                                       "ON CONFLICT (key) DO NOTHING", it.key)
                    used = await conn.fetchval(
                        "UPDATE quota_buckets SET used = used + $2, updated_at = now() "
                        "WHERE key = $1 AND used + $2 <= $3 RETURNING used", it.key, it.cost, it.limit)
                    if used is None:
                        cur = await conn.fetchval("SELECT used FROM quota_buckets WHERE key=$1", it.key)
                        raise _NoRoom(_exceeded(it, int(cur), now))
                    charged[it.key] = {"rule": it.rule_id, "used": int(used), "limit": it.limit}
        except _NoRoom as e:
            return e.result
        return QuotaResult(True, [charged[i.key] for i in items])


class _NoRoom(Exception):
    def __init__(self, result: QuotaResult) -> None:
        super().__init__(result.exceeded)
        self.result = result


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------
class RecordedQuota:
    """Serves the quota outcomes recorded in a trace's authz events, keyed by
    action id, so a replay never touches (or double-charges) the real store."""

    def __init__(self, events: list[Any]) -> None:
        self._by_action = {f"s{e.step}:c{e.data['idx']}": e.data["quota"] for e in events
                           if e.type == "authz" and "quota" in e.data}

    def charge(self, p: Principal, cap: CapabilityDescriptor, run_id: str, action_id: str) -> QuotaResult:
        rec = self._by_action.get(action_id)
        return QuotaResult(True) if rec is None else QuotaResult.from_event(rec)
