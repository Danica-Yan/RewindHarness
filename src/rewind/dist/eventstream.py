"""Trace event bus on Redis Streams — the plan's "NATS JetStream (or Redis
Streams)" choice, Redis Streams half (section 4.5/architecture diagram's
Trace Writer box).

Each run gets its own stream key (`rewind:trace:<run_id>`), so per-run
ordering is free: a single Redis stream is always delivered in the order
entries were added. A consumer group lets multiple Trace Writer instances
share the work, but delivery through a consumer group is at-least-once —
a consumer that dies after XREADGROUP but before XACK will have its pending
entries re-delivered to someone else — so the consumer side must dedupe,
which is exactly what `Event.seq` is for (see events.py: strictly increasing
per run_id). `TraceStreamConsumer.drain` acks every entry it reads (even a
duplicate) but only returns events whose `seq` it hasn't already returned,
so a re-delivered duplicate is a no-op for the caller.
"""
from __future__ import annotations

import redis.asyncio as aioredis

from ..events import Event


class StreamEventBus:
    def __init__(self, redis_client: aioredis.Redis, key_prefix: str = "rewind:trace:") -> None:
        self.redis = redis_client
        self.prefix = key_prefix

    def stream_key(self, run_id: str) -> str:
        return f"{self.prefix}{run_id}"

    async def ensure_group(self, run_id: str, group: str = "writers") -> None:
        key = self.stream_key(run_id)
        try:
            await self.redis.xgroup_create(key, group, id="0", mkstream=True)
        except aioredis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def publish(self, event: Event) -> str:
        key = self.stream_key(event.run_id)
        return await self.redis.xadd(key, {"payload": event.model_dump_json()})


class TraceStreamConsumer:
    """Models the "Trace Writer" service consuming a run's stream: reads via
    a consumer group, dedupes by `seq`, returns newly-seen events in order."""

    def __init__(self, redis_client: aioredis.Redis, run_id: str, group: str = "writers",
                consumer: str = "c1", key_prefix: str = "rewind:trace:") -> None:
        self.redis = redis_client
        self.run_id = run_id
        self.group = group
        self.consumer = consumer
        self.key = f"{key_prefix}{run_id}"
        self.seen_seqs: set[int] = set()

    async def drain(self, count: int = 100, block_ms: int = 100) -> list[Event]:
        """Read once (pending re-deliveries first is Redis's own `>` semantics
        for a fresh read — pass id='0' instead of '>' to a call site that
        wants this consumer's own still-pending backlog), ack everything read,
        and return only the events not already seen, sorted by seq."""
        resp = await self.redis.xreadgroup(self.group, self.consumer, {self.key: ">"},
                                           count=count, block=block_ms)
        out: list[Event] = []
        if not resp:
            return out
        for _stream_key, entries in resp:
            for entry_id, fields in entries:
                ev = Event.model_validate_json(fields["payload"])
                if ev.seq not in self.seen_seqs:
                    self.seen_seqs.add(ev.seq)
                    out.append(ev)
                await self.redis.xack(self.key, self.group, entry_id)
        out.sort(key=lambda e: e.seq)
        return out

    async def read_without_ack(self, count: int = 100, block_ms: int = 100) -> list[Event]:
        """Like `drain()` but deliberately skips the XACK — a fresh read (`>`)
        marks these entries as this consumer's *pending*, then we stop, just
        like a process that reads a batch and crashes before it finishes
        processing (and therefore before it would normally ack). Only used by
        crash-recovery tests to set that scenario up; real callers use
        `drain()`, which acks once it has actually handed events back."""
        resp = await self.redis.xreadgroup(self.group, self.consumer, {self.key: ">"},
                                           count=count, block=block_ms)
        out: list[Event] = []
        if not resp:
            return out
        for _stream_key, entries in resp:
            for _entry_id, fields in entries:
                out.append(Event.model_validate_json(fields["payload"]))
        out.sort(key=lambda e: e.seq)
        return out

    async def reclaim_stale(self, min_idle_ms: int = 30_000, count: int = 100) -> list[Event]:
        """Recover entries some other consumer (or this one, in an earlier
        life) read via XREADGROUP but never acked — most likely because that
        process crashed between reading and finishing. `XAUTOCLAIM` reassigns
        any pending entry idle for at least `min_idle_ms` to *this* consumer
        (this is the "someone died mid-processing" recovery path; a live
        consumer's own in-flight entries are simply not idle long enough yet
        to be swept up here). Claimed entries go through the same seq-based
        dedup as `drain()` before being acked, so a duplicate delivery is
        still a no-op for the caller even though it did genuinely happen
        twice at the transport level."""
        out: list[Event] = []
        start_id = "0-0"
        while True:
            next_id, entries, _deleted = await self.redis.xautoclaim(
                self.key, self.group, self.consumer, min_idle_time=min_idle_ms,
                start_id=start_id, count=count)
            if not entries:
                break
            for entry_id, fields in entries:
                ev = Event.model_validate_json(fields["payload"])
                if ev.seq not in self.seen_seqs:
                    self.seen_seqs.add(ev.seq)
                    out.append(ev)
                await self.redis.xack(self.key, self.group, entry_id)
            if next_id in ("0-0", "0"):
                break
            start_id = next_id
        out.sort(key=lambda e: e.seq)
        return out
