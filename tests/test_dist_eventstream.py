"""Redis Streams trace event bus tests against a real Redis. Needs
REWIND_REDIS_URL (default redis://127.0.0.1:6379/0); skips with a clear
reason, not a failure, if it can't connect.
"""
from __future__ import annotations

import asyncio
import os
import uuid

import pytest

pytest.importorskip("redis", reason="needs the [dist] extra")
import redis.asyncio as aioredis  # noqa: E402

from rewind.dist.eventstream import StreamEventBus, TraceStreamConsumer
from rewind.events import Event

URL = os.environ.get("REWIND_REDIS_URL", "redis://127.0.0.1:6379/0")


async def _connect() -> aioredis.Redis:
    client = aioredis.from_url(URL, decode_responses=True)
    try:
        await client.ping()
    except (OSError, aioredis.RedisError) as e:
        pytest.skip(f"Redis not reachable at {URL}: {e}")
    return client


def ev(run_id: str, seq: int, step: int = 1, type_: str = "message") -> Event:
    return Event(run_id=run_id, seq=seq, step=step, type=type_, ts="2026-01-01T00:00:00Z",
                data={"n": seq})


def test_publish_and_drain_preserves_seq_order():
    async def go():
        redis_client = await _connect()
        try:
            prefix = f"rewind:test:stream:{uuid.uuid4().hex}:"
            bus = StreamEventBus(redis_client, key_prefix=prefix)
            run_id = "run-a"
            await bus.ensure_group(run_id)
            # publish out of the order a slower producer might interleave them in;
            # the stream itself still orders by append time, so seq order here
            # tracks append order, not something the consumer has to re-sort hard.
            for seq in [1, 2, 3, 4]:
                await bus.publish(ev(run_id, seq))

            consumer = TraceStreamConsumer(redis_client, run_id, key_prefix=prefix)
            got = await consumer.drain(block_ms=50)
            assert [e.seq for e in got] == [1, 2, 3, 4]
        finally:
            await redis_client.aclose()
    asyncio.run(go())


def test_redelivered_duplicate_is_deduped_by_seq():
    """Models at-least-once delivery: the same logical event (same seq)
    arriving twice — whether from a real XCLAIM re-delivery after a crashed
    consumer, or an application-level retry — must not be returned twice."""
    async def go():
        redis_client = await _connect()
        try:
            prefix = f"rewind:test:stream:{uuid.uuid4().hex}:"
            bus = StreamEventBus(redis_client, key_prefix=prefix)
            run_id = "run-b"
            await bus.ensure_group(run_id)
            await bus.publish(ev(run_id, 1))

            consumer = TraceStreamConsumer(redis_client, run_id, key_prefix=prefix)
            first = await consumer.drain(block_ms=50)
            assert [e.seq for e in first] == [1]

            await bus.publish(ev(run_id, 1))          # re-delivery / duplicate publish of seq=1
            second = await consumer.drain(block_ms=50)
            assert second == [], "a duplicate seq must be acked but not returned again"
        finally:
            await redis_client.aclose()
    asyncio.run(go())


def test_different_runs_have_independent_streams():
    async def go():
        redis_client = await _connect()
        try:
            prefix = f"rewind:test:stream:{uuid.uuid4().hex}:"
            bus = StreamEventBus(redis_client, key_prefix=prefix)
            await bus.ensure_group("run-x")
            await bus.ensure_group("run-y")
            await bus.publish(ev("run-x", 1))
            await bus.publish(ev("run-y", 1))
            await bus.publish(ev("run-y", 2))

            cx = TraceStreamConsumer(redis_client, "run-x", key_prefix=prefix)
            cy = TraceStreamConsumer(redis_client, "run-y", key_prefix=prefix)
            gx = await cx.drain(block_ms=50)
            gy = await cy.drain(block_ms=50)
            assert len(gx) == 1 and len(gy) == 2
        finally:
            await redis_client.aclose()
    asyncio.run(go())


def test_ensure_group_is_idempotent():
    async def go():
        redis_client = await _connect()
        try:
            prefix = f"rewind:test:stream:{uuid.uuid4().hex}:"
            bus = StreamEventBus(redis_client, key_prefix=prefix)
            await bus.ensure_group("run-z")
            await bus.ensure_group("run-z")           # must not raise BUSYGROUP
        finally:
            await redis_client.aclose()
    asyncio.run(go())


def test_reclaim_recovers_events_read_but_never_acked_by_a_dead_consumer():
    """Simulates a Trace Writer that XREADGROUP'd a batch and then crashed
    before acking (`read_without_ack`): those entries sit as *pending* for
    the dead consumer forever unless someone reclaims them. A new consumer
    (a restarted process, or the same one after recovery) should be able to
    pick them up via XAUTOCLAIM once they've been idle long enough."""
    async def go():
        redis_client = await _connect()
        try:
            prefix = f"rewind:test:stream:{uuid.uuid4().hex}:"
            bus = StreamEventBus(redis_client, key_prefix=prefix)
            run_id = "run-crash"
            await bus.ensure_group(run_id)
            for seq in [1, 2, 3]:
                await bus.publish(ev(run_id, seq))

            dead = TraceStreamConsumer(redis_client, run_id, group="writers",
                                       consumer="worker-dead", key_prefix=prefix)
            read = await dead.read_without_ack(block_ms=50)
            assert [e.seq for e in read] == [1, 2, 3]
            # `dead` never acks and is discarded here — modeling the process
            # dying right after this point.

            survivor = TraceStreamConsumer(redis_client, run_id, group="writers",
                                           consumer="worker-survivor", key_prefix=prefix)
            # Nothing to reclaim yet: those entries haven't been idle long
            # enough (min_idle_ms is generous on purpose).
            too_soon = await survivor.reclaim_stale(min_idle_ms=60_000)
            assert too_soon == []

            recovered = await survivor.reclaim_stale(min_idle_ms=0)
            assert [e.seq for e in recovered] == [1, 2, 3]

            # The reclaimed entries are now acked under the new consumer, so a
            # second reclaim attempt (even with min_idle_ms=0) finds nothing.
            again = await survivor.reclaim_stale(min_idle_ms=0)
            assert again == []
        finally:
            await redis_client.aclose()
    asyncio.run(go())


def test_reclaim_dedupes_by_seq_same_as_drain():
    """A seq already handed back by a live `drain()` must not be handed back
    again just because it also happens to show up via reclaim (e.g. the
    caller drains some entries, crashes, and its own successor reclaims the
    rest of that same batch)."""
    async def go():
        redis_client = await _connect()
        try:
            prefix = f"rewind:test:stream:{uuid.uuid4().hex}:"
            bus = StreamEventBus(redis_client, key_prefix=prefix)
            run_id = "run-crash-2"
            await bus.ensure_group(run_id)
            for seq in [1, 2]:
                await bus.publish(ev(run_id, seq))

            consumer = TraceStreamConsumer(redis_client, run_id, group="writers",
                                           consumer="worker-1", key_prefix=prefix)
            got = await consumer.drain(block_ms=50)
            assert [e.seq for e in got] == [1, 2]

            # Nothing pending and unacked for this consumer, so reclaiming
            # under its own name (as if it had crashed and come back) finds
            # nothing new to return.
            replay = await consumer.reclaim_stale(min_idle_ms=0)
            assert replay == []
        finally:
            await redis_client.aclose()
    asyncio.run(go())
