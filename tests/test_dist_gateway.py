"""LLMGateway (cache + single-flight + rate limiter) tests against a real
Redis. Needs REWIND_REDIS_URL (default redis://127.0.0.1:6379/0); skips with
a clear reason, not a failure, if it can't connect.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid

import pytest

pytest.importorskip("redis", reason="needs the [dist] extra")
import redis.asyncio as aioredis  # noqa: E402

from rewind.dist.gateway import (
    CircuitBreaker,
    CircuitOpen,
    DualTokenBucket,
    LLMGateway,
    RateLimitExceeded,
    TokenBucket,
)
from rewind.providers import LLMResponse, ProviderError

URL = os.environ.get("REWIND_REDIS_URL", "redis://127.0.0.1:6379/0")


async def _connect() -> aioredis.Redis:
    client = aioredis.from_url(URL, decode_responses=True)
    try:
        await client.ping()
    except (OSError, aioredis.RedisError) as e:
        pytest.skip(f"Redis not reachable at {URL}: {e}")
    return client


class CountingProvider:
    def __init__(self, delay: float = 0.0) -> None:
        self.calls = 0
        self.delay = delay

    async def complete(self, messages, tools, params) -> LLMResponse:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return LLMResponse(text=f"call #{self.calls}", usage={"input_tokens": 1, "output_tokens": 1})


class FailingProvider:
    """Always raises a ProviderError; `status` lets a test pick between a
    generic failure (counted toward the threshold) and a 429 (opens the
    circuit immediately)."""

    def __init__(self, status: int = 500, delay: float = 0.0) -> None:
        self.calls = 0
        self.status = status
        self.delay = delay

    async def complete(self, messages, tools, params) -> LLMResponse:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        raise ProviderError(f"boom (status={self.status})", retryable=True, status=self.status)


MSGS = [{"role": "user", "content": "hi"}]
TOOLS: list[dict] = []


def test_identical_request_is_served_from_cache_not_a_second_real_call():
    async def go():
        redis_client = await _connect()
        try:
            inner = CountingProvider()
            gw = LLMGateway(inner, redis_client, key_prefix=f"rewind:test:{uuid.uuid4().hex}:")
            r1 = await gw.complete(MSGS, TOOLS, {"temperature": 0})
            r2 = await gw.complete(MSGS, TOOLS, {"temperature": 0})
            assert inner.calls == 1
            assert r1.text == r2.text == "call #1"
            assert gw.stats == {"hits": 1, "misses": 1, "single_flight_waits": 0, "real_calls": 1}
        finally:
            await redis_client.aclose()
    asyncio.run(go())


def test_different_params_are_different_cache_entries():
    async def go():
        redis_client = await _connect()
        try:
            inner = CountingProvider()
            gw = LLMGateway(inner, redis_client, key_prefix=f"rewind:test:{uuid.uuid4().hex}:")
            await gw.complete(MSGS, TOOLS, {"temperature": 0, "sample_idx": 0})
            await gw.complete(MSGS, TOOLS, {"temperature": 0, "sample_idx": 1})
            assert inner.calls == 2, "live multi-sampling must not collapse into one cache entry"
        finally:
            await redis_client.aclose()
    asyncio.run(go())


def test_concurrent_identical_requests_single_flight_to_one_real_call():
    async def go():
        redis_client = await _connect()
        try:
            inner = CountingProvider(delay=0.2)
            gw = LLMGateway(inner, redis_client, key_prefix=f"rewind:test:{uuid.uuid4().hex}:",
                            wait_timeout=5.0)
            results = await asyncio.gather(*[gw.complete(MSGS, TOOLS, {"temperature": 0})
                                             for _ in range(6)])
            assert inner.calls == 1, "6 identical concurrent requests should make 1 real call"
            assert all(r.text == results[0].text for r in results)
            assert gw.stats["single_flight_waits"] == 5
        finally:
            await redis_client.aclose()
    asyncio.run(go())


def test_rate_limiter_blocks_once_bucket_is_exhausted():
    async def go():
        redis_client = await _connect()
        try:
            bucket_key = f"rewind:test:bucket:{uuid.uuid4().hex}"
            limiter = TokenBucket(redis_client, bucket_key, capacity=1, refill_per_second=0.01)
            inner = CountingProvider()
            gw = LLMGateway(inner, redis_client, key_prefix=f"rewind:test:{uuid.uuid4().hex}:",
                            rate_limiter=limiter)
            await gw.complete(MSGS, TOOLS, {"temperature": 0, "sample_idx": 0})   # consumes the 1 token
            with pytest.raises(RateLimitExceeded):
                await gw.complete(MSGS, TOOLS, {"temperature": 0, "sample_idx": 1})  # different key, no cache
            assert inner.calls == 1
        finally:
            await redis_client.aclose()
    asyncio.run(go())


def test_token_bucket_refills_over_time():
    async def go():
        redis_client = await _connect()
        try:
            bucket_key = f"rewind:test:bucket:{uuid.uuid4().hex}"
            limiter = TokenBucket(redis_client, bucket_key, capacity=1, refill_per_second=10)
            await limiter.acquire()
            with pytest.raises(RateLimitExceeded):
                await limiter.acquire()
            await asyncio.sleep(0.15)               # refill_per_second=10 -> ~1.5 tokens back
            await limiter.acquire()                  # should succeed now
        finally:
            await redis_client.aclose()
    asyncio.run(go())


def test_unlock_is_compare_and_delete_not_a_blind_delete():
    """If this worker's lock TTL expired and a *different* worker has since
    acquired the same lock, releasing must not delete the new holder's lock."""
    async def go():
        redis_client = await _connect()
        try:
            gw = LLMGateway(CountingProvider(), redis_client,
                            key_prefix=f"rewind:test:{uuid.uuid4().hex}:")
            lock_key = "rewind:test:cas-lock:" + uuid.uuid4().hex
            my_token = "worker-A-token"
            other_token = "worker-B-token"
            await redis_client.set(lock_key, my_token, nx=True, px=100)
            await asyncio.sleep(0.15)                                    # my lease expires
            await redis_client.set(lock_key, other_token, nx=True)       # B acquires it

            await gw._unlock(keys=[lock_key], args=[my_token])           # A tries to release its (stale) lock
            assert await redis_client.get(lock_key) == other_token, \
                "a stale unlock must never delete a newer worker's lock"
        finally:
            await redis_client.aclose()
    asyncio.run(go())


def test_dual_bucket_blocks_on_rpm_even_when_tpm_has_room():
    async def go():
        redis_client = await _connect()
        try:
            key = f"rewind:test:dual:{uuid.uuid4().hex}"
            bucket = DualTokenBucket(redis_client, key, rpm_capacity=1, rpm_refill_per_second=0.001,
                                     tpm_capacity=1_000_000, tpm_refill_per_second=1000)
            await bucket.acquire(cost=1, token_cost=10)           # consumes the sole RPM token
            with pytest.raises(RateLimitExceeded):
                await bucket.acquire(cost=1, token_cost=10)       # TPM is wide open, RPM is not
        finally:
            await redis_client.aclose()
    asyncio.run(go())


def test_dual_bucket_blocks_on_tpm_even_when_rpm_has_room():
    async def go():
        redis_client = await _connect()
        try:
            key = f"rewind:test:dual:{uuid.uuid4().hex}"
            bucket = DualTokenBucket(redis_client, key, rpm_capacity=1_000_000, rpm_refill_per_second=1000,
                                     tpm_capacity=100, tpm_refill_per_second=0.001)
            await bucket.acquire(cost=1, token_cost=100)          # drains the TPM budget
            with pytest.raises(RateLimitExceeded):
                await bucket.acquire(cost=1, token_cost=1)        # RPM is wide open, TPM is not
        finally:
            await redis_client.aclose()
    asyncio.run(go())


def test_dual_bucket_is_all_or_nothing_neither_dimension_partially_spent():
    """A request that fails on TPM must not have silently consumed RPM budget
    (that would make the two dimensions drift out of sync with reality)."""
    async def go():
        redis_client = await _connect()
        try:
            key = f"rewind:test:dual:{uuid.uuid4().hex}"
            bucket = DualTokenBucket(redis_client, key, rpm_capacity=5, rpm_refill_per_second=0.001,
                                     tpm_capacity=10, tpm_refill_per_second=0.001)
            with pytest.raises(RateLimitExceeded):
                await bucket.acquire(cost=1, token_cost=999)      # fails on TPM alone
            # RPM must still have all 5 tokens: this should succeed 5 times in a row.
            for _ in range(5):
                await bucket.acquire(cost=1, token_cost=1)
            with pytest.raises(RateLimitExceeded):
                await bucket.acquire(cost=1, token_cost=1)        # now RPM really is exhausted
        finally:
            await redis_client.aclose()
    asyncio.run(go())


def test_circuit_breaker_opens_after_failure_threshold_and_blocks_calls():
    async def go():
        redis_client = await _connect()
        try:
            name = f"test-{uuid.uuid4().hex}"
            inner = FailingProvider(status=500)
            cb = CircuitBreaker(redis_client, name, failure_threshold=3, window_seconds=30,
                                cooldown_seconds=1)
            gw = LLMGateway(inner, redis_client, key_prefix=f"rewind:test:{uuid.uuid4().hex}:",
                            circuit_breaker=cb)
            for i in range(3):
                with pytest.raises(ProviderError):
                    await gw.complete(MSGS, TOOLS, {"temperature": 0, "sample_idx": i})
            assert inner.calls == 3, "breaker must not trip before the threshold is reached"
            # 4th distinct request: breaker should now be open and short-circuit
            # *without* even reaching the inner provider.
            with pytest.raises(CircuitOpen):
                await gw.complete(MSGS, TOOLS, {"temperature": 0, "sample_idx": 99})
            assert inner.calls == 3, "an open circuit must not call the inner provider at all"
        finally:
            await redis_client.aclose()
    asyncio.run(go())


def test_circuit_breaker_opens_immediately_on_a_single_429():
    async def go():
        redis_client = await _connect()
        try:
            name = f"test-{uuid.uuid4().hex}"
            inner = FailingProvider(status=429)
            cb = CircuitBreaker(redis_client, name, failure_threshold=5, cooldown_seconds=5)
            gw = LLMGateway(inner, redis_client, key_prefix=f"rewind:test:{uuid.uuid4().hex}:",
                            circuit_breaker=cb)
            with pytest.raises(ProviderError):
                await gw.complete(MSGS, TOOLS, {"temperature": 0})
            assert inner.calls == 1
            # A single 429 opens the circuit right away, well below the
            # generic failure_threshold of 5.
            with pytest.raises(CircuitOpen):
                await gw.complete(MSGS, TOOLS, {"temperature": 0, "sample_idx": "other"})
            assert inner.calls == 1
        finally:
            await redis_client.aclose()
    asyncio.run(go())


def test_circuit_breaker_recovers_after_cooldown_and_a_success_clears_it():
    async def go():
        redis_client = await _connect()
        try:
            name = f"test-{uuid.uuid4().hex}"
            failing = FailingProvider(status=500)
            cb = CircuitBreaker(redis_client, name, failure_threshold=1, cooldown_seconds=1)
            gw = LLMGateway(failing, redis_client, key_prefix=f"rewind:test:{uuid.uuid4().hex}:",
                            circuit_breaker=cb)
            with pytest.raises(ProviderError):
                await gw.complete(MSGS, TOOLS, {"temperature": 0})
            with pytest.raises(CircuitOpen):
                await gw.complete(MSGS, TOOLS, {"temperature": 0, "sample_idx": "x"})
            await asyncio.sleep(1.1)                        # cooldown_seconds (rounded up to 1s) expires
            good = CountingProvider()
            gw2 = LLMGateway(good, redis_client, key_prefix=gw.prefix, circuit_breaker=cb)
            r = await gw2.complete(MSGS, TOOLS, {"temperature": 0, "sample_idx": "y"})
            assert r.text == "call #1"
            assert good.calls == 1, "circuit should allow calls again once the open flag's TTL expires"
        finally:
            await redis_client.aclose()
    asyncio.run(go())


def test_single_flight_waiter_wakes_on_failure_and_retries_itself():
    """If the leader's real call *fails*, waiters must not hang for the full
    wait_timeout: the failure publish should wake them immediately so they
    fall through and try their own call."""
    async def go():
        redis_client = await _connect()
        try:
            prefix = f"rewind:test:{uuid.uuid4().hex}:"
            leader_inner = FailingProvider(status=500, delay=0.2)
            leader_gw = LLMGateway(leader_inner, redis_client, key_prefix=prefix, wait_timeout=5.0)
            follower_inner = CountingProvider()
            follower_gw = LLMGateway(follower_inner, redis_client, key_prefix=prefix, wait_timeout=5.0)

            async def leader():
                with pytest.raises(ProviderError):
                    await leader_gw.complete(MSGS, TOOLS, {"temperature": 0})

            async def follower():
                await asyncio.sleep(0.02)          # let the leader grab the lock first
                start = time.monotonic()
                r = await follower_gw.complete(MSGS, TOOLS, {"temperature": 0})
                return r, time.monotonic() - start

            _, (resp, elapsed) = await asyncio.gather(leader(), follower())
            assert resp.text == "call #1"
            assert follower_inner.calls == 1, "follower must retry for itself after the leader failed"
            assert elapsed < 1.0, "pub/sub wake-on-failure should be near-instant, not a timeout wait"
        finally:
            await redis_client.aclose()
    asyncio.run(go())


def test_slow_real_call_keeps_the_single_flight_lock_via_heartbeat():
    """Regression test for a real gap: the single-flight lock originally had
    no heartbeat (unlike the job lease, which has heartbeat()), so a real
    call slower than `lock_ttl_ms` would let the lock expire mid-call and a
    second worker's `SET NX` would steal it -- single-flight silently
    defeated, the SAME request paying for a real call twice. Here
    `lock_ttl_ms=100` is deliberately much shorter than the provider's
    400ms delay: without the heartbeat renewing it roughly every 33ms, the
    lock would expire 3-4 times over before the leader's call even returns,
    and a second gateway racing in partway through would have grabbed it
    and made its own real call. With the heartbeat, the lock survives for
    the whole call and the follower waits for the leader's result instead."""
    async def go():
        redis_client = await _connect()
        try:
            prefix = f"rewind:test:{uuid.uuid4().hex}:"
            shared_inner = CountingProvider(delay=0.4)
            leader_gw = LLMGateway(shared_inner, redis_client, key_prefix=prefix,
                                   lock_ttl_ms=100, wait_timeout=5.0)
            follower_gw = LLMGateway(shared_inner, redis_client, key_prefix=prefix,
                                     lock_ttl_ms=100, wait_timeout=5.0)

            async def leader():
                return await leader_gw.complete(MSGS, TOOLS, {"temperature": 0})

            async def follower():
                await asyncio.sleep(0.15)          # well past the old (buggy) 100ms lock TTL
                return await follower_gw.complete(MSGS, TOOLS, {"temperature": 0})

            r_leader, r_follower = await asyncio.gather(leader(), follower())
            assert r_leader.text == r_follower.text == "call #1"
            assert shared_inner.calls == 1, (
                "the lock should have survived the whole slow call via its heartbeat -- "
                "a second real call means it was stolen mid-flight")
        finally:
            await redis_client.aclose()
    asyncio.run(go())
