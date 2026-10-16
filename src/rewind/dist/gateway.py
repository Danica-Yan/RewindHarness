"""LLM Gateway: shared response cache, single-flight, a Redis-backed dual
(RPM+TPM) token bucket rate limiter, and a shared circuit breaker — wrapping
any Provider so AgentRunner needs no changes. `LLMGateway` implements the
exact same `async def complete(messages, tools, params)` shape as any other
provider in `providers.py`.

Why this matters once there's more than one worker process: they all need
(1) a shared cache keyed by exact request identity, so a replayed or forked
request that happens to repeat doesn't re-pay for an identical call, (2)
single-flight, so N workers asking the identical question at the same
instant make ONE real call instead of N, (3) a rate limiter that is actually
shared across processes — an in-process limiter is meaningless once there's
more than one process talking to the same provider quota — and (4) a
circuit breaker that is *also* shared: one worker discovering the provider
is down (or rate-limiting everyone) should make every other worker back off
immediately, not rediscover the same 429 independently N times.

Needs `pip install -e ".[dist]"` and a reachable Redis. `tests/test_dist_gateway.py`
looks for `REWIND_REDIS_URL` (falling back to `redis://127.0.0.1:6379/0`) and
skips itself with a clear reason if it can't connect.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from typing import Any

import redis.asyncio as aioredis

from ..providers import LLMResponse, ProviderError
from ..util import request_hash

# Atomic single-dimension token bucket: refill by elapsed time, then try to
# consume `requested` tokens. KEYS[1] = bucket key.
# ARGV = capacity, refill_per_second, now, requested.
_TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if tokens == nil then
  tokens = capacity
  ts = now
end

local elapsed = math.max(0, now - ts)
tokens = math.min(capacity, tokens + elapsed * refill_rate)

local allowed = 0
if tokens >= requested then
  tokens = tokens - requested
  allowed = 1
end

redis.call('HMSET', key, 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('EXPIRE', key, 3600)
return {allowed, tostring(tokens)}
"""

# Two buckets (requests/min and tokens/min) consumed atomically together in
# one script: an all-or-nothing acquire, so a request never partially
# reserves RPM budget while failing on TPM budget (or vice versa) — with two
# separate Lua calls that "all or nothing" property would need a manual
# rollback path; one script makes it free.
# KEYS[1]=rpm key, KEYS[2]=tpm key.
# ARGV = rpm_capacity, rpm_refill, tpm_capacity, tpm_refill, now, req_cost, token_cost.
_DUAL_BUCKET_LUA = """
local function refill(key, capacity, rate, now)
  local data = redis.call('HMGET', key, 'tokens', 'ts')
  local tokens = tonumber(data[1])
  local ts = tonumber(data[2])
  if tokens == nil then tokens = capacity; ts = now end
  local elapsed = math.max(0, now - ts)
  return math.min(capacity, tokens + elapsed * rate)
end

local rpm_key, tpm_key = KEYS[1], KEYS[2]
local rpm_capacity, rpm_rate = tonumber(ARGV[1]), tonumber(ARGV[2])
local tpm_capacity, tpm_rate = tonumber(ARGV[3]), tonumber(ARGV[4])
local now = tonumber(ARGV[5])
local req_cost, tok_cost = tonumber(ARGV[6]), tonumber(ARGV[7])

local rpm_tokens = refill(rpm_key, rpm_capacity, rpm_rate, now)
local tpm_tokens = refill(tpm_key, tpm_capacity, tpm_rate, now)

local allowed = 0
local reason = 'none'
if rpm_tokens < req_cost then
  reason = 'rpm'
elseif tpm_tokens < tok_cost then
  reason = 'tpm'
else
  allowed = 1
  rpm_tokens = rpm_tokens - req_cost
  tpm_tokens = tpm_tokens - tok_cost
end

redis.call('HMSET', rpm_key, 'tokens', tostring(rpm_tokens), 'ts', tostring(now))
redis.call('EXPIRE', rpm_key, 3600)
redis.call('HMSET', tpm_key, 'tokens', tostring(tpm_tokens), 'ts', tostring(now))
redis.call('EXPIRE', tpm_key, 3600)

return {allowed, reason, tostring(rpm_tokens), tostring(tpm_tokens)}
"""

# Compare-and-delete: only release the lock if it still holds *our* token, so
# a worker can never release a lock some other worker has since acquired
# (e.g. after this worker stalled past the lock's own TTL).
_UNLOCK_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
else
  return 0
end
"""

# Compare-and-extend: the single-flight heartbeat's version of the same
# "only touch it if it's still ours" guard as _UNLOCK_LUA -- renews the
# lock's TTL without ever resurrecting a lock this process has already
# lost (e.g. because an earlier heartbeat lagged past the old TTL and
# someone else's `SET ... NX` already won the key).
_EXTEND_LOCK_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('PEXPIRE', KEYS[1], ARGV[2])
else
  return 0
end
"""

# Atomic failure counter with a window, tripping the breaker once a
# threshold is reached. KEYS[1]=failure counter, KEYS[2]=open flag.
# ARGV = window_seconds, threshold, cooldown_seconds.
_RECORD_FAILURE_LUA = """
local fails = redis.call('INCR', KEYS[1])
redis.call('EXPIRE', KEYS[1], math.max(1, math.ceil(tonumber(ARGV[1]))))
if fails >= tonumber(ARGV[2]) then
  redis.call('SET', KEYS[2], '1', 'EX', math.max(1, math.ceil(tonumber(ARGV[3]))))
end
return fails
"""


class RateLimitExceeded(ProviderError):
    def __init__(self, wait_hint: float) -> None:
        super().__init__(f"rate limit exceeded, retry in ~{wait_hint:.2f}s", retryable=True, status=429)
        self.wait_hint = wait_hint


class CircuitOpen(ProviderError):
    def __init__(self, name: str, retry_after: float, reason: str) -> None:
        super().__init__(f"circuit '{name}' open ({reason}), retry in ~{retry_after:.1f}s",
                         retryable=True, status=503)
        self.retry_after = retry_after


class TokenBucket:
    """A single-dimension rate limit shared across every process talking to
    the same Redis — the point being that N worker processes see ONE limit,
    not N. See `DualTokenBucket` for a separate RPM+TPM pair."""

    def __init__(self, redis_client: aioredis.Redis, key: str, capacity: float,
                refill_per_second: float) -> None:
        self.redis = redis_client
        self.key = key
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self._script = self.redis.register_script(_TOKEN_BUCKET_LUA)

    async def acquire(self, cost: float = 1.0) -> None:
        allowed, tokens = await self._script(keys=[self.key],
                                             args=[self.capacity, self.refill_per_second,
                                                   time.time(), cost])
        if not int(allowed):
            deficit = cost - float(tokens)
            wait = deficit / self.refill_per_second if self.refill_per_second > 0 else 1.0
            raise RateLimitExceeded(wait)


class DualTokenBucket:
    """RPM + TPM together, all-or-nothing per request. `token_cost` defaults
    to `cost * tokens_per_request_estimate` so callers that don't have a real
    token estimate handy (the common case before the model has actually
    responded) still get a reasonable TPM approximation; pass `token_cost`
    explicitly once you have a better number (e.g. from the previous turn)."""

    def __init__(self, redis_client: aioredis.Redis, key: str, rpm_capacity: float,
                rpm_refill_per_second: float, tpm_capacity: float, tpm_refill_per_second: float,
                tokens_per_request_estimate: float = 500.0) -> None:
        self.redis = redis_client
        self.rpm_key = key + ":rpm"
        self.tpm_key = key + ":tpm"
        self.rpm_capacity = rpm_capacity
        self.rpm_refill = rpm_refill_per_second
        self.tpm_capacity = tpm_capacity
        self.tpm_refill = tpm_refill_per_second
        self.tokens_per_request_estimate = tokens_per_request_estimate
        self._script = self.redis.register_script(_DUAL_BUCKET_LUA)

    async def acquire(self, cost: float = 1.0, token_cost: float | None = None) -> None:
        token_cost = self.tokens_per_request_estimate * cost if token_cost is None else token_cost
        allowed, reason, rpm_tokens, tpm_tokens = await self._script(
            keys=[self.rpm_key, self.tpm_key],
            args=[self.rpm_capacity, self.rpm_refill, self.tpm_capacity, self.tpm_refill,
                 time.time(), cost, token_cost])
        if not int(allowed):
            if reason == "rpm":
                deficit = cost - float(rpm_tokens)
                wait = deficit / self.rpm_refill if self.rpm_refill > 0 else 1.0
            else:
                deficit = token_cost - float(tpm_tokens)
                wait = deficit / self.tpm_refill if self.tpm_refill > 0 else 1.0
            raise RateLimitExceeded(wait)


class CircuitBreaker:
    """Shared (Redis-backed) circuit breaker per provider name. Two ways to
    trip: `record_failure()` N times within `window_seconds` (a generic
    failure-rate breaker), or a single `record_rate_limited()` (a 429 is
    already definitive — no need to wait for a threshold). "Half-open" here
    is approximated by the open-flag's own TTL expiring — simpler than a
    formal single-trial half-open state, and documented as such; see the
    module's README notes on this simplification."""

    def __init__(self, redis_client: aioredis.Redis, name: str, failure_threshold: int = 5,
                window_seconds: float = 30, cooldown_seconds: float = 20,
                key_prefix: str = "rewind:gw:cb:") -> None:
        self.redis = redis_client
        self.name = name
        self.failure_threshold = failure_threshold
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.fail_key = f"{key_prefix}{name}:fails"
        self.open_key = f"{key_prefix}{name}:open"
        self._record_failure_script = self.redis.register_script(_RECORD_FAILURE_LUA)

    async def allow(self) -> None:
        ttl = await self.redis.ttl(self.open_key)
        if ttl and ttl > 0:
            raise CircuitOpen(self.name, ttl, "open")

    async def record_success(self) -> None:
        await self.redis.delete(self.fail_key)

    async def record_failure(self) -> None:
        await self._record_failure_script(
            keys=[self.fail_key, self.open_key],
            args=[self.window_seconds, self.failure_threshold, self.cooldown_seconds])

    async def record_rate_limited(self, retry_after: float | None = None) -> None:
        await self.redis.set(self.open_key, "1", ex=max(1, int(retry_after or self.cooldown_seconds)))


class LLMGateway:
    """Drop-in Provider decorator: `AgentRunner(LLMGateway(real_provider, redis), ...)`."""

    def __init__(self, inner: Any, redis_client: aioredis.Redis, *, cache_ttl: int = 3600,
                rate_limiter: TokenBucket | DualTokenBucket | None = None,
                circuit_breaker: CircuitBreaker | None = None, lock_ttl_ms: int = 15_000,
                lock_heartbeat_ms: int | None = None, wait_timeout: float = 10.0,
                key_prefix: str = "rewind:gw:") -> None:
        self.inner = inner
        self.redis = redis_client
        self.cache_ttl = cache_ttl
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.lock_ttl_ms = lock_ttl_ms
        # The job lease has heartbeat() so a slow-but-alive worker never
        # loses its lease mid-step; this lock originally had no equivalent,
        # so any real call slower than lock_ttl_ms (large/slow completions
        # are not an edge case) let a second worker's `SET NX` steal the key
        # while the first was still genuinely working -- single-flight
        # silently defeated, and the SAME request paid for a real call
        # twice. Default heartbeat interval is a third of the TTL, matching
        # the usual "renew well before expiry, not right at the deadline"
        # rule of thumb for any lease/lock.
        self.lock_heartbeat_ms = lock_heartbeat_ms or max(1, lock_ttl_ms // 3)
        self.wait_timeout = wait_timeout
        self.prefix = key_prefix
        self._unlock = self.redis.register_script(_UNLOCK_LUA)
        self._extend_lock = self.redis.register_script(_EXTEND_LOCK_LUA)
        self.stats = {"hits": 0, "misses": 0, "single_flight_waits": 0, "real_calls": 0}

    def _key(self, messages: list[dict], tools: list[dict], params: dict) -> str:
        return self.prefix + "resp:" + request_hash(messages, tools, params)

    async def complete(self, messages: list[dict], tools: list[dict], params: dict) -> LLMResponse:
        key = self._key(messages, tools, params)
        cached = await self.redis.get(key)
        if cached is not None:
            self.stats["hits"] += 1
            return LLMResponse.model_validate_json(cached)

        lock_key, channel = key + ":lock", key + ":done"
        token = uuid.uuid4().hex
        got_lock = bool(await self.redis.set(lock_key, token, nx=True, px=self.lock_ttl_ms))
        if not got_lock:
            # Someone else is already making this exact call; wait for their
            # result (via pub/sub, not polling) instead of making a second
            # real call.
            self.stats["single_flight_waits"] += 1
            resp = await self._wait_for(key, channel)
            if resp is not None:
                return resp
            # No cached result after waking (or timing out): either the
            # holder crashed before publishing, or it published because its
            # own call *failed*. Either way, fall through and try ourselves
            # rather than hanging or returning nothing.

        self.stats["misses"] += 1
        resp: LLMResponse | None = None
        heartbeat_task = (asyncio.ensure_future(self._renew_lock_until_cancelled(lock_key, token))
                          if got_lock else None)
        try:
            if self.circuit_breaker is not None:
                await self.circuit_breaker.allow()
            if self.rate_limiter is not None:
                await self.rate_limiter.acquire()
            self.stats["real_calls"] += 1
            try:
                resp = await self.inner.complete(messages, tools, params)
            except ProviderError as e:
                if self.circuit_breaker is not None:
                    if e.status == 429:
                        await self.circuit_breaker.record_rate_limited(getattr(e, "wait_hint", None))
                    else:
                        await self.circuit_breaker.record_failure()
                raise
            else:
                if self.circuit_breaker is not None:
                    await self.circuit_breaker.record_success()
        finally:
            if heartbeat_task is not None:
                # Stop renewing *before* deleting the lock -- otherwise a
                # heartbeat tick racing the unlock could re-extend a key
                # we're about to delete out from under ourselves (harmless
                # either way since compare-and-delete/compare-and-extend
                # both check the token, but there's no reason to race it).
                heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat_task
            if got_lock:
                await self._unlock(keys=[lock_key], args=[token])
                # Wake every waiter whether we succeeded or not: on failure
                # there's nothing to hand them, so they immediately fall
                # through to trying their own call instead of idling out the
                # full wait_timeout for nothing.
                await self.redis.publish(channel, "1")
        await self.redis.set(key, resp.model_dump_json(), ex=self.cache_ttl)
        return resp

    async def _renew_lock_until_cancelled(self, lock_key: str, token: str) -> None:
        """Background heartbeat for the single-flight lock -- keeps extending
        its TTL at `lock_heartbeat_ms` intervals for as long as the real call
        is still in flight, so a call slower than `lock_ttl_ms` never loses
        the lock out from under it. Uses compare-and-extend (`_EXTEND_LOCK_LUA`):
        if this process ever DID already lose the lock (e.g. a long GC pause
        let an earlier tick lag past the old TTL before this one ran), the
        script is a no-op instead of resurrecting a lock someone else now
        legitimately owns."""
        interval = self.lock_heartbeat_ms / 1000
        try:
            while True:
                await asyncio.sleep(interval)
                await self._extend_lock(keys=[lock_key], args=[token, self.lock_ttl_ms])
        except asyncio.CancelledError:
            pass

    async def _wait_for(self, key: str, channel: str) -> LLMResponse | None:
        async with self.redis.pubsub() as pubsub:
            await pubsub.subscribe(channel)
            # The leader may have already finished (and published) between
            # our failed lock attempt and this subscribe — check once more
            # now that we're guaranteed not to miss any *future* publish.
            cached = await self.redis.get(key)
            if cached is not None:
                return LLMResponse.model_validate_json(cached)
            try:
                await asyncio.wait_for(self._first_message(pubsub), timeout=self.wait_timeout)
            except asyncio.TimeoutError:
                pass
            cached = await self.redis.get(key)
            return LLMResponse.model_validate_json(cached) if cached is not None else None

    @staticmethod
    async def _first_message(pubsub: Any) -> None:
        async for msg in pubsub.listen():
            if msg.get("type") == "message":
                return
