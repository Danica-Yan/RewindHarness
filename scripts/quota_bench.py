"""Throughput and latency of PgQuotaStore.charge under concurrency.

Two cases, same number of charges and workers:
  spread  every charge hits its own bucket (no row contention)
  hot     every charge hits ONE bucket (all writers serialise on its row lock)

    python scripts/quota_bench.py --n 2000 --workers 20
Needs the [dist] extra and Postgres at REWIND_PG_DSN.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time

from rewind.platform.quota import Charge, PgQuotaStore

DSN = os.environ.get("REWIND_PG_DSN", "postgresql://rewind:rewind@127.0.0.1:5432/rewind_test")


async def bench(store: PgQuotaStore, n: int, workers: int, hot: bool) -> dict:
    async with store.pool.acquire() as conn:
        await conn.execute("TRUNCATE quota_buckets, quota_charges")
    lat: list[float] = []
    q: asyncio.Queue[int] = asyncio.Queue()
    for i in range(n):
        q.put_nowait(i)

    async def worker():
        while not q.empty():
            i = q.get_nowait()
            key = "hot" if hot else f"b{i}"
            t = time.perf_counter()
            await store.charge(f"c{i}", [Charge(key, "r", n, 1, None)], 0.0)
            lat.append(time.perf_counter() - t)

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(workers)))
    wall = time.perf_counter() - t0
    lat.sort()
    return {"case": "hot" if hot else "spread", "charges/s": round(n / wall),
            "p50_ms": round(statistics.median(lat) * 1000, 2),
            "p99_ms": round(lat[int(len(lat) * 0.99) - 1] * 1000, 2)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    store = await PgQuotaStore.connect(DSN)
    try:
        for hot in (False, True):
            print(await bench(store, a.n, a.workers, hot))
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
