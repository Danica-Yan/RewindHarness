"""Sandbox call latency: a fresh container per call (cold) vs a pre-started,
single-use container from the pool (warm).

    python scripts/sandbox_bench.py --n 30            # back-to-back calls
    python scripts/sandbox_bench.py --n 20 --gap 1    # with think time between calls
Needs Docker and the sandbox image (docker/sandbox.Dockerfile).
"""
from __future__ import annotations

import argparse
import os
import statistics
import tempfile
import time

from rewind.platform.sandbox import DockerSandbox, SandboxPolicy, Workspaces


def bench(pool: int, n: int, policy: SandboxPolicy, gap: float = 0.0) -> dict:
    sb = DockerSandbox(policy, Workspaces(tempfile.mkdtemp()), pool_size=pool)
    try:
        if pool:
            sb.warm("t1")
        lat, warm = [], 0
        for _ in range(n):
            time.sleep(gap)               # think time between calls (an LLM turn is seconds)
            r = sb.run("t1", "import json,sys; print(json.dumps(json.load(sys.stdin)))", stdin='{"x": 1}')
            assert r.exit_code == 0, r.stderr
            lat.append(r.duration_ms)
            warm += r.warm
        lat.sort()
        return {"mode": "warm pool" if pool else "cold", "gap_s": gap, "p50_ms": statistics.median(lat),
                "p90_ms": lat[int(n * 0.9) - 1], "warm_hits": f"{warm}/{n}"}
    finally:
        sb.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--gap", type=float, default=0.0, help="seconds between calls")
    a = ap.parse_args()
    policy = SandboxPolicy(image=os.environ.get("REWIND_SANDBOX_IMAGE", "rewind-sandbox:py"))
    for pool in (0, 2):
        print(bench(pool, a.n, policy, a.gap))


if __name__ == "__main__":
    main()
