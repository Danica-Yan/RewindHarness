"""Throughput of the offline policy what-if (rewind.platform.whatif).

    python scripts/whatif_bench.py --runs 2000
Builds a corpus of real rewind runs (tests/whatif_corpus.py), writes them as
JSONL, then judges them under a changed policy: once from memory and once
from disk (parse included). No model calls.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import whatif_corpus as wc  # noqa: E402
from test_platform_whatif import MIXED_CHANGE  # noqa: E402

from rewind.platform.whatif import what_if  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2000)
    a = ap.parse_args()
    t = time.perf_counter()
    c = wc.build(a.runs, seed=7)
    print(f"corpus: {len(c.traces)} runs, built in {time.perf_counter() - t:.1f}s")
    d = Path(tempfile.mkdtemp())
    for tr in c.traces:
        (d / f"{tr[0].run_id}.jsonl").write_text("".join(e.model_dump_json() + "\n" for e in tr))
    size = sum(p.stat().st_size for p in d.glob("*.jsonl"))
    access = wc.new_access(MIXED_CHANGE)
    for label, src in (("in memory", c.traces), ("from JSONL", sorted(d.glob("*.jsonl")))):
        t = time.perf_counter()
        rep = what_if(src, access)
        dt = time.perf_counter() - t
        print(f"{label:11}: {rep.calls} calls in {len(c.traces)} runs, {dt:.2f}s -> "
              f"{rep.calls / dt:,.0f} calls/s, {len(c.traces) / dt:,.0f} runs/s"
              + (f"  ({size / 1e6:.1f} MB of traces)" if label != "in memory" else ""))
    print(rep.summary()["changes"])


if __name__ == "__main__":
    main()
