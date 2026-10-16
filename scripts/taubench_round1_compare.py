#!/usr/bin/env python3
"""Compare a baseline tau-bench results json against a round-1 (post-patch)
results json. Thin wrapper around the generic scripts/taubench_compare.py
with labels "baseline"/"round 1" and --targets 72,98,105 (the tasks
scripts/taubench_round1_wiki.patch specifically targeted) -- kept as its own
script/CLI because earlier README instructions already point at this name.

Usage:
    python scripts/taubench_round1_compare.py baseline.json round1.json
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "scripts")
from taubench_compare import compare  # noqa: E402

TARGETS = [72, 98, 105]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline_json")
    ap.add_argument("round1_json")
    args = ap.parse_args()
    compare(args.baseline_json, args.round1_json, "baseline", "round 1", TARGETS)


if __name__ == "__main__":
    main()
