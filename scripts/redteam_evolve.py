#!/usr/bin/env python3
"""Red-team the self-improvement loop -- NO API calls.

Feeds a catalogue of attacks (edits and lessons that would raise the
measured reward, or slip through, without making the agent better) to the
layer that should stop each one, and prints which layer did. Exit code 1
if any attack is NOT CAUGHT.

  export REWIND_TAUBENCH_SRC=../tau-bench
  python scripts/redteam_evolve.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rewind.env.taubench_dialogue import build_dialogue_registry  # noqa: E402
from rewind.evolve.redteam import run_offline  # noqa: E402


def main() -> None:
    rows = run_offline(set(build_dialogue_registry()._tools))  # noqa: SLF001
    w = max(len(r["attack"]) for r in rows)
    for r in rows:
        print(f"{r['attack']:{w}s}  {r['exploits']}\n{'':{w}s}  -> {r['stopped_by']}")
    missed = [r for r in rows if r["stopped_by"] == "NOT CAUGHT"]
    print(f"\n{len(rows) - len(missed)}/{len(rows)} attacks stopped"
          + ("" if not missed else f"; NOT CAUGHT: {[r['attack'] for r in missed]}"))
    print("Attacks that need real episodes (a policy violation the reward does not see) are tested\n"
          "against the gate's compliance monitor in tests/test_redteam.py.")
    sys.exit(1 if missed else 0)


if __name__ == "__main__":
    main()
