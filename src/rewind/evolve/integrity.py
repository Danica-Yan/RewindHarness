"""The judge cannot drift silently.

The loop's own outputs (rules, notes, checks, skills, lessons) are data the
loop writes; the JUDGE -- the code and settings that decide whether those
outputs are adopted -- is not. By construction the proposer can only emit
`Edit`s and lessons, which never touch code. This module adds the second
half of the guarantee: the judge's fingerprint (hashes of the judging code,
the compliance checks it replays, and the gate / screening / backtest
settings) is recorded in the ledger on first use, and every command that
can adopt something verifies it. A mismatch stops the run until a human
acknowledges the change with a reason (`--accept-judge-change "..."`),
which is recorded in the ledger's judge history -- so every verdict in the
ledger can be traced to the judge that made it.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
JUDGE_FILES = [
    _HERE / "gate.py", _HERE / "stats.py", _HERE / "screen.py", _HERE / "stability.py",
    _HERE / "backtest.py", _HERE / "monitors.py", _HERE.parent / "policy.py",
    _HERE.parents[2] / "data" / "policy_checks" / "taubench_retail.json",
]
KEY = "judge_fingerprint"


class JudgeChanged(RuntimeError):
    pass


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def fingerprint(cfg: Any) -> dict:
    files = {}
    for f in JUDGE_FILES:
        files[f.name] = _sha(f.read_bytes()) if f.exists() else "missing"
    settings = {"gate": cfg.gate.model_dump(), "screen": cfg.screen.model_dump(),
                "backtest": cfg.backtest.model_dump(), "stability_n": cfg.stability_n,
                "stable_max": cfg.stable_max}
    return {"files": files, "settings": settings,
            "id": _sha(json.dumps([files, settings], sort_keys=True).encode())}


def _diff(old: dict, new: dict) -> list[str]:
    out = [f"code {k}" for k in sorted(set(old["files"]) | set(new["files"]))
           if old["files"].get(k) != new["files"].get(k)]
    for sect in sorted(set(old["settings"]) | set(new["settings"])):
        a, b = old["settings"].get(sect), new["settings"].get(sect)
        if isinstance(a, dict) and isinstance(b, dict):
            out += [f"setting {sect}.{k}: {a.get(k)!r} -> {b.get(k)!r}"
                    for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
        elif a != b:
            out.append(f"setting {sect}: {a!r} -> {b!r}")
    return out


def history(ledger: Any) -> list[dict]:
    raw = ledger._get(KEY + "_history")  # noqa: SLF001
    return json.loads(raw) if raw else []


def verify(ledger: Any, cfg: Any, accept_reason: str | None = None) -> dict:
    """Record the judge on first use; afterwards refuse to run under a
    different judge unless a human gives a reason, which is recorded."""
    new = fingerprint(cfg)
    raw = ledger._get(KEY)  # noqa: SLF001
    if raw is None:
        _store(ledger, new, "first use")
        return new
    old = json.loads(raw)
    if old["id"] == new["id"]:
        return new
    changes = _diff(old, new)
    if not accept_reason:
        raise JudgeChanged(
            "the judge differs from the one recorded in the ledger:\n  " + "\n  ".join(changes)
            + "\nIf a human made this change on purpose, rerun with --accept-judge-change \"<why>\"; "
              "it is recorded so later verdicts can be traced to the judge that made them.")
    _store(ledger, new, accept_reason, changes)
    return new


def _store(ledger: Any, fp: dict, reason: str, changes: list[str] | None = None) -> None:
    h = history(ledger)
    h.append({"id": fp["id"], "reason": reason, "changes": changes or [], "at": time.time()})
    ledger._set(KEY, json.dumps(fp))  # noqa: SLF001
    ledger._set(KEY + "_history", json.dumps(h))  # noqa: SLF001
    ledger.db.commit()
