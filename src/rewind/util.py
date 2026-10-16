"""Hashing and clock helpers. Everything that touches time goes through a Clock
so that replay never depends on wall time."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def request_hash(messages: list[dict], tools: list[dict], params: dict) -> str:
    """Identity of one LLM request. Replay asserts this matches the recording."""
    return sha({"messages": messages, "tools": tools, "params": params})


class SystemClock:
    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def perf(self) -> float:
        return time.perf_counter()


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.t = start

    def now_iso(self) -> str:
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        return (base + timedelta(seconds=self.t)).isoformat()

    def perf(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt
