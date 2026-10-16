from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .events import Event
from .util import SystemClock


class TraceWriter:
    """Append-only event log. Keeps events in memory and (optionally) JSONL on disk."""

    def __init__(self, run_id: str, path: str | Path | None = None, clock=None) -> None:
        self.run_id = run_id
        self.clock = clock or SystemClock()
        self.events: list[Event] = []
        self._seq = 0
        self._fh = None
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._fh = p.open("w", encoding="utf-8")

    def emit(self, type: str, step: int, data: dict[str, Any] | None = None,
             inherited: bool = False) -> Event:
        self._seq += 1
        ev = Event(run_id=self.run_id, seq=self._seq, step=step, type=type,  # type: ignore[arg-type]
                   ts=self.clock.now_iso(), inherited=inherited,
                   data=copy.deepcopy(data or {}))
        self.events.append(ev)
        if self._fh:
            self._fh.write(ev.model_dump_json() + "\n")
            self._fh.flush()
        return ev

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    @property
    def last_seq(self) -> int:
        return self._seq

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @classmethod
    def resume_from(cls, path: str | Path, run_id: str, upto_seq: int, clock=None) -> "TraceWriter":
        """Reopen a trace file for a resumed worker, discarding any event with
        seq > upto_seq. Those events belong to a crashed attempt that never
        reached the checkpoint at upto_seq, so they were never durably
        confirmed — this is what makes a "torn write" (events on disk with no
        checkpoint backing them) safe to just throw away."""
        p = Path(path)
        kept = [e for e in load_trace(p) if e.seq <= upto_seq] if p.exists() else []
        for e in kept:
            if e.run_id != run_id:
                raise ValueError(f"trace at {p} belongs to run {e.run_id}, not {run_id}")
        with p.open("w", encoding="utf-8") as f:
            for e in kept:
                f.write(e.model_dump_json() + "\n")
        tw = cls(run_id, path=None, clock=clock)
        tw.events = kept
        tw._seq = upto_seq
        tw._fh = p.open("a", encoding="utf-8")
        return tw


def load_trace(path: str | Path) -> list[Event]:
    with Path(path).open(encoding="utf-8") as f:
        return [Event.model_validate_json(line) for line in f if line.strip()]
