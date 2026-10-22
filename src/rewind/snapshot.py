from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .util import canonical_json, sha


def _ordered_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


class SnapshotStore:
    """Content-addressed store for environment states.

    Default keys hash canonical JSON (sorted keys), so two states that differ
    only in dict key order share one entry -- right for states whose order
    never reaches the model. `put(..., ordered=True)` is for states whose
    key order IS observable: tau-bench's tools return a DB record by
    `json.dumps`-ing the dict as stored, so a restored DB with sorted keys
    shows the agent differently-ordered tool results than the run it was
    forked from. Ordered entries hash and store the JSON in insertion order
    (`json.loads` preserves it on the way back) under a `sha256o:` key.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else None
        self._mem: dict[str, dict[str, Any]] = {}
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def _file(self, h: str) -> Path:
        assert self.root is not None
        return self.root / (h.replace(":", "_") + ".json")

    def put(self, state: dict[str, Any], *, ordered: bool = False) -> str:
        if ordered:
            text = _ordered_json(state)
            h = "sha256o:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        else:
            h = sha(state)
            text = None
        if h not in self._mem:
            self._mem[h] = copy.deepcopy(state)
            if self.root and not self._file(h).exists():
                self._write(h, text if text is not None else canonical_json(state))
        return h

    def _write(self, h: str, text: str) -> None:
        # atomic: a crash mid-write must never leave a truncated snapshot that
        # a later process would fail to parse
        f = self._file(h)
        tmp = f.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(f)

    def get(self, h: str) -> dict[str, Any]:
        if h in self._mem:
            return copy.deepcopy(self._mem[h])
        if self.root and self._file(h).exists():
            return json.loads(self._file(h).read_text(encoding="utf-8"))
        raise KeyError(f"unknown snapshot {h}")
