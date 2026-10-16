from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .util import canonical_json, sha


class SnapshotStore:
    """Content-addressed store for environment states (sha256 of canonical JSON)."""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else None
        self._mem: dict[str, dict[str, Any]] = {}
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def _file(self, h: str) -> Path:
        assert self.root is not None
        return self.root / (h.replace(":", "_") + ".json")

    def put(self, state: dict[str, Any]) -> str:
        h = sha(state)
        if h not in self._mem:
            self._mem[h] = copy.deepcopy(state)
            if self.root and not self._file(h).exists():
                self._file(h).write_text(canonical_json(state), encoding="utf-8")
        return h

    def get(self, h: str) -> dict[str, Any]:
        if h in self._mem:
            return copy.deepcopy(self._mem[h])
        if self.root and self._file(h).exists():
            return json.loads(self._file(h).read_text(encoding="utf-8"))
        raise KeyError(f"unknown snapshot {h}")
