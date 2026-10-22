"""Edit ledger: every context version, every decision, and what the proposer
has seen. SQLite, one file, so a whole evolution run is inspectable and
reproducible.

* versions   -- each committed ContextVersion, its parent, the edits that
                made it, and the results file that is its gate evidence (and
                the next round's incumbent rows)
* decisions  -- every candidate edit and where it stopped (invalid, screen,
                gate) with the numbers -- rejected ideas are fed back to the
                proposer so it does not repeat them
* shown      -- train tasks whose failures were shown to the proposer or used
                for screening; they are excluded from the gate pool forever
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .surface import ContextVersion, Edit

SCHEMA = """
CREATE TABLE IF NOT EXISTS versions(
  version INTEGER PRIMARY KEY, parent INTEGER, ctx_json TEXT NOT NULL,
  edits_json TEXT NOT NULL, rows_path TEXT, verdict_json TEXT, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS decisions(
  id INTEGER PRIMARY KEY AUTOINCREMENT, round INTEGER, version_from INTEGER, edit_id TEXT,
  cluster TEXT, edit_json TEXT, stage TEXT, accepted INTEGER, reason TEXT, data_json TEXT,
  created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS shown(task_id INTEGER PRIMARY KEY, round INTEGER, role TEXT);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""


class EditLedger:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)
        self.db.commit()

    # -- versions ----------------------------------------------------------------
    def init_base(self, rows_path: str) -> None:
        """Register v0 (no edits) with the baseline results file, once."""
        if self.db.execute("SELECT 1 FROM versions WHERE version=0").fetchone():
            return
        self.db.execute("INSERT INTO versions VALUES(0,NULL,?,?,?,NULL,?)",
                        (ContextVersion().model_dump_json(), "[]", rows_path, time.time()))
        self._set("current", "0")
        self.db.commit()

    def current(self) -> ContextVersion:
        v = int(self._get("current") or 0)
        return self.version(v)

    def version(self, v: int) -> ContextVersion:
        row = self.db.execute("SELECT ctx_json FROM versions WHERE version=?", (v,)).fetchone()
        if row is None:
            raise KeyError(f"no version {v}")
        return ContextVersion.model_validate_json(row[0])

    def rows_path(self, v: int) -> str:
        return self.db.execute("SELECT rows_path FROM versions WHERE version=?", (v,)).fetchone()[0]

    def next_version_number(self) -> int:
        return (self.db.execute("SELECT MAX(version) FROM versions").fetchone()[0] or 0) + 1

    def commit(self, ctx: ContextVersion, edits: list[Edit], rows_path: str, verdict: dict) -> int:
        self.db.execute("INSERT INTO versions VALUES(?,?,?,?,?,?,?)",
                        (ctx.version, ctx.parent, ctx.model_dump_json(),
                         json.dumps([e.model_dump() for e in edits], ensure_ascii=False),
                         rows_path, json.dumps(verdict), time.time()))
        self._set("current", str(ctx.version))
        self.db.commit()
        return ctx.version

    def rollback(self, to_version: int) -> ContextVersion:
        ctx = self.version(to_version)  # raises if unknown
        self._set("current", str(to_version))
        self.db.commit()
        return ctx

    def lineage(self, v: int) -> list[tuple[int, list[dict]]]:
        out = []
        while v is not None:
            parent, edits = self.db.execute(
                "SELECT parent, edits_json FROM versions WHERE version=?", (v,)).fetchone()
            out.append((v, json.loads(edits)))
            v = parent
        return list(reversed(out))

    # -- decisions -----------------------------------------------------------------
    def record(self, round_no: int, version_from: int, edit: Edit | None, stage: str, accepted: bool,
               reason: str, data: dict | None = None, cluster: str | None = None) -> None:
        self.db.execute(
            "INSERT INTO decisions(round,version_from,edit_id,cluster,edit_json,stage,accepted,reason,"
            "data_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (round_no, version_from, edit.edit_id if edit else None,
             cluster or (edit.source_cluster if edit else None),
             edit.model_dump_json() if edit else None, stage, int(accepted), reason,
             json.dumps(data or {}, default=str), time.time()))
        self.db.commit()

    def rejected_for_prompt(self, cluster: str | None = None, limit: int = 12) -> list[dict]:
        q = ("SELECT edit_json, stage, reason FROM decisions WHERE accepted=0 AND edit_json IS NOT NULL"
             + (" AND cluster=?" if cluster else "") + " ORDER BY id DESC LIMIT ?")
        args = (cluster, limit) if cluster else (limit,)
        out = []
        for ej, stage, reason in self.db.execute(q, args):
            e = json.loads(ej)
            out.append({"kind": e["kind"], "target": e["target"], "text": e.get("text"),
                        "reason": f"{stage}: {reason}"})
        return out

    def tried_edit_ids(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT edit_id FROM decisions WHERE edit_id IS NOT NULL")}

    def decisions(self) -> list[dict]:
        cols = ["id", "round", "version_from", "edit_id", "cluster", "stage", "accepted", "reason"]
        return [dict(zip(cols, r)) for r in self.db.execute(
            "SELECT id,round,version_from,edit_id,cluster,stage,accepted,reason FROM decisions ORDER BY id")]

    # -- shown tasks -----------------------------------------------------------------
    def mark_shown(self, tasks: list[int], round_no: int, role: str) -> None:
        self.db.executemany("INSERT OR IGNORE INTO shown VALUES(?,?,?)",
                            [(t, round_no, role) for t in tasks])
        self.db.commit()

    def shown(self) -> set[int]:
        return {r[0] for r in self.db.execute("SELECT task_id FROM shown")}

    def round_no(self) -> int:
        return int(self._get("round") or 0)

    def bump_round(self) -> int:
        n = self.round_no() + 1
        self._set("round", str(n))
        self.db.commit()
        return n

    # -- meta ------------------------------------------------------------------------
    def _get(self, k: str) -> str | None:
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (k,)).fetchone()
        return r[0] if r else None

    def _set(self, k: str, v: str) -> None:
        self.db.execute("INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (k, v))
