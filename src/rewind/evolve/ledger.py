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
* regressions -- regression memory: tasks an accepted version fixed, with a
                trace to fork them from; every later gate re-checks them
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
CREATE TABLE IF NOT EXISTS regressions(task_id INTEGER PRIMARY KEY, trace TEXT NOT NULL,
  version INTEGER, source TEXT, created_at REAL NOT NULL);
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
        # fixes made by versions after the target no longer hold
        self.db.execute("DELETE FROM regressions WHERE version>?", (to_version,))
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
               reason: str, data: dict | None = None, cluster: str | None = None) -> int:
        cur = self.db.execute(
            "INSERT INTO decisions(round,version_from,edit_id,cluster,edit_json,stage,accepted,reason,"
            "data_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (round_no, version_from, edit.edit_id if edit else None,
             cluster or (edit.source_cluster if edit else None),
             edit.model_dump_json() if edit else None, stage, int(accepted), reason,
             json.dumps(data or {}, default=str), time.time()))
        self.db.commit()
        return cur.lastrowid

    def rejected_for_prompt(self, cluster: str | None = None, limit: int = 12) -> list[dict]:
        """Rejected edits for the proposer prompt: this cluster's first (with the
        full gate feedback, if any), then others, newest first."""
        rows = [r for r in self.db.execute(
            "SELECT edit_json, stage, reason, data_json, cluster, id FROM decisions "
            "WHERE accepted=0 AND edit_json IS NOT NULL AND stage != 'ablation' ORDER BY id DESC").fetchall()
            if r[5] not in self.overturned()]
        rows.sort(key=lambda r: r[4] != cluster)  # stable: same cluster first, newest first within
        out = []
        for ej, stage, reason, dj, cl, _id in rows[:limit]:
            e = json.loads(ej)
            fb = ((json.loads(dj) if dj else {}) or {}).get("feedback") or {}
            out.append({"kind": e["kind"], "target": e["target"], "text": e.get("text"),
                        "trigger": e.get("trigger"), "reason": f"{stage}: {reason}",
                        "feedback": fb.get("text") if cl == cluster else None})
        return out

    def set_feedback(self, decision_id: int, feedback: dict) -> None:
        dj = self.db.execute("SELECT data_json FROM decisions WHERE id=?", (decision_id,)).fetchone()[0]
        data = json.loads(dj) if dj else {}
        data["feedback"] = feedback
        self.db.execute("UPDATE decisions SET data_json=? WHERE id=?", (json.dumps(data, default=str),
                                                                        decision_id))
        self.db.commit()

    def mark_regated(self, decision_id: int) -> None:
        dj = self.db.execute("SELECT data_json FROM decisions WHERE id=?", (decision_id,)).fetchone()[0]
        data = json.loads(dj) if dj else {}
        data["regated"] = True
        self.db.execute("UPDATE decisions SET data_json=? WHERE id=?", (json.dumps(data, default=str),
                                                                        decision_id))
        self.db.commit()

    def overturned(self) -> set[int]:
        """Gate rejections that a later re-gate reversed (the edit was accepted)."""
        out = set()
        for dj, acc in self.db.execute("SELECT data_json, accepted FROM decisions WHERE stage='gate'"):
            d = json.loads(dj) if dj else {}
            if acc and d.get("regate_of") is not None:
                out.add(int(d["regate_of"]))
        return out

    def gate_rejections_without_feedback(self) -> list[tuple[int, int, dict]]:
        """(decision id, version_from, verdict) of rejected gate decisions that
        have no feedback yet -- e.g. recorded before feedback existed."""
        out = []
        for did, v, dj in self.db.execute(
                "SELECT id, version_from, data_json FROM decisions WHERE stage='gate' AND accepted=0"):
            data = json.loads(dj) if dj else {}
            if "feedback" not in data and data.get("rows_path"):
                out.append((did, v, data))
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

    # -- regression memory -------------------------------------------------------------
    def remember(self, items: list[tuple[int, str]], version: int, source: str) -> int:
        """Add fixed tasks (task_id, trace to fork from); returns how many were new."""
        n = 0
        for task_id, trace in items:
            cur = self.db.execute("INSERT OR IGNORE INTO regressions VALUES(?,?,?,?,?)",
                                  (task_id, trace, version, source, time.time()))
            n += cur.rowcount
        self.db.commit()
        return n

    def regression_suite(self) -> list[tuple[int, str]]:
        return [(t, tr) for t, tr in self.db.execute(
            "SELECT task_id, trace FROM regressions ORDER BY task_id")]

    def forget(self, version: int) -> None:
        """Drop the memories a version added (used when it is rolled back)."""
        self.db.execute("DELETE FROM regressions WHERE version=?", (version,))
        self.db.commit()

    # -- meta ------------------------------------------------------------------------
    def _get(self, k: str) -> str | None:
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (k,)).fetchone()
        return r[0] if r else None

    def _set(self, k: str, v: str) -> None:
        self.db.execute("INSERT INTO meta VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (k, v))
