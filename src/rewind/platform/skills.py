"""Skill management: versioned, gated, rolled back, and authorised at load time.

`rewind.skills` defines what a skill IS for the agent (a procedure behind
`load_skill`); Evolve's ContextVersion keeps one benchmark agent's skills.
This module is the multi-tenant registry around them:

  SkillManifest   a skill version as data: the procedure, and the
                  capabilities it REQUIRES (concrete capability ids). Who may
                  see it: every tenant, or the listed ones.
  SkillStore      immutable versions (name + semver, content-hashed), the
                  `candidate` and `stable` channels, and an append-only audit
                  log of every move. Rollback pops the stable stack; a
                  rolled-back version can be yanked -- no new run binds it,
                  but it stays readable so old runs still replay.
  Gates           promotion runs gates: `StaticSkillGate` (requirements
                  exist; the text names no tool it did not declare; no case
                  data) for candidate, plus any quality gate for stable --
                  Evolve's statistical gate plugs in via `verdict_from_evolve`.
  bind_skills     what a principal actually gets for a run: each requested
                  skill is offered only if policy allows `skill.<name>` AND
                  the principal could use every capability it requires.
                  The result is pinned to exact versions (put them in
                  AgentConfig.skill_pins), so a stable release mid-run or
                  later does not change what a run -- or its replay -- sees.

A skill never grants anything: loading one only shows the agent text, and
every call it leads to is still decided by the policy engine. Where a skill
gets its own sub-agent, `principal_for_skill` runs it with the parent's
grants intersected with the skill's declared requirements.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, field_validator

from ..skills import LOAD_SKILL, SKILLS_HEADER, Skill
from ..tools import Tool, ToolError, ToolRegistry
from ..util import sha
from .authz import AuthzContext
from .capability import CapabilityCatalog, CapabilityDescriptor
from .identity import Principal

Channel = Literal["candidate", "stable"]
_SEMVER = re.compile(r"(\d+)\.(\d+)\.(\d+)")
_CAP_ID = re.compile(r"[a-z]+\.[a-z0-9_.\-]+")
LOAD_SKILL_CAPABILITY = f"local.{LOAD_SKILL}"


def semver(v: str) -> tuple[int, int, int]:
    m = _SEMVER.fullmatch(v)
    if not m:
        raise ValueError(f"version {v!r} is not MAJOR.MINOR.PATCH")
    return int(m[1]), int(m[2]), int(m[3])


class SkillManifest(BaseModel):
    name: str
    version: str
    description: str
    steps: list[str]
    requires: list[str]                                  # concrete capability ids
    tenants: list[str] = Field(default_factory=list)     # empty: visible to every tenant
    source: Literal["human", "evolve"] = "human"
    author: str = ""

    @field_validator("version")
    @classmethod
    def _version(cls, v: str) -> str:
        semver(v)
        return v

    @field_validator("requires")
    @classmethod
    def _requires(cls, v: list[str]) -> list[str]:
        bad = [c for c in v if not _CAP_ID.fullmatch(c)]
        if bad:
            raise ValueError(f"requires must be concrete capability ids (no globs): {bad}")
        return sorted(set(v))

    def model_post_init(self, _ctx: Any) -> None:
        # name / description / steps validated exactly like an agent skill
        Skill(name=self.name, description=self.description, steps=self.steps)

    def content_hash(self) -> str:
        return sha(self.model_dump(exclude={"author"}))[7:23]

    def as_skill(self, catalog: CapabilityCatalog | None = None) -> Skill:
        tools = []
        for c in self.requires:
            cap = catalog.by_id(c) if catalog is not None else None
            tools.append(cap.name if cap is not None else c.rsplit(".", 1)[-1])
        return Skill(name=self.name, description=self.description, steps=self.steps, tools=tools)


class GateVerdict(BaseModel):
    gate: str
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class Gate(Protocol):
    name: str

    def evaluate(self, m: SkillManifest) -> GateVerdict: ...


class PromotionRejected(Exception):
    def __init__(self, verdicts: list[GateVerdict]) -> None:
        super().__init__("; ".join(f"{v.gate}: {', '.join(v.reasons)}" for v in verdicts if not v.passed))
        self.verdicts = verdicts


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
class StaticSkillGate:
    """Checks that need no episode:

      * every required capability exists in the catalog
      * the text does not name a tool it did not declare -- a skill that
        says "first call admin_reset" without requiring it would have the
        agent attempt something its manifest hides from the loader's check
      * no case data in the text (order ids, emails, ...), Evolve's leak rules
    """

    name = "static"

    def __init__(self, catalog: CapabilityCatalog) -> None:
        self.catalog = catalog

    def evaluate(self, m: SkillManifest) -> GateVerdict:
        from ..evolve.surface import leak_reasons
        reasons = [f"requires unknown capability {c}" for c in m.requires if self.catalog.by_id(c) is None]
        text = " ".join([m.description, *m.steps])
        declared = set(m.requires)
        for cap in self.catalog:
            if cap.id not in declared and cap.name != LOAD_SKILL and \
                    re.search(rf"(?<![\w.]){re.escape(cap.name)}(?![\w])", text):
                reasons.append(f"mentions {cap.name} ({cap.id}) without requiring it")
        reasons += [f"contains {r}" for r in leak_reasons(text)]
        return GateVerdict(gate=self.name, passed=not reasons, reasons=reasons)


def verdict_from_evolve(decision: dict) -> GateVerdict:
    """An Evolve ledger decision (EditLedger.decisions() row) as a gate
    verdict, so a statistically gated edit can be promoted to stable."""
    data = decision.get("data") or decision.get("data_json") or {}
    if isinstance(data, str):
        data = json.loads(data)
    passed = bool(decision.get("accepted"))
    return GateVerdict(gate=f"evolve.{decision.get('stage', 'gate')}", passed=passed,
                       reasons=[] if passed else [str(decision.get("reason", "rejected"))],
                       details={"decision_id": decision.get("id"), **data})


def manifest_from_evolve(skill: dict, version: str, *, catalog: CapabilityCatalog,
                         author: str = "evolve") -> SkillManifest:
    """An Evolve skill (Skill fields: name, description, steps, tools) as a
    manifest. Its tool names become required capability ids via the catalog;
    a tool the catalog does not know stays unresolved and fails the static gate."""
    req = []
    for t in skill.get("tools", []):
        cap = catalog.resolve(t)
        req.append(cap.id if cap is not None else f"local.{t}")
    return SkillManifest(name=skill["name"], version=version, description=skill["description"],
                         steps=skill["steps"], requires=req, source="evolve", author=author)


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS skill_versions(
  name TEXT NOT NULL, version TEXT NOT NULL, manifest TEXT NOT NULL, content_hash TEXT NOT NULL,
  yanked INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, PRIMARY KEY (name, version));
CREATE TABLE IF NOT EXISTS skill_events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, version TEXT, action TEXT NOT NULL,
  channel TEXT, actor TEXT NOT NULL, verdicts TEXT, note TEXT, at REAL NOT NULL);
"""


@dataclass
class SkillVersion:
    manifest: SkillManifest
    content_hash: str
    yanked: bool = False

    @property
    def ref(self) -> str:
        return f"{self.manifest.name}@{self.manifest.version}"


class SkillStore:
    """sqlite-backed (":memory:" by default). Versions are immutable; channel
    positions are derived from the append-only event log, so the audit log
    and the current state cannot disagree."""

    def __init__(self, path: str = ":memory:", clock=time.time) -> None:
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self._lock = threading.Lock()
        self.clock = clock

    # ---- versions --------------------------------------------------------------
    def register(self, m: SkillManifest, actor: str) -> SkillVersion:
        h = m.content_hash()
        with self._lock, self.db:
            row = self.db.execute("SELECT content_hash FROM skill_versions WHERE name=? AND version=?",
                                  (m.name, m.version)).fetchone()
            if row is not None:
                if row[0] != h:
                    raise ValueError(f"{m.name}@{m.version} already exists with different content; "
                                     "versions are immutable -- publish a new version")
                return self._get(m.name, m.version)
            latest = self._versions(m.name)
            if latest and semver(m.version) <= semver(latest[-1]):
                raise ValueError(f"{m.name}@{m.version} is not newer than {latest[-1]}")
            self.db.execute("INSERT INTO skill_versions VALUES (?,?,?,?,0,?)",
                            (m.name, m.version, m.model_dump_json(), h, self.clock()))
            self._event(m.name, m.version, "register", None, actor, None, None)
        return self._get(m.name, m.version)

    def _versions(self, name: str) -> list[str]:
        vs = [r[0] for r in self.db.execute("SELECT version FROM skill_versions WHERE name=?", (name,))]
        return sorted(vs, key=semver)

    def versions(self, name: str) -> list[str]:
        with self._lock:
            return self._versions(name)

    def _get(self, name: str, version: str) -> SkillVersion:
        row = self.db.execute("SELECT manifest, content_hash, yanked FROM skill_versions "
                              "WHERE name=? AND version=?", (name, version)).fetchone()
        if row is None:
            raise KeyError(f"no skill {name}@{version}")
        return SkillVersion(SkillManifest.model_validate_json(row[0]), row[1], bool(row[2]))

    def get(self, name: str, version: str) -> SkillVersion:
        with self._lock:
            return self._get(name, version)

    # ---- channels --------------------------------------------------------------
    def _event(self, name, version, action, channel, actor, verdicts, note) -> None:
        self.db.execute("INSERT INTO skill_events(name,version,action,channel,actor,verdicts,note,at) "
                        "VALUES (?,?,?,?,?,?,?,?)",
                        (name, version, action, channel, actor,
                         None if verdicts is None else json.dumps([v.model_dump() for v in verdicts]),
                         note, self.clock()))

    def _stack(self, name: str, channel: Channel) -> list[str]:
        st: list[str] = []
        for version, action in self.db.execute(
                "SELECT version, action FROM skill_events WHERE name=? AND channel=? ORDER BY id",
                (name, channel)):
            if action == "promote":
                st.append(version)
            elif action == "rollback" and st:
                st.pop()
        return st

    def channel(self, name: str, channel: Channel) -> str | None:
        with self._lock:
            st = self._stack(name, channel)
            return st[-1] if st else None

    def promote(self, name: str, version: str, channel: Channel, *, actor: str,
                gates: Iterable[Gate] = (), verdicts: Iterable[GateVerdict] = ()) -> list[GateVerdict]:
        """Move `version` onto `channel` if every gate passes. `verdicts` are
        results of gates that ran elsewhere (an Evolve gate decision). Stable
        needs at least one gate beyond `static` and the version to be the
        current candidate. Every attempt -- rejected ones too -- is logged."""
        sv = self.get(name, version)
        if sv.yanked:
            raise ValueError(f"{sv.ref} was yanked and cannot be promoted again")
        results = [g.evaluate(sv.manifest) for g in gates] + list(verdicts)
        problems = [v for v in results if not v.passed]
        if channel == "stable":
            if self.channel(name, "candidate") != version:
                problems.append(GateVerdict(gate="process", passed=False,
                                            reasons=[f"{sv.ref} is not the current candidate"]))
            if not any(v.gate != "static" for v in results):
                problems.append(GateVerdict(gate="process", passed=False,
                                            reasons=["stable needs a quality gate, not only static checks"]))
        with self._lock, self.db:
            self._event(name, version, "promote" if not problems else "promote_rejected", channel,
                        actor, results + [p for p in problems if p.gate == "process"], None)
        if problems:
            raise PromotionRejected(problems)
        return results

    def rollback(self, name: str, *, actor: str, reason: str, yank: bool = True,
                 channel: Channel = "stable") -> str | None:
        """Undo the last promotion on `channel`; returns the version now
        there. With `yank`, the rolled-back version can never be bound by a
        new run or promoted again (old runs pinned to it still replay)."""
        with self._lock, self.db:
            st = self._stack(name, channel)
            if not st:
                raise ValueError(f"nothing to roll back on {name} {channel}")
            bad = st[-1]
            self._event(name, bad, "rollback", channel, actor, None, reason)
            if yank:
                self.db.execute("UPDATE skill_versions SET yanked=1 WHERE name=? AND version=?", (name, bad))
                self._event(name, bad, "yank", None, actor, None, reason)
            return st[-2] if len(st) > 1 else None

    def history(self, name: str) -> list[dict]:
        with self._lock:
            rows = self.db.execute("SELECT version, action, channel, actor, verdicts, note FROM skill_events "
                                   "WHERE name=? ORDER BY id", (name,)).fetchall()
        return [{"version": r[0], "action": r[1], "channel": r[2], "actor": r[3],
                 "verdicts": json.loads(r[4]) if r[4] else None, "note": r[5]} for r in rows]

    # ---- resolution ------------------------------------------------------------
    def resolve(self, ref: str, tenant: str, *, allow_yanked: bool = False) -> SkillVersion:
        """`name` (stable), `name@candidate`, or `name@1.2.0`."""
        name, _, which = ref.partition("@")
        which = which or "stable"
        version = self.channel(name, which) if which in ("stable", "candidate") else which
        if version is None:
            raise KeyError(f"{name} has no {which} version")
        sv = self.get(name, version)
        if sv.manifest.tenants and tenant not in sv.manifest.tenants:
            raise KeyError(f"no skill {name} for tenant {tenant}")    # same answer as "does not exist"
        if sv.yanked and not allow_yanked:
            raise KeyError(f"{sv.ref} was yanked")
        return sv


# ---------------------------------------------------------------------------
# binding: what a principal gets for one run
# ---------------------------------------------------------------------------
@dataclass
class SkillBinding:
    offered: dict[str, SkillVersion] = field(default_factory=dict)
    excluded: dict[str, str] = field(default_factory=dict)       # requested ref -> why

    def pins(self) -> list[str]:
        return [sv.ref for sv in self.offered.values()]

    def index_text(self) -> str:
        if not self.offered:
            return ""
        lines = [f"- {n}: {sv.manifest.description}" for n, sv in self.offered.items()]
        return (f"{SKILLS_HEADER}\n\nBefore handling a request one of these covers, call "
                f"{LOAD_SKILL}(name) and follow its steps.\n\n" + "\n".join(lines))


def skill_capability(name: str) -> CapabilityDescriptor:
    return CapabilityDescriptor(id=f"skill.{name}", name=f"skill:{name}", source="local", risk="read")


def bind_skills(store: SkillStore, refs: Iterable[str], principal: Principal, registry: ToolRegistry,
                *, for_replay: bool = False) -> SkillBinding:
    """Resolve `refs` for `principal`, keep the skills it may load, and
    install a `load_skill` tool that serves exactly those versions. Uses the
    registry's AccessControl (required). For replay, pass the run's recorded
    pins with for_replay=True so yanked versions still resolve."""
    access = getattr(registry, "access", None)
    if access is None:
        raise ValueError("bind_skills needs a registry with access control (registry.access)")
    b = SkillBinding()
    for ref in refs:
        try:
            sv = store.resolve(ref, principal.tenant_id, allow_yanked=for_replay)
        except KeyError as e:
            b.excluded[ref] = str(e.args[0])
            continue
        m = sv.manifest
        d = access.engine.decide(AuthzContext(principal, skill_capability(m.name), {}, False))
        if d.effect != "allow":
            b.excluded[ref] = f"policy does not allow skill.{m.name} ({d.reason})"
            continue
        missing = []
        for cap_id in m.requires:
            cap = access.catalog.by_id(cap_id)
            if cap is None or not access.engine.visible(principal, cap):
                missing.append(cap_id)
        if missing:
            b.excluded[ref] = f"principal cannot use required {', '.join(missing)}"
            continue
        b.offered[m.name] = sv
    _install_loader(b, registry, access.catalog)
    return b


def _install_loader(b: SkillBinding, registry: ToolRegistry, catalog: CapabilityCatalog) -> None:
    rendered = {n: sv.manifest.as_skill(catalog).render() for n, sv in b.offered.items()}
    names = sorted(rendered)

    class _Args(BaseModel):
        name: str

    def fn(_env, a: _Args) -> str:
        if a.name.strip() not in rendered:
            raise ToolError(f"unknown skill {a.name!r}; available: {', '.join(names)}")
        return rendered[a.name.strip()]

    registry.register(Tool(
        name=LOAD_SKILL, args_model=_Args, fn=fn, mutating=False, raw_text_result=True,
        description=("Load the step-by-step procedure of one of the skills listed in your instructions. "
                     "Call it before handling a request the skill covers, then follow the steps."),
        input_schema={"type": "object", "properties": {"name": {
            "type": "string", "enum": names, "description": "The skill's name."}}, "required": ["name"]}))
    if catalog.resolve(LOAD_SKILL) is None:
        catalog.add(CapabilityDescriptor(id=LOAD_SKILL_CAPABILITY, name=LOAD_SKILL, risk="read"))


def principal_for_skill(principal: Principal, sv: SkillVersion) -> Principal:
    """Principal for a sub-agent dedicated to one skill: the parent's grants
    intersected with what the skill declares (scopes = requires)."""
    return principal.attenuate(f"{principal.agent_id}.{sv.manifest.name}", scopes=tuple(sv.manifest.requires))
