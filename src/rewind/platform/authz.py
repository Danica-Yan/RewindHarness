"""Policy engine: the single decision point for "may this principal call this
capability with these arguments, right now?"

Not to be confused with `rewind.policy`, which enforces *business* rules of a
domain (confirm before a write, act only for the authenticated customer).
This module is access control: identity, tenancy, roles, scopes, risk.

Semantics (evaluated in this order):

  1. Scope: if the principal carries `scopes` and the capability id is not
     covered, deny. Nothing below can widen it.
  2. Collect every rule that matches -- statically (tenant, role, agent,
     capability glob, risk level) AND all of its `when` conditions hold.
  3. Any matching deny -> deny (deny overrides).
     No matching allow -> deny (default deny).
     Else any matching require_approval -> require_approval.
     Else allow.

Only `allow` grants. `require_approval` is an obligation layered on top of a
grant, never a grant itself: otherwise a broad escalation rule ("anything
external needs approval once the context is tainted") would hand every
principal every external capability, one human click away.

Rules and conditions are data. Conditions come from a small fixed library
(like `rewind.policy`'s), each a pure function of the principal, the
capability, the call's arguments and whether the conversation has already
taken in untrusted content -- so a decision depends on nothing that replay
cannot reproduce, and the same engine can re-judge recorded traces offline.
"""
from __future__ import annotations

import json
import posixpath
import re
import string
from collections.abc import Callable
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

from ..util import sha
from .capability import CapabilityDescriptor, Risk
from .identity import Principal

Effect = Literal["allow", "deny", "require_approval"]
TEMPLATE_VARS = {"tenant_id", "user_id", "agent_id"}


@dataclass
class AuthzContext:
    principal: Principal
    capability: CapabilityDescriptor
    args: dict
    tainted: bool               # an untrusted-output capability already returned in this run


# ---------------------------------------------------------------------------
# condition library: (ctx, params) -> does the condition hold?
# ---------------------------------------------------------------------------
CondFn = Callable[[AuthzContext, dict], bool]
CONDITIONS: dict[str, CondFn] = {}
COND_PARAMS: dict[str, set[str]] = {}


def condition(name: str, params: set[str]):
    def deco(fn: CondFn) -> CondFn:
        CONDITIONS[name] = fn
        COND_PARAMS[name] = params
        return fn
    return deco


def _render(template: str, p: Principal) -> str:
    return string.Template(template).substitute(p.template_vars())


@condition("arg_prefix", {"arg", "prefix"})
def _arg_prefix(ctx: AuthzContext, p: dict) -> bool:
    """A path argument stays under `prefix` after normalisation, so
    `/ws/t1/../t2/x` does not pass as being under `/ws/t1/`. The prefix may
    use ${tenant_id}, ${user_id}, ${agent_id}."""
    v = ctx.args.get(p["arg"])
    if not isinstance(v, str) or not v:
        return False
    root = posixpath.normpath(_render(p["prefix"], ctx.principal)).rstrip("/") or "/"
    path = posixpath.normpath(v)
    return path == root or path.startswith(root if root == "/" else root + "/")


@condition("arg_in", {"arg", "values"})
def _arg_in(ctx: AuthzContext, p: dict) -> bool:
    return ctx.args.get(p["arg"]) in p["values"]


@condition("arg_host_in", {"arg", "hosts"})
def _arg_host_in(ctx: AuthzContext, p: dict) -> bool:
    """An http(s) URL argument whose host is one of `hosts` or a subdomain of one."""
    v = ctx.args.get(p["arg"])
    if not isinstance(v, str):
        return False
    u = urlparse(v)
    host = (u.hostname or "").lower()
    if u.scheme not in ("http", "https") or not host:
        return False
    return any(host == h or host.endswith("." + h) for h in (h.lower() for h in p["hosts"]))


@condition("context_tainted", set())
def _tainted(ctx: AuthzContext, _p: dict) -> bool:
    return ctx.tainted


class Cond(BaseModel):
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)
    negate: bool = False

    @model_validator(mode="after")
    def _known(self) -> Cond:
        if self.kind not in CONDITIONS:
            raise ValueError(f"unknown condition {self.kind!r}; known: {sorted(CONDITIONS)}")
        if set(self.params) != COND_PARAMS[self.kind]:
            raise ValueError(f"{self.kind} takes exactly {sorted(COND_PARAMS[self.kind])}, "
                             f"got {sorted(self.params)}")
        if "prefix" in self.params:
            names = {m[1] or m[2] for m in string.Template.pattern.findall(self.params["prefix"])
                     if m[1] or m[2]}
            if names - TEMPLATE_VARS:
                raise ValueError(f"unknown template variables {sorted(names - TEMPLATE_VARS)}")
        return self

    def holds(self, ctx: AuthzContext) -> bool:
        return CONDITIONS[self.kind](ctx, self.params) != self.negate


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------
class Rule(BaseModel):
    id: str
    effect: Effect
    capabilities: list[str]                 # globs over capability ids
    tenants: list[str] = Field(default_factory=list)   # empty = any tenant
    roles: list[str] = Field(default_factory=list)     # any-of; empty = any principal
    agents: list[str] = Field(default_factory=list)    # globs over agent ids; empty = any
    risk: list[Risk] = Field(default_factory=list)     # empty = any risk level
    when: list[Cond] = Field(default_factory=list)     # all must hold
    description: str = ""

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not re.fullmatch(r"[a-z][a-z0-9_.\-]{2,63}", v):
            raise ValueError("rule id must be lowercase, 3-64 chars")
        return v

    @field_validator("capabilities")
    @classmethod
    def _caps(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("a rule must name at least one capability glob")
        return v

    def matches_statically(self, p: Principal, cap: CapabilityDescriptor) -> bool:
        return ((not self.tenants or p.tenant_id in self.tenants)
                and (not self.roles or bool(set(self.roles) & set(p.roles)))
                and (not self.agents or any(fnmatchcase(p.agent_id, a) for a in self.agents))
                and (not self.risk or cap.risk in self.risk)
                and any(fnmatchcase(cap.id, c) for c in self.capabilities))

    def applies(self, ctx: AuthzContext) -> bool:
        return (self.matches_statically(ctx.principal, ctx.capability)
                and all(c.holds(ctx) for c in self.when))


@dataclass
class Decision:
    effect: Effect
    capability_id: str
    reason: str                              # "default_deny", "outside_scope", "rule:<id>"
    matched: list[str] = field(default_factory=list)
    policy_version: str = ""

    def to_event(self) -> dict[str, Any]:
        return {"effect": self.effect, "capability": self.capability_id, "reason": self.reason,
                "matched": list(self.matched), "policy_version": self.policy_version}


class PolicyEngine:
    def __init__(self, rules: list[Rule]) -> None:
        ids = [r.id for r in rules]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate rule ids: {sorted(dupes)}")
        self.rules = list(rules)
        self.version = sha([r.model_dump() for r in self.rules])[7:23]

    @classmethod
    def from_json(cls, text: str) -> PolicyEngine:
        return cls([Rule.model_validate(r) for r in json.loads(text)])

    def decide(self, ctx: AuthzContext) -> Decision:
        cap_id = ctx.capability.id
        if not ctx.principal.in_scope(cap_id):
            return Decision("deny", cap_id, "outside_scope", [], self.version)
        hits = [r for r in self.rules if r.applies(ctx)]
        matched = [r.id for r in hits]
        first = {e: next((r for r in hits if r.effect == e), None)
                 for e in ("deny", "allow", "require_approval")}
        if first["deny"] is not None:
            return Decision("deny", cap_id, f"rule:{first['deny'].id}", matched, self.version)
        if first["allow"] is None:
            return Decision("deny", cap_id, "default_deny", matched, self.version)
        if first["require_approval"] is not None:
            return Decision("require_approval", cap_id, f"rule:{first['require_approval'].id}",
                            matched, self.version)
        return Decision("allow", cap_id, f"rule:{first['allow'].id}", matched, self.version)

    def visible(self, principal: Principal, cap: CapabilityDescriptor) -> bool:
        """Should the model be shown this capability at all? Yes if some
        allow rule could apply to it (conditions aside) and no unconditional
        deny does. Hiding a tool is least privilege for the
        context window; it is NOT what stops a call -- `decide` is."""
        if not principal.in_scope(cap.id):
            return False
        static = [r for r in self.rules if r.matches_statically(principal, cap)]
        if any(r.effect == "deny" and not r.when for r in static):
            return False
        return any(r.effect == "allow" for r in static)
