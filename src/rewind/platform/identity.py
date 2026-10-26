"""Who a run acts as.

A `Principal` is the identity every access decision is made for: which tenant
and end user the agent is working on behalf of, which agent it is, the roles
it holds, and -- optionally -- an explicit allowlist of capabilities
(`scopes`) that narrows whatever its roles would otherwise grant.

The principal travels in `AgentConfig.principal`, so it is written into the
trace's `run_start` like the rest of the config: replay and fork rebuild the
exact same identity without any extra plumbing.

Sub-agents get their principal from `attenuate`, which can only take
permissions away -- same tenant, same user, a subset of the parent's roles,
and scopes the parent's scopes already cover. A child can never be handed
more than its parent had.
"""
from __future__ import annotations

import re
from fnmatch import fnmatchcase

from pydantic import BaseModel, ConfigDict, field_validator

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,63}")
_WILDCARDS = set("*?[")


class AttenuationError(ValueError):
    """A child principal asked for something its parent does not have."""


def _check_id(v: str) -> str:
    if not _ID.fullmatch(v):
        raise ValueError(f"invalid id {v!r}")
    return v


class Principal(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant_id: str
    user_id: str
    agent_id: str
    roles: tuple[str, ...] = ()
    # None: no narrowing beyond what the roles grant. Otherwise a list of
    # capability-id globs; anything outside them is denied before any rule runs.
    scopes: tuple[str, ...] | None = None
    parent_agent_id: str | None = None

    @field_validator("tenant_id", "user_id", "agent_id")
    @classmethod
    def _ids(cls, v: str) -> str:
        return _check_id(v)

    @field_validator("roles")
    @classmethod
    def _roles(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted({_check_id(r) for r in v}))

    def in_scope(self, capability_id: str) -> bool:
        return self.scopes is None or any(fnmatchcase(capability_id, s) for s in self.scopes)

    def template_vars(self) -> dict[str, str]:
        """Values a policy condition may substitute, e.g. a path prefix
        `/workspaces/${tenant_id}/`. Deliberately only identity fields."""
        return {"tenant_id": self.tenant_id, "user_id": self.user_id, "agent_id": self.agent_id}

    def attenuate(self, agent_id: str, *, roles: tuple[str, ...] | None = None,
                  scopes: tuple[str, ...] | None = None) -> Principal:
        """Principal for a sub-agent spawned by this one. Raises
        AttenuationError if the child would gain anything."""
        child_roles = self.roles if roles is None else tuple(roles)
        extra = set(child_roles) - set(self.roles)
        if extra:
            raise AttenuationError(f"child roles {sorted(extra)} not held by parent {self.agent_id}")
        if scopes is None:
            child_scopes = self.scopes
        else:
            if self.scopes is not None:
                for s in scopes:
                    if not any(_covers(p, s) for p in self.scopes):
                        raise AttenuationError(f"child scope {s!r} is not covered by the parent's scopes")
            child_scopes = tuple(scopes)
        return Principal(tenant_id=self.tenant_id, user_id=self.user_id, agent_id=agent_id,
                         roles=child_roles, scopes=child_scopes, parent_agent_id=self.agent_id)


def _covers(parent: str, child: str) -> bool:
    """Conservative: True only when every id `child` can match is matched by
    `parent`. A concrete child is checked directly; a wildcard child is
    accepted only under a parent of the form `<literal prefix>*`."""
    if not _WILDCARDS & set(child):
        return fnmatchcase(child, parent)
    if parent == child:
        return True
    if parent.endswith("*") and not _WILDCARDS & set(parent[:-1]):
        return child.startswith(parent[:-1])
    return False
