"""Capabilities: one uniform description of everything an agent can call.

The policy engine never looks at a `Tool` directly. It looks at a
`CapabilityDescriptor`: a stable id (`local.cancel_order`, later
`mcp.<server>.<tool>`), where it comes from, how risky it is, whether its
output is untrusted (web pages, third-party server replies), and a hash of
the schema the model is shown -- so local tools and MCP tools are governed by
exactly the same rules.

Risk levels, least to most dangerous:

  read         reads state, no side effect
  write        changes state inside the tenant's own data
  external     sends something outside the system (email, HTTP POST, ...)
  destructive  irreversible or wide-blast-radius changes

The default for a local tool is `write` if it is mutating, else `read`;
anything else is an explicit override at catalog build time.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from ..tools import ToolRegistry
from ..util import sha

Risk = Literal["read", "write", "external", "destructive"]
RISK_ORDER: dict[str, int] = {"read": 0, "write": 1, "external": 2, "destructive": 3}
Source = Literal["local", "mcp"]


class CapabilityDescriptor(BaseModel):
    id: str                      # what policies match on
    name: str                    # what the model calls
    source: Source = "local"
    risk: Risk = "read"
    untrusted_output: bool = False
    schema_hash: str = ""
    description: str = ""


class CapabilityCatalog:
    """name the model uses -> descriptor."""

    def __init__(self, caps: list[CapabilityDescriptor] | None = None) -> None:
        self._by_name: dict[str, CapabilityDescriptor] = {}
        for c in caps or []:
            self.add(c)

    def add(self, cap: CapabilityDescriptor) -> None:
        if cap.name in self._by_name and self._by_name[cap.name].id != cap.id:
            raise ValueError(f"tool name {cap.name!r} already maps to {self._by_name[cap.name].id}")
        self._by_name[cap.name] = cap

    def resolve(self, name: str) -> CapabilityDescriptor | None:
        return self._by_name.get(name)

    def by_id(self, cap_id: str) -> CapabilityDescriptor | None:
        return next((c for c in self._by_name.values() if c.id == cap_id), None)

    def __iter__(self):
        return iter(self._by_name.values())

    def __len__(self) -> int:
        return len(self._by_name)

    @classmethod
    def from_registry(cls, registry: ToolRegistry,
                      overrides: dict[str, dict[str, Any]] | None = None) -> CapabilityCatalog:
        """Describe every local tool in `registry`. `overrides` maps a tool
        name to fields to set (`risk`, `untrusted_output`, ...)."""
        overrides = overrides or {}
        unknown = set(overrides) - {t.name for t in registry._tools.values()}
        if unknown:
            raise ValueError(f"overrides for unknown tools: {sorted(unknown)}")
        cat = cls()
        for schema in registry.schemas():
            name = schema["name"]
            fields: dict[str, Any] = {
                "id": f"local.{name}", "name": name, "source": "local",
                "risk": "write" if registry.is_mutating(name) else "read",
                "schema_hash": sha(schema), "description": schema["description"],
            }
            fields.update(overrides.get(name, {}))
            cat.add(CapabilityDescriptor(**fields))
        return cat
