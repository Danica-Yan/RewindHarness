"""MCP tool definitions an admin has approved, and how risky a tool is.

Nothing here talks to a server; `mcp_gateway.py` does. This is the policy
half, kept free of the MCP SDK so it is testable on its own.

Pinning. An MCP server describes its own tools, and it can change that
description at any time -- after you approved it ("rug pull"), possibly with
instructions aimed at the model hidden in the text. So what gets approved is
a fingerprint of the exact definition (name, description, input schema,
annotations). On every sync each listed tool is one of:

  approved   fingerprint matches the pin            -> exposed
  new        never approved                         -> not exposed
  changed    pinned, but the definition differs     -> quarantined
  removed    pinned, no longer listed               -> not exposed

Risk. MCP annotations (readOnlyHint, destructiveHint, openWorldHint) are
claims made by the server. They may RAISE a tool's risk above the admin's
floor for that server, never lower it -- a server that lies "read-only" on a
write tool gains nothing. Only an explicit per-tool admin override can set a
risk below the floor.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Literal

from ..util import sha
from .capability import RISK_ORDER, Risk

ToolStatus = Literal["approved", "new", "changed", "removed"]


def fingerprint(name: str, description: str | None, input_schema: dict | None,
                annotations: dict | None) -> str:
    return sha({"name": name, "description": description or "", "input_schema": input_schema or {},
                "annotations": {k: v for k, v in (annotations or {}).items() if v is not None}})[7:31]


def hinted_risk(annotations: dict | None) -> Risk:
    a = annotations or {}
    if a.get("destructive_hint"):
        return "destructive"
    if a.get("read_only_hint"):
        return "read"
    if a.get("open_world_hint"):
        return "external"
    return "write"


def derive_risk(annotations: dict | None, floor: Risk, override: Risk | None = None) -> Risk:
    if override is not None:
        return override
    hint = hinted_risk(annotations)
    return hint if RISK_ORDER[hint] > RISK_ORDER[floor] else floor


def reconcile(pins: dict[str, str], listed: dict[str, str]) -> dict[str, ToolStatus]:
    """pins / listed: tool name -> fingerprint."""
    out: dict[str, ToolStatus] = {}
    for name, fp in listed.items():
        if name not in pins:
            out[name] = "new"
        else:
            out[name] = "approved" if pins[name] == fp else "changed"
    for name in pins:
        if name not in listed:
            out[name] = "removed"
    return out


class PinStore:
    """server id -> {tool name -> approved fingerprint}. In memory, or backed
    by a JSON file when `path` is given (written on every change)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._pins: dict[str, dict[str, dict[str, Any]]] = {}
        if self.path and self.path.exists():
            self._pins = json.loads(self.path.read_text())

    def pins(self, server: str) -> dict[str, str]:
        with self._lock:
            return {t: p["fingerprint"] for t, p in self._pins.get(server, {}).items()}

    def approve(self, server: str, tool: str, fp: str, approver: str) -> None:
        with self._lock:
            self._pins.setdefault(server, {})[tool] = {"fingerprint": fp, "approved_by": approver}
            self._save()

    def revoke(self, server: str, tool: str) -> None:
        with self._lock:
            self._pins.get(server, {}).pop(tool, None)
            self._save()

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._pins, indent=1, sort_keys=True))
            tmp.replace(self.path)
