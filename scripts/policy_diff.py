"""Judge recorded runs under a new access policy, before it ships (no model calls).

    python scripts/policy_diff.py runs/traces/ --policy data/access_policies/new.json \\
        --registry mypkg.tools:build_registry --fail-on loosened

Prints the report (rewind.platform.whatif) and exits 1 when the policy change
does something --fail-on forbids -- the form a CI check on a policy pull
request takes.

  traces          .jsonl trace files, or directories of them
  --policy        the new rules (JSON list, as for PolicyEngine)
  --registry      module:function returning the ToolRegistry whose tools the
                  traces called (the capability catalog is built from it)
  --catalog       JSON {tool: {risk, untrusted_output}} overrides for the catalog
  --as            tenant:user:agent[:role,role] to judge runs recorded without
                  a principal (e.g. before access control was switched on)
  --json          also write the full report (every changed call) here
  --fail-on       none | loosened | tightened | any   (default: none)
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from rewind.platform import AccessControl, CapabilityCatalog, PolicyEngine, Principal
from rewind.platform.whatif import what_if


def _load_registry(spec: str):
    mod, _, fn = spec.partition(":")
    return getattr(importlib.import_module(mod), fn or "build_registry")()


def _traces(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        out += sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    return out


def _as(spec: str | None):
    if not spec:
        return None
    parts = spec.split(":")
    if len(parts) not in (3, 4):
        raise SystemExit("--as is tenant:user:agent[:role,role]")
    p = Principal(tenant_id=parts[0], user_id=parts[1], agent_id=parts[2],
                  roles=tuple(r for r in (parts[3].split(",") if len(parts) == 4 else []) if r))

    def principal_for(start: dict):
        rec = (start.get("cfg") or {}).get("principal")
        return Principal.model_validate(rec) if rec else p
    return principal_for


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--registry", default="rewind.env:build_registry")
    ap.add_argument("--catalog", default=None)
    ap.add_argument("--as", dest="as_", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--fail-on", choices=["none", "loosened", "tightened", "any"], default="none")
    a = ap.parse_args(argv)

    registry = _load_registry(a.registry)
    overrides = json.loads(Path(a.catalog).read_text()) if a.catalog else None
    access = AccessControl(PolicyEngine.from_json(Path(a.policy).read_text()),
                           CapabilityCatalog.from_registry(registry, overrides))
    files = _traces(a.traces)
    if not files:
        raise SystemExit("no traces found")
    rep = what_if(files, access, principal_for=_as(a.as_))
    print(rep.render_text())
    if a.json:
        Path(a.json).write_text(rep.to_json())
    s = rep.summary()["changes"]
    bad = {"none": 0, "loosened": s["loosened"], "tightened": s["tightened"],
           "any": s["loosened"] + s["tightened"] + s["reclassified"]}[a.fail_on]
    if bad:
        print(f"\nFAIL: {bad} call(s) change under --fail-on {a.fail_on}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
