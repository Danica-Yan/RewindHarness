"""AccessControl: the enforcement point the agent loop calls.

The policy engine (`authz.PolicyEngine`) only decides. This class turns a
model's tool call into the engine's inputs and the engine's answer into what
the loop does:

  resolve   tool name -> CapabilityDescriptor. A name the catalog does not
            know is never executed (fail closed), and a capability the
            principal cannot see is answered exactly like an unknown tool,
            so probing does not reveal what exists.
  taint     derived from the conversation: has a capability marked
            `untrusted_output` already returned successfully?
  approve   a require_approval decision becomes allow only if the approval
            store holds a grant for this exact call (run, action, tool, args).

It is attached to the ToolRegistry (`registry.access`), next to
`registry.policy`, so every path that builds an AgentRunner -- run, worker
resume, fork, replay, the Evolve gate -- enforces the same thing without
being told to.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from ..routing import ESCALATE_TOOL
from ..util import sha
from .authz import AuthzContext, Decision, PolicyEngine
from .capability import CapabilityCatalog, CapabilityDescriptor
from .identity import Principal

Verdict = Literal["allow", "deny", "hidden", "unknown", "pending_approval", "require_approval"]


# ---------------------------------------------------------------------------
# approvals
# ---------------------------------------------------------------------------
@dataclass
class ApprovalRequest:
    key: str
    run_id: str
    action_id: str
    capability: str
    args: dict
    principal: dict
    status: Literal["pending", "granted", "rejected"] = "pending"
    decided_by: str | None = None


class ApprovalStore(Protocol):
    def get(self, key: str) -> ApprovalRequest | None: ...
    def request(self, req: ApprovalRequest) -> ApprovalRequest: ...


class InMemoryApprovalStore:
    """Process-local store, enough for tests and single-process demos. A
    multi-worker deployment needs the same three methods on shared storage."""

    def __init__(self) -> None:
        self._reqs: dict[str, ApprovalRequest] = {}

    def get(self, key: str) -> ApprovalRequest | None:
        return self._reqs.get(key)

    def request(self, req: ApprovalRequest) -> ApprovalRequest:
        return self._reqs.setdefault(req.key, req)

    def decide(self, key: str, approver: str, grant: bool) -> ApprovalRequest:
        req = self._reqs[key]
        if req.status != "pending":
            raise ValueError(f"approval {key} already {req.status}")
        req.status = "granted" if grant else "rejected"
        req.decided_by = approver
        return req

    def pending(self) -> list[ApprovalRequest]:
        return [r for r in self._reqs.values() if r.status == "pending"]


def approval_key(run_id: str, action_id: str, name: str, args: dict) -> str:
    """Identity of one concrete call. A resumed run replays the same recorded
    action with the same action_id, so its key -- and its grant -- match;
    any change to the arguments needs a fresh approval."""
    return sha({"run": run_id, "action": action_id, "tool": name, "args": args})[7:31]


# ---------------------------------------------------------------------------
# the enforcement point
# ---------------------------------------------------------------------------
@dataclass
class CallVerdict:
    verdict: Verdict
    capability: CapabilityDescriptor | None
    decision: Decision | None
    approval_key: str | None = None
    message: str = ""                         # what the model is told (deny/hidden/unknown)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_event(self) -> dict[str, Any]:
        out: dict[str, Any] = {"verdict": self.verdict}
        if self.decision is not None:
            out.update(self.decision.to_event())
        elif self.capability is not None:
            out["capability"] = self.capability.id
        if self.approval_key:
            out["approval_key"] = self.approval_key
        out.update(self.extra)
        return out


class AccessControl:
    """`meta_tools`: harness tools that grant nothing on their own and are
    always visible and allowed -- by default the tool router's escalation
    tool, which only switches back to the toolset the principal could
    already see (routing filters what access control allows, never adds)."""

    def __init__(self, engine: PolicyEngine, catalog: CapabilityCatalog,
                 approvals: ApprovalStore | None = None,
                 meta_tools: frozenset[str] = frozenset({ESCALATE_TOOL})) -> None:
        self.engine = engine
        self.catalog = catalog
        self.approvals = approvals if approvals is not None else InMemoryApprovalStore()
        self.meta_tools = frozenset(meta_tools)

    @property
    def version(self) -> str:
        return self.engine.version

    def visible_names(self, principal: Principal) -> set[str]:
        return {c.name for c in self.catalog if self.engine.visible(principal, c)} | set(self.meta_tools)

    def filter_schemas(self, principal: Principal, schemas: list[dict]) -> list[dict]:
        names = self.visible_names(principal)
        return [s for s in schemas if s["name"] in names]

    def tainted(self, history: list[dict]) -> bool:
        """True once any untrusted-output capability has returned a
        successful result earlier in this conversation."""
        names = {c["id"]: c.get("name", "") for m in history if m.get("role") == "assistant"
                 for c in m.get("tool_calls") or []}
        for m in history:
            if m.get("role") != "tool" or m.get("is_error"):
                continue
            cap = self.catalog.resolve(names.get(m.get("tool_call_id"), ""))
            if cap is not None and cap.untrusted_output:
                return True
        return False

    def evaluate(self, principal: Principal, name: str, args: dict, history: list[dict]) -> CallVerdict:
        """The policy's answer for one call, with no side effects: never reads
        or writes the approval store. Verdict is unknown / hidden / allow /
        deny / require_approval. `check` builds on it; offline re-evaluation
        of recorded traces (platform/whatif.py) uses it directly."""
        if name in self.meta_tools:
            return CallVerdict("allow", None, None, extra={"meta": True})
        cap = self.catalog.resolve(name)
        tainted = self.tainted(history)
        if cap is None or not self.engine.visible(principal, cap):
            shown = ", ".join(sorted(self.visible_names(principal)))
            return CallVerdict("unknown" if cap is None else "hidden", cap,
                               None if cap is None else self.engine.decide(
                                   AuthzContext(principal, cap, args or {}, tainted)),
                               message=f"unknown tool '{name}'. Available tools: {shown}")
        d = self.engine.decide(AuthzContext(principal, cap, args or {}, tainted))
        extra = {"tainted": tainted}
        if d.effect == "deny":
            return CallVerdict("deny", cap, d, extra=extra,
                               message=f"permission denied: this agent may not call {name}"
                                       + (" with these arguments" if d.reason.startswith("rule:") else ""))
        return CallVerdict(d.effect, cap, d, extra=extra)       # allow / require_approval

    def check(self, principal: Principal, run_id: str, action_id: str, name: str,
              args: dict, history: list[dict]) -> CallVerdict:
        """`evaluate`, plus resolving require_approval against the approval
        store (and filing a request when there is none yet)."""
        v = self.evaluate(principal, name, args, history)
        if v.verdict != "require_approval":
            return v
        cap, d = v.capability, v.decision
        key = approval_key(run_id, action_id, name, args or {})
        req = self.approvals.get(key)
        if req is not None and req.status == "granted":
            return CallVerdict("allow", cap, d, key, extra={**v.extra, "approved_by": req.decided_by})
        if req is not None and req.status == "rejected":
            return CallVerdict("deny", cap, d, key, extra={**v.extra, "rejected_by": req.decided_by},
                               message=f"a human reviewer rejected this call to {name}; "
                                       "do not retry it, tell the user")
        self.approvals.request(ApprovalRequest(key, run_id, action_id, cap.id, dict(args or {}),
                                               principal.model_dump(mode="json")))
        return CallVerdict("pending_approval", cap, d, key, extra=v.extra)
