"""Agent platform: identity, capabilities and access control for tool calls.

P0 of the platform plan -- one enforcement point in front of every tool call:

  identity.Principal         who the run acts as (tenant, user, agent, roles,
                             scopes); sub-agents only ever get less
  capability.Catalog         uniform description of what can be called
                             (local tools now, MCP tools later)
  authz.PolicyEngine         default-deny decisions: allow / deny /
                             require_approval, deny overrides
  access.AccessControl       the loop-facing enforcement point: hides what a
                             principal may not see, answers hidden tools like
                             unknown ones, tracks untrusted content, resolves
                             approvals
  quota.QuotaMeter           how much: per tenant/user/agent/run limits,
                             charged all-or-nothing, exactly once per action
                             (in-memory and Postgres stores)

Usage: attach `AccessControl` to the registry and put the principal in the
agent config.

    registry.access = AccessControl(PolicyEngine(rules), CapabilityCatalog.from_registry(registry))
    registry.quota = QuotaMeter(quota_rules, InMemoryQuotaStore())    # optional
    cfg = AgentConfig(..., principal=principal.model_dump(mode="json"))

Every decision is written to the trace as an `authz` event (with the policy
version), so replay re-checks enforcement along with everything else.
"""
from .access import (AccessControl, ApprovalRequest, ApprovalStore, CallVerdict,
                     InMemoryApprovalStore, approval_key)
from .authz import CONDITIONS, AuthzContext, Cond, Decision, PolicyEngine, Rule
from .capability import CapabilityCatalog, CapabilityDescriptor
from .identity import AttenuationError, Principal
from .quota import (Charge, InMemoryQuotaStore, PgQuotaStore, QuotaMeter, QuotaResult, QuotaRule,
                    RecordedQuota)

__all__ = [
    "AccessControl", "ApprovalRequest", "ApprovalStore", "AttenuationError", "AuthzContext",
    "CONDITIONS", "CallVerdict", "CapabilityCatalog", "CapabilityDescriptor", "Cond", "Decision",
    "Charge", "InMemoryApprovalStore", "InMemoryQuotaStore", "PgQuotaStore", "PolicyEngine", "Principal",
    "QuotaMeter", "QuotaResult", "QuotaRule", "RecordedQuota", "Rule", "approval_key",
]
